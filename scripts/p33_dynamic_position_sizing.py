"""P33: 動態倉位方案設計與回測.

設計思路:
  P32 發現真正的風險在「廣度40-50%灰色地帶」，過濾未觸發但持倉已暴露風險。
  方案: 根據廣度動態調整最大持倉數。

方案:
  A: 現行 (固定5倉, 廣度<40%阻擋)
  B: 三階梯 (>50%→5倉, 40-50%→3倉, <40%→0)
  C: 四階梯 (>60%→5倉, 50-60%→4倉, 40-50%→2倉, <40%→0)
  D: 線性縮放 (positions = 5 * (breadth-0.40)/0.20, cap 5)
  E: B + 單股止損-15%
  F: C + 單股止損-15%
  G: 保守版 (>55%→5倉, 45-55%→3倉, <45%→0)
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
import finlab
from finlab import data

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p33_dynamic_sizing")
OUT.mkdir(parents=True, exist_ok=True)

MIN_TURNOVER = 20_000_000
GAP_LOW = -0.08
GAP_HIGH = 0.04
PANIC_THRESHOLD = 400
PANIC_COOLDOWN = 9
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")
COST_RATE = 0.001425 + 0.003 + 0.003
STOP_LOSS = -0.15

def get_max_positions(breadth_val, scheme):
    """Return max allowed positions given breadth and scheme."""
    if np.isnan(breadth_val):
        return 5
    if scheme == "A":
        return 5 if breadth_val >= 0.40 else 0
    elif scheme == "B":
        if breadth_val >= 0.50: return 5
        elif breadth_val >= 0.40: return 3
        else: return 0
    elif scheme == "C":
        if breadth_val >= 0.60: return 5
        elif breadth_val >= 0.50: return 4
        elif breadth_val >= 0.40: return 2
        else: return 0
    elif scheme == "D":
        if breadth_val < 0.40: return 0
        elif breadth_val > 0.60: return 5
        else: return max(1, int(5 * (breadth_val - 0.40) / 0.20))
    elif scheme == "E":
        if breadth_val >= 0.50: return 5
        elif breadth_val >= 0.40: return 3
        else: return 0
    elif scheme == "F":
        if breadth_val >= 0.60: return 5
        elif breadth_val >= 0.50: return 4
        elif breadth_val >= 0.40: return 2
        else: return 0
    elif scheme == "G":
        if breadth_val >= 0.55: return 5
        elif breadth_val >= 0.45: return 3
        else: return 0
    return 5


def simulate(trades, cal, close, open_p, n_cal, scheme, use_stop_loss=False):
    """Simulate portfolio with dynamic position sizing."""
    n = n_cal
    daily_pnl = np.zeros(n)
    position_count = np.zeros(n, dtype=int)

    # Sort trades by entry
    trades_sorted = sorted(trades, key=lambda t: t["entry_idx"])

    # Track active positions per slot
    # Each slot: list of (exit_idx, stock_id, entry_price, allocated_weight)
    active_positions = []  # list of dicts

    for t in trades_sorted:
        ei = t["entry_idx"]
        xi = t["exit_idx"]
        sym = t["stock_id"]

        # Determine max positions at entry time
        breadth_at_entry = t["breadth_at_entry"]
        max_pos = get_max_positions(breadth_at_entry, scheme)

        if max_pos == 0:
            continue

        # Remove expired positions
        active_positions = [p for p in active_positions if p["exit_idx"] > ei]

        # Check if we have room
        if len(active_positions) >= max_pos:
            continue

        # Allocate weight = 1 / max_pos (not 1/current_count, to avoid over-concentration)
        weight = 1.0 / max_pos

        # Calculate daily P&L for this position
        try:
            entry_price = open_p.iloc[ei][sym]
        except (IndexError, KeyError):
            continue
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        actual_exit = xi
        if use_stop_loss:
            # Check for stop loss during holding period
            for d in range(ei + 1, xi + 1):
                try:
                    px = close.iloc[d][sym]
                    if np.isnan(px):
                        continue
                    unrealized = px / entry_price - 1
                    if unrealized <= STOP_LOSS:
                        actual_exit = d
                        break
                except (IndexError, KeyError):
                    continue

        pos_info = {
            "exit_idx": actual_exit,
            "stock_id": sym,
            "entry_price": entry_price,
            "weight": weight,
            "entry_idx": ei,
        }
        active_positions.append(pos_info)

        # Record daily P&L
        for d in range(ei, actual_exit + 1):
            try:
                px = close.iloc[d][sym]
                if d == ei:
                    prev_px = entry_price
                else:
                    prev_px = close.iloc[d-1][sym]
                if np.isnan(px) or np.isnan(prev_px) or prev_px <= 0:
                    continue
                daily_ret = px / prev_px - 1
                daily_pnl[d] += daily_ret * weight
                position_count[d] += 1
            except (IndexError, KeyError):
                continue

    return daily_pnl, position_count


def compute_metrics(daily_ret, label=""):
    n = len(daily_ret)
    cum = np.cumprod(1 + daily_ret)
    total_yrs = n / 252
    cagr = cum[-1] ** (1/max(total_yrs, 0.1)) - 1 if cum[-1] > 0 else -1
    sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) > 0 else 0
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    mdd = dd.min()

    # Yearly returns
    yearly = {}
    for yr in range(2011, 2027):
        yr_mask = np.array([pd.Timestamp(d).year == yr for d in cal])
        if yr_mask.sum() > 50:
            yearly[yr] = np.prod(1 + daily_ret[yr_mask]) - 1
    n_pos_years = sum(1 for v in yearly.values() if v > 0)

    # Rolling 3yr Sharpe minimum
    rolling_sharpes = []
    for start in range(0, n - 756, 252):
        window = daily_ret[start:start+756]
        if np.std(window) > 0:
            rolling_sharpes.append(np.mean(window) / np.std(window) * np.sqrt(252))
    rolling_min = min(rolling_sharpes) if rolling_sharpes else 0

    # Worst drawdown episodes
    dd_episodes = []
    in_dd = False
    ep_min = 0
    for i in range(n):
        if dd[i] < -0.05 and not in_dd:
            in_dd = True
            ep_min = dd[i]
        elif in_dd:
            if dd[i] < ep_min:
                ep_min = dd[i]
            if dd[i] > -0.02:
                in_dd = False
                dd_episodes.append(ep_min)
    n_dd_gt10 = sum(1 for x in dd_episodes if x < -0.10)
    n_dd_gt20 = sum(1 for x in dd_episodes if x < -0.20)

    return {
        "label": label, "cagr": cagr, "sharpe": sharpe, "mdd": mdd,
        "years_pos": f"{n_pos_years}/{len(yearly)}",
        "rolling_min_sharpe": rolling_min,
        "n_dd_gt10": n_dd_gt10, "n_dd_gt20": n_dd_gt20,
        "yearly": yearly,
    }


def main():
    global cal
    print("=" * 70)
    print("P33: 動態倉位方案設計與回測")
    print("=" * 70)

    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # === Build trades ===
    dis_raw = pd.DataFrame(data.get("disposal_information"))
    dis = dis_raw.copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$")
        & dis["stock_id"].isin(valid_stocks)
        & ~dis["stock_id"].str.startswith(("00", "91"))
    ].copy()

    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1
    dis["is_new_regime"] = dis["announce"] >= NEW_REGIME_DATE

    valid = dis[
        (dis["start_idx"] >= 0)
        & (dis["end_idx"] >= 0)
        & (dis["end_idx"] < n_cal)
        & (dis["duration"] >= 3)
    ].copy()

    # Indicators
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()
    daily_ret_all = close.pct_change()
    panic_count = (daily_ret_all < -0.06).sum(axis=1)
    is_panic = panic_count > PANIC_THRESHOLD
    danger_zone = is_panic.rolling(PANIC_COOLDOWN, min_periods=1).max() > 0

    ma20 = close.rolling(20).mean()
    above_ma20 = (close > ma20).sum(axis=1) / close.notna().sum(axis=1)

    # Generate all candidate trades (without breadth filter — let dynamic sizing handle it)
    trades = []
    for _, row in valid.iterrows():
        sym = row["stock_id"]
        si = int(row["start_idx"])
        ei = int(row["end_idx"])

        if row["is_new_regime"]:
            entry_idx = si
            exit_idx = ei - 1
        else:
            entry_idx = si + 3
            exit_idx = ei - 1

        if entry_idx < 1 or exit_idx >= n_cal or entry_idx >= exit_idx:
            continue
        if danger_zone.iloc[entry_idx]:
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[entry_idx - 1][sym]
            avg_to = avg_turnover_5d.iloc[entry_idx - 1][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan([entry_price, prev_close])):
            continue
        if entry_price <= 0:
            continue
        if np.isnan(avg_to) or avg_to < MIN_TURNOVER:
            continue

        gap = entry_price / prev_close - 1
        if gap < GAP_LOW or gap > GAP_HIGH:
            continue

        breadth_val = above_ma20.iloc[entry_idx]

        trades.append({
            "stock_id": sym,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "breadth_at_entry": breadth_val,
        })

    print(f"\n  候選交易總數: {len(trades)}")

    # === Run all schemes ===
    schemes = [
        ("A", "現行(固定5倉, <40%阻擋)", False),
        ("B", "三階梯(>50%→5, 40-50%→3, <40%→0)", False),
        ("C", "四階梯(>60%→5, 50-60%→4, 40-50%→2, <40%→0)", False),
        ("D", "線性縮放(5*(breadth-0.4)/0.2)", False),
        ("E", "三階梯 + 止損-15%", True),
        ("F", "四階梯 + 止損-15%", True),
        ("G", "保守(>55%→5, 45-55%→3, <45%→0)", False),
    ]

    results = []
    all_daily = {}

    for scheme_id, label, use_sl in schemes:
        print(f"\n  模擬方案 {scheme_id}: {label}...")
        daily_pnl, pos_count = simulate(trades, cal, close, open_p, n_cal, scheme_id, use_sl)
        metrics = compute_metrics(daily_pnl, label)
        results.append(metrics)
        all_daily[scheme_id] = daily_pnl

    # === Print comparison table ===
    print("\n" + "=" * 70)
    print("方案比較")
    print("=" * 70)

    print(f"\n  {'方案':>4} {'描述':<40} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'年正':>6} {'3yr最低SR':>9} {'DD>10%':>6} {'DD>20%':>6}")
    print("  " + "-" * 110)
    for r in results:
        print(f"  {r['label'][0]:>4} {r['label'][1:]:<40} {r['cagr']:>8.1%} {r['sharpe']:>7.2f} {r['mdd']:>8.1%} {r['years_pos']:>6} {r['rolling_min_sharpe']:>9.2f} {r['n_dd_gt10']:>6} {r['n_dd_gt20']:>6}")

    # === Yearly breakdown for top schemes ===
    print("\n" + "=" * 70)
    print("逐年報酬比較 (A vs B vs C vs F)")
    print("=" * 70)

    key_schemes = ["A", "B", "C", "F"]
    print(f"\n  {'年份':>6}", end="")
    for s in key_schemes:
        print(f"  {s:>8}", end="")
    print()
    print("  " + "-" * 45)

    for yr in range(2011, 2027):
        yr_mask = np.array([pd.Timestamp(d).year == yr for d in cal])
        if yr_mask.sum() < 50:
            continue
        print(f"  {yr:>6}", end="")
        for s in key_schemes:
            ret = np.prod(1 + all_daily[s][yr_mask]) - 1
            print(f"  {ret:>8.1%}", end="")
        print()

    # === MDD comparison during extreme events ===
    print("\n" + "=" * 70)
    print("極端事件中的回撤比較")
    print("=" * 70)

    extreme_events = [
        ("2026/3 灰色地帶", "2026-03-01", "2026-04-30"),
        ("2024/7-12 漫長消耗", "2024-07-01", "2024-12-31"),
        ("2020/7 個股崩塌", "2020-07-01", "2020-08-31"),
        ("2025/4 關稅", "2025-04-01", "2025-05-15"),
    ]

    print(f"\n  {'事件':<25}", end="")
    for s in key_schemes:
        print(f"  {s:>8}", end="")
    print()
    print("  " + "-" * 60)

    for label, start_str, end_str in extreme_events:
        start_dt = pd.Timestamp(start_str)
        end_dt = pd.Timestamp(end_str)
        mask = np.array([(start_dt <= pd.Timestamp(d) <= end_dt) for d in cal])
        if mask.sum() == 0:
            continue
        print(f"  {label:<25}", end="")
        for s in key_schemes:
            ep_ret = all_daily[s][mask]
            ep_cum = np.cumprod(1 + ep_ret)
            ep_dd = ((ep_cum - np.maximum.accumulate(ep_cum)) / np.maximum.accumulate(ep_cum)).min()
            print(f"  {ep_dd:>8.1%}", end="")
        print()

    # === Position count analysis ===
    print("\n" + "=" * 70)
    print("平均持倉數比較")
    print("=" * 70)

    for scheme_id, label, _ in schemes[:4]:
        _, pos_count = simulate(trades, cal, close, open_p, n_cal, scheme_id, False)
        active_days = (pos_count > 0).sum()
        avg_pos = pos_count[pos_count > 0].mean() if active_days > 0 else 0
        max_pos = pos_count.max()
        print(f"  {scheme_id}: 有持倉天={active_days}, 平均持倉={avg_pos:.1f}, 最大={max_pos}")

    # === Recommendation ===
    print("\n" + "=" * 70)
    print("推薦方案分析")
    print("=" * 70)

    # Find best risk-adjusted
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    best_cagr = max(results, key=lambda x: x["cagr"])
    best_mdd = max(results, key=lambda x: x["mdd"])  # least negative
    best_rolling = max(results, key=lambda x: x["rolling_min_sharpe"])

    print(f"\n  最高 Sharpe: {best_sharpe['label']} ({best_sharpe['sharpe']:.2f})")
    print(f"  最高 CAGR: {best_cagr['label']} ({best_cagr['cagr']:.1%})")
    print(f"  最小 MDD: {best_mdd['label']} ({best_mdd['mdd']:.1%})")
    print(f"  最高 3yr最低SR: {best_rolling['label']} ({best_rolling['rolling_min_sharpe']:.2f})")

    # Save results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "yearly"} for r in results])
    results_df.to_csv(OUT / "p33_scheme_comparison.csv", index=False, encoding="utf-8-sig")

    # Save daily returns for all schemes
    daily_out = pd.DataFrame({"date": cal})
    for s_id, d_ret in all_daily.items():
        daily_out[f"scheme_{s_id}"] = d_ret
    daily_out.to_csv(OUT / "p33_daily_returns.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUT}")

if __name__ == "__main__":
    main()
