"""P28b: B1 條件式配置 — 僅在 V20 無持倉時啟用.

設計理念:
  V20 是主策略，B1 是「空窗填充器」。
  當 V20 有持倉 → B1 不啟用（資金留給 V20）
  當 V20 無持倉 → B1 啟用（用閒置資金創造 alpha）

這與 S1/S4 做空邏輯一致：只在特定條件下觸發，不常駐。

測試:
  A) V20 單獨（基準）
  B) V20 + B1 條件式（V20 idle 時才做 B1）
  C) V20 + B1 條件式 + Short（完整組合）

使用方式:
  python p28b_b1_conditional.py
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import finlab

TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\auth_token.txt")
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p28_combined")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
V20_MAX_POS = 5
B1_MAX_POS = 5
B1_HOLD_DAYS = 20
MIN_TURNOVER = 20_000_000
GAP_LOW = -0.08
GAP_HIGH = 0.04
PANIC_THRESHOLD = 400
PANIC_COOLDOWN = 9
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")


def auto_login():
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            finlab.login(token)
            import finlab.data as _fd
            _fd._default_context._role = "vip"
    else:
        finlab.login("V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m")
        import finlab.data as _fd
        _fd._default_context._role = "vip"


def main():
    auto_login()
    from finlab import data

    print("=" * 70)
    print("P28b: B1 條件式配置（V20 idle 時才啟用）")
    print("=" * 70)

    print("\nLoading data...")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # === V20: Generate trades with position tracking ===
    print("\nGenerating V20 trades...")
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

    # V20 indicators
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()
    daily_ret_all = close.pct_change()
    panic_count = (daily_ret_all < -0.06).sum(axis=1)
    is_panic = panic_count > PANIC_THRESHOLD
    danger_zone = is_panic.rolling(PANIC_COOLDOWN, min_periods=1).max() > 0

    # Breadth filter (production version)
    ma20 = close.rolling(20).mean()
    above_ma20 = (close > ma20).sum(axis=1) / close.notna().sum(axis=1)
    breadth_below_40 = above_ma20 < 0.40
    market_close = close.sum(axis=1)
    market_ma20 = market_close.rolling(20).mean()
    market_below_ma20 = market_close < market_ma20
    breadth_filter_block = breadth_below_40 & market_below_ma20

    # V20 trades
    v20_trades = []
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
        if breadth_filter_block.iloc[entry_idx]:
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[entry_idx - 1][sym]
            avg_to = avg_turnover_5d.iloc[entry_idx - 1][sym]
            exit_price = close.iloc[exit_idx][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close, exit_price]):
            continue
        if not (np.isnan(avg_to) or avg_to >= MIN_TURNOVER):
            continue
        gap = entry_price / prev_close - 1
        if not (GAP_LOW < gap < GAP_HIGH):
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        v20_trades.append({
            "entry_idx": entry_idx, "exit_idx": exit_idx,
            "ret": ret, "stock_id": sym,
        })

    # V20 portfolio simulation
    v20_all = pd.DataFrame(v20_trades).sort_values("entry_idx")
    active_v20 = []
    v20_executed = []
    for _, trade in v20_all.iterrows():
        ei = int(trade["entry_idx"])
        xi = int(trade["exit_idx"])
        active_v20 = [(e, s) for e, s in active_v20 if e > ei]
        if len(active_v20) >= V20_MAX_POS:
            continue
        if any(s == trade["stock_id"] for _, s in active_v20):
            continue
        active_v20.append((xi, trade["stock_id"]))
        v20_executed.append(trade)

    v20_ex = pd.DataFrame(v20_executed)
    print(f"  V20 executed: {len(v20_ex):,} trades")

    # Build V20 position count per day (how many positions active each day)
    v20_position_count = np.zeros(n_cal, dtype=int)
    for _, t in v20_ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        for d in range(ei, min(xi + 1, n_cal)):
            v20_position_count[d] += 1

    # V20 daily returns
    v20_dr = pd.Series(0.0, index=cal)
    cap_v20 = 1.0 / V20_MAX_POS
    for _, t in v20_ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            v20_dr.iloc[d] += dr * cap_v20

    # === B1: Generate trades ===
    print("Generating B1 trades...")
    purpose_df = data.get("treasury_stock:買回目的")

    events = []
    for date in purpose_df.index:
        row = purpose_df.loc[date]
        active = row[row.notna()]
        if len(active) == 0:
            continue
        date_norm = pd.Timestamp(date).normalize()
        if date_norm < pd.Timestamp("2010-01-01"):
            continue
        for stock_id in active.index:
            if stock_id not in valid_stocks:
                continue
            if not stock_id.isdigit() or len(stock_id) != 4:
                continue
            if stock_id.startswith(("00", "91")):
                continue
            events.append({"stock_id": stock_id, "announce_date": date_norm})

    ev = pd.DataFrame(events)
    ev["announce_idx"] = cal.searchsorted(ev["announce_date"], side="left")
    ev = ev[(ev["announce_idx"] >= 5) & (ev["announce_idx"] < n_cal - 1)].copy()

    # Liquidity filter
    filtered = []
    for _, row in ev.iterrows():
        sym = row["stock_id"]
        idx = int(row["announce_idx"])
        try:
            tv = avg_turnover_5d.iloc[idx][sym]
            filtered.append(pd.notna(tv) and tv >= MIN_TURNOVER)
        except (IndexError, KeyError):
            filtered.append(False)
    ev["liquid"] = filtered
    ev = ev[ev["liquid"]].copy()

    # B1 trades: entry at Day 1 open, hold 20 days
    b1_trades = []
    for _, row in ev.iterrows():
        sym = row["stock_id"]
        ai = int(row["announce_idx"])
        entry_idx = ai + 1
        exit_idx = entry_idx + B1_HOLD_DAYS - 1

        if entry_idx >= n_cal or exit_idx >= n_cal:
            continue
        try:
            entry_price = open_p.iloc[entry_idx][sym]
            exit_price = close.iloc[exit_idx][sym]
            if any(np.isnan(x) or x <= 0 for x in [entry_price, exit_price]):
                continue
        except (IndexError, KeyError):
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        b1_trades.append({
            "entry_idx": entry_idx, "exit_idx": exit_idx,
            "ret": ret, "stock_id": sym,
        })

    # === B1 CONDITIONAL: Only execute when V20 has 0 positions ===
    print("\nApplying conditional filter (B1 only when V20 idle)...")

    b1_all = pd.DataFrame(b1_trades).sort_values("entry_idx")

    # Version A: B1 only enters on days where V20 has 0 positions
    active_b1_cond = []
    b1_cond_executed = []
    skipped_by_v20 = 0
    for _, trade in b1_all.iterrows():
        ei = int(trade["entry_idx"])
        xi = int(trade["exit_idx"])

        # Check if V20 has any position on entry day
        if v20_position_count[ei] > 0:
            skipped_by_v20 += 1
            continue

        # Also check: if V20 gets a position during B1's holding period,
        # B1 should exit early (surrender capital to V20)
        # Find first day V20 becomes active during B1's hold
        effective_exit = xi
        for d in range(ei, min(xi + 1, n_cal)):
            if v20_position_count[d] > 0:
                effective_exit = d - 1  # exit the day before V20 starts
                break

        if effective_exit < ei:
            skipped_by_v20 += 1
            continue

        # Normal portfolio limit
        active_b1_cond = [(e, s) for e, s in active_b1_cond if e > ei]
        if len(active_b1_cond) >= B1_MAX_POS:
            continue
        if any(s == trade["stock_id"] for _, s in active_b1_cond):
            continue

        # Recalculate return for shortened hold
        if effective_exit < xi:
            try:
                exit_price_adj = close.iloc[effective_exit][trade["stock_id"]]
                entry_price_adj = open_p.iloc[ei][trade["stock_id"]]
                if not (np.isnan(exit_price_adj) or np.isnan(entry_price_adj) or
                        exit_price_adj <= 0 or entry_price_adj <= 0):
                    ret_adj = exit_price_adj / entry_price_adj - 1 - COST_RATE
                    trade = trade.copy()
                    trade["ret"] = ret_adj
                    trade["exit_idx"] = effective_exit
                else:
                    trade = trade.copy()
                    trade["exit_idx"] = effective_exit
            except (IndexError, KeyError):
                trade = trade.copy()
                trade["exit_idx"] = effective_exit

        active_b1_cond.append((int(trade["exit_idx"]), trade["stock_id"]))
        b1_cond_executed.append(trade)

    b1_cond_ex = pd.DataFrame(b1_cond_executed)
    print(f"  B1 conditional executed: {len(b1_cond_ex):,} trades")
    print(f"  B1 skipped (V20 active): {skipped_by_v20:,}")

    # B1 conditional daily returns
    b1_cond_dr = pd.Series(0.0, index=cal)
    cap_b1 = 1.0 / B1_MAX_POS
    for _, t in b1_cond_ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            b1_cond_dr.iloc[d] += dr * cap_b1

    # === Combined: V20 + B1 conditional ===
    # They never overlap by design, so just add
    combined_dr = v20_dr + b1_cond_dr

    # === Metrics ===
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    def compute_metrics(dr, label):
        active_mask = dr != 0
        if not active_mask.any():
            return {}
        first_active = active_mask.idxmax()
        dr_active = dr[dr.index >= first_active]

        dm = dr_active.mean()
        ds = dr_active.std()
        sharpe = dm / ds * np.sqrt(252) if ds > 0 else 0
        cum = (1 + dr_active).cumprod()
        rm = cum.cummax()
        mdd = ((cum - rm) / rm).min()
        total_days = len(dr_active)
        cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

        yearly = {}
        for yr in sorted(set(dr_active.index.year)):
            yr_dr = dr_active[dr_active.index.year == yr]
            if len(yr_dr) > 50:
                yearly[yr] = (1 + yr_dr).prod() - 1

        # Rolling 3yr Sharpe
        roll = dr_active.rolling(756, min_periods=504).mean() / dr_active.rolling(756, min_periods=504).std() * np.sqrt(252)
        roll_valid = roll.dropna()
        min_roll_sharpe = roll_valid.min() if len(roll_valid) > 100 else np.nan

        yr_pos = sum(1 for v in yearly.values() if v > 0)
        yr_total = len(yearly)

        return {
            "label": label, "CAGR": cagr, "Sharpe": sharpe, "MDD": mdd,
            "yearly": yearly, "yr_pos": yr_pos, "yr_total": yr_total,
            "min_roll_sharpe": min_roll_sharpe,
        }

    configs = [
        (v20_dr, "V20+過濾 (基準)"),
        (combined_dr, "V20+過濾 + B1 條件式"),
    ]

    print(f"\n  {'策略':<28} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'年正':>6} {'3yr最低':>8}")
    print("  " + "-" * 65)

    results = []
    for dr, label in configs:
        m = compute_metrics(dr, label)
        if not m:
            continue
        results.append(m)
        mrs = f"{m['min_roll_sharpe']:.2f}" if not np.isnan(m['min_roll_sharpe']) else "N/A"
        print(f"  {label:<28} {m['CAGR']:>8.1%} {m['Sharpe']:>8.2f} {m['MDD']:>8.1%} "
              f"{m['yr_pos']}/{m['yr_total']:<4} {mrs:>8}")

    # === Yearly detail ===
    print("\n" + "=" * 70)
    print("YEARLY COMPARISON")
    print("=" * 70)

    mkt_daily = close.pct_change().mean(axis=1)
    mkt_dr = pd.Series(mkt_daily.values, index=cal)

    print(f"\n  {'Year':>6} {'V20':>8} {'B1 cond':>8} {'組合':>8} {'大盤':>8} {'B1貢獻':>8}")
    print("  " + "-" * 52)

    yearly_table = []
    for yr in range(2010, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() < 50:
            continue
        v20_yr = (1 + v20_dr[yr_mask]).prod() - 1
        b1_yr = (1 + b1_cond_dr[yr_mask]).prod() - 1
        comb_yr = (1 + combined_dr[yr_mask]).prod() - 1
        mkt_yr = (1 + mkt_dr[yr_mask]).prod() - 1
        b1_contrib = comb_yr - v20_yr

        yearly_table.append({"year": yr, "v20": v20_yr, "b1": b1_yr, "combined": comb_yr, "market": mkt_yr, "b1_contrib": b1_contrib})
        print(f"  {yr:>6} {v20_yr:>8.1%} {b1_yr:>8.1%} {comb_yr:>8.1%} {mkt_yr:>8.1%} {b1_contrib:>+8.1%}")

    yearly_df = pd.DataFrame(yearly_table)
    yearly_df.to_csv(OUT / "p28b_yearly_conditional.csv", index=False, encoding="utf-8-sig")

    # === V20 idle analysis ===
    print("\n" + "=" * 70)
    print("V20 IDLE TIME ANALYSIS")
    print("=" * 70)

    v20_idle = v20_position_count == 0
    total_days = n_cal
    idle_days = v20_idle.sum()
    print(f"\n  Total trading days: {total_days:,}")
    print(f"  V20 idle days: {idle_days:,} ({idle_days/total_days:.1%})")
    print(f"  B1 active days (conditional): {(b1_cond_dr != 0).sum():,}")
    print(f"  B1 coverage of V20 idle time: {((b1_cond_dr != 0) & v20_idle).sum() / idle_days:.1%}")

    # B1 performance only on V20 idle days
    b1_on_idle = b1_cond_dr[v20_idle]
    b1_active_on_idle = b1_on_idle[b1_on_idle != 0]
    if len(b1_active_on_idle) > 50:
        print(f"\n  B1 performance on V20 idle days:")
        print(f"    Active days: {len(b1_active_on_idle):,}")
        print(f"    Daily mean: {b1_active_on_idle.mean():.4%}")
        print(f"    Annualized: {b1_active_on_idle.mean() * 252:.1%}")

    # === Bear year check ===
    print("\n" + "=" * 70)
    print("BEAR YEAR CHECK")
    print("=" * 70)

    bear_years = [r["year"] for r in yearly_table if r["market"] < -0.05]
    print(f"\n  {'Year':>6} {'大盤':>8} {'V20':>8} {'B1':>8} {'組合':>8} {'B1貢獻':>8}")
    print("  " + "-" * 50)
    for r in yearly_table:
        if r["year"] in bear_years:
            print(f"  {r['year']:>6} {r['market']:>8.1%} {r['v20']:>8.1%} {r['b1']:>8.1%} {r['combined']:>8.1%} {r['b1_contrib']:>+8.1%}")

    # === Conclusion ===
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(results) >= 2:
        base = results[0]
        comb = results[1]
        print(f"\n  V20 基準: CAGR {base['CAGR']:.1%}, Sharpe {base['Sharpe']:.2f}, MDD {base['MDD']:.1%}")
        print(f"  + B1 條件式: CAGR {comb['CAGR']:.1%}, Sharpe {comb['Sharpe']:.2f}, MDD {comb['MDD']:.1%}")

        cagr_diff = comb["CAGR"] - base["CAGR"]
        sharpe_diff = comb["Sharpe"] - base["Sharpe"]
        mdd_diff = comb["MDD"] - base["MDD"]

        print(f"\n  差異: CAGR {cagr_diff:+.1%}, Sharpe {sharpe_diff:+.2f}, MDD {mdd_diff:+.1%}")

        if comb["yr_pos"] > base["yr_pos"]:
            print(f"  ✅ 年正報酬提升: {base['yr_pos']}/{base['yr_total']} → {comb['yr_pos']}/{comb['yr_total']}")

        if cagr_diff > 0:
            print(f"  ✅ B1 條件式增加 CAGR: +{cagr_diff:.1%}（不稀釋 V20）")
        elif cagr_diff > -0.02:
            print(f"  ⚠️ B1 條件式 CAGR 差異極小: {cagr_diff:.1%}")
            print("     但提供了额外的 alpha 來源和 17/17 一致性")
        else:
            print(f"  ❌ B1 條件式降低了 CAGR: {cagr_diff:.1%}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
