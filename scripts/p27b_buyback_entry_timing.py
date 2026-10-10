"""P27b: 庫藏股入場時機驗證 — 收盤後公告 → 次日開盤買入是否仍有效.

台股庫藏股公告通常在收盤後發布（14:00後或收盤後），
因此實際可執行的最早入場是「次日開盤」。

本腳本比較:
  A) 理論入場: 公告日收盤 (Day 0 close) — 不可執行
  B) 實際入場: 次日開盤 (Day 1 open) — 可執行
  C) 保守入場: 次日收盤 (Day 1 close) — 最保守

量化:
  - 隔夜跳空吃掉了多少 alpha
  - 剩餘 alpha 是否仍可交易
  - 不同入場時機的淨報酬比較

使用方式:
  python p27b_buyback_entry_timing.py
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import finlab
from scipy import stats

TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\auth_token.txt")
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p27_treasury_stock")
OUT.mkdir(parents=True, exist_ok=True)

MIN_TURNOVER = 20_000_000
MAX_TRACK_DAYS = 30
COST = 0.001425 + 0.003  # buy + sell


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

    print("Loading data...")
    close_p = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol_p = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close_p.columns)

    # Load buyback events
    purpose_df = data.get("treasury_stock:買回目的")

    # Turnover for liquidity filter
    turnover = close_p * vol_p
    avg_turnover_5d = turnover.rolling(5).mean()

    # Market MA20
    market_close = close_p.sum(axis=1)
    market_ma20 = market_close.rolling(20).mean()
    market_above_ma20 = (market_close > market_ma20)

    # === Extract events ===
    events = []
    for date in purpose_df.index:
        row = purpose_df.loc[date]
        active = row[row.notna()]
        if len(active) == 0:
            continue
        date_norm = pd.Timestamp(date).normalize()
        if date_norm < pd.Timestamp("2010-01-01"):
            continue
        for stock_id, purpose in active.items():
            if stock_id not in valid_stocks:
                continue
            if not stock_id.isdigit() or len(stock_id) != 4:
                continue
            if stock_id.startswith(("00", "91")):
                continue
            events.append({
                "stock_id": stock_id,
                "announce_date": date_norm,
                "purpose": str(purpose),
            })

    ev = pd.DataFrame(events)
    ev["announce_idx"] = cal.searchsorted(ev["announce_date"], side="left")
    ev = ev[(ev["announce_idx"] < n_cal) & (ev["announce_idx"] >= 5)].copy()
    ev["year"] = ev["announce_date"].dt.year

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

    # Need Day 1 + MAX_TRACK_DAYS after announcement
    ev = ev[ev["announce_idx"] + 1 + MAX_TRACK_DAYS < n_cal].copy()
    print(f"Events with full window: {len(ev):,}")

    if len(ev) < 100:
        print("ERROR: Too few events.")
        return

    # === Compute returns from THREE different entry points ===
    results = []
    for _, row in ev.iterrows():
        sym = row["stock_id"]
        ai = int(row["announce_idx"])
        day1_idx = ai + 1  # next trading day

        try:
            # Entry A: announcement day close (theoretical, not executable)
            entry_a = close_p.iloc[ai][sym]
            # Entry B: next day open (realistic — buy at open)
            entry_b = open_p.iloc[day1_idx][sym]
            # Entry C: next day close (conservative — buy at close)
            entry_c = close_p.iloc[day1_idx][sym]

            if any(np.isnan(x) or x <= 0 for x in [entry_a, entry_b, entry_c]):
                continue
        except (IndexError, KeyError):
            continue

        # Overnight gap: how much alpha is captured before you can trade
        overnight_gap = entry_b / entry_a - 1
        # Intraday on Day 1: open to close
        day1_intraday = entry_c / entry_b - 1

        # Returns from each entry point
        returns_a = {}  # from Day 0 close
        returns_b = {}  # from Day 1 open
        returns_c = {}  # from Day 1 close

        for d in range(1, MAX_TRACK_DAYS + 1):
            target_idx = day1_idx + d - 1
            if target_idx >= n_cal:
                break
            try:
                tc = close_p.iloc[target_idx][sym]
                if pd.isna(tc) or tc <= 0:
                    continue
                returns_a[d] = tc / entry_a - 1
                returns_b[d] = tc / entry_b - 1
                returns_c[d] = tc / entry_c - 1
            except (IndexError, KeyError):
                continue

        if len(returns_a) < MAX_TRACK_DAYS:
            continue

        # Market regime
        try:
            regime = bool(market_above_ma20.iloc[min(ai, n_cal - 1)])
        except Exception:
            regime = True

        results.append({
            "stock_id": sym,
            "announce_date": row["announce_date"],
            "year": row["year"],
            "purpose": row["purpose"],
            "market_bull": regime,
            "overnight_gap": overnight_gap,
            "day1_intraday": day1_intraday,
            **{f"a_day_{d}": returns_a.get(d, np.nan) for d in range(1, MAX_TRACK_DAYS + 1)},
            **{f"b_day_{d}": returns_b.get(d, np.nan) for d in range(1, MAX_TRACK_DAYS + 1)},
            **{f"c_day_{d}": returns_c.get(d, np.nan) for d in range(1, MAX_TRACK_DAYS + 1)},
        })

    res = pd.DataFrame(results)
    print(f"Final events: {len(res):,}")

    if len(res) < 100:
        print("ERROR: Too few valid events.")
        return

    res.to_csv(OUT / "p27b_entry_timing.csv", index=False, encoding="utf-8-sig")

    # === Analysis ===
    print("\n" + "=" * 70)
    print("P27b: ENTRY TIMING ANALYSIS")
    print("=" * 70)

    # 1. Overnight gap
    print("\n--- OVERNIGHT GAP (Day 0 close → Day 1 open) ---")
    og = res["overnight_gap"].dropna()
    t_og, p_og = stats.ttest_1samp(og, 0)
    print(f"  Mean: {og.mean():.4%}, Median: {og.median():.4%}, Win: {(og > 0).mean():.1%}")
    print(f"  t={t_og:.2f}, p={p_og:.2e}")
    print(f"  → 隔夜跳空吃掉了 {og.mean() / (og.mean() + res['b_day_30'].dropna().mean()) * 100:.1f}% 的總 alpha")

    # 2. Day 1 intraday
    print("\n--- DAY 1 INTRADAY (Day 1 open → Day 1 close) ---")
    d1 = res["day1_intraday"].dropna()
    t_d1, p_d1 = stats.ttest_1samp(d1, 0)
    print(f"  Mean: {d1.mean():.4%}, Median: {d1.median():.4%}, Win: {(d1 > 0).mean():.1%}")
    print(f"  t={t_d1:.2f}, p={p_d1:.2e}")

    # 3. Compare three entry points
    print("\n" + "=" * 70)
    print("ENTRY POINT COMPARISON")
    print("=" * 70)

    print(f"\n{'Period':<12} {'A: Day0 Close':>14} {'B: Day1 Open':>14} {'C: Day1 Close':>14} {'B net':>10} {'C net':>10}")
    print("-" * 80)
    for d in [1, 3, 5, 10, 20, 30]:
        ra = res[f"a_day_{d}"].dropna()
        rb = res[f"b_day_{d}"].dropna()
        rc = res[f"c_day_{d}"].dropna()
        if len(rb) < 30:
            continue
        net_b = rb - COST
        net_c = rc - COST
        print(f"Day {d:>2}      {ra.mean():>13.4%} {rb.mean():>13.4%} {rc.mean():>13.4%} "
              f"{net_b.mean():>9.4%} {net_c.mean():>9.4%}")

    # 4. Detailed stats for Entry B (Day 1 open) — the realistic one
    print("\n" + "=" * 70)
    print("ENTRY B (Day 1 Open) — REALISTIC EXECUTION")
    print("=" * 70)

    print(f"\n{'Period':<12} {'n':>6} {'Gross':>8} {'Net':>8} {'Median':>8} {'Win%':>6} {'t-stat':>8} {'p-value':>10}")
    print("-" * 75)
    for d in [1, 3, 5, 10, 20, 30]:
        col = f"b_day_{d}"
        valid = res[col].dropna()
        if len(valid) < 30:
            continue
        net = valid - COST
        t_stat, p_val = stats.ttest_1samp(net, 0)
        print(f"Day {d:>2}      {len(valid):>6,} {valid.mean():>8.4%} {net.mean():>8.4%} "
              f"{valid.median():>8.4%} {(net > 0).mean():>6.1%} {t_stat:>8.2f} {p_val:>10.2e}")

    # 5. Entry B yearly stability
    print("\n" + "=" * 70)
    print("ENTRY B YEARLY STABILITY (Day 1 Open, Hold 10d)")
    print("=" * 70)

    yearly = res.groupby("year").apply(
        lambda g: pd.Series({
            "n": len(g),
            "b_day_5_net": (g["b_day_5"] - COST).mean(),
            "b_day_5_win": ((g["b_day_5"] - COST) > 0).mean(),
            "b_day_10_net": (g["b_day_10"] - COST).mean(),
            "b_day_10_win": ((g["b_day_10"] - COST) > 0).mean(),
            "b_day_30_net": (g["b_day_30"] - COST).mean(),
            "b_day_30_win": ((g["b_day_30"] - COST) > 0).mean(),
        })
    ).reset_index()
    yearly = yearly[yearly["n"] >= 20]
    print(yearly.to_string(index=False))
    yearly.to_csv(OUT / "p27b_yearly_entry_b.csv", index=False, encoding="utf-8-sig")

    # 6. Entry B by market regime
    print("\n" + "=" * 70)
    print("ENTRY B BY MARKET REGIME")
    print("=" * 70)

    by_regime = res.groupby("market_bull").apply(
        lambda g: pd.Series({
            "n": len(g),
            "b_day_5_net": (g["b_day_5"] - COST).mean(),
            "b_day_10_net": (g["b_day_10"] - COST).mean(),
            "b_day_30_net": (g["b_day_30"] - COST).mean(),
            "b_day_30_win": ((g["b_day_30"] - COST) > 0).mean(),
        })
    ).reset_index()
    by_regime["regime"] = by_regime["market_bull"].map({True: "多頭", False: "空頭"})
    print(by_regime[["regime", "n", "b_day_5_net", "b_day_10_net", "b_day_30_net", "b_day_30_win"]].to_string(index=False))

    # 7. Control group for Entry B
    print("\n" + "=" * 70)
    print("CONTROL GROUP (Entry B, shift 60 days)")
    print("=" * 70)

    control_results = []
    for _, row in ev.iterrows():
        sym = row["stock_id"]
        ai = int(row["announce_idx"])
        ctrl_ai = ai + 60
        ctrl_day1 = ctrl_ai + 1
        if ctrl_day1 + MAX_TRACK_DAYS >= n_cal or ctrl_ai < 0:
            continue
        try:
            ctrl_entry = open_p.iloc[ctrl_day1][sym]
            if np.isnan(ctrl_entry) or ctrl_entry <= 0:
                continue
        except (IndexError, KeyError):
            continue

        ctrl_returns = {}
        for d in range(1, MAX_TRACK_DAYS + 1):
            target_idx = ctrl_day1 + d - 1
            if target_idx >= n_cal:
                break
            try:
                tc = close_p.iloc[target_idx][sym]
                if pd.isna(tc) or tc <= 0:
                    continue
                ctrl_returns[d] = tc / ctrl_entry - 1
            except (IndexError, KeyError):
                continue

        if len(ctrl_returns) < MAX_TRACK_DAYS:
            continue
        control_results.append({f"b_day_{d}": ctrl_returns.get(d, np.nan) for d in range(1, MAX_TRACK_DAYS + 1)})

    ctrl = pd.DataFrame(control_results)
    print(f"Control events: {len(ctrl):,}")

    if len(ctrl) > 50:
        for d in [5, 10, 30]:
            col = f"b_day_{d}"
            treat = res[col].dropna()
            ctrl_sub = ctrl[col].dropna()
            if len(treat) < 30 or len(ctrl_sub) < 30:
                continue
            diff = treat.mean() - ctrl_sub.mean()
            t_stat, p_val = stats.ttest_ind(treat, ctrl_sub, equal_var=False)
            print(f"  Day {d}: Treatment={treat.mean():.4%}, Control={ctrl_sub.mean():.4%}, "
                  f"Excess={diff:.4%}, t={t_stat:.2f}, p={p_val:.2e}")

    # 8. Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    b5 = (res["b_day_5"] - COST).dropna()
    b10 = (res["b_day_10"] - COST).dropna()
    b30 = (res["b_day_30"] - COST).dropna()

    t5, p5 = stats.ttest_1samp(b5, 0)
    t10, p10 = stats.ttest_1samp(b10, 0)
    t30, p30 = stats.ttest_1samp(b30, 0)

    print(f"\n  Entry B (Day 1 Open) net returns:")
    print(f"    Day 5:  {b5.mean():.4%} (win {(b5 > 0).mean():.1%}, p={p5:.2e})")
    print(f"    Day 10: {b10.mean():.4%} (win {(b10 > 0).mean():.1%}, p={p10:.2e})")
    print(f"    Day 30: {b30.mean():.4%} (win {(b30 > 0).mean():.1%}, p={p30:.2e})")

    if p10 < 0.001 and b10.mean() > 0.01:
        print(f"\n✅ 次日開盤買入仍然有效。Day 10 淨報酬 {b10.mean():.2%}，勝率 {(b10 > 0).mean():.1%}。")
        print("   策略可執行。建議入場：公告次日開盤買入。")
    elif p10 < 0.01 and b10.mean() > 0.005:
        print(f"\n⚠️ 次日開盤買入邊際有效。Day 10 淨報酬 {b10.mean():.2%}。")
        print("   建議：仍可使用，但需搭配其他濾網增強。")
    else:
        print(f"\n❌ 次日開盤買入效果不足。Day 10 淨報酬 {b10.mean():.2%}。")
        print("   隔夜跳空吃掉了大部分 alpha。需要尋找其他入場時機。")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
