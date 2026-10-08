"""P11: New regime monitoring - does Strategy A still work?

Key change: disposal duration went from 10 days to 5 days.
This fundamentally changes the strategy mechanics.

Tests:
  1. Segment returns under new regime
  2. Entry day optimization (Day 1/2/3 with 5-day window)
  3. Exit timing (end of disposal vs next day)
  4. Condition breakdown (limited sample)
  5. Comparison with old regime
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p11_new_regime"
)
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003  # commission + tax + slippage


def main():
    # Load disposal events
    dis = pd.DataFrame(data.get("disposal_information"))
    dis["stock_key"] = dis["symbol"].astype(str).str.lstrip("0")
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis = dis[dis["symbol"].str.match(r"^\d{4}$")].copy()

    # Load prices
    close_p = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    high_p = data.get("price:最高價")
    low_p = data.get("price:最低價")
    vol_p = data.get("price:成交股數")
    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Split regimes
    new = dis[dis["announce"] >= "2026-08-10"].copy()
    old = dis[(dis["announce"] >= "2025-01-01") & (dis["announce"] < "2026-08-10")].copy()

    for subset in [new, old]:
        subset["start_idx"] = cal.searchsorted(subset["start"], side="left")
        subset["end_idx"] = cal.searchsorted(subset["end"], side="right") - 1
        subset["duration_days"] = subset["end_idx"] - subset["start_idx"] + 1

    # Filter valid
    new_valid = new[(new["start_idx"] >= 0) & (new["end_idx"] >= 0) &
                    (new["end_idx"] < n_cal) & (new["duration_days"] >= 2)].copy()
    old_valid = old[(old["start_idx"] >= 0) & (old["end_idx"] >= 0) &
                    (old["end_idx"] < n_cal) & (old["duration_days"] >= 3)].copy()

    print(f"New regime valid events: {len(new_valid):,}")
    print(f"  Duration: median={new_valid['duration_days'].median():.0f} "
          f"mean={new_valid['duration_days'].mean():.1f}")
    print(f"  Date range: {new_valid['announce'].min().date()} to {new_valid['announce'].max().date()}")
    print(f"\nOld regime (2025+) valid events: {len(old_valid):,}")
    print(f"  Duration: median={old_valid['duration_days'].median():.0f}")

    # === Compute returns for both regimes ===
    def compute_returns(subset, label):
        results = []
        stocks = set(close_p.columns)

        for _, row in subset.iterrows():
            sym = row["symbol"]
            if sym not in stocks:
                continue
            si = row["start_idx"]
            ei = row["end_idx"]
            exit_idx = ei + 1
            if exit_idx >= n_cal:
                continue

            try:
                start_open = open_p.iloc[si][sym]
                end_close = close_p.iloc[ei][sym]
                exit_open = open_p.iloc[exit_idx][sym]
                exit_close = close_p.iloc[exit_idx][sym]
            except (IndexError, KeyError):
                continue

            if any(np.isnan(x) or x <= 0 for x in [start_open, end_close, exit_open, exit_close]):
                continue

            # Full period return
            seg1 = end_close / start_open - 1
            seg2 = exit_open / end_close - 1
            seg3 = exit_close / exit_open - 1
            total = exit_close / start_open - 1

            # Entry day returns (test all possible entry days)
            day_returns = {}
            for d in range(1, min(row["duration_days"] + 1, 8)):
                entry_idx = si + d - 1
                if entry_idx > ei:
                    continue
                try:
                    ep = open_p.iloc[entry_idx][sym]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    # Hold to end of disposal
                    day_returns[f"day{d}_to_end"] = end_close / ep - 1
                    # Hold to exit day close
                    day_returns[f"day{d}_to_exit"] = exit_close / ep - 1
                    # Hold to exit day open
                    day_returns[f"day{d}_to_exit_open"] = exit_open / ep - 1
                except (IndexError, KeyError):
                    continue

            # Tradability check
            try:
                prev_close_entry = close_p.iloc[si][sym]  # for day 1 entry
                entry_vol = vol_p.iloc[si][sym]
            except (IndexError, KeyError):
                prev_close_entry = np.nan
                entry_vol = np.nan

            results.append({
                "symbol": sym, "announce": row["announce"],
                "start": row["start"], "end": row["end"],
                "duration": row["duration_days"],
                "condition": row.get("處置條件", ""),
                "seg1_during": seg1, "seg2_gap": seg2,
                "seg3_exit_day": seg3, "total": total,
                **day_returns,
            })

        return pd.DataFrame(results)

    print("\nComputing new regime returns...")
    new_res = compute_returns(new_valid, "NEW")
    print(f"  Computed: {len(new_res):,}")

    print("Computing old regime (2025+) returns...")
    old_res = compute_returns(old_valid, "OLD")
    print(f"  Computed: {len(old_res):,}")

    # === Analysis ===
    print("\n" + "=" * 70)
    print("NEW REGIME ANALYSIS")
    print("=" * 70)

    # 1. Segment returns
    print("\n--- 1. SEGMENT RETURNS ---")
    for label, res in [("NEW REGIME", new_res), ("OLD REGIME (2025+)", old_res)]:
        print(f"\n  {label} (n={len(res):,}):")
        for col, desc in [("seg1_during", "處置期間"),
                          ("seg2_gap", "隔夜跳空"),
                          ("seg3_exit_day", "出關日盤中"),
                          ("total", "合計")]:
            valid = res[col].notna()
            if not valid.any():
                continue
            g = res.loc[valid, col]
            net = g - COST_RATE
            print(f"    {desc}: mean={g.mean():.4%} net={net.mean():.4%} "
                  f"win={(net>0).mean():.1%} median={g.median():.4%}")

    # 2. Entry day optimization
    print("\n--- 2. ENTRY DAY OPTIMIZATION ---")
    for label, res in [("NEW REGIME", new_res), ("OLD REGIME (2025+)", old_res)]:
        print(f"\n  {label} (n={len(res):,}):")
        print(f"    {'Entry':<10} {'Exit':<12} {'n':>5} {'Gross':>8} {'Net':>8} {'Win':>6}")
        for d in range(1, 8):
            for exit_type, exit_label in [("to_end", "處置末"),
                                           ("to_exit", "出關收盤"),
                                           ("to_exit_open", "出關開盤")]:
                col = f"day{d}_{exit_type}"
                if col not in res.columns:
                    continue
                valid = res[col].notna()
                if valid.sum() < 5:
                    continue
                g = res.loc[valid, col]
                net = g - COST_RATE
                print(f"    Day {d:<5} {exit_label:<12} {int(valid.sum()):>5} "
                      f"{g.mean():>8.2%} {net.mean():>8.2%} {(net>0).mean():>6.1%}")

    # 3. Duration distribution impact
    print("\n--- 3. BY DURATION (new regime) ---")
    for dur in sorted(new_res["duration"].unique()):
        sub = new_res[new_res["duration"] == dur]
        if len(sub) < 3:
            continue
        col = "seg1_during"
        valid = sub[col].notna()
        if not valid.any():
            continue
        g = sub.loc[valid, col]
        print(f"  Duration={dur}d: n={len(sub)}, seg1 mean={g.mean():.4%} "
              f"win={(g-COST_RATE>0).mean():.1%}")

    # 4. Condition breakdown (new regime)
    print("\n--- 4. BY CONDITION (new regime) ---")
    for cond, grp in new_res.groupby("condition"):
        if len(grp) < 3:
            continue
        col = "seg1_during"
        valid = grp[col].notna()
        if not valid.any():
            continue
        g = grp.loc[valid, col]
        print(f"  {cond[:30]}: n={len(grp)}, mean={g.mean():.4%} "
              f"win={(g-COST_RATE>0).mean():.1%}")

    # 5. Key structural differences
    print("\n--- 5. STRUCTURAL COMPARISON ---")
    print(f"""
    | 項目 | 舊制 | 新制 |
    |------|------|------|
    | 處置期間 | 10 個交易日 | 5 個交易日 |
    | 交易機制 | 5分鐘撮價 | 5分鐘撮價（不變） |
    | 漲跌幅 | ±10% | ±10%（不變） |
    | 策略A Day 3 進場 | 第3天=30%處置期 | 第3天=60%處置期 |
    | 持有天數 | ~7天 | ~2-3天 |
    """)

    # 6. What entry day works best for 5-day disposal?
    print("\n--- 6. OPTIMAL STRATEGY FOR 5-DAY DISPOSAL ---")
    five_day = new_res[new_res["duration"] == 5]
    if len(five_day) >= 10:
        print(f"  5-day disposal events: n={len(five_day)}")
        print(f"  {'Entry':<10} {'Exit':<12} {'n':>5} {'Gross':>8} {'Net':>8} {'Win':>6}")
        for d in range(1, 6):
            for exit_type, exit_label in [("to_end", "處置末"),
                                           ("to_exit", "出關收盤")]:
                col = f"day{d}_{exit_type}"
                if col not in five_day.columns:
                    continue
                valid = five_day[col].notna()
                if valid.sum() < 5:
                    continue
                g = five_day.loc[valid, col]
                net = g - COST_RATE
                print(f"  Day {d:<5} {exit_label:<12} {int(valid.sum()):>5} "
                      f"{g.mean():>8.2%} {net.mean():>8.2%} {(net>0).mean():>6.1%}")

    # Save results
    new_res.to_csv(OUT / "p11_new_regime_all.csv", index=False, encoding="utf-8-sig")
    old_res.to_csv(OUT / "p11_old_regime_2025.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
