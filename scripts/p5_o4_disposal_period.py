"""P5: O4 disposal period return analysis.

Answers: Can you profit DURING the disposal period? Where does the profit come from?

Segments:
  Segment 1: Buy at disposal start -> hold during disposal -> sell before end
  Segment 2: Last day of disposal -> next day open (overnight gap)
  Segment 3: Exit day open -> exit day close

Entry rules (pre-specified, not cherry-picked):
  Rule A: Fixed day entry (day 1/2/3 of disposal)
  Rule B: Dip entry (price dropped X% from entry)
  Rule C: Strength entry (relative strength > Y%)

All results segmented by: old/new regime, disposal type, tradability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p5_o4"
)
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003


def main():
    # Load disposal events
    dis = pd.DataFrame(data.get("disposal_information"))
    dis["stock_key"] = dis["symbol"].astype(str).str.lstrip("0")
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis = dis[dis["symbol"].str.match(r"^\d{4}$")].copy()
    dis = dis[dis["announce"] >= "2018-01-01"].copy()
    dis["year"] = dis["announce"].dt.year
    dis["regime"] = np.where(dis["announce"] >= "2026-08-10", "new", "old")

    # Compute disposal duration in trading days
    close_p = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    high_p = data.get("price:最高價")
    low_p = data.get("price:最低價")
    vol_p = data.get("price:成交股數")
    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)

    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration_days"] = dis["end_idx"] - dis["start_idx"] + 1

    # Filter valid events
    dis = dis[(dis["start_idx"] >= 0) & (dis["end_idx"] >= 0) &
              (dis["end_idx"] < n_cal) & (dis["duration_days"] >= 3)].copy()
    print(f"Valid disposal events: {len(dis):,}")
    print(f"Duration distribution: median={dis['duration_days'].median():.0f} "
          f"mean={dis['duration_days'].mean():.1f}")
    print(f"By regime: old={int((dis['regime']=='old').sum()):,} "
          f"new={int((dis['regime']=='new').sum()):,}")

    # === Segment returns for each disposal event ===
    results = []
    stocks = set(close_p.columns) & set(dis["stock_key"])

    for _, row in dis.iterrows():
        sym = row["symbol"]
        if sym not in stocks:
            continue
        si = row["start_idx"]
        ei = row["end_idx"]
        exit_idx = ei + 1  # day after disposal ends (exit day)
        if exit_idx >= n_cal:
            continue

        # Get prices
        try:
            start_open = open_p.iloc[si][sym]
            start_close = close_p.iloc[si][sym]
            end_close = close_p.iloc[ei][sym]
            exit_open = open_p.iloc[exit_idx][sym]
            exit_close = close_p.iloc[exit_idx][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in
               [start_open, end_close, exit_open, exit_close]):
            continue

        # Segment 1: Buy at start open -> hold to end close (during disposal)
        seg1 = end_close / start_open - 1
        # Segment 2: End close -> exit day open (overnight gap)
        seg2 = exit_open / end_close - 1
        # Segment 3: Exit day open -> exit day close
        seg3 = exit_close / exit_open - 1
        # Total: start open -> exit close
        total = exit_close / start_open - 1

        # Fixed day entries (Rule A)
        day_returns = {}
        for d in [1, 2, 3]:
            entry_idx = si + d - 1
            if entry_idx > ei:
                continue
            try:
                ep = open_p.iloc[entry_idx][sym]
                if np.isnan(ep) or ep <= 0:
                    continue
                # Hold to end close
                day_returns[f"day{d}_to_end"] = end_close / ep - 1
                # Hold to exit open
                day_returns[f"day{d}_to_exit_open"] = exit_open / ep - 1
                # Hold to exit close
                day_returns[f"day{d}_to_exit_close"] = exit_close / ep - 1
            except (IndexError, KeyError):
                continue

        results.append({
            "symbol": sym, "announce": row["announce"],
            "start": row["start"], "end": row["end"],
            "year": row["year"], "regime": row["regime"],
            "duration": row["duration_days"],
            "condition": row.get("處置條件", ""),
            "seg1_during": seg1, "seg2_gap": seg2,
            "seg3_exit_day": seg3, "total": total,
            **day_returns,
        })

    res = pd.DataFrame(results)
    res.to_csv(OUT / "p5_o4_all_events.csv", index=False, encoding="utf-8-sig")

    # === Summary by segment ===
    print("\n" + "=" * 60)
    print("O4 SEGMENTED RETURNS (all valid events)")
    print("=" * 60)

    for regime in ["old", "new"]:
        sub = res[res["regime"] == regime]
        if len(sub) < 10:
            continue
        print(f"\n--- {regime.upper()} REGIME (n={len(sub):,}) ---")
        for col, label in [("seg1_during", "段1: 處置期間"),
                           ("seg2_gap", "段2: 隔夜跳空"),
                           ("seg3_exit_day", "段3: 出關日盤中"),
                           ("total", "合計: 進場到出關收盤")]:
            valid = sub[col].notna()
            if not valid.any():
                continue
            g = sub.loc[valid, col]
            print(f"  {label}: mean={g.mean():.4%} median={g.median():.4%} "
                  f"win={(g>0).mean():.1%} p5={g.quantile(0.05):.4%}")

    # === Rule A: Fixed day entry ===
    print("\n" + "=" * 60)
    print("RULE A: FIXED DAY ENTRY (old regime, 2018-2026)")
    print("=" * 60)

    old = res[res["regime"] == "old"]
    for d in [1, 2, 3]:
        for exit_type in ["to_end", "to_exit_open", "to_exit_close"]:
            col = f"day{d}_{exit_type}"
            if col not in old.columns:
                continue
            valid = old[col].notna()
            if not valid.any():
                continue
            g = old.loc[valid, col]
            net = g - COST_RATE
            print(f"  Day {d} -> {exit_type}: n={int(valid.sum()):,} "
                  f"mean={g.mean():.4%} net={net.mean():.4%} "
                  f"win_net={(net>0).mean():.1%}")

    # === Yearly stability of Day 3 -> exit close ===
    print("\n--- Day 3 -> exit close by year (old regime) ---")
    col = "day3_to_exit_close"
    if col in old.columns:
        yearly = old.groupby("year").apply(
            lambda g: pd.Series({
                "n": g[col].notna().sum(),
                "mean": g.loc[g[col].notna(), col].mean(),
                "net_mean": (g.loc[g[col].notna(), col] - COST_RATE).mean(),
                "win_net": ((g.loc[g[col].notna(), col] - COST_RATE) > 0).mean(),
            })
        ).reset_index()
        yearly = yearly[yearly["n"] >= 10]
        print(yearly.to_string(index=False))
        yearly.to_csv(OUT / "p5_o4_day3_yearly.csv",
                      index=False, encoding="utf-8-sig")

    # === By disposal condition ===
    print("\n--- By condition (Day 3 -> exit close, old regime) ---")
    if col in old.columns:
        cond = old.groupby("condition").apply(
            lambda g: pd.Series({
                "n": g[col].notna().sum(),
                "mean": g.loc[g[col].notna(), col].mean(),
                "net_mean": (g.loc[g[col].notna(), col] - COST_RATE).mean(),
            })
        ).reset_index()
        cond = cond[cond["n"] >= 20].sort_values("n", ascending=False)
        print(cond.head(10).to_string(index=False))
        cond.to_csv(OUT / "p5_o4_by_condition.csv",
                    index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
