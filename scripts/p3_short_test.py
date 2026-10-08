"""P3: Test short-side alpha (high strength / high momentum = short signal).

Logic:
  - High A4 strength -> high upgrade probability -> disposal -> trading restrictions -> price decline
  - High recent momentum -> mean reversion -> price decline
  - Short = -net_ret (profit when price falls)

Tests:
  1. Short high-strength (Q1 in reversed A4 = Q5 in original)
  2. Short high-momentum (A5 high = short)
  3. Combined short signal
  4. Yearly stability
  5. Can you actually short? (one-word limit down check, borrow availability)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

O2_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_o2\p3_o2_returns.csv"
)
A4_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\a01_clause_parser\a04d_clause1_text_features.csv"
)
ENRICHED_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_followup\p3_enriched_features.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_short"
)
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Load enriched data (has A4 + A5 + A6 + O2 returns)
    enriched = pd.read_csv(ENRICHED_SRC, dtype={"stock_key": str},
                           encoding="utf-8-sig", low_memory=False)
    enriched["attention_date"] = pd.to_datetime(enriched["attention_date"])
    enriched["year"] = enriched["attention_date"].dt.year

    # Also load O2 for can_sell flag
    o2 = pd.read_csv(O2_SRC, dtype={"stock_key": str},
                     encoding="utf-8-sig", low_memory=False)
    o2["attention_date"] = pd.to_datetime(o2["attention_date"])
    can_sell_map = o2.set_index(["stock_key", "attention_date"])["can_buy"]

    enriched = enriched.merge(
        o2[["stock_key", "attention_date", "can_buy", "one_word_limit_down"]]
        .rename(columns={"can_buy": "can_buy_check"}),
        on=["stock_key", "attention_date"], how="left",
    )
    enriched["can_sell"] = ~enriched["one_word_limit_down"].astype(
        str).str.lower().eq("true")

    # Test period, sellable
    test = enriched[(enriched["year"] >= 2023) & enriched["can_sell"]].copy()
    print(f"Test period sellable: {len(test):,}")

    # Short return = -net_ret (profit when price drops)
    for k in [3, 5, 10]:
        test[f"short_net_{k}d"] = -test[f"net_ret_{k}d"]

    # === TEST 1: Short high A4 strength ===
    print("\n" + "=" * 60)
    print("TEST 1: SHORT HIGH A4 STRENGTH (original direction)")
    print("=" * 60)

    results = []
    for k in [3, 5, 10]:
        ret_col = f"short_net_{k}d"
        valid = test[ret_col].notna() & test["margin_to_threshold"].notna()
        vd = test.loc[valid].copy()
        if len(vd) < 100:
            continue
        vd["q"] = pd.qcut(vd["margin_to_threshold"], 5, labels=[1,2,3,4,5],
                          duplicates="drop")
        for q, grp in vd.groupby("q", observed=True):
            if len(grp) < 20:
                continue
            results.append({
                "horizon": k, "quintile": int(q), "n": len(grp),
                "mean_short_net": grp[ret_col].mean(),
                "median_short_net": grp[ret_col].median(),
                "short_win_rate": (grp[ret_col] > 0).mean(),
            })
    short_df = pd.DataFrame(results)
    short_df.to_csv(OUT / "p3_short_a4_quintiles.csv",
                    index=False, encoding="utf-8-sig")
    for k in [3, 5, 10]:
        sub = short_df[short_df["horizon"] == k]
        if not sub.empty:
            print(f"\n--- Short A4: Q5=highest strength (most bearish) ---")
            print(f"  {k}d:")
            print(sub[["quintile", "n", "mean_short_net", "median_short_net",
                       "short_win_rate"]].to_string(index=False))

    # === TEST 2: Short high momentum (A5) ===
    print("\n" + "=" * 60)
    print("TEST 2: SHORT HIGH MOMENTUM (A5 high = short)")
    print("=" * 60)

    mom_results = []
    for k in [3, 5, 10]:
        ret_col = f"short_net_{k}d"
        valid = test[ret_col].notna() & test["ret5"].notna()
        vd = test.loc[valid].copy()
        if len(vd) < 100:
            continue
        vd["q"] = pd.qcut(vd["ret5"], 5, labels=[1,2,3,4,5], duplicates="drop")
        for q, grp in vd.groupby("q", observed=True):
            if len(grp) < 20:
                continue
            mom_results.append({
                "horizon": k, "momentum_quintile": int(q), "n": len(grp),
                "mean_short_net": grp[ret_col].mean(),
                "median_short_net": grp[ret_col].median(),
                "short_win_rate": (grp[ret_col] > 0).mean(),
                "mean_ret5": grp["ret5"].mean(),
            })
    mom_df = pd.DataFrame(mom_results)
    mom_df.to_csv(OUT / "p3_short_momentum.csv", index=False, encoding="utf-8-sig")
    for k in [3, 5]:
        sub = mom_df[mom_df["horizon"] == k]
        if not sub.empty:
            print(f"\n--- Short momentum: Q5=highest ret5 (most bearish) ---")
            print(f"  {k}d:")
            print(sub[["momentum_quintile", "n", "mean_short_net",
                       "median_short_net", "short_win_rate", "mean_ret5"]].to_string(index=False))

    # === TEST 3: Combined short (high A4 + high momentum) ===
    print("\n" + "=" * 60)
    print("TEST 3: COMBINED SHORT (high A4 + high momentum)")
    print("=" * 60)

    for k in [3, 5, 10]:
        ret_col = f"short_net_{k}d"
        valid = (test[ret_col].notna() & test["margin_to_threshold"].notna()
                 & test["ret5"].notna())
        vd = test.loc[valid].copy()
        if len(vd) < 100:
            continue
        # Composite short score: high A4 + high momentum = most bearish
        vd["short_score"] = (vd["margin_to_threshold"].rank(pct=True) +
                             vd["ret5"].rank(pct=True)) / 2
        vd["q"] = pd.qcut(vd["short_score"], 5, labels=[1,2,3,4,5],
                          duplicates="drop")
        print(f"\n  {k}d (Q5=most bearish composite):")
        for q, grp in vd.groupby("q", observed=True):
            if len(grp) < 20:
                continue
            print(f"    Q{int(q)}: n={len(grp)} short_net={grp[ret_col].mean():.4%} "
                  f"win={(grp[ret_col]>0).mean():.1%}")

    # === TEST 4: Yearly stability of short signal ===
    print("\n" + "=" * 60)
    print("TEST 4: YEARLY STABILITY (Short Q5 - Q1 spread)")
    print("=" * 60)

    year_rows = []
    for k in [3, 5]:
        ret_col = f"short_net_{k}d"
        for yr in sorted(test["year"].unique()):
            yr_data = test[test["year"] == yr]
            valid = (yr_data[ret_col].notna() &
                     yr_data["margin_to_threshold"].notna() &
                     yr_data["ret5"].notna())
            vd = yr_data.loc[valid].copy()
            if len(vd) < 50:
                continue
            vd["short_score"] = (vd["margin_to_threshold"].rank(pct=True) +
                                 vd["ret5"].rank(pct=True)) / 2
            try:
                vd["q"] = pd.qcut(vd["short_score"], 5, labels=[1,2,3,4,5],
                                  duplicates="drop")
            except ValueError:
                continue
            q5 = vd.loc[vd["q"] == 5, ret_col].mean()
            q1 = vd.loc[vd["q"] == 1, ret_col].mean()
            year_rows.append({
                "year": yr, "horizon": k, "n": len(vd),
                "q5_short": q5, "q1_short": q1, "spread": q5 - q1,
            })
    yr_df = pd.DataFrame(year_rows)
    yr_df.to_csv(OUT / "p3_short_yearly.csv", index=False, encoding="utf-8-sig")
    for k in [3, 5]:
        sub = yr_df[yr_df["horizon"] == k]
        if not sub.empty:
            print(f"\n  {k}d:")
            print(sub[["year", "n", "q5_short", "q1_short", "spread"]].to_string(index=False))

    # === TEST 5: Short feasibility ===
    print("\n" + "=" * 60)
    print("TEST 5: SHORT FEASIBILITY")
    print("=" * 60)
    total_test = len(enriched[enriched["year"] >= 2023])
    sellable = int(enriched.loc[enriched["year"] >= 2023, "can_sell"].sum())
    print(f"  Test period events: {total_test:,}")
    print(f"  Can sell (not one-word limit down): {sellable:,} ({sellable/total_test:.1%})")
    print(f"  Note: 融券 availability and borrow cost not modeled here.")
    print(f"  融券 in Taiwan requires: stock on eligible list, 6-month lock, ~2.5% annual cost.")
    print(f"  For research purposes, we test price direction; real short cost to be added in P9.")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
