"""P8: Factor combination for Strategy A enhancement.

Goal: Can reversed A4 (low text strength) + reversed A5 (low momentum) 
improve Strategy A (disposal period long) by filtering/ranking which 
disposal stocks to buy?

Timeline:
  Attention announced -> A4/A5 computed -> Disposal starts -> Day 3 entry -> Exit
  (A4/A5 available BEFORE entry, valid as pre-entry filter)

Tests:
  1. Link attention events to disposal events
  2. IC of reversed A4, reversed A5, composite within disposal universe
  3. Quintile returns (does the factor add value?)
  4. Enhanced Strategy A: only buy top-quartile composite
  5. Yearly stability of enhanced strategy
  6. Compare: base A vs enhanced A
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

P7A_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p7_strategy_a\p7_all_trades.csv"
)
ENRICHED_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_followup\p3_enriched_features.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p8_factor_combo"
)
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # === Load P7A trades (disposal universe) ===
    p7a = pd.read_csv(P7A_SRC, encoding="utf-8-sig")
    p7a["stock_key"] = p7a["symbol"].astype(str).str.lstrip("0")
    p7a["entry_date"] = pd.to_datetime(p7a["entry_date"])
    p7a["exit_date"] = pd.to_datetime(p7a["exit_date"])
    # Only executable trades
    p7a = p7a[p7a["executable"]].copy()
    print(f"P7A executable trades: {len(p7a):,}")

    # === Load enriched features (attention events with A4/A5) ===
    enriched = pd.read_csv(ENRICHED_SRC, dtype={"stock_key": str},
                           encoding="utf-8-sig", low_memory=False)
    enriched["attention_date"] = pd.to_datetime(enriched["attention_date"])
    print(f"Enriched attention events: {len(enriched):,}")

    # === Link: For each disposal trade, find the most recent preceding attention ===
    # Strategy: merge_asof on stock_key, matching attention_date <= entry_date
    enriched_sorted = enriched.sort_values("attention_date").reset_index(drop=True)
    p7a_sorted = p7a.sort_values("entry_date").reset_index(drop=True)

    linked = pd.merge_asof(
        p7a_sorted,
        enriched_sorted[["stock_key", "attention_date", "margin_to_threshold",
                         "ret5", "ret10", "ret20", "vol20"]],
        left_on="entry_date",
        right_on="attention_date",
        by="stock_key",
        direction="backward",
        tolerance=pd.Timedelta("30D"),  # attention must be within 30 days before entry
    )

    # Check match rate
    matched = linked["margin_to_threshold"].notna().sum()
    print(f"Matched (attention within 30d before entry): {matched:,} / {len(linked):,} ({matched/len(linked):.1%})")

    # Filter to matched only
    linked = linked[linked["margin_to_threshold"].notna() & linked["ret5"].notna()].copy()
    print(f"Usable for factor test: {len(linked):,}")

    # === Build factor scores ===
    # Reversed A4: LOW strength = good for long → negate
    linked["rev_a4"] = -linked["margin_to_threshold"]
    # Reversed A5: LOW momentum = good for long → negate
    linked["rev_a5"] = -linked["ret5"]

    # Composite: equal-weight z-scores
    linked["z_rev_a4"] = (linked["rev_a4"] - linked["rev_a4"].mean()) / linked["rev_a4"].std()
    linked["z_rev_a5"] = (linked["rev_a5"] - linked["rev_a5"].mean()) / linked["rev_a5"].std()
    linked["composite"] = (linked["z_rev_a4"] + linked["z_rev_a5"]) / 2

    # === TEST 1: IC Analysis ===
    print("\n" + "=" * 70)
    print("TEST 1: INFORMATION COEFFICIENT (within disposal universe)")
    print("=" * 70)

    ret_col = "net_return"
    factors = {
        "rev_a4 (低強度=做多)": "rev_a4",
        "rev_a5 (低動量=做多)": "rev_a5",
        "composite (等權組合)": "composite",
        "raw_a4 (高強度)": "margin_to_threshold",
        "raw_a5 (高動量)": "ret5",
    }

    ic_results = []
    for name, col in factors.items():
        valid = linked[col].notna() & linked[ret_col].notna()
        if valid.sum() < 50:
            continue
        ic, pval = stats.spearmanr(linked.loc[valid, col], linked.loc[valid, ret_col])
        ic_results.append({"factor": name, "IC": ic, "p_value": pval,
                          "n": int(valid.sum())})
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {name}: IC={ic:.4f} p={pval:.4f} {sig} (n={int(valid.sum()):,})")

    ic_df = pd.DataFrame(ic_results)
    ic_df.to_csv(OUT / "p8_ic_results.csv", index=False, encoding="utf-8-sig")

    # === TEST 2: Quintile Returns ===
    print("\n" + "=" * 70)
    print("TEST 2: QUINTILE RETURNS (composite factor)")
    print("=" * 70)

    valid = linked["composite"].notna() & linked[ret_col].notna()
    vd = linked[valid].copy()
    vd["q"] = pd.qcut(vd["composite"], 5, labels=[1,2,3,4,5], duplicates="drop")

    quintile_rows = []
    for q, grp in vd.groupby("q", observed=True):
        quintile_rows.append({
            "quintile": int(q), "n": len(grp),
            "mean_net": grp[ret_col].mean(),
            "median_net": grp[ret_col].median(),
            "win_rate": (grp[ret_col] > 0).mean(),
            "p5_net": grp[ret_col].quantile(0.05),
        })
    q_df = pd.DataFrame(quintile_rows)
    print(q_df.to_string(index=False))
    q_df.to_csv(OUT / "p8_quintile_returns.csv", index=False, encoding="utf-8-sig")

    spread = q_df.iloc[-1]["mean_net"] - q_df.iloc[0]["mean_net"]
    print(f"\n  Q5-Q1 spread: {spread:.4%}")

    # === TEST 3: Enhanced Strategy A (top quartile only) ===
    print("\n" + "=" * 70)
    print("TEST 3: ENHANCED STRATEGY A (top 25% composite only)")
    print("=" * 70)

    # Base strategy (all executable)
    base = linked[ret_col].dropna()
    print(f"\n  Base Strategy A: n={len(base):,} mean={base.mean():.4%} "
          f"win={(base>0).mean():.1%}")

    # Enhanced: top quartile
    top_q = vd[vd["q"] == 5]
    top_ret = top_q[ret_col]
    print(f"  Enhanced (Q5 only): n={len(top_ret):,} mean={top_ret.mean():.4%} "
          f"win={(top_ret>0).mean():.1%}")

    # Top tercile (more practical)
    vd["t"] = pd.qcut(vd["composite"], 3, labels=[1,2,3], duplicates="drop")
    top_t = vd[vd["t"] == 3]
    top_t_ret = top_t[ret_col]
    print(f"  Enhanced (top 1/3): n={len(top_t_ret):,} mean={top_t_ret.mean():.4%} "
          f"win={(top_t_ret>0).mean():.1%}")

    # Bottom quartile (skip these)
    bot_q = vd[vd["q"] == 1]
    bot_ret = bot_q[ret_col]
    print(f"  Bottom Q1 (skip): n={len(bot_ret):,} mean={bot_ret.mean():.4%} "
          f"win={(bot_ret>0).mean():.1%}")

    # Improvement
    improvement = top_ret.mean() - base.mean()
    print(f"\n  Improvement (Q5 vs base): {improvement:+.4%}")
    print(f"  Avoidance (skip Q1): base improves by {base.mean() - vd[vd['q'].isin([2,3,4,5])][ret_col].mean():.4%} if Q1 removed")

    # === TEST 4: Yearly stability of enhanced strategy ===
    print("\n" + "=" * 70)
    print("TEST 4: YEARLY STABILITY (Enhanced Q5 vs Base)")
    print("=" * 70)

    yearly_rows = []
    for yr in sorted(vd["year"].unique()):
        yr_all = vd[vd["year"] == yr]
        yr_q5 = yr_all[yr_all["q"] == 5]
        yr_q1 = yr_all[yr_all["q"] == 1]
        if len(yr_all) < 20:
            continue
        row = {
            "year": yr,
            "n_base": len(yr_all),
            "base_mean": yr_all[ret_col].mean(),
            "n_q5": len(yr_q5),
            "q5_mean": yr_q5[ret_col].mean() if len(yr_q5) >= 5 else np.nan,
            "q5_win": (yr_q5[ret_col] > 0).mean() if len(yr_q5) >= 5 else np.nan,
            "n_q1": len(yr_q1),
            "q1_mean": yr_q1[ret_col].mean() if len(yr_q1) >= 5 else np.nan,
        }
        yearly_rows.append(row)

    yr_df = pd.DataFrame(yearly_rows)
    print(yr_df.to_string(index=False))
    yr_df.to_csv(OUT / "p8_yearly_comparison.csv", index=False, encoding="utf-8-sig")

    # Count positive years
    base_pos = (yr_df["base_mean"] > 0).sum()
    q5_pos = (yr_df["q5_mean"].dropna() > 0).sum()
    q5_total = yr_df["q5_mean"].notna().sum()
    print(f"\n  Base positive years: {base_pos}/{len(yr_df)}")
    print(f"  Q5 positive years: {q5_pos}/{q5_total}")

    # === TEST 5: IC by year (factor stability) ===
    print("\n" + "=" * 70)
    print("TEST 5: IC BY YEAR (factor stability)")
    print("=" * 70)

    ic_yearly = []
    for yr in sorted(vd["year"].unique()):
        yr_data = vd[vd["year"] == yr]
        if len(yr_data) < 30:
            continue
        ic_comp, p_comp = stats.spearmanr(yr_data["composite"], yr_data[ret_col])
        ic_a4, p_a4 = stats.spearmanr(yr_data["rev_a4"], yr_data[ret_col])
        ic_a5, p_a5 = stats.spearmanr(yr_data["rev_a5"], yr_data[ret_col])
        ic_yearly.append({
            "year": yr, "n": len(yr_data),
            "ic_composite": ic_comp, "p_composite": p_comp,
            "ic_rev_a4": ic_a4, "p_rev_a4": p_a4,
            "ic_rev_a5": ic_a5, "p_rev_a5": p_a5,
        })

    ic_yr_df = pd.DataFrame(ic_yearly)
    print(ic_yr_df.to_string(index=False))
    ic_yr_df.to_csv(OUT / "p8_ic_by_year.csv", index=False, encoding="utf-8-sig")

    # === TEST 6: Signal frequency impact ===
    print("\n" + "=" * 70)
    print("TEST 6: SIGNAL FREQUENCY IMPACT")
    print("=" * 70)

    # How many signals per month if we filter to Q5?
    vd["month"] = pd.to_datetime(vd["entry_date"]).dt.to_period("M")
    monthly_all = vd.groupby("month").size()
    monthly_q5 = vd[vd["q"] == 5].groupby("month").size()
    print(f"  Base: monthly avg={monthly_all.mean():.1f}, median={monthly_all.median():.0f}")
    print(f"  Q5 only: monthly avg={monthly_q5.mean():.1f}, median={monthly_q5.median():.0f}")
    print(f"  Signal reduction: {(1 - monthly_q5.mean()/monthly_all.mean()):.0%}")

    # With Q5, how many concurrent positions?
    q5_trades = vd[vd["q"] == 5].copy()
    q5_trades["entry_dt"] = pd.to_datetime(q5_trades["entry_date"])
    q5_trades["exit_dt"] = pd.to_datetime(q5_trades["exit_date"])
    all_dates = pd.date_range(q5_trades["entry_dt"].min(), q5_trades["exit_dt"].max(), freq="B")
    concurrent = []
    for d in all_dates:
        n = ((q5_trades["entry_dt"] <= d) & (q5_trades["exit_dt"] >= d)).sum()
        concurrent.append(n)
    concurrent = pd.Series(concurrent)
    print(f"  Q5 concurrent positions: mean={concurrent.mean():.1f}, median={concurrent.median():.0f}, P95={concurrent.quantile(0.95):.0f}")

    # === TEST 7: Practical recommendation ===
    print("\n" + "=" * 70)
    print("TEST 7: PRACTICAL RECOMMENDATION")
    print("=" * 70)

    # Compare: base vs Q5 vs top-half (Q4+Q5)
    top_half = vd[vd["q"].isin([4, 5])]
    th_ret = top_half[ret_col]
    print(f"\n  {'Strategy':<25} {'n':>6} {'Mean':>8} {'Win':>6} {'Monthly':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Base (all executable)':<25} {len(base):>6,} {base.mean():>8.2%} {(base>0).mean():>6.1%} {monthly_all.mean():>8.1f}")
    print(f"  {'Top half (Q4+Q5)':<25} {len(th_ret):>6,} {th_ret.mean():>8.2%} {(th_ret>0).mean():>6.1%} {vd[vd['q'].isin([4,5])].groupby('month').size().mean():>8.1f}")
    print(f"  {'Top quartile (Q5)':<25} {len(top_ret):>6,} {top_ret.mean():>8.2%} {(top_ret>0).mean():>6.1%} {monthly_q5.mean():>8.1f}")

    # Save linked data for further analysis
    linked.to_csv(OUT / "p8_linked_trades.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
