"""P10: Condition-weighted allocation + Sector/Size analysis.

Part 1: Condition-weighted allocation
  - Test if overweighting high-return conditions improves portfolio
  - Check yearly stability of condition returns

Part 2: Sector/Size analysis
  - Industry from stock code prefix (broad groups)
  - Size from average daily trading value before event
  - Test if returns differ by group
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

P7A_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p7_strategy_a\p7_all_trades.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p10_condition_sector"
)
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003  # commission + tax + slippage


def classify_industry(symbol: str) -> str:
    """Map stock code to broad industry group based on TWSE code ranges."""
    code = int(symbol)
    if 1100 <= code <= 1299:
        return "水泥"
    elif 1300 <= code <= 1399:
        return "塑膠"
    elif 1400 <= code <= 1499:
        return "紡織"
    elif 1500 <= code <= 1699:
        return "電器電纜"
    elif 1700 <= code <= 1799:
        return "食品"
    elif 1800 <= code <= 1999:
        return "建材造紙"
    elif 2000 <= code <= 2199:
        return "鋼鐵車輛"
    elif 2200 <= code <= 2299:
        return "交通工具"
    elif 2300 <= code <= 2499:
        return "電子"
    elif 2500 <= code <= 2699:
        return "營建"
    elif 2700 <= code <= 2799:
        return "休閒"
    elif 2800 <= code <= 2899:
        return "金融保險"
    elif 2900 <= code <= 2999:
        return "零售"
    elif 3000 <= code <= 3199:
        return "貿易"
    elif 3200 <= code <= 3699:
        return "電子"
    elif 3700 <= code <= 3999:
        return "電子"
    elif 4100 <= code <= 4599:
        return "生化機械"
    elif 4600 <= code <= 4999:
        return "機械電子"
    elif 5000 <= code <= 5499:
        return "金屬化工"
    elif 5500 <= code <= 5999:
        return "電子"
    elif 6000 <= code <= 6999:
        return "電子"
    elif 7000 <= code <= 7999:
        return "服務"
    elif 8000 <= code <= 8999:
        return "電子"
    elif 9000 <= code <= 9999:
        return "其他"
    else:
        return "其他"


def main():
    # Load P7A trades
    p7a = pd.read_csv(P7A_SRC, encoding="utf-8-sig")
    p7a = p7a[p7a["executable"]].copy()
    p7a["entry_date"] = pd.to_datetime(p7a["entry_date"])
    p7a["exit_date"] = pd.to_datetime(p7a["exit_date"])
    p7a["symbol"] = p7a["symbol"].astype(str).str.zfill(4)

    # Add live return (with slippage)
    p7a["live_net"] = p7a["net_return"] - 0.003  # subtract slippage

    print(f"Loaded {len(p7a):,} executable trades")

    # === PART 1: CONDITION-WEIGHTED ALLOCATION ===
    print("\n" + "=" * 70)
    print("PART 1: CONDITION-WEIGHTED ALLOCATION")
    print("=" * 70)

    # 1a. Condition returns with yearly stability
    print("\n--- 1a. Condition Returns by Year ---")
    cond_yearly = p7a.groupby(["condition", "year"]).agg(
        n=("live_net", "count"),
        mean_net=("live_net", "mean"),
        win=("live_net", lambda x: (x > 0).mean()),
    ).reset_index()

    # Only conditions with enough data
    cond_totals = p7a.groupby("condition").agg(
        n=("live_net", "count"),
        mean_net=("live_net", "mean"),
        median_net=("live_net", "median"),
        win=("live_net", lambda x: (x > 0).mean()),
        p5=("live_net", lambda x: x.quantile(0.05)),
    ).reset_index().sort_values("mean_net", ascending=False)
    cond_totals = cond_totals[cond_totals["n"] >= 30]

    print("\nOverall condition performance (n>=30):")
    print(cond_totals.to_string(index=False))
    cond_totals.to_csv(OUT / "p10_condition_overall.csv",
                       index=False, encoding="utf-8-sig")

    # Yearly stability per condition
    print("\n--- Yearly stability (top conditions) ---")
    top_conditions = cond_totals.head(6)["condition"].tolist()
    for cond in top_conditions:
        sub = cond_yearly[cond_yearly["condition"] == cond]
        sub = sub[sub["n"] >= 10].sort_values("year")
        pos_years = (sub["mean_net"] > 0).sum()
        total_years = len(sub)
        print(f"\n  {cond[:30]}... ({pos_years}/{total_years} years positive)")
        print(f"    {'Year':<6} {'n':>5} {'Mean':>8} {'Win':>6}")
        for _, r in sub.iterrows():
            print(f"    {int(r['year']):<6} {int(r['n']):>5} {r['mean_net']:>8.2%} {r['win']:>6.1%}")

    # 1b. Weighted allocation test
    print("\n" + "=" * 70)
    print("1b. ALLOCATION STRATEGIES COMPARISON")
    print("=" * 70)

    # Strategy 1: Equal weight (baseline)
    baseline = p7a["live_net"]
    print(f"\n  Equal weight (baseline): mean={baseline.mean():.4%} "
          f"win={(baseline>0).mean():.1%}")

    # Strategy 2: Condition-weighted (proportional to historical edge)
    # Weight = max(0, condition_mean - overall_mean) normalized
    overall_mean = baseline.mean()
    cond_means = p7a.groupby("condition")["live_net"].mean()
    cond_counts = p7a.groupby("condition")["live_net"].count()

    # Only use conditions with n >= 50 for stable estimates
    valid_conds = cond_counts[cond_counts >= 50].index
    cond_means_valid = cond_means[cond_means.index.isin(valid_conds)]

    # Edge = how much better than average
    edges = (cond_means_valid - overall_mean).clip(lower=0)
    weights = edges / edges.sum() if edges.sum() > 0 else pd.Series(1/len(edges), index=edges.index)

    print(f"\n  Condition weights (edge-proportional):")
    weight_df = pd.DataFrame({
        "condition": weights.index,
        "weight": weights.values,
        "mean_return": cond_means_valid.values,
        "n": cond_counts[cond_means_valid.index].values,
    }).sort_values("weight", ascending=False)
    print(weight_df.to_string(index=False))

    # Simulate weighted: for each trade, multiply return by its condition weight
    # (This is a simplified version - real implementation would adjust position size)
    p7a["cond_weight"] = p7a["condition"].map(weights).fillna(0)
    # Normalize so average weight = 1 (same total capital deployed)
    p7a["cond_weight"] = p7a["cond_weight"] / p7a["cond_weight"].mean()

    weighted_returns = p7a["live_net"] * p7a["cond_weight"]
    print(f"\n  Condition-weighted: mean={weighted_returns.mean():.4%} "
          f"win={(weighted_returns>0).mean():.1%}")
    print(f"  Improvement: {weighted_returns.mean() - baseline.mean():+.4%}")

    # Strategy 3: Tiered (high/medium/low based on condition performance)
    # Tier 1 (2x weight): conditions with mean > 6%
    # Tier 2 (1x weight): conditions with mean 3-6%
    # Tier 3 (0.5x weight): conditions with mean < 3%
    tier_map = {}
    for cond in cond_means_valid.index:
        m = cond_means_valid[cond]
        if m > 0.06:
            tier_map[cond] = 2.0
        elif m > 0.03:
            tier_map[cond] = 1.0
        else:
            tier_map[cond] = 0.5

    p7a["tier_weight"] = p7a["condition"].map(tier_map).fillna(1.0)
    p7a["tier_weight"] = p7a["tier_weight"] / p7a["tier_weight"].mean()
    tier_returns = p7a["live_net"] * p7a["tier_weight"]
    print(f"\n  Tiered (2x/1x/0.5x): mean={tier_returns.mean():.4%} "
          f"win={(tier_returns>0).mean():.1%}")
    print(f"  Improvement: {tier_returns.mean() - baseline.mean():+.4%}")

    # Strategy 4: Skip worst conditions entirely
    worst_conds = cond_means_valid[cond_means_valid < overall_mean * 0.5].index
    skip_mask = ~p7a["condition"].isin(worst_conds)
    skip_returns = p7a.loc[skip_mask, "live_net"]
    print(f"\n  Skip worst conditions ({len(worst_conds)} skipped): "
          f"mean={skip_returns.mean():.4%} win={(skip_returns>0).mean():.1%}")
    print(f"  Improvement: {skip_returns.mean() - baseline.mean():+.4%}")
    print(f"  Signal loss: {(1-skip_mask.mean()):.1%}")

    # Yearly check of weighted strategy
    print("\n--- Yearly check: Tiered vs Baseline ---")
    p7a["yr"] = p7a["entry_date"].dt.year
    yr_compare = p7a.groupby("yr").agg(
        base_mean=("live_net", "mean"),
        tier_mean=("live_net", lambda x: (x * p7a.loc[x.index, "tier_weight"]).mean()),
        n=("live_net", "count"),
    ).reset_index()
    yr_compare["improvement"] = yr_compare["tier_mean"] - yr_compare["base_mean"]
    print(yr_compare.to_string(index=False))
    yr_compare.to_csv(OUT / "p10_weighted_yearly.csv",
                      index=False, encoding="utf-8-sig")

    # === PART 2: SECTOR / SIZE ANALYSIS ===
    print("\n" + "=" * 70)
    print("PART 2: SECTOR & SIZE ANALYSIS")
    print("=" * 70)

    # 2a. Industry classification
    p7a["industry"] = p7a["symbol"].apply(classify_industry)

    # Group small industries into broader categories
    industry_map = {
        "水泥": "傳產", "塑膠": "傳產", "紡織": "傳產", "電器電纜": "傳產",
        "食品": "傳產", "建材造紙": "傳產", "鋼鐵車輛": "傳產", "交通工具": "傳產",
        "營建": "傳產", "休閒": "傳產", "零售": "傳產", "貿易": "傳產",
        "金屬化工": "傳產", "生化機械": "傳產", "機械電子": "傳產",
        "電子": "電子",
        "金融保險": "金融",
        "服務": "其他", "其他": "其他",
    }
    p7a["industry_broad"] = p7a["industry"].map(industry_map).fillna("其他")

    # Industry returns
    ind_stats = p7a.groupby("industry_broad").agg(
        n=("live_net", "count"),
        mean_net=("live_net", "mean"),
        median_net=("live_net", "median"),
        win=("live_net", lambda x: (x > 0).mean()),
        p5=("live_net", lambda x: x.quantile(0.05)),
        std=("live_net", "std"),
    ).reset_index().sort_values("mean_net", ascending=False)

    print("\n--- Industry (broad) ---")
    print(ind_stats.to_string(index=False))
    ind_stats.to_csv(OUT / "p10_industry_broad.csv",
                     index=False, encoding="utf-8-sig")

    # Detailed industry
    ind_detail = p7a.groupby("industry").agg(
        n=("live_net", "count"),
        mean_net=("live_net", "mean"),
        win=("live_net", lambda x: (x > 0).mean()),
    ).reset_index()
    ind_detail = ind_detail[ind_detail["n"] >= 20].sort_values("mean_net", ascending=False)
    print("\n--- Industry (detailed, n>=20) ---")
    print(ind_detail.to_string(index=False))
    ind_detail.to_csv(OUT / "p10_industry_detail.csv",
                      index=False, encoding="utf-8-sig")

    # 2b. Size proxy: average daily trading value before event
    print("\nLoading trading value for size proxy...")
    amount = data.get("price:成交金額")
    cal = pd.DatetimeIndex(amount.index).normalize().unique().sort_values()

    # For each trade, get avg daily amount in 20 days before entry
    size_data = []
    stocks_in_amount = set(amount.columns)
    for _, row in p7a.iterrows():
        sym = row["symbol"]
        if sym not in stocks_in_amount:
            size_data.append(np.nan)
            continue
        entry_idx = cal.searchsorted(row["entry_date"], side="left")
        start_idx = max(0, entry_idx - 20)
        try:
            avg_amt = amount.iloc[start_idx:entry_idx][sym].mean()
            size_data.append(avg_amt)
        except (IndexError, KeyError):
            size_data.append(np.nan)

    p7a["avg_daily_amount"] = size_data

    # Size quintiles
    valid_size = p7a["avg_daily_amount"].notna()
    p7a_size = p7a[valid_size].copy()
    p7a_size["size_q"] = pd.qcut(p7a_size["avg_daily_amount"], 5,
                                  labels=["極小盤", "小盤", "中盤", "中大盤", "大盤"],
                                  duplicates="drop")

    size_stats = p7a_size.groupby("size_q", observed=True).agg(
        n=("live_net", "count"),
        mean_net=("live_net", "mean"),
        median_net=("live_net", "median"),
        win=("live_net", lambda x: (x > 0).mean()),
        p5=("live_net", lambda x: x.quantile(0.05)),
        avg_amount=("avg_daily_amount", "mean"),
    ).reset_index()

    print("\n--- Size (by avg daily trading value, 5 quintiles) ---")
    print(size_stats.to_string(index=False))
    size_stats.to_csv(OUT / "p10_size_quintiles.csv",
                      index=False, encoding="utf-8-sig")

    # Size by year stability
    print("\n--- Size x Year (top 2 vs bottom 2) ---")
    p7a_size["size_group"] = p7a_size["size_q"].map({
        "極小盤": "小盤", "小盤": "小盤", "中盤": "中盤",
        "中大盤": "大盤", "大盤": "大盤"
    })
    size_year = p7a_size.groupby(["size_group", "year"]).agg(
        n=("live_net", "count"),
        mean_net=("live_net", "mean"),
    ).reset_index()
    size_year = size_year[size_year["n"] >= 10]
    pivot = size_year.pivot_table(index="year", columns="size_group",
                                   values="mean_net")
    print(pivot.to_string())

    # 2c. Industry x Size interaction
    print("\n--- Industry x Size (mean return) ---")
    p7a_size["industry_broad"] = p7a_size["industry"].map(industry_map).fillna("其他")
    cross = p7a_size.groupby(["industry_broad", "size_q"], observed=True).agg(
        n=("live_net", "count"),
        mean_net=("live_net", "mean"),
    ).reset_index()
    cross = cross[cross["n"] >= 20]
    cross_pivot = cross.pivot_table(index="industry_broad", columns="size_q",
                                     values="mean_net")
    print(cross_pivot.to_string())
    cross.to_csv(OUT / "p10_industry_size.csv", index=False, encoding="utf-8-sig")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    # Best conditions
    best_conds = cond_totals.head(3)
    print(f"\n  Top conditions to overweight:")
    for _, r in best_conds.iterrows():
        print(f"    {r['condition'][:30]}: {r['mean_net']:.2%} (n={int(r['n'])})")

    # Industry insight
    best_ind = ind_stats.iloc[0]
    worst_ind = ind_stats.iloc[-1]
    print(f"\n  Best industry: {best_ind['industry_broad']} ({best_ind['mean_net']:.2%})")
    print(f"  Worst industry: {worst_ind['industry_broad']} ({worst_ind['mean_net']:.2%})")

    # Size insight
    best_size = size_stats.iloc[size_stats["mean_net"].idxmax()]
    print(f"\n  Best size: {best_size['size_q']} ({best_size['mean_net']:.2%})")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
