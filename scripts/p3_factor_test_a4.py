"""P3: Test A4 text strength factor against O2 returns.

Steps:
  1. Load O2 returns + A4 text strength features
  2. Group by A4 strength quintile (within armed a3 days)
  3. Report returns by group × tradability layer
  4. Test marginal contribution over A3 baseline
  5. Stability: split by year, check direction consistency

This is the core P3 test for the strongest factor.
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
A3_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\a01_clause_parser\a03_state_table.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_factor_a4"
)
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 3, 5, 10)


def main():
    # Load O2 returns
    o2 = pd.read_csv(O2_SRC, dtype={"stock_key": str},
                     encoding="utf-8-sig", low_memory=False)
    o2["attention_date"] = pd.to_datetime(o2["attention_date"])
    o2["year"] = o2["attention_date"].dt.year
    print(f"O2 returns: {len(o2):,} events")

    # Load A4 features (only a3 armed days with parsed text strength)
    a4 = pd.read_csv(A4_SRC, dtype={"stock_key": str},
                     encoding="utf-8-sig", low_memory=False)
    a4["date"] = pd.to_datetime(a4["date"])
    # Only keep successfully parsed, upward direction
    a4 = a4[(a4["parse_status"] == "ok") & (a4["direction_text"] == "up")].copy()
    print(f"A4 features (up, parsed): {len(a4):,} events")

    # Merge A4 into O2
    merged = o2.merge(
        a4[["stock_key", "date", "reported_pct", "margin_to_threshold",
            "wording", "year"]].rename(columns={"date": "attention_date",
                                                 "year": "a4_year"}),
        on=["stock_key", "attention_date"], how="inner",
    )
    print(f"Merged (A4 ∩ O2): {len(merged):,} events")

    # Filter to test period (2023+) for out-of-sample validation
    test = merged[merged["a4_year"] >= 2023].copy()
    train = merged[merged["a4_year"] <= 2022].copy()
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    # === TEST 1: Grouped returns by A4 strength quintile ===
    print("\n" + "=" * 60)
    print("TEST 1: A4 STRENGTH QUINTILE → O2 RETURNS (TEST PERIOD)")
    print("=" * 60)

    results = []
    for wording in ["收盤價", "最後成交價"]:
        w_data = test[test["wording"] == wording].copy()
        if len(w_data) < 100:
            continue
        try:
            w_data["quintile"] = pd.qcut(
                w_data["margin_to_threshold"], 5, labels=[1, 2, 3, 4, 5],
                duplicates="drop"
            )
        except ValueError:
            continue

        for k in HORIZONS:
            ret_col = f"net_ret_{k}d"
            for layer in ["all", "tradable"]:
                mask = w_data[ret_col].notna()
                if layer == "tradable":
                    mask &= w_data["layer"] == "tradable"
                if not mask.any():
                    continue
                g = w_data.loc[mask]
                for q, grp in g.groupby("quintile", observed=True):
                    if len(grp) < 20:
                        continue
                    results.append({
                        "wording": wording, "horizon": k, "layer": layer,
                        "quintile": int(q), "n": len(grp),
                        "mean_net_ret": grp[ret_col].mean(),
                        "median_net_ret": grp[ret_col].median(),
                        "win_rate": (grp[ret_col] > 0).mean(),
                        "p5": grp[ret_col].quantile(0.05),
                        "p95": grp[ret_col].quantile(0.95),
                        "mean_margin": grp["margin_to_threshold"].mean(),
                    })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "p3_a4_quintile_returns.csv",
                  index=False, encoding="utf-8-sig")

    # Print key results
    for wording in ["收盤價", "最後成交價"]:
        for k in [3, 5]:
            sub = res_df[(res_df["wording"] == wording) &
                         (res_df["horizon"] == k) &
                         (res_df["layer"] == "tradable")]
            if sub.empty:
                continue
            print(f"\n--- {wording}, {k}d, tradable ---")
            print(sub[["quintile", "n", "mean_net_ret", "median_net_ret",
                       "win_rate", "mean_margin"]].to_string(index=False))

    # === TEST 2: Monotonicity test (Spearman) ===
    print("\n" + "=" * 60)
    print("TEST 2: MONOTONICITY (SPEARMAN RANK CORRELATION)")
    print("=" * 60)

    mono_results = []
    for wording in ["收盤價", "最後成交價"]:
        w_data = test[(test["wording"] == wording) &
                      (test["layer"] == "tradable")].copy()
        for k in HORIZONS:
            ret_col = f"net_ret_{k}d"
            valid = w_data[ret_col].notna() & w_data["margin_to_threshold"].notna()
            if valid.sum() < 50:
                continue
            rho, pval = stats.spearmanr(
                w_data.loc[valid, "margin_to_threshold"],
                w_data.loc[valid, ret_col],
            )
            mono_results.append({
                "wording": wording, "horizon": k,
                "n": int(valid.sum()),
                "spearman_rho": rho, "p_value": pval,
            })
    mono_df = pd.DataFrame(mono_results)
    mono_df.to_csv(OUT / "p3_a4_monotonicity.csv",
                   index=False, encoding="utf-8-sig")
    print(mono_df.to_string(index=False))

    # === TEST 3: Year-by-year stability ===
    print("\n" + "=" * 60)
    print("TEST 3: YEAR-BY-YEAR STABILITY (Q5 vs Q1 spread)")
    print("=" * 60)

    year_results = []
    for wording in ["收盤價", "最後成交價"]:
        w_data = merged[(merged["wording"] == wording) &
                        (merged["layer"] == "tradable")].copy()
        for yr in sorted(w_data["a4_year"].unique()):
            yr_data = w_data[w_data["a4_year"] == yr]
            if len(yr_data) < 50:
                continue
            try:
                yr_data["q"] = pd.qcut(
                    yr_data["margin_to_threshold"], 5,
                    labels=[1, 2, 3, 4, 5], duplicates="drop"
                )
            except ValueError:
                continue
            for k in [3, 5]:
                ret_col = f"net_ret_{k}d"
                valid = yr_data[ret_col].notna()
                if valid.sum() < 50:
                    continue
                vd = yr_data.loc[valid]
                q5 = vd.loc[vd["q"] == 5, ret_col].mean()
                q1 = vd.loc[vd["q"] == 1, ret_col].mean()
                year_results.append({
                    "wording": wording, "year": yr, "horizon": k,
                    "n": int(valid.sum()),
                    "q5_mean": q5, "q1_mean": q1,
                    "spread": q5 - q1,
                })
    year_df = pd.DataFrame(year_results)
    year_df.to_csv(OUT / "p3_a4_yearly_stability.csv",
                   index=False, encoding="utf-8-sig")
    # Print spread summary
    for wording in ["收盤價", "最後成交價"]:
        sub = year_df[(year_df["wording"] == wording) &
                      (year_df["horizon"] == 5)]
        if sub.empty:
            continue
        print(f"\n--- {wording}, 5d, Q5-Q1 spread by year ---")
        print(sub[["year", "n", "q5_mean", "q1_mean", "spread"]].to_string(index=False))

    # === TEST 4: Marginal contribution (regression) ===
    print("\n" + "=" * 60)
    print("TEST 4: MARGINAL CONTRIBUTION (OLS with cluster SE)")
    print("=" * 60)

    # Simple OLS: net_ret ~ margin_to_threshold + controls
    # Controls: streak, c10 (as proxies for A3 state)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    reg_data = test[test["layer"] == "tradable"].copy()
    reg_data["log_margin"] = np.log1p(reg_data["margin_to_threshold"].clip(lower=0))

    for k in [3, 5]:
        ret_col = f"net_ret_{k}d"
        valid = reg_data[ret_col].notna() & reg_data["log_margin"].notna()
        rd = reg_data.loc[valid].copy()
        if len(rd) < 100:
            continue

        # Model 1: baseline (A3 proxies only)
        rd["c10_clip"] = rd["c10"].clip(upper=7)
        rd["streak_clip"] = rd["streak"].clip(upper=6)
        X_base = pd.DataFrame({
            "c10": rd["c10_clip"], "streak": rd["streak_clip"],
        })
        X_base = sm.add_constant(X_base)
        m1 = sm.OLS(rd[ret_col], X_base).fit(
            cov_type="cluster", cov_kwds={"groups": rd["stock_key"]})

        # Model 2: add A4
        X_full = X_base.copy()
        X_full["log_margin"] = rd["log_margin"]
        m2 = sm.OLS(rd[ret_col], X_full).fit(
            cov_type="cluster", cov_kwds={"groups": rd["stock_key"]})

        print(f"\n--- {k}d horizon, n={len(rd):,} ---")
        print(f"  A4 coefficient: {m2.params['log_margin']:.6f} "
              f"(t={m2.tvalues['log_margin']:.2f}, p={m2.pvalues['log_margin']:.4f})")
        print(f"  R2 base: {m1.rsquared:.4f} -> full: {m2.rsquared:.4f} "
              f"(dR2={m2.rsquared - m1.rsquared:.4f})")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
