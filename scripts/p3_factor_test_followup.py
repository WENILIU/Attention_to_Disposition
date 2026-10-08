"""P3 Follow-up: Reverse A4, non-upgrade path, A5/A6 incremental test.

Tests:
  1. Reverse A4: use -margin_to_threshold as long signal
  2. Non-upgrade path: low-strength armed days that DON'T get disposed
  3. A5 (price momentum) and A6 (volatility/turnover) marginal contribution
     over reversed-A4 baseline
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from finlab import data

O2_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_o2\p3_o2_returns.csv"
)
A4_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\a01_clause_parser\a04d_clause1_text_features.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_followup"
)
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 3, 5, 10)


def load_merged():
    o2 = pd.read_csv(O2_SRC, dtype={"stock_key": str},
                     encoding="utf-8-sig", low_memory=False)
    o2["attention_date"] = pd.to_datetime(o2["attention_date"])
    o2["year"] = o2["attention_date"].dt.year

    a4 = pd.read_csv(A4_SRC, dtype={"stock_key": str},
                     encoding="utf-8-sig", low_memory=False)
    a4["date"] = pd.to_datetime(a4["date"])
    a4 = a4[(a4["parse_status"] == "ok") & (a4["direction_text"] == "up")].copy()

    merged = o2.merge(
        a4[["stock_key", "date", "margin_to_threshold", "wording",
            "disposal_tomorrow"]].rename(columns={"date": "attention_date"}),
        on=["stock_key", "attention_date"], how="inner",
    )
    merged["disposal_tomorrow"] = merged["disposal_tomorrow"].astype(
        str).str.lower().eq("true")
    merged["layer"] = merged["layer"].astype(str)
    return merged


def compute_a5_a6_features(merged):
    """Compute A5 (momentum) and A6 (volatility/turnover) for each event."""
    close_p = data.get("price:收盤價")
    vol_p = data.get("price:成交股數")
    amount_p = data.get("price:成交金額")
    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()

    merged = merged.copy()
    merged["t_idx"] = merged["t_idx"].astype(int)

    # A5: 5/10/20 day momentum (return over past N days)
    # A6: 20-day volatility (std of daily returns), turnover level
    n = len(merged)
    ret5 = np.full(n, np.nan)
    ret10 = np.full(n, np.nan)
    ret20 = np.full(n, np.nan)
    vol20 = np.full(n, np.nan)
    vol_ratio = np.full(n, np.nan)  # volume vs 20d avg

    stocks = set(close_p.columns) & set(merged["stock_key"])
    for key in stocks:
        mask = (merged["stock_key"] == key).to_numpy()
        idx = np.where(mask)[0]
        t = merged.iloc[idx]["t_idx"].to_numpy()
        valid = (t >= 20) & (t < len(cal))
        if not valid.any():
            continue
        vi = idx[valid]
        vt = t[valid]
        col = key
        # Momentum: close[t] / close[t-k] - 1
        c_now = close_p.iloc[vt][col].to_numpy()
        c5 = close_p.iloc[vt - 5][col].to_numpy()
        c10 = close_p.iloc[vt - 10][col].to_numpy()
        c20 = close_p.iloc[vt - 20][col].to_numpy()
        ret5[vi] = np.where((c5 > 0) & ~np.isnan(c5), c_now / c5 - 1, np.nan)
        ret10[vi] = np.where((c10 > 0) & ~np.isnan(c10), c_now / c10 - 1, np.nan)
        ret20[vi] = np.where((c20 > 0) & ~np.isnan(c20), c_now / c20 - 1, np.nan)
        # Volatility: std of daily returns over 20 days
        daily_ret = close_p[col].pct_change(fill_method=None)
        vol_series = daily_ret.rolling(20).std()
        vol20[vi] = vol_series.iloc[vt].to_numpy()
        # Volume ratio: today's volume / 20d avg volume
        v_now = vol_p.iloc[vt][col].to_numpy()
        v_avg = vol_p[col].rolling(20).mean()
        v20 = v_avg.iloc[vt].to_numpy()
        vol_ratio[vi] = np.where((v20 > 0) & ~np.isnan(v20), v_now / v20, np.nan)

    merged["ret5"] = ret5
    merged["ret10"] = ret10
    merged["ret20"] = ret20
    merged["vol20"] = vol20
    merged["vol_ratio"] = vol_ratio
    return merged


def main():
    merged = load_merged()
    test = merged[merged["year"] >= 2023].copy()
    tradable = test[test["layer"] == "tradable"].copy()
    print(f"Test period tradable: {len(tradable):,}")

    # === TEST 1: Reverse A4 (low strength = long signal) ===
    print("\n" + "=" * 60)
    print("TEST 1: REVERSED A4 (-margin = long signal)")
    print("=" * 60)

    tradable["inv_margin"] = -tradable["margin_to_threshold"]
    results = []
    for k in [3, 5]:
        ret_col = f"net_ret_{k}d"
        valid = tradable[ret_col].notna() & tradable["inv_margin"].notna()
        vd = tradable.loc[valid].copy()
        if len(vd) < 100:
            continue
        vd["q"] = pd.qcut(vd["inv_margin"], 5, labels=[1, 2, 3, 4, 5],
                          duplicates="drop")
        for q, grp in vd.groupby("q", observed=True):
            if len(grp) < 20:
                continue
            results.append({
                "horizon": k, "quintile": int(q), "n": len(grp),
                "mean_net": grp[ret_col].mean(),
                "median_net": grp[ret_col].median(),
                "win_rate": (grp[ret_col] > 0).mean(),
            })
    rev_df = pd.DataFrame(results)
    rev_df.to_csv(OUT / "p3_reverse_a4_quintiles.csv",
                  index=False, encoding="utf-8-sig")
    for k in [3, 5]:
        sub = rev_df[rev_df["horizon"] == k]
        if not sub.empty:
            print(f"\n--- Reversed A4, {k}d (Q5=lowest strength=most bullish) ---")
            print(sub[["quintile", "n", "mean_net", "median_net", "win_rate"]].to_string(index=False))

    # === TEST 2: Non-upgrade path ===
    print("\n" + "=" * 60)
    print("TEST 2: NON-UPGRADE PATH (armed but NOT disposed tomorrow)")
    print("=" * 60)

    # Among armed days, split by whether disposal actually happened
    not_disposed = tradable[~tradable["disposal_tomorrow"]].copy()
    disposed = tradable[tradable["disposal_tomorrow"]].copy()
    print(f"Armed + NOT disposed: {len(not_disposed):,}")
    print(f"Armed + disposed: {len(disposed):,}")

    for k in [3, 5, 10]:
        ret_col = f"net_ret_{k}d"
        nd_valid = not_disposed[ret_col].notna()
        d_valid = disposed[ret_col].notna()
        if nd_valid.any():
            nd = not_disposed.loc[nd_valid, ret_col]
            print(f"\n  NOT disposed {k}d: n={nd_valid.sum()} "
                  f"mean={nd.mean():.4%} median={nd.median():.4%} "
                  f"win={((nd>0).mean()):.1%}")
        if d_valid.any():
            d = disposed.loc[d_valid, ret_col]
            print(f"  Disposed {k}d: n={d_valid.sum()} "
                  f"mean={d.mean():.4%} median={d.median():.4%} "
                  f"win={((d>0).mean()):.1%}")

    # Further split non-disposed by A4 strength
    print("\n  --- Non-disposed path by A4 strength quintile ---")
    nd_results = []
    for k in [3, 5]:
        ret_col = f"net_ret_{k}d"
        valid = not_disposed[ret_col].notna() & not_disposed["margin_to_threshold"].notna()
        vd = not_disposed.loc[valid].copy()
        if len(vd) < 50:
            continue
        try:
            vd["q"] = pd.qcut(vd["margin_to_threshold"], 5, labels=[1,2,3,4,5],
                              duplicates="drop")
        except ValueError:
            continue
        for q, grp in vd.groupby("q", observed=True):
            if len(grp) < 15:
                continue
            nd_results.append({
                "horizon": k, "quintile": int(q), "n": len(grp),
                "mean_net": grp[ret_col].mean(),
                "median_net": grp[ret_col].median(),
                "win_rate": (grp[ret_col] > 0).mean(),
            })
    nd_df = pd.DataFrame(nd_results)
    nd_df.to_csv(OUT / "p3_nonupgrade_path.csv", index=False, encoding="utf-8-sig")
    for k in [3, 5]:
        sub = nd_df[nd_df["horizon"] == k]
        if not sub.empty:
            print(f"\n  {k}d:")
            print(sub[["quintile", "n", "mean_net", "median_net", "win_rate"]].to_string(index=False))

    # === TEST 3: A5/A6 incremental over reversed-A4 ===
    print("\n" + "=" * 60)
    print("TEST 3: A5/A6 INCREMENTAL OVER REVERSED-A4 BASELINE")
    print("=" * 60)

    # Compute A5/A6 features
    print("Computing A5/A6 features...")
    enriched = compute_a5_a6_features(tradable)
    enriched["inv_margin"] = -enriched["margin_to_threshold"]

    for k in [3, 5]:
        ret_col = f"net_ret_{k}d"
        valid = (enriched[ret_col].notna() &
                 enriched["inv_margin"].notna() &
                 enriched["ret5"].notna() &
                 enriched["vol20"].notna())
        rd = enriched.loc[valid].copy()
        if len(rd) < 100:
            continue

        # Baseline: reversed A4 + A3 proxies
        rd["c10_clip"] = rd["c10"].clip(upper=7)
        rd["streak_clip"] = rd["streak"].clip(upper=6)
        X_base = pd.DataFrame({
            "inv_margin": rd["inv_margin"],
            "c10": rd["c10_clip"], "streak": rd["streak_clip"],
        })
        X_base = sm.add_constant(X_base)
        m1 = sm.OLS(rd[ret_col], X_base).fit(
            cov_type="cluster", cov_kwds={"groups": rd["stock_key"]})

        # Add A5 (momentum)
        X_a5 = X_base.copy()
        X_a5["ret5"] = rd["ret5"]
        X_a5["ret20"] = rd["ret20"]
        m_a5 = sm.OLS(rd[ret_col], X_a5).fit(
            cov_type="cluster", cov_kwds={"groups": rd["stock_key"]})

        # Add A6 (volatility, volume ratio)
        X_a6 = X_base.copy()
        X_a6["vol20"] = rd["vol20"]
        X_a6["vol_ratio"] = rd["vol_ratio"].clip(upper=10)
        m_a6 = sm.OLS(rd[ret_col], X_a6).fit(
            cov_type="cluster", cov_kwds={"groups": rd["stock_key"]})

        # Add both
        X_full = X_a5.copy()
        X_full["vol20"] = rd["vol20"]
        X_full["vol_ratio"] = rd["vol_ratio"].clip(upper=10)
        m_full = sm.OLS(rd[ret_col], X_full).fit(
            cov_type="cluster", cov_kwds={"groups": rd["stock_key"]})

        print(f"\n--- {k}d horizon, n={len(rd):,} ---")
        print(f"  Baseline (inv_A4 + A3): R2={m1.rsquared:.4f}")
        print(f"  + A5 (momentum): ret5 coef={m_a5.params['ret5']:.4f} "
              f"(t={m_a5.tvalues['ret5']:.2f}, p={m_a5.pvalues['ret5']:.3f}) "
              f"ret20 coef={m_a5.params['ret20']:.4f} "
              f"(t={m_a5.tvalues['ret20']:.2f}, p={m_a5.pvalues['ret20']:.3f}) "
              f"dR2={m_a5.rsquared - m1.rsquared:.4f}")
        print(f"  + A6 (vol/volume): vol20 coef={m_a6.params['vol20']:.4f} "
              f"(t={m_a6.tvalues['vol20']:.2f}, p={m_a6.pvalues['vol20']:.3f}) "
              f"vol_ratio coef={m_a6.params['vol_ratio']:.4f} "
              f"(t={m_a6.tvalues['vol_ratio']:.2f}, p={m_a6.pvalues['vol_ratio']:.3f}) "
              f"dR2={m_a6.rsquared - m1.rsquared:.4f}")
        print(f"  + Both: dR2={m_full.rsquared - m1.rsquared:.4f}")

    # Save enriched data for future use
    save_cols = ["stock_key", "attention_date", "year", "layer",
                 "margin_to_threshold", "inv_margin", "disposal_tomorrow",
                 "ret5", "ret10", "ret20", "vol20", "vol_ratio",
                 "c10", "streak"]
    for k in HORIZONS:
        save_cols += [f"net_ret_{k}d", f"raw_ret_{k}d"]
    enriched[save_cols].to_csv(OUT / "p3_enriched_features.csv",
                               index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
