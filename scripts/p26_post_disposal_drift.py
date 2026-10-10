"""P26: Post-Disposal Drift — 處置出關後 30 天漂移驗證.

研究假設: 處置出關後，機構重新覆蓋 + 散戶跟進 → 持續上漲漂移。

分析維度:
  1. 分段報酬拆解: 處置期間 / 出關日 / 出關後 1-5d / 6-10d / 11-20d / 21-30d
  2. 逐年穩定性
  3. 按處置期間漲幅分組（動量假設）
  4. 按市場環境分組（多頭/空頭）
  5. 按處置條件分組
  6. 統計顯著性檢驗

使用方式:
  python p26_post_disposal_drift.py
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
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p26_post_drift")
OUT.mkdir(parents=True, exist_ok=True)

MIN_TURNOVER = 20_000_000
MAX_DRIFT_DAYS = 30


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
    dis_raw = pd.DataFrame(data.get("disposal_information"))
    close_p = data.get("price:收盤價")
    vol_p = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close_p.columns)

    # Build disposal events (same logic as V20)
    dis = dis_raw.copy()
    dis["stock_key"] = dis["symbol"].astype(str).str.lstrip("0")
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis["condition"] = dis.get("處置條件", "").astype(str)
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$")
        & dis["stock_id"].isin(valid_stocks)
        & ~dis["stock_id"].str.startswith(("00", "91"))
        & (dis["announce"] >= "2010-01-01")
    ].copy()

    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1
    dis["year"] = dis["announce"].dt.year

    # Filter: valid duration, enough post-disposal data
    dis = dis[
        (dis["start_idx"] >= 0)
        & (dis["end_idx"] >= 0)
        & (dis["duration"] >= 3)
        & (dis["end_idx"] + MAX_DRIFT_DAYS < n_cal)
    ].copy()

    # Liquidity filter (same as V20)
    turnover = close_p * vol_p
    avg_turnover_5d = turnover.rolling(5).mean()

    print(f"Valid events with 30d post-window: {len(dis):,}")

    # Market MA20 for regime classification
    market_close = close_p.sum(axis=1)
    market_ma20 = market_close.rolling(20).mean()
    market_above_ma20 = (market_close > market_ma20)

    # === Compute segment returns ===
    results = []

    for _, row in dis.iterrows():
        sym = row["stock_key"]
        if sym not in valid_stocks:
            continue

        si = int(row["start_idx"])
        ei = int(row["end_idx"])
        exit_idx = ei + 1  # first day after disposal

        # Check liquidity at entry
        try:
            entry_turnover = avg_turnover_5d.iloc[si][sym]
            if np.isnan(entry_turnover) or entry_turnover < MIN_TURNOVER:
                continue
        except (IndexError, KeyError):
            continue

        # Get price at disposal end (our reference point)
        try:
            end_close = close_p.iloc[ei][sym]
            if np.isnan(end_close) or end_close <= 0:
                continue
        except (IndexError, KeyError):
            continue

        # Also get start price for V20 segment
        try:
            start_close = close_p.iloc[si][sym]
            if np.isnan(start_close) or start_close <= 0:
                continue
        except (IndexError, KeyError):
            continue

        # Compute returns at each post-disposal day
        post_returns = {}
        for d in range(1, MAX_DRIFT_DAYS + 1):
            target_idx = exit_idx + d - 1
            if target_idx >= n_cal:
                break
            try:
                target_close = close_p.iloc[target_idx][sym]
                if np.isnan(target_close) or target_close <= 0:
                    continue
                post_returns[d] = target_close / end_close - 1
            except (IndexError, KeyError):
                continue

        if len(post_returns) < MAX_DRIFT_DAYS:
            continue

        # Segment returns
        seg_during = end_close / start_close - 1  # during disposal
        seg_day1 = post_returns.get(1, np.nan)  # exit day
        seg_1_5 = post_returns.get(5, np.nan)  # cumulative to day 5
        seg_6_10 = ((1 + post_returns.get(10, np.nan)) / (1 + post_returns.get(5, np.nan)) - 1) if 5 in post_returns and 10 in post_returns else np.nan
        seg_11_20 = ((1 + post_returns.get(20, np.nan)) / (1 + post_returns.get(10, np.nan)) - 1) if 10 in post_returns and 20 in post_returns else np.nan
        seg_21_30 = ((1 + post_returns.get(30, np.nan)) / (1 + post_returns.get(20, np.nan)) - 1) if 20 in post_returns and 30 in post_returns else np.nan
        seg_total_30 = post_returns.get(30, np.nan)

        # Disposal period return (for grouping)
        disposal_return = seg_during

        # Market regime at exit
        try:
            regime_at_exit = bool(market_above_ma20.iloc[min(exit_idx, n_cal - 1)])
        except Exception:
            regime_at_exit = True

        results.append({
            "symbol": row["stock_id"],
            "announce": row["announce"],
            "start": row["start"],
            "end": row["end"],
            "year": row["year"],
            "duration": row["duration"],
            "condition": row["condition"],
            "disposal_return": disposal_return,
            "seg_during": seg_during,
            "seg_day1": seg_day1,
            "seg_1_5": seg_1_5,
            "seg_6_10": seg_6_10,
            "seg_11_20": seg_11_20,
            "seg_21_30": seg_21_30,
            "seg_total_30": seg_total_30,
            "market_bull": regime_at_exit,
            **{f"day_{d}": post_returns.get(d, np.nan) for d in range(1, MAX_DRIFT_DAYS + 1)},
        })

    res = pd.DataFrame(results)
    print(f"Events with full 30d window: {len(res):,}")

    if len(res) < 100:
        print("ERROR: Too few events. Aborting.")
        return

    res.to_csv(OUT / "p26_all_events.csv", index=False, encoding="utf-8-sig")

    # === 1. Overall segment summary ===
    print("\n" + "=" * 70)
    print("P26: POST-DISPOSAL DRIFT ANALYSIS")
    print("=" * 70)

    print(f"\nTotal events: {len(res):,}")
    print(f"Period: {res['year'].min()}-{res['year'].max()}")

    segments = [
        ("seg_during", "處置期間 (V20 alpha)"),
        ("seg_day1", "出關日 (Day 1)"),
        ("seg_1_5", "出關後 1-5 天 (累計)"),
        ("seg_6_10", "出關後 6-10 天 (邊際)"),
        ("seg_11_20", "出關後 11-20 天 (邊際)"),
        ("seg_21_30", "出關後 21-30 天 (邊際)"),
        ("seg_total_30", "出關後 30 天 (累計)"),
    ]

    print(f"\n{'Segment':<30} {'n':>6} {'Mean':>8} {'Median':>8} {'Win%':>6} {'t-stat':>8} {'p-value':>10}")
    print("-" * 85)
    for col, label in segments:
        valid = res[col].dropna()
        if len(valid) < 30:
            continue
        t_stat, p_val = stats.ttest_1samp(valid, 0)
        print(f"{label:<30} {len(valid):>6,} {valid.mean():>8.4%} {valid.median():>8.4%} "
              f"{(valid > 0).mean():>6.1%} {t_stat:>8.2f} {p_val:>10.2e}")

    # === 2. Yearly stability of post-disposal 30d ===
    print("\n" + "=" * 70)
    print("YEARLY STABILITY: Post-Disposal 30d Cumulative Return")
    print("=" * 70)

    yearly = res.groupby("year").apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_30d": g["seg_total_30"].mean(),
            "median_30d": g["seg_total_30"].median(),
            "win_30d": (g["seg_total_30"] > 0).mean(),
            "mean_5d": g["seg_1_5"].mean(),
            "mean_10d": g["seg_6_10"].mean(),
            "mean_20d": g["seg_11_20"].mean(),
        })
    ).reset_index()
    yearly = yearly[yearly["n"] >= 20]
    print(yearly.to_string(index=False))
    yearly.to_csv(OUT / "p26_yearly.csv", index=False, encoding="utf-8-sig")

    # === 3. By disposal period return (momentum hypothesis) ===
    print("\n" + "=" * 70)
    print("BY DISPOSAL PERIOD RETURN (Momentum Hypothesis)")
    print("=" * 70)
    print("Hypothesis: stocks that rose more during disposal continue to drift up")

    res["drift_quintile"] = pd.qcut(res["disposal_return"], 5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"], duplicates="drop")

    by_drift = res.groupby("drift_quintile", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "disposal_ret_mean": g["disposal_return"].mean(),
            "post_5d": g["seg_1_5"].mean(),
            "post_10d": g["seg_1_5"].mean() + g["seg_6_10"].mean(),
            "post_30d": g["seg_total_30"].mean(),
            "win_30d": (g["seg_total_30"] > 0).mean(),
        })
    ).reset_index()
    print(by_drift.to_string(index=False))
    by_drift.to_csv(OUT / "p26_by_disposal_return.csv", index=False, encoding="utf-8-sig")

    # === 4. By market regime ===
    print("\n" + "=" * 70)
    print("BY MARKET REGIME (Bull vs Bear at Exit)")
    print("=" * 70)

    by_regime = res.groupby("market_bull").apply(
        lambda g: pd.Series({
            "n": len(g),
            "post_5d": g["seg_1_5"].mean(),
            "post_10d": g["seg_1_5"].mean() + g["seg_6_10"].mean(),
            "post_30d": g["seg_total_30"].mean(),
            "win_30d": (g["seg_total_30"] > 0).mean(),
        })
    ).reset_index()
    by_regime["regime"] = by_regime["market_bull"].map({True: "多頭(MA20上)", False: "空頭(MA20下)"})
    print(by_regime[["regime", "n", "post_5d", "post_10d", "post_30d", "win_30d"]].to_string(index=False))

    # === 5. By disposal condition ===
    print("\n" + "=" * 70)
    print("BY DISPOSAL CONDITION (top conditions by count)")
    print("=" * 70)

    by_cond = res.groupby("condition").apply(
        lambda g: pd.Series({
            "n": len(g),
            "post_5d": g["seg_1_5"].mean(),
            "post_30d": g["seg_total_30"].mean(),
            "win_30d": (g["seg_total_30"] > 0).mean(),
        })
    ).reset_index()
    by_cond = by_cond[by_cond["n"] >= 50].sort_values("n", ascending=False)
    print(by_cond.head(10).to_string(index=False))

    # === 6. Day-by-day cumulative return curve ===
    print("\n" + "=" * 70)
    print("DAY-BY-DAY CUMULATIVE RETURN (post-disposal)")
    print("=" * 70)

    day_cols = [f"day_{d}" for d in range(1, MAX_DRIFT_DAYS + 1)]
    day_means = res[day_cols].mean()
    day_medians = res[day_cols].median()
    day_wins = res[day_cols].apply(lambda x: (x > 0).mean())

    print(f"\n{'Day':>4} {'Mean':>8} {'Median':>8} {'Win%':>6}")
    print("-" * 30)
    for d in range(1, MAX_DRIFT_DAYS + 1):
        col = f"day_{d}"
        if col in day_means.index:
            print(f"{d:>4} {day_means[col]:>8.4%} {day_medians[col]:>8.4%} {day_wins[col]:>6.1%}")

    # Save curve
    curve = pd.DataFrame({
        "day": list(range(1, MAX_DRIFT_DAYS + 1)),
        "mean": [day_means.get(f"day_{d}", np.nan) for d in range(1, MAX_DRIFT_DAYS + 1)],
        "median": [day_medians.get(f"day_{d}", np.nan) for d in range(1, MAX_DRIFT_DAYS + 1)],
        "win_rate": [day_wins.get(f"day_{d}", np.nan) for d in range(1, MAX_DRIFT_DAYS + 1)],
    })
    curve.to_csv(OUT / "p26_drift_curve.csv", index=False, encoding="utf-8-sig")

    # === 7. Statistical test: is there drift beyond noise? ===
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)

    # Compare post-disposal returns vs random baseline
    # Random baseline: same stocks, random entry dates (shift by 60 days)
    print("\nMethod: Compare post-disposal returns vs 60-day-shifted control")

    control_results = []
    for _, row in dis.iterrows():
        sym = row["stock_key"]
        if sym not in valid_stocks:
            continue
        si = int(row["start_idx"])
        ei = int(row["end_idx"])
        # Control: shift the entire window by 60 trading days
        ctrl_end = ei + 60
        if ctrl_end + MAX_DRIFT_DAYS >= n_cal or ctrl_end < 0:
            continue
        try:
            ctrl_end_close = close_p.iloc[ctrl_end][sym]
            if np.isnan(ctrl_end_close) or ctrl_end_close <= 0:
                continue
        except (IndexError, KeyError):
            continue

        ctrl_returns = {}
        for d in range(1, MAX_DRIFT_DAYS + 1):
            target_idx = ctrl_end + d
            if target_idx >= n_cal:
                break
            try:
                tc = close_p.iloc[target_idx][sym]
                if np.isnan(tc) or tc <= 0:
                    continue
                ctrl_returns[d] = tc / ctrl_end_close - 1
            except (IndexError, KeyError):
                continue

        if len(ctrl_returns) < MAX_DRIFT_DAYS:
            continue

        control_results.append({
            "seg_1_5": ctrl_returns.get(5, np.nan),
            "seg_total_30": ctrl_returns.get(30, np.nan),
        })

    ctrl = pd.DataFrame(control_results)
    print(f"Control events: {len(ctrl):,}")

    if len(ctrl) > 50:
        for col, label in [("seg_1_5", "5d"), ("seg_total_30", "30d")]:
            treat = res[col].dropna()
            ctrl_sub = ctrl[col].dropna()
            t_stat, p_val = stats.ttest_ind(treat, ctrl_sub, equal_var=False)
            print(f"  {label}: Treatment mean={treat.mean():.4%}, Control mean={ctrl_sub.mean():.4%}, "
                  f"Diff={treat.mean() - ctrl_sub.mean():.4%}, t={t_stat:.2f}, p={p_val:.2e}")

    # === 8. Conclusion ===
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    total_30 = res["seg_total_30"].dropna()
    t_final, p_final = stats.ttest_1samp(total_30, 0)

    # The correct test is vs control (market drift is not alpha)
    has_control = len(ctrl) > 50
    if has_control:
        treat_30 = res["seg_total_30"].dropna()
        ctrl_30 = ctrl["seg_total_30"].dropna()
        diff = treat_30.mean() - ctrl_30.mean()
        t_ctrl, p_ctrl = stats.ttest_ind(treat_30, ctrl_30, equal_var=False)

        print(f"\n  Raw 30d mean (vs zero): {total_30.mean():.4%}, p={p_final:.2e}")
        print(f"  Control-adjusted 30d: {diff:.4%}, t={t_ctrl:.2f}, p={p_ctrl:.2e}")
        print(f"  Win rate: {(total_30 > 0).mean():.1%}, Median: {total_30.median():.4%}")

        if p_ctrl < 0.01 and diff > 0.01:
            print(f"\n✅ POST-DISPOSAL DRIFT EXISTS (control-adjusted): {diff:.4%}, p={p_ctrl:.2e}")
            print("   This is a statistically significant extension of V20 alpha.")
        elif p_ctrl < 0.05 and diff > 0.005:
            print(f"\n⚠️ MARGINAL DRIFT (control-adjusted): {diff:.4%}, p={p_ctrl:.2e}")
            print("   Weak evidence. May not survive transaction costs.")
            print("   Note: mean driven by outliers (median is negative).")
        else:
            print(f"\n❌ NO SIGNIFICANT DRIFT vs control: {diff:.4%}, p={p_ctrl:.2e}")
            print("   Post-disposal returns are explained by general market drift.")
            print("   V20's finding confirmed: all excess alpha is during disposal period.")
    else:
        print(f"\n  Raw 30d mean: {total_30.mean():.4%}, p={p_final:.2e}")
        print("   Cannot determine alpha without control comparison.")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
