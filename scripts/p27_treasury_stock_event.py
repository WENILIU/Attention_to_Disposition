"""P27: 庫藏股事件策略驗證 — 買回公告後股價效應.

研究假設: 公司買回自家股票 = 強制買入 → 價格支撐 → 上漲。
這是處置策略的鏡像: 處置 = 強制賣出 → 超賣 → 反彈。

分析維度:
  1. 公告後逐日報酬曲線 (Day 0-30)
  2. 按買回規模分組 (% of shares outstanding)
  3. 按買回目的分組
  4. 按市場環境分組
  5. 逐年穩定性
  6. 控制組比較
  7. 可交易性評估

使用方式:
  python p27_treasury_stock_event.py
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
    vol_p = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close_p.columns)

    # Load treasury_stock event data
    purpose_df = data.get("treasury_stock:買回目的")
    pct_shares_df = data.get("treasury_stock:本次買回股數佔公司已發行股份總數比例(%)")
    period_start_df = data.get("treasury_stock:預定買回期間-起")
    period_end_df = data.get("treasury_stock:預定買回期間-迄")
    planned_shares_df = data.get("treasury_stock:預定買回股數")

    # Turnover for liquidity filter
    turnover = close_p * vol_p
    avg_turnover_5d = turnover.rolling(5).mean()

    # Market MA20 for regime
    market_close = close_p.sum(axis=1)
    market_ma20 = market_close.rolling(20).mean()
    market_above_ma20 = (market_close > market_ma20)

    # === Extract buyback events ===
    # purpose_df index = announcement dates, columns = stock IDs
    # Any non-null cell = a buyback announcement for that stock on that date
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

            # Get buyback details
            pct_shares = np.nan
            try:
                pct_shares = pct_shares_df.loc[date, stock_id]
            except (KeyError, Exception):
                pass

            period_start = None
            period_end = None
            try:
                period_start = period_start_df.loc[date, stock_id]
                period_end = period_end_df.loc[date, stock_id]
            except (KeyError, Exception):
                pass

            events.append({
                "stock_id": stock_id,
                "announce_date": date_norm,
                "purpose": str(purpose),
                "pct_shares": float(pct_shares) if pd.notna(pct_shares) else np.nan,
                "period_start": period_start,
                "period_end": period_end,
            })

    ev = pd.DataFrame(events)
    print(f"Raw buyback events (2010+): {len(ev):,}")

    # Map to calendar index
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
            if pd.notna(tv) and tv >= MIN_TURNOVER:
                filtered.append(True)
            else:
                filtered.append(False)
        except (IndexError, KeyError):
            filtered.append(False)
    ev["liquid"] = filtered
    ev = ev[ev["liquid"]].copy()
    print(f"After liquidity filter: {len(ev):,}")

    # Ensure enough post-announcement data
    ev = ev[ev["announce_idx"] + MAX_TRACK_DAYS < n_cal].copy()
    print(f"With full {MAX_TRACK_DAYS}d window: {len(ev):,}")

    if len(ev) < 100:
        print("ERROR: Too few events.")
        return

    # === Compute post-announcement returns ===
    results = []
    for _, row in ev.iterrows():
        sym = row["stock_id"]
        ai = int(row["announce_idx"])

        try:
            base_price = close_p.iloc[ai][sym]
            if np.isnan(base_price) or base_price <= 0:
                continue
        except (IndexError, KeyError):
            continue

        # Also get pre-announcement return (for context)
        pre_returns = {}
        for d in [1, 5, 10]:
            pre_idx = ai - d
            if pre_idx < 0:
                continue
            try:
                pp = close_p.iloc[pre_idx][sym]
                if pd.notna(pp) and pp > 0:
                    pre_returns[f"pre_{d}d"] = base_price / pp - 1
            except (IndexError, KeyError):
                continue

        # Post-announcement returns
        post_returns = {}
        for d in range(0, MAX_TRACK_DAYS + 1):
            target_idx = ai + d
            if target_idx >= n_cal:
                break
            try:
                tc = close_p.iloc[target_idx][sym]
                if pd.isna(tc) or tc <= 0:
                    continue
                post_returns[d] = tc / base_price - 1
            except (IndexError, KeyError):
                continue

        if len(post_returns) < MAX_TRACK_DAYS:
            continue

        # Market regime at announcement
        try:
            regime = bool(market_above_ma20.iloc[min(ai, n_cal - 1)])
        except Exception:
            regime = True

        results.append({
            "stock_id": sym,
            "announce_date": row["announce_date"],
            "year": row["year"],
            "purpose": row["purpose"],
            "pct_shares": row["pct_shares"],
            "market_bull": regime,
            **{f"pre_{k}": v for k, v in pre_returns.items()},
            **{f"day_{d}": post_returns.get(d, np.nan) for d in range(MAX_TRACK_DAYS + 1)},
        })

    res = pd.DataFrame(results)
    print(f"Final events with full window: {len(res):,}")

    if len(res) < 100:
        print("ERROR: Too few valid events.")
        return

    res.to_csv(OUT / "p27_all_events.csv", index=False, encoding="utf-8-sig")

    # === 1. Day-by-day return curve ===
    print("\n" + "=" * 70)
    print("P27: TREASURY STOCK BUYBACK — POST-ANNOUNCEMENT RETURNS")
    print("=" * 70)
    print(f"\nTotal events: {len(res):,}")
    print(f"Period: {res['year'].min()}-{res['year'].max()}")

    day_cols = [f"day_{d}" for d in range(MAX_TRACK_DAYS + 1)]
    day_means = res[day_cols].mean()
    day_medians = res[day_cols].median()
    day_wins = res[day_cols].apply(lambda x: (x > 0).mean())

    print(f"\n{'Day':>4} {'Mean':>8} {'Median':>8} {'Win%':>6} {'t-stat':>8}")
    print("-" * 40)
    for d in range(MAX_TRACK_DAYS + 1):
        col = f"day_{d}"
        valid = res[col].dropna()
        if len(valid) < 30:
            continue
        t_stat, _ = stats.ttest_1samp(valid, 0)
        print(f"{d:>4} {day_means[col]:>8.4%} {day_medians[col]:>8.4%} "
              f"{day_wins[col]:>6.1%} {t_stat:>8.2f}")

    curve = pd.DataFrame({
        "day": list(range(MAX_TRACK_DAYS + 1)),
        "mean": [day_means.get(f"day_{d}", np.nan) for d in range(MAX_TRACK_DAYS + 1)],
        "median": [day_medians.get(f"day_{d}", np.nan) for d in range(MAX_TRACK_DAYS + 1)],
        "win_rate": [day_wins.get(f"day_{d}", np.nan) for d in range(MAX_TRACK_DAYS + 1)],
    })
    curve.to_csv(OUT / "p27_return_curve.csv", index=False, encoding="utf-8-sig")

    # === 2. Key summary statistics ===
    print("\n" + "=" * 70)
    print("KEY PERIODS SUMMARY")
    print("=" * 70)

    key_periods = [
        (0, "公告日 (Day 0)"),
        (1, "公告後 1 天"),
        (3, "公告後 3 天"),
        (5, "公告後 5 天"),
        (10, "公告後 10 天"),
        (20, "公告後 20 天"),
        (30, "公告後 30 天"),
    ]

    print(f"\n{'Period':<20} {'n':>6} {'Mean':>8} {'Median':>8} {'Win%':>6} {'t-stat':>8} {'p-value':>10}")
    print("-" * 75)
    for d, label in key_periods:
        col = f"day_{d}"
        valid = res[col].dropna()
        if len(valid) < 30:
            continue
        t_stat, p_val = stats.ttest_1samp(valid, 0)
        print(f"{label:<20} {len(valid):>6,} {valid.mean():>8.4%} {valid.median():>8.4%} "
              f"{(valid > 0).mean():>6.1%} {t_stat:>8.2f} {p_val:>10.2e}")

    # === 3. By buyback size (% of shares) ===
    print("\n" + "=" * 70)
    print("BY BUYBACK SIZE (% of shares outstanding)")
    print("=" * 70)

    res_valid_pct = res[res["pct_shares"].notna() & (res["pct_shares"] > 0)].copy()
    if len(res_valid_pct) > 100:
        res_valid_pct["size_group"] = pd.cut(
            res_valid_pct["pct_shares"],
            bins=[0, 0.5, 1, 2, 5, 100],
            labels=["<0.5%", "0.5-1%", "1-2%", "2-5%", ">5%"],
        )
        by_size = res_valid_pct.groupby("size_group", observed=True).apply(
            lambda g: pd.Series({
                "n": len(g),
                "day_5": g["day_5"].mean(),
                "day_10": g["day_10"].mean(),
                "day_30": g["day_30"].mean(),
                "win_30": (g["day_30"] > 0).mean(),
            })
        ).reset_index()
        print(by_size.to_string(index=False))
        by_size.to_csv(OUT / "p27_by_size.csv", index=False, encoding="utf-8-sig")
    else:
        print("  Insufficient data for size grouping.")

    # === 4. By buyback purpose ===
    print("\n" + "=" * 70)
    print("BY BUYBACK PURPOSE")
    print("=" * 70)

    by_purpose = res.groupby("purpose").apply(
        lambda g: pd.Series({
            "n": len(g),
            "day_5": g["day_5"].mean(),
            "day_10": g["day_10"].mean(),
            "day_30": g["day_30"].mean(),
            "win_30": (g["day_30"] > 0).mean(),
        })
    ).reset_index()
    by_purpose = by_purpose[by_purpose["n"] >= 30].sort_values("n", ascending=False)
    print(by_purpose.to_string(index=False))
    by_purpose.to_csv(OUT / "p27_by_purpose.csv", index=False, encoding="utf-8-sig")

    # === 5. By market regime ===
    print("\n" + "=" * 70)
    print("BY MARKET REGIME")
    print("=" * 70)

    by_regime = res.groupby("market_bull").apply(
        lambda g: pd.Series({
            "n": len(g),
            "day_5": g["day_5"].mean(),
            "day_10": g["day_10"].mean(),
            "day_30": g["day_30"].mean(),
            "win_30": (g["day_30"] > 0).mean(),
        })
    ).reset_index()
    by_regime["regime"] = by_regime["market_bull"].map({True: "多頭(MA20上)", False: "空頭(MA20下)"})
    print(by_regime[["regime", "n", "day_5", "day_10", "day_30", "win_30"]].to_string(index=False))

    # === 6. Yearly stability ===
    print("\n" + "=" * 70)
    print("YEARLY STABILITY")
    print("=" * 70)

    yearly = res.groupby("year").apply(
        lambda g: pd.Series({
            "n": len(g),
            "day_5_mean": g["day_5"].mean(),
            "day_5_win": (g["day_5"] > 0).mean(),
            "day_10_mean": g["day_10"].mean(),
            "day_30_mean": g["day_30"].mean(),
            "day_30_win": (g["day_30"] > 0).mean(),
        })
    ).reset_index()
    yearly = yearly[yearly["n"] >= 20]
    print(yearly.to_string(index=False))
    yearly.to_csv(OUT / "p27_yearly.csv", index=False, encoding="utf-8-sig")

    # === 7. Control group comparison ===
    print("\n" + "=" * 70)
    print("CONTROL GROUP COMPARISON (shift 60 days)")
    print("=" * 70)

    control_results = []
    for _, row in ev.iterrows():
        sym = row["stock_id"]
        ai = int(row["announce_idx"])
        ctrl_idx = ai + 60
        if ctrl_idx + MAX_TRACK_DAYS >= n_cal or ctrl_idx < 0:
            continue
        try:
            base = close_p.iloc[ctrl_idx][sym]
            if np.isnan(base) or base <= 0:
                continue
        except (IndexError, KeyError):
            continue

        ctrl_returns = {}
        for d in range(MAX_TRACK_DAYS + 1):
            target_idx = ctrl_idx + d
            if target_idx >= n_cal:
                break
            try:
                tc = close_p.iloc[target_idx][sym]
                if pd.isna(tc) or tc <= 0:
                    continue
                ctrl_returns[d] = tc / base - 1
            except (IndexError, KeyError):
                continue

        if len(ctrl_returns) < MAX_TRACK_DAYS:
            continue
        control_results.append({f"day_{d}": ctrl_returns.get(d, np.nan) for d in range(MAX_TRACK_DAYS + 1)})

    ctrl = pd.DataFrame(control_results)
    print(f"Control events: {len(ctrl):,}")

    if len(ctrl) > 50:
        for d in [5, 10, 30]:
            col = f"day_{d}"
            treat = res[col].dropna()
            ctrl_sub = ctrl[col].dropna()
            if len(treat) < 30 or len(ctrl_sub) < 30:
                continue
            t_stat, p_val = stats.ttest_ind(treat, ctrl_sub, equal_var=False)
            print(f"  Day {d}: Treatment={treat.mean():.4%}, Control={ctrl_sub.mean():.4%}, "
                  f"Diff={treat.mean() - ctrl_sub.mean():.4%}, t={t_stat:.2f}, p={p_val:.2e}")

    # === 8. Pre-announcement context ===
    print("\n" + "=" * 70)
    print("PRE-ANNOUNCEMENT CONTEXT (what happens before buyback?)")
    print("=" * 70)

    for col, label in [("pre_pre_1d", "前1天"), ("pre_pre_5d", "前5天"), ("pre_pre_10d", "前10天")]:
        if col in res.columns:
            valid = res[col].dropna()
            if len(valid) > 30:
                print(f"  {label}: mean={valid.mean():.4%}, median={valid.median():.4%}, win={(valid > 0).mean():.1%}")

    # === 9. Strategy simulation: Buy at announcement, hold N days ===
    print("\n" + "=" * 70)
    print("STRATEGY SIMULATION (buy at announcement close, hold N days)")
    print("=" * 70)

    COST = 0.001425 + 0.003  # buy + sell commission + tax

    for hold_days in [1, 3, 5, 10, 20, 30]:
        col = f"day_{hold_days}"
        valid = res[col].dropna()
        if len(valid) < 30:
            continue
        net = valid - COST
        t_stat, p_val = stats.ttest_1samp(net, 0)
        print(f"  Hold {hold_days:>2}d: n={len(valid):>5,} gross={valid.mean():.4%} "
              f"net={net.mean():.4%} win_net={(net > 0).mean():.1%} t={t_stat:.2f} p={p_val:.2e}")

    # === 10. Filtered strategy: large buyback + bear market ===
    print("\n" + "=" * 70)
    print("FILTERED STRATEGY: Large buyback (>2%) + Bear market")
    print("=" * 70)

    if len(res_valid_pct) > 100:
        # Large buyback
        large = res_valid_pct[res_valid_pct["pct_shares"] >= 2]
        print(f"\n  Large buyback (>2%): n={len(large):,}")
        for d in [5, 10, 30]:
            col = f"day_{d}"
            valid = large[col].dropna()
            if len(valid) >= 20:
                net = valid - COST
                print(f"    Day {d}: n={len(valid):,} gross={valid.mean():.4%} net={net.mean():.4%} win={(valid > 0).mean():.1%}")

        # Bear market
        bear = res[~res["market_bull"]]
        print(f"\n  Bear market: n={len(bear):,}")
        for d in [5, 10, 30]:
            col = f"day_{d}"
            valid = bear[col].dropna()
            if len(valid) >= 20:
                net = valid - COST
                print(f"    Day {d}: n={len(valid):,} gross={valid.mean():.4%} net={net.mean():.4%} win={(valid > 0).mean():.1%}")

        # Combined: large + bear
        large_bear = res_valid_pct[(res_valid_pct["pct_shares"] >= 2) & (~res_valid_pct["market_bull"])]
        print(f"\n  Large + Bear: n={len(large_bear):,}")
        for d in [5, 10, 30]:
            col = f"day_{d}"
            valid = large_bear[col].dropna()
            if len(valid) >= 20:
                net = valid - COST
                print(f"    Day {d}: n={len(valid):,} gross={valid.mean():.4%} net={net.mean():.4%} win={(valid > 0).mean():.1%}")

    # === 11. Conclusion ===
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    day5 = res["day_5"].dropna()
    day30 = res["day_30"].dropna()
    t5, p5 = stats.ttest_1samp(day5, 0)
    t30, p30 = stats.ttest_1samp(day30, 0)

    # Control-adjusted
    if len(ctrl) > 50:
        ctrl5 = ctrl["day_5"].dropna()
        ctrl30 = ctrl["day_30"].dropna()
        diff5 = day5.mean() - ctrl5.mean()
        diff30 = day30.mean() - ctrl30.mean()
        tc5, pc5 = stats.ttest_ind(day5, ctrl5, equal_var=False)
        tc30, pc30 = stats.ttest_ind(day30, ctrl30, equal_var=False)

        print(f"\n  Raw Day 5: {day5.mean():.4%} (p={p5:.2e})")
        print(f"  Control-adjusted Day 5: {diff5:.4%} (p={pc5:.2e})")
        print(f"  Raw Day 30: {day30.mean():.4%} (p={p30:.2e})")
        print(f"  Control-adjusted Day 30: {diff30:.4%} (p={pc30:.2e})")
        print(f"  Win rate Day 5: {(day5 > 0).mean():.1%}, Day 30: {(day30 > 0).mean():.1%}")

        if pc5 < 0.01 and diff5 > 0.005:
            print(f"\n✅ BUYBACK ANNOUNCEMENT ALPHA EXISTS (Day 5): {diff5:.4%}")
            print("   Statistically significant after control adjustment.")
        elif pc5 < 0.05 and diff5 > 0.002:
            print(f"\n⚠️ MARGINAL BUYBACK ALPHA (Day 5): {diff5:.4%}")
            print("   Weak but present. May need additional filters.")
        else:
            print(f"\n❌ NO SIGNIFICANT BUYBACK ALPHA vs control: {diff5:.4%} (p={pc5:.2e})")
            print("   Announcement effect is explained by general market drift.")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
