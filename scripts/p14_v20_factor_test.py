"""P14: Factor testing for V20 (Day 1 entry, disposal universe).

Tests factors that could improve V20 by ranking/filtering which
disposal stocks to buy on Day 1.

Available factors:
  A11: 融資融券變化 (margin balance change)
  A6:  波動率 (pre-disposal volatility)
  A7:  技術狀態 (MA deviation, RSI)
  S1:  市值分層 (trading value)
  S2:  融資比例 (margin ratio)
  A8:  距門檻距離 (proxied by how many times disposed before)
  A9:  注意間隔 (days since last attention)

Method: IC + quintile returns within disposal universe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from finlab import data

OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p14_v20_factors"
)
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    # Load data
    print("Loading data...")
    dis_raw = data.get("disposal_information")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    high_p = data.get("price:最高價")
    low_p = data.get("price:最低價")
    vol = data.get("price:成交股數")
    amount = data.get("price:成交金額")
    margin_buy = data.get("margin_transactions:融資買進")
    margin_sell = data.get("margin_transactions:融券賣出")

    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()

    valid_stocks = set(close.columns)
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$") &
        dis["stock_id"].isin(valid_stocks) &
        ~dis["stock_id"].str.startswith(("00", "91")) &
        (dis["announce"] >= "2018-01-01")
    ].copy()

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1

    # V20 style: Day 1 entry, exit at end_idx - 1
    # Use old regime (10-day) as proxy for testing (larger sample)
    # The factor effects should be similar regardless of duration
    dis = dis[(dis["start_idx"] >= 2) & (dis["end_idx"] >= 0) &
              (dis["end_idx"] < n_cal) & (dis["duration"] >= 5)].copy()

    print(f"Events: {len(dis):,}")

    # Compute indicators
    print("Computing factors...")
    all_stocks = list(set(dis["stock_id"]) & valid_stocks)

    # Pre-compute rolling indicators
    returns = close[all_stocks].pct_change()
    vol_20d = returns.rolling(20).std()  # A6: volatility
    ma5 = close[all_stocks].rolling(5).mean()
    ma20 = close[all_stocks].rolling(20).mean()
    ma60 = close[all_stocks].rolling(60).mean()

    # RSI (14-day)
    delta = close[all_stocks].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Margin data
    margin_dates = pd.DatetimeIndex(margin_buy.index).normalize().unique().sort_values()

    # Count prior disposals (A9 proxy)
    dis_sorted = dis.sort_values("announce")
    prior_count = {}
    for _, row in dis_sorted.iterrows():
        sym = row["stock_id"]
        key = (sym, row["announce"])
        prior_count[key] = prior_count.get(sym, 0)
        prior_count[sym] = prior_count.get(sym, 0) + 1

    # Build event features
    print("Building features...")
    events = []
    for _, row in dis.iterrows():
        sym = row["stock_id"]
        if sym not in all_stocks:
            continue
        si = row["start_idx"]
        ei = row["end_idx"]
        exit_trade_idx = ei - 1
        if exit_trade_idx < si:
            continue

        try:
            entry_open = open_p.iloc[si][sym]
            exit_open = open_p.iloc[exit_trade_idx][sym]
            prev_close = close.iloc[si - 1][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_open, exit_open, prev_close]):
            continue

        net_ret = exit_open / entry_open - 1 - COST_RATE

        # === FACTORS ===
        # A6: Volatility (20d std of returns, before disposal)
        try:
            a6_vol = vol_20d.iloc[si - 1][sym] if sym in vol_20d.columns else np.nan
        except (IndexError, KeyError):
            a6_vol = np.nan

        # A7: Technical state (price vs MA20, MA60)
        try:
            price = close.iloc[si - 1][sym]
            ma20_val = ma20.iloc[si - 1][sym] if sym in ma20.columns else np.nan
            ma60_val = ma60.iloc[si - 1][sym] if sym in ma60.columns else np.nan
            rsi_val = rsi.iloc[si - 1][sym] if sym in rsi.columns else np.nan
            a7_bias20 = (price / ma20_val - 1) if not np.isnan(ma20_val) and ma20_val > 0 else np.nan
            a7_bias60 = (price / ma60_val - 1) if not np.isnan(ma60_val) and ma60_val > 0 else np.nan
            a7_rsi = rsi_val
        except (IndexError, KeyError):
            a7_bias20 = a7_bias60 = a7_rsi = np.nan

        # S1: Size (avg daily trading value, 20d before)
        try:
            s1_amount = amount.iloc[max(0, si-20):si][sym].mean() if sym in amount.columns else np.nan
        except (IndexError, KeyError):
            s1_amount = np.nan

        # A11: Margin activity (融資買進 ratio, 5d before)
        try:
            m_idx = margin_dates.searchsorted(row["start"], side="left")
            if m_idx >= 5 and sym in margin_buy.columns:
                m_recent = margin_buy.iloc[m_idx-5:m_idx][sym].sum()
                m_prev = margin_buy.iloc[max(0,m_idx-10):m_idx-5][sym].sum()
                a11_margin_change = (m_recent - m_prev) / (abs(m_prev) + 1)
            else:
                a11_margin_change = np.nan
        except (IndexError, KeyError):
            a11_margin_change = np.nan

        # A9: Prior disposal count
        a9_prior = prior_count.get((sym, row["announce"]), 0)

        # Gap (for reference)
        gap = entry_open / prev_close - 1

        events.append({
            "symbol": sym,
            "entry_date": cal[si],
            "year": row["announce"].year,
            "condition": row.get("處置條件", ""),
            "net_ret": net_ret,
            "gap": gap,
            "a6_vol": a6_vol,
            "a7_bias20": a7_bias20,
            "a7_bias60": a7_bias60,
            "a7_rsi": a7_rsi,
            "s1_amount": s1_amount,
            "a11_margin": a11_margin_change,
            "a9_prior": a9_prior,
        })

    ev = pd.DataFrame(events)
    print(f"Events with factors: {len(ev):,}")

    # Apply V20 filters (liquidity + gap)
    ev_filtered = ev[
        (ev["gap"] > -0.08) & (ev["gap"] < 0.04) &
        (ev["s1_amount"].isna() | (ev["s1_amount"] >= 20_000_000))
    ].copy()
    print(f"After V20 filters: {len(ev_filtered):,}")

    # === FACTOR TESTING ===
    print("\n" + "=" * 70)
    print("FACTOR IC WITHIN V20 UNIVERSE (Day 1 entry)")
    print("=" * 70)

    factors = {
        "A6 波動率": "a6_vol",
        "A7 乖離MA20": "a7_bias20",
        "A7 乖離MA60": "a7_bias60",
        "A7 RSI": "a7_rsi",
        "S1 成交額(市值代理)": "s1_amount",
        "A11 融資變化": "a11_margin",
        "A9 歷史處置次數": "a9_prior",
    }

    ic_results = []
    for name, col in factors.items():
        valid = ev_filtered[col].notna() & ev_filtered["net_ret"].notna()
        if valid.sum() < 100:
            print(f"  {name}: insufficient data (n={valid.sum()})")
            continue
        ic, pval = stats.spearmanr(ev_filtered.loc[valid, col],
                                    ev_filtered.loc[valid, "net_ret"])
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {name}: IC={ic:.4f} p={pval:.4f} {sig} (n={int(valid.sum()):,})")
        ic_results.append({"factor": name, "col": col, "IC": ic, "p": pval,
                          "n": int(valid.sum())})

    ic_df = pd.DataFrame(ic_results)
    ic_df.to_csv(OUT / "p14_ic_results.csv", index=False, encoding="utf-8-sig")

    # === QUINTILE ANALYSIS for significant factors ===
    significant = ic_df[ic_df["p"] < 0.1]
    if len(significant) > 0:
        print("\n" + "=" * 70)
        print("QUINTILE RETURNS (significant factors only)")
        print("=" * 70)

        for _, row in significant.iterrows():
            col = row["col"]
            name = row["factor"]
            valid = ev_filtered[col].notna() & ev_filtered["net_ret"].notna()
            vd = ev_filtered[valid].copy()
            try:
                vd["q"] = pd.qcut(vd[col], 5, labels=[1,2,3,4,5], duplicates="drop")
            except ValueError:
                continue

            print(f"\n  {name} (IC={row['IC']:.4f}):")
            print(f"    {'Q':>3} {'n':>6} {'Mean':>8} {'Win':>6} {'Median':>8}")
            for q, grp in vd.groupby("q", observed=True):
                if len(grp) < 20:
                    continue
                r = grp["net_ret"]
                print(f"    Q{int(q)} {len(grp):>6} {r.mean():>8.2%} "
                      f"{(r>0).mean():>6.1%} {r.median():>8.2%}")

            spread = (vd[vd["q"]==5]["net_ret"].mean() -
                      vd[vd["q"]==1]["net_ret"].mean())
            print(f"    Q5-Q1 spread: {spread:.2%}")

    # === YEARLY IC STABILITY ===
    print("\n" + "=" * 70)
    print("IC BY YEAR (top factors)")
    print("=" * 70)

    for _, row in ic_df.sort_values("p").head(3).iterrows():
        col = row["col"]
        name = row["factor"]
        print(f"\n  {name}:")
        for yr in sorted(ev_filtered["year"].unique()):
            yr_data = ev_filtered[ev_filtered["year"] == yr]
            valid = yr_data[col].notna() & yr_data["net_ret"].notna()
            if valid.sum() < 30:
                continue
            ic, p = stats.spearmanr(yr_data.loc[valid, col],
                                     yr_data.loc[valid, "net_ret"])
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"    {yr}: IC={ic:.4f} p={p:.4f} {sig} (n={int(valid.sum())})")

    # === COMBINED FACTOR TEST ===
    print("\n" + "=" * 70)
    print("COMBINED FACTOR (if any significant)")
    print("=" * 70)

    sig_cols = [r["col"] for _, r in significant.iterrows()]
    if len(sig_cols) >= 2:
        # Build composite from significant factors
        vd = ev_filtered.copy()
        z_scores = []
        for col in sig_cols:
            valid = vd[col].notna()
            z = pd.Series(0, index=vd.index)
            z[valid] = (vd.loc[valid, col] - vd.loc[valid, col].mean()) / vd.loc[valid, col].std()
            # Flip sign if IC is negative (we want positive = good)
            ic_val = ic_df[ic_df["col"] == col]["IC"].values[0]
            if ic_val < 0:
                z = -z
            z_scores.append(z)

        vd["composite"] = sum(z_scores) / len(z_scores)
        valid = vd["composite"].notna() & vd["net_ret"].notna()
        vd = vd[valid]

        ic_comp, p_comp = stats.spearmanr(vd["composite"], vd["net_ret"])
        print(f"  Composite IC: {ic_comp:.4f} p={p_comp:.4f}")

        vd["q"] = pd.qcut(vd["composite"], 5, labels=[1,2,3,4,5], duplicates="drop")
        print(f"\n  {'Q':>3} {'n':>6} {'Mean':>8} {'Win':>6}")
        for q, grp in vd.groupby("q", observed=True):
            if len(grp) < 20:
                continue
            r = grp["net_ret"]
            print(f"  Q{int(q)} {len(grp):>6} {r.mean():>8.2%} {(r>0).mean():>6.1%}")
    else:
        print("  No significant factors found. V20 event signal is sufficient alone.")

    ev_filtered.to_csv(OUT / "p14_events_with_factors.csv",
                       index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
