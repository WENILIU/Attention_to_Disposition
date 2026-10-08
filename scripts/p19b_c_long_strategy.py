"""P19b: C-Long Strategy — 做多注意→處置 with upgrade prediction.

Key question: Can we filter attention events to find those likely to upgrade?
If yes, we get +6.47%/trade with no 融券 constraints.

Approach:
1. Use A4 text strength (disposal probability) as filter
2. Test returns for high-probability subset
3. Simulate full strategy with position sizing
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19b_c_long")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P19b: C-Long Strategy (注意→處置 做多)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    att_raw = data.get("trading_attention")
    dis_raw = data.get("disposal_information")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Build attention events
    att = pd.DataFrame(att_raw).copy()
    att["stock_id"] = att["symbol"].astype(str).str.zfill(4)
    att["announce"] = pd.to_datetime(att["date"]).dt.normalize()
    att = att[
        att["stock_id"].str.match(r"^\d{4}$") &
        att["stock_id"].isin(valid_stocks) &
        ~att["stock_id"].str.startswith(("00", "91")) &
        (att["announce"] >= "2018-01-01")
    ].copy()
    att["ann_idx"] = cal.searchsorted(att["announce"], side="left")
    att = att[(att["ann_idx"] >= 1) & (att["ann_idx"] < n_cal - 20)].copy()

    # Build disposal events
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()

    # For each attention, find if/when disposal follows
    dis_by_stock = {}
    for _, row in dis.iterrows():
        sid = row["stock_id"]
        if sid not in dis_by_stock:
            dis_by_stock[sid] = []
        dis_by_stock[sid].append((row["dis_announce"], row["dis_start"]))

    att["leads_to_disposal"] = False
    att["dis_announce_date"] = pd.NaT
    att["dis_start_date"] = pd.NaT
    att["days_to_dis_announce"] = np.nan
    att["days_to_dis_start"] = np.nan

    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann_date = row["announce"]
        if sid in dis_by_stock:
            for dis_ann, dis_start in dis_by_stock[sid]:
                delta_ann = (dis_ann - ann_date).days
                delta_start = (dis_start - ann_date).days
                if 0 < delta_ann <= 30:
                    att.loc[idx, "leads_to_disposal"] = True
                    att.loc[idx, "dis_announce_date"] = dis_ann
                    att.loc[idx, "dis_start_date"] = dis_start
                    att.loc[idx, "days_to_dis_announce"] = delta_ann
                    att.loc[idx, "days_to_dis_start"] = delta_start
                    break

    # Compute returns
    print("Computing returns...")
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    results = []
    for _, row in att.iterrows():
        sym = row["stock_id"]
        ai = row["ann_idx"]
        if sym not in valid_stocks:
            continue

        entry_idx = ai + 1
        if entry_idx >= n_cal - 1:
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[ai][sym]
            avg_to = avg_turnover_5d.iloc[ai][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            continue
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            continue
        gap = entry_price / prev_close - 1
        if not (-0.08 < gap < 0.04):
            continue

        # Returns at various horizons
        rets = {}
        for hold in [1, 3, 5, 7, 10, 15, 20]:
            exit_idx = entry_idx + hold - 1
            if exit_idx >= n_cal:
                rets[f'ret_{hold}d'] = np.nan
                continue
            try:
                exit_price = close.iloc[exit_idx][sym]
                rets[f'ret_{hold}d'] = exit_price / entry_price - 1 - COST_RATE if not np.isnan(exit_price) and exit_price > 0 else np.nan
            except (IndexError, KeyError):
                rets[f'ret_{hold}d'] = np.nan

        # Return to disposal announcement (the real target)
        ret_to_dis_ann = np.nan
        if row["leads_to_disposal"] and not pd.isna(row["dis_announce_date"]):
            dis_ann_idx = cal.searchsorted(row["dis_announce_date"], side="left")
            if dis_ann_idx > entry_idx and dis_ann_idx < n_cal:
                try:
                    p = close.iloc[dis_ann_idx - 1][sym]
                    if not np.isnan(p) and p > 0:
                        ret_to_dis_ann = p / entry_price - 1 - COST_RATE
                except (IndexError, KeyError):
                    pass

        # Return to disposal start (day before)
        ret_to_dis_start = np.nan
        if row["leads_to_disposal"] and not pd.isna(row["dis_start_date"]):
            dis_start_idx = cal.searchsorted(row["dis_start_date"], side="left")
            if dis_start_idx > entry_idx and dis_start_idx < n_cal:
                try:
                    p = close.iloc[dis_start_idx - 1][sym]
                    if not np.isnan(p) and p > 0:
                        ret_to_dis_start = p / entry_price - 1 - COST_RATE
                except (IndexError, KeyError):
                    pass

        results.append({
            "symbol": sym,
            "announce_date": row["announce"],
            "entry_date": cal[entry_idx],
            "year": row["announce"].year,
            "leads_to_disposal": row["leads_to_disposal"],
            "days_to_dis_announce": row["days_to_dis_announce"],
            "days_to_dis_start": row["days_to_dis_start"],
            "gap": gap,
            "ret_to_dis_announce": ret_to_dis_ann,
            "ret_to_dis_start": ret_to_dis_start,
            **rets,
        })

    res = pd.DataFrame(results)
    print(f"Valid events: {len(res):,}")

    upg = res[res["leads_to_disposal"]].copy()
    non_upg = res[~res["leads_to_disposal"]].copy()
    print(f"Upgraded: {len(upg):,}, Not upgraded: {len(non_upg):,}")

    # === STRATEGY: Entry at attention, exit at disposal announcement ===
    print("\n" + "=" * 70)
    print("C-Long: 注意→處置公告前 做多")
    print("=" * 70)

    # Best exit: day before disposal announcement
    valid = upg["ret_to_dis_announce"].notna()
    if valid.sum() > 100:
        r = upg.loc[valid, "ret_to_dis_announce"]
        t_stat, p_val = stats.ttest_1samp(r, 0)
        print(f"\n  注意→處置公告前一日: mean={r.mean():.2%}, median={r.median():.2%}, win={(r>0).mean():.1%}, t={t_stat:.2f} (n={valid.sum()})")

    valid2 = upg["ret_to_dis_start"].notna()
    if valid2.sum() > 100:
        r2 = upg.loc[valid2, "ret_to_dis_start"]
        t_stat2, p_val2 = stats.ttest_1samp(r2, 0)
        print(f"  注意→處置開始前一日: mean={r2.mean():.2%}, median={r2.median():.2%}, win={(r2>0).mean():.1%}, t={t_stat2:.2f} (n={valid2.sum()})")

    # === YEARLY STABILITY (ret_to_dis_announce) ===
    print("\n" + "=" * 70)
    print("逐年穩定性: 注意→處置公告前")
    print("=" * 70)

    for year in sorted(upg["year"].unique()):
        yr = upg[upg["year"] == year]
        valid = yr["ret_to_dis_announce"].notna()
        if valid.sum() < 20:
            continue
        r = yr.loc[valid, "ret_to_dis_announce"]
        sign = "PASS" if r.mean() > 0 else "FAIL"
        print(f"  {year}: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={valid.sum()} {sign}")

    # === SIMULATION: Fixed 10d hold (no look-ahead) ===
    print("\n" + "=" * 70)
    print("固定10天持有（無前視偏差）")
    print("=" * 70)

    # For the "blind" strategy: hold all attention events for 10d
    # Expected return = upgrade_rate * upgraded_return + (1-upgrade_rate) * not_upgraded_return
    col = "ret_10d"
    valid_all = res[col].notna()
    if valid_all.sum() > 100:
        r_all = res.loc[valid_all, col]
        t_all, p_all = stats.ttest_1samp(r_all, 0)
        print(f"\n  全部注意 10d: mean={r_all.mean():.2%}, win={(r_all>0).mean():.1%}, t={t_all:.2f} p={p_all:.4f} (n={valid_all.sum():,})")

    # Yearly for blind 10d
    print("\n  逐年 (全部注意 10d):")
    for year in sorted(res["year"].unique()):
        yr = res[res["year"] == year]
        valid = yr[col].notna()
        if valid.sum() < 20:
            continue
        r = yr.loc[valid, col]
        sign = "+" if r.mean() > 0 else "-"
        print(f"    {year}: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={valid.sum()} {sign}")

    # === KEY: What if we only trade the ones that DO upgrade? ===
    # This has look-ahead bias but shows the ceiling
    print("\n" + "=" * 70)
    print("上限測試: 只做升級的（有前視偏差）")
    print("=" * 70)

    for hold in [5, 7, 10, 15]:
        col = f'ret_{hold}d'
        valid = upg[col].notna()
        if valid.sum() < 50:
            continue
        r = upg.loc[valid, col]
        print(f"  升級 10d: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={valid.sum()}")

    # === CAGR ESTIMATE ===
    print("\n" + "=" * 70)
    print("CAGR 估算")
    print("=" * 70)

    # Scenario 1: Blind (all attention, 10d hold)
    n_per_year = len(res) / 9
    trades_per_year_5pos = 5 / 10 * 252  # 5 positions, 10d hold
    blind_ret = res.loc[res["ret_10d"].notna(), "ret_10d"].mean()
    print(f"\n  場景1: 全部注意, 10d持有, 5檔")
    print(f"    事件/年: {n_per_year:.0f}")
    print(f"    容量/年: {trades_per_year_5pos:.0f}")
    print(f"    每筆報酬: {blind_ret:.2%}")
    print(f"    年化: {trades_per_year_5pos / 5 * blind_ret:.1%}")

    # Scenario 2: Oracle (only upgraded, exit at disposal)
    oracle_ret = upg.loc[upg["ret_to_dis_announce"].notna(), "ret_to_dis_announce"].mean()
    oracle_n_per_year = len(upg) / 9
    avg_hold = upg["days_to_dis_announce"].dropna().mean()
    trades_oracle = 5 / avg_hold * 252 if avg_hold > 0 else 0
    print(f"\n  場景2: 只做升級(完美預測), 平均持有{avg_hold:.0f}天, 5檔")
    print(f"    事件/年: {oracle_n_per_year:.0f}")
    print(f"    容量/年: {trades_oracle:.0f}")
    print(f"    每筆報酬: {oracle_ret:.2%}")
    print(f"    年化: {trades_oracle / 5 * oracle_ret:.1%}")

    # Scenario 3: Realistic (50% precision filter)
    # If we can filter to 50% upgrade rate (from 29.4% base)
    # We'd capture ~50% of upgraded events
    realistic_ret = 0.5 * oracle_ret + 0.5 * res.loc[res["ret_10d"].notna(), "ret_10d"].mean()
    realistic_trades = trades_oracle * 0.7  # fewer events pass filter
    print(f"\n  場景3: 50%精確度過濾, 5檔")
    print(f"    每筆期望: {realistic_ret:.2%}")
    print(f"    容量/年: {realistic_trades:.0f}")
    print(f"    年化: {realistic_trades / 5 * realistic_ret:.1%}")

    # Save
    res.to_csv(OUT / "p19b_c_long_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
