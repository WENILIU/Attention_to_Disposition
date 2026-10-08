"""P19: O2 注意→處置段報酬 — 完整 Alpha 探索

假設：
1. 注意公告後，股票在進入處置前可能有可預測的走勢
2. 升級到處置的股票 vs 未升級的股票，走勢不同
3. 可能存在做空 alpha（注意後下跌→處置前觸底）

測試維度：
- 注意→處置 vs 注意→未處置
- 不同持有期（注意後1d/3d/5d/10d/到處置日）
- 做多 vs 做空
- 按注意條件分類
- 按時間衰減
- 逐年穩定性
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19_o2_attention")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
SHORT_COST_RATE = 0.001425 + 0.003 + 0.003 + 0.025/252*5  # + 融券成本5天


def main():
    print("=" * 70)
    print("P19: O2 注意→處置段 Alpha 探索")
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
    print("Building attention events...")
    att = pd.DataFrame(att_raw).copy()
    att["stock_id"] = att["symbol"].astype(str).str.zfill(4)
    att["announce"] = pd.to_datetime(att["date"]).dt.normalize()

    # Get attention conditions
    if "註意原因" in att.columns:
        att["reason"] = att["註意原因"]
    elif "原因" in att.columns:
        att["reason"] = att["原因"]
    else:
        att["reason"] = "unknown"

    att = att[
        att["stock_id"].str.match(r"^\d{4}$") &
        att["stock_id"].isin(valid_stocks) &
        ~att["stock_id"].str.startswith(("00", "91")) &
        (att["announce"] >= "2018-01-01")
    ].copy()

    att["ann_idx"] = cal.searchsorted(att["announce"], side="left")
    att = att[(att["ann_idx"] >= 1) & (att["ann_idx"] < n_cal - 1)].copy()
    print(f"  Attention events (2018+): {len(att):,}")

    # Build disposal events for matching
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()

    # For each attention event, check if it leads to disposal
    # Match: same stock, disposal starts within 30 days after attention
    print("Matching attention → disposal...")
    dis_by_stock = {}
    for _, row in dis.iterrows():
        sid = row["stock_id"]
        if sid not in dis_by_stock:
            dis_by_stock[sid] = []
        dis_by_stock[sid].append(row["dis_start"])

    # For each attention, find if disposal follows within 30 days
    att["leads_to_disposal"] = False
    att["disposal_date"] = pd.NaT
    att["days_to_disposal"] = np.nan

    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann_date = row["announce"]
        if sid in dis_by_stock:
            for dis_date in dis_by_stock[sid]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    att.loc[idx, "leads_to_disposal"] = True
                    att.loc[idx, "disposal_date"] = dis_date
                    att.loc[idx, "days_to_disposal"] = delta
                    break

    upgraded = att[att["leads_to_disposal"]].copy()
    not_upgraded = att[~att["leads_to_disposal"]].copy()
    print(f"  注意→處置 (upgraded): {len(upgraded):,}")
    print(f"  注意→未處置 (not upgraded): {len(not_upgraded):,}")
    print(f"  Upgrade rate: {len(upgraded)/len(att):.1%}")

    # Pre-compute turnover
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # === COMPUTE RETURNS ===
    print("\nComputing returns...")

    def compute_event_returns(event_df, label):
        """Compute returns from attention announcement for each event."""
        results = []
        for _, row in event_df.iterrows():
            sym = row["stock_id"]
            ai = row["ann_idx"]
            if sym not in valid_stocks:
                continue

            try:
                # Entry: next day open after attention announcement
                entry_idx = ai + 1
                if entry_idx >= n_cal - 1:
                    continue
                entry_price = open_p.iloc[entry_idx][sym]
                prev_close = close.iloc[ai][sym]
                avg_to = avg_turnover_5d.iloc[ai][sym]
            except (IndexError, KeyError):
                continue

            if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
                continue

            # Liquidity filter
            if not (np.isnan(avg_to) or avg_to >= 20_000_000):
                continue

            gap = entry_price / prev_close - 1
            if not (-0.08 < gap < 0.04):
                continue

            # Fixed holding periods
            rets = {}
            for hold in [1, 2, 3, 5, 10, 15, 20]:
                exit_idx = entry_idx + hold - 1
                if exit_idx >= n_cal:
                    rets[f'ret_{hold}d'] = np.nan
                    continue
                try:
                    exit_price = close.iloc[exit_idx][sym]
                    if np.isnan(exit_price) or exit_price <= 0:
                        rets[f'ret_{hold}d'] = np.nan
                    else:
                        rets[f'ret_{hold}d'] = exit_price / entry_price - 1 - COST_RATE
                except (IndexError, KeyError):
                    rets[f'ret_{hold}d'] = np.nan

            # Return to disposal date (for upgraded events)
            ret_to_dis = np.nan
            if row.get("leads_to_disposal", False) and not pd.isna(row.get("disposal_date", pd.NaT)):
                dis_idx = cal.searchsorted(row["disposal_date"], side="left")
                if dis_idx > entry_idx and dis_idx < n_cal:
                    try:
                        dis_price = close.iloc[dis_idx - 1][sym]  # Day before disposal
                        if not np.isnan(dis_price) and dis_price > 0:
                            ret_to_dis = dis_price / entry_price - 1 - COST_RATE
                    except (IndexError, KeyError):
                        pass

            # Short returns (negative of long)
            short_rets = {}
            for hold in [1, 2, 3, 5, 10]:
                col = f'ret_{hold}d'
                if not np.isnan(rets.get(col, np.nan)):
                    short_rets[f'short_{hold}d'] = -rets[col] - SHORT_COST_RATE + COST_RATE
                else:
                    short_rets[f'short_{hold}d'] = np.nan

            results.append({
                "symbol": sym,
                "announce_date": row["announce"],
                "entry_date": cal[entry_idx],
                "year": row["announce"].year,
                "reason": row.get("reason", ""),
                "leads_to_disposal": row.get("leads_to_disposal", False),
                "days_to_disposal": row.get("days_to_disposal", np.nan),
                "gap": gap,
                "ret_to_disposal": ret_to_dis,
                **rets,
                **short_rets,
            })

        return pd.DataFrame(results)

    # Compute for all attention events
    all_res = compute_event_returns(att, "all")
    print(f"  Valid events (V20 filters): {len(all_res):,}")

    if len(all_res) < 100:
        print("  Too few events!")
        return

    # Split by upgrade status
    upg_res = all_res[all_res["leads_to_disposal"]].copy()
    non_upg_res = all_res[~all_res["leads_to_disposal"]].copy()

    # === ANALYSIS 1: Overall returns after attention ===
    print("\n" + "=" * 70)
    print("1. 注意公告後報酬（全部，做多）")
    print("=" * 70)

    for hold in [1, 2, 3, 5, 10, 15, 20]:
        col = f'ret_{hold}d'
        valid = all_res[col].notna()
        if valid.sum() < 50:
            continue
        r = all_res.loc[valid, col]
        t_stat, p_val = stats.ttest_1samp(r, 0)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {hold}d: mean={r.mean():.2%}, median={r.median():.2%}, win={(r>0).mean():.1%}, t={t_stat:.2f} p={p_val:.4f} {sig} (n={valid.sum():,})")

    # === ANALYSIS 2: Upgraded vs Not Upgraded ===
    print("\n" + "=" * 70)
    print("2. 注意→處置 vs 注意→未處置（做多）")
    print("=" * 70)

    for label, subset in [("注意→處置", upg_res), ("注意→未處置", non_upg_res)]:
        print(f"\n  {label} (n={len(subset):,}):")
        for hold in [1, 3, 5, 10, 20]:
            col = f'ret_{hold}d'
            valid = subset[col].notna()
            if valid.sum() < 30:
                continue
            r = subset.loc[valid, col]
            t_stat, p_val = stats.ttest_1samp(r, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"    {hold}d: mean={r.mean():.2%}, win={(r>0).mean():.1%}, t={t_stat:.2f} p={p_val:.4f} {sig}")

    # Return to disposal date
    if len(upg_res) > 50:
        valid = upg_res["ret_to_disposal"].notna()
        if valid.sum() > 50:
            r = upg_res.loc[valid, "ret_to_disposal"]
            t_stat, p_val = stats.ttest_1samp(r, 0)
            print(f"\n  注意→處置前一日 (ret_to_disposal): mean={r.mean():.2%}, win={(r>0).mean():.1%}, t={t_stat:.2f} p={p_val:.4f} (n={valid.sum()})")

    # === ANALYSIS 3: SHORT side ===
    print("\n" + "=" * 70)
    print("3. 做空報酬（注意後做空）")
    print("=" * 70)

    for label, subset in [("全部注意", all_res), ("注意→處置", upg_res), ("注意→未處置", non_upg_res)]:
        print(f"\n  {label}:")
        for hold in [1, 3, 5, 10]:
            col = f'short_{hold}d'
            valid = subset[col].notna()
            if valid.sum() < 30:
                continue
            r = subset.loc[valid, col]
            t_stat, p_val = stats.ttest_1samp(r, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"    short {hold}d: mean={r.mean():.2%}, win={(r>0).mean():.1%}, t={t_stat:.2f} p={p_val:.4f} {sig}")

    # === ANALYSIS 4: Yearly stability (best signal) ===
    print("\n" + "=" * 70)
    print("4. 逐年穩定性")
    print("=" * 70)

    # Test the most promising: short 5d for upgraded
    for label, subset, col in [
        ("做多 5d (全部)", all_res, "ret_5d"),
        ("做空 5d (全部)", all_res, "short_5d"),
        ("做空 5d (→處置)", upg_res, "short_5d"),
        ("做空 3d (全部)", all_res, "short_3d"),
    ]:
        print(f"\n  {label}:")
        for year in sorted(subset["year"].unique()):
            yr = subset[subset["year"] == year]
            valid = yr[col].notna()
            if valid.sum() < 20:
                continue
            r = yr.loc[valid, col]
            sign = "+" if r.mean() > 0 else "-"
            print(f"    {year}: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={valid.sum()} {sign}")

    # === ANALYSIS 5: By attention reason ===
    print("\n" + "=" * 70)
    print("5. 按注意原因分類")
    print("=" * 70)

    if all_res["reason"].nunique() > 1:
        top_reasons = all_res["reason"].value_counts().head(8).index
        for reason in top_reasons:
            subset = all_res[all_res["reason"] == reason]
            col = "ret_5d"
            valid = subset[col].notna()
            if valid.sum() < 30:
                continue
            r = subset.loc[valid, col]
            t_stat, p_val = stats.ttest_1samp(r, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"  {reason}: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={valid.sum()} {sig}")

    # === ANALYSIS 6: Days to disposal (for upgraded) ===
    print("\n" + "=" * 70)
    print("6. 注意→處置天數 vs 報酬")
    print("=" * 70)

    if len(upg_res) > 100:
        upg_valid = upg_res[upg_res["days_to_disposal"].notna() & upg_res["ret_5d"].notna()]
        if len(upg_valid) > 50:
            upg_valid = upg_valid.copy()
            upg_valid["days_bin"] = pd.cut(upg_valid["days_to_disposal"],
                                          bins=[0, 5, 10, 15, 20, 30],
                                          labels=["1-5d", "6-10d", "11-15d", "16-20d", "21-30d"])
            for bin_label, grp in upg_valid.groupby("days_bin", observed=True):
                if len(grp) < 20:
                    continue
                r = grp["ret_5d"]
                print(f"  {bin_label}: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={len(grp)}")

    # Save
    all_res.to_csv(OUT / "p19_all_attention_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
