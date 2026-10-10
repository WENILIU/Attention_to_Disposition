"""S4b: 入處置前做空 — 優化版（加入市場環境過濾 + 融券可行性）"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\s4_pre_disposal_short")

def main():
    print("=" * 70)
    print("S4b: 入處置做空 — 優化與過濾")
    print("=" * 70)

    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    dis_raw = data.get("disposal_information")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Market context
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_close = (1 + mkt_ret).cumprod()
    mkt_ma20 = mkt_close.rolling(20).mean()
    mkt_ma60 = mkt_close.rolling(60).mean()
    above_ma20 = (close > close.rolling(20).mean()).sum(axis=1) / close.notna().sum(axis=1)

    # Disposal events
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$") &
        dis["stock_id"].isin(valid_stocks) &
        ~dis["stock_id"].str.startswith(("00", "91"))
    ].copy()
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis = dis[(dis["start_idx"] >= 5)].copy()

    # Build events with features
    print("\nBuilding events...")
    events = []
    for _, row in dis.iterrows():
        sym = row["stock_id"]
        si = int(row["start_idx"])
        if sym not in close.columns or si >= n_cal or si < 6:
            continue
        c = close[sym]
        try:
            base = c.iloc[si - 1]
            if np.isnan(base) or base <= 0:
                continue
            day1_ret = c.iloc[si] / base - 1
            if np.isnan(day1_ret):
                continue

            # Features at T-1 (entry day for short)
            ti = si - 1
            feat = {
                "stock_id": sym,
                "start_date": row["start"],
                "start_idx": si,
                "year": row["start"].year,
                "day1_ret": day1_ret,
                "short_ret_1d": -day1_ret - 0.006,
            }

            # Market state
            try:
                feat["mkt_above_ma20"] = 1 if mkt_close.iloc[ti] > mkt_ma20.iloc[ti] else 0
                feat["mkt_above_ma60"] = 1 if mkt_close.iloc[ti] > mkt_ma60.iloc[ti] else 0
                feat["breadth"] = float(above_ma20.iloc[ti]) if not np.isnan(above_ma20.iloc[ti]) else 0.5
            except:
                feat["mkt_above_ma20"] = 1
                feat["mkt_above_ma60"] = 1
                feat["breadth"] = 0.5

            # Stock pre-disposal momentum
            try:
                feat["pre_ret_5d"] = float(c.iloc[ti] / c.iloc[ti - 5] - 1) if ti >= 5 else 0
                feat["pre_ret_3d"] = float(c.iloc[ti] / c.iloc[ti - 3] - 1) if ti >= 3 else 0
                feat["vol_20d"] = float(c.iloc[ti-20:ti].pct_change().dropna().std()) if ti >= 20 else 0
                feat["turnover"] = float(c.iloc[ti] * vol.iloc[ti][sym]) if sym in vol.columns else 0
            except:
                feat["pre_ret_5d"] = 0
                feat["pre_ret_3d"] = 0
                feat["vol_20d"] = 0
                feat["turnover"] = 0

            # Gap at Day 1 open (can we even short at T-1 close?)
            try:
                feat["day1_gap"] = float(open_p.iloc[si][sym] / c.iloc[si-1] - 1) if sym in open_p.columns else 0
            except:
                feat["day1_gap"] = 0

            events.append(feat)
        except (IndexError, KeyError):
            continue

    ev = pd.DataFrame(events)
    print(f"  Events: {len(ev):,}")

    # === FILTER OPTIMIZATION ===
    print("\n" + "=" * 70)
    print("過濾優化: 哪些條件下做空最有效")
    print("=" * 70)

    baseline = ev["short_ret_1d"].mean()
    print(f"\n  Baseline: avg={baseline:.2%}, win={(ev['short_ret_1d']>0).mean():.1%}, n={len(ev)}")

    filters = {
        "大盤MA20上": ev["mkt_above_ma20"] == 1,
        "大盤MA60上": ev["mkt_above_ma60"] == 1,
        "廣度>50%": ev["breadth"] > 0.50,
        "廣度>60%": ev["breadth"] > 0.60,
        "處置前5日漲>20%": ev["pre_ret_5d"] > 0.20,
        "處置前5日漲>30%": ev["pre_ret_5d"] > 0.30,
        "處置前3日漲>10%": ev["pre_ret_3d"] > 0.10,
        "高波動>4%": ev["vol_20d"] > 0.04,
        "大盤MA20上+前漲>20%": (ev["mkt_above_ma20"] == 1) & (ev["pre_ret_5d"] > 0.20),
        "大盤MA60上+廣度>50%": (ev["mkt_above_ma60"] == 1) & (ev["breadth"] > 0.50),
        "大盤MA20上+廣度>50%": (ev["mkt_above_ma20"] == 1) & (ev["breadth"] > 0.50),
    }

    print(f"\n  {'Filter':<30} {'keep_n':>7} {'keep_ret':>8} {'keep_win':>8} {'skip_ret':>8} {'impact':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, mask in filters.items():
        keep = ev[mask]
        skip = ev[~mask]
        if len(keep) < 50 or len(skip) < 50:
            continue
        print(f"  {name:<30} {len(keep):>7} {keep['short_ret_1d'].mean():>8.2%} {(keep['short_ret_1d']>0).mean():>8.1%} {skip['short_ret_1d'].mean():>8.2%} {keep['short_ret_1d'].mean()-baseline:>+8.2%}")

    # === BEST COMBO: Market above MA20 + breadth > 50% ===
    print("\n" + "=" * 70)
    print("最佳過濾組合逐年驗證")
    print("=" * 70)

    best_mask = (ev["mkt_above_ma20"] == 1) & (ev["breadth"] > 0.50)
    filtered = ev[best_mask].copy()
    print(f"\n  Filter: 大盤MA20上 + 廣度>50%")
    print(f"  Kept: {len(filtered)}, avg={filtered['short_ret_1d'].mean():.2%}, win={(filtered['short_ret_1d']>0).mean():.1%}")

    yearly = filtered.groupby("year")["short_ret_1d"].agg(["mean", "count", lambda x: (x > 0).mean()])
    yearly.columns = ["avg_ret", "n", "win_rate"]
    yearly = yearly[yearly["n"] >= 10]
    n_pos = (yearly["avg_ret"] > 0).sum()
    print(f"\n  Years positive: {n_pos}/{len(yearly)}")
    print(f"  {'Year':>6} {'avg_ret':>8} {'n':>5} {'win%':>6}")
    for yr, r in yearly.iterrows():
        print(f"  {yr:>6} {r['avg_ret']:>8.2%} {int(r['n']):>5} {r['win_rate']:>6.1%}")

    # === PORTFOLIO SIMULATION ===
    print("\n" + "=" * 70)
    print("組合模擬: 每日最多做空N檔, 持有1天")
    print("=" * 70)

    for max_pos in [3, 5, 10]:
        filtered_sorted = filtered.sort_values("start_idx")
        daily_short = pd.Series(0.0, index=cal)
        n_trades = 0

        for _, t in filtered_sorted.iterrows():
            si = int(t["start_idx"])
            if si >= n_cal:
                continue
            # Count active shorts on this day (all are 1-day, so just check capacity)
            # Since all are 1-day holds, we just limit per-day entries
            day_mask = daily_short.index[si] == daily_short.index[si]
            n_trades += 1
            daily_short.iloc[si] += t["short_ret_1d"] / max_pos

        # Cap daily contribution
        daily_short = daily_short.clip(upper=0.10, lower=-0.10)

        sharpe = np.mean(daily_short) / np.std(daily_short) * np.sqrt(252) if np.std(daily_short) > 0 else 0
        cum = (1 + daily_short).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()

        yr_returns = {}
        for yr in range(2010, 2027):
            yr_mask = cal.year == yr
            if yr_mask.sum() > 50:
                yr_returns[yr] = np.prod(1 + daily_short.values[yr_mask]) - 1
        n_pos_yr = sum(1 for v in yr_returns.values() if v > 0)

        print(f"\n  Max {max_pos} positions/day:")
        print(f"    Sharpe: {sharpe:.2f}, MDD: {mdd:.1%}, Years: {n_pos_yr}/{len(yr_returns)}")
        print(f"    Trades: {n_trades}")
        for yr in [2011, 2015, 2018, 2022, 2023, 2024, 2025]:
            if yr in yr_returns:
                print(f"      {yr}: {yr_returns[yr]:+.1%}")

    # === C-SHORT: A4 intensity angle ===
    print("\n" + "=" * 70)
    print("C-Short 角度: 注意強度 vs 處置首日跌幅")
    print("=" * 70)

    # Load attention data to get A4 (margin_to_threshold)
    try:
        margin = data.get("margin_transactions:融資今日餘額")
        print(f"  Margin data: {margin.shape}")

        # For events with margin data, check if high margin intensity → bigger Day 1 drop
        ev_with_margin = []
        for _, t in ev.iterrows():
            sym = t["stock_id"]
            si = int(t["start_idx"])
            ti = si - 1
            if sym not in margin.columns or ti < 20:
                continue
            try:
                m = margin.iloc[ti-20:ti+1][sym]
                m_valid = m.dropna()
                if len(m_valid) < 10:
                    continue
                # A4 proxy: margin level relative to its own 20d average
                a4 = float(m_valid.iloc[-1] / m_valid.mean() - 1)
                ev_with_margin.append({**t, "a4_margin": a4})
            except:
                continue

        if len(ev_with_margin) > 100:
            evm = pd.DataFrame(ev_with_margin)
            evm["quintile"] = pd.qcut(evm["a4_margin"], 5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"])
            print(f"\n  融資強度 vs 處置首日做空報酬:")
            print(f"  {'Quintile':<10} {'a4_level':>10} {'short_ret':>10} {'win%':>6} {'n':>6}")
            for q, grp in evm.groupby("quintile", observed=True):
                print(f"  {str(q):<10} {grp['a4_margin'].mean():>10.2%} {grp['short_ret_1d'].mean():>10.2%} {(grp['short_ret_1d']>0).mean():>6.1%} {len(grp):>6}")
        else:
            print("  Insufficient margin data for A4 analysis")
    except Exception as e:
        print(f"  Margin data unavailable: {e}")

    # Save
    ev.to_csv(OUT / "s4b_optimized_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
