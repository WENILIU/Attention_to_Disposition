"""S4: 入處置前/第一天下跌 — 做空 alpha 測試

假設: 處置公告日→處置開始前，股價因恐慌性賣壓下跌
策略: 公告日收盤做空，持有2-3天後回補
數據: 所有處置事件 (2011-2026)
"""
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
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("S4: 入處置前做空 — 處置開始前的下跌 alpha")
    print("=" * 70)

    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    dis_raw = data.get("disposal_information")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Build disposal events
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

    # Compute key indices
    dis["announce_idx"] = cal.searchsorted(dis["announce"], side="right") - 1
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1

    # Filter: valid events where we can observe pre-disposal prices
    dis = dis[(dis["start_idx"] > dis["announce_idx"]) & (dis["start_idx"] >= 5)].copy()
    print(f"\nValid disposal events: {len(dis):,}")

    # === ANALYSIS 1: Pre-disposal price action ===
    print("\n" + "=" * 70)
    print("分析1: 處置開始前N天的報酬")
    print("=" * 70)

    results = []
    for _, row in dis.iterrows():
        sym = row["stock_id"]
        si = int(row["start_idx"])
        ai = int(row["announce_idx"])

        if sym not in close.columns:
            continue
        if si < 5 or si >= n_cal:
            continue

        c = close[sym]
        # Returns relative to day before disposal starts (T-1 close)
        try:
            base_price = c.iloc[si - 1]
            if np.isnan(base_price) or base_price <= 0:
                continue

            # Pre-disposal returns (negative = stock drops before disposal)
            ret_t_minus_1 = c.iloc[si - 1] / c.iloc[si - 2] - 1 if si >= 2 else np.nan
            ret_t_minus_2 = c.iloc[si - 1] / c.iloc[si - 3] - 1 if si >= 3 else np.nan
            ret_t_minus_3 = c.iloc[si - 1] / c.iloc[si - 4] - 1 if si >= 4 else np.nan
            ret_t_minus_5 = c.iloc[si - 1] / c.iloc[si - 6] - 1 if si >= 6 else np.nan

            # Day 1 of disposal (open to close)
            ret_day1 = c.iloc[si] / c.iloc[si - 1] - 1 if si < n_cal else np.nan
            # Day 1-2 cumulative
            ret_day1_2 = c.iloc[si + 1] / c.iloc[si - 1] - 1 if si + 1 < n_cal else np.nan
            # Day 1-3 cumulative
            ret_day1_3 = c.iloc[si + 2] / c.iloc[si - 1] - 1 if si + 2 < n_cal else np.nan

            # Announcement day return
            ret_announce = c.iloc[ai] / c.iloc[ai - 1] - 1 if ai >= 1 and ai < n_cal else np.nan
            # Announcement to day 1
            ret_ann_to_day1 = c.iloc[si] / c.iloc[ai] - 1 if ai >= 0 and si < n_cal else np.nan

            results.append({
                "stock_id": sym,
                "announce_date": row["announce"],
                "start_date": row["start"],
                "start_idx": si,
                "announce_idx": ai,
                "ret_t_minus_1": ret_t_minus_1,
                "ret_t_minus_2": ret_t_minus_2,
                "ret_t_minus_3": ret_t_minus_3,
                "ret_t_minus_5": ret_t_minus_5,
                "ret_day1": ret_day1,
                "ret_day1_2": ret_day1_2,
                "ret_day1_3": ret_day1_3,
                "ret_announce": ret_announce,
                "ret_ann_to_day1": ret_ann_to_day1,
                "year": row["start"].year,
            })
        except (IndexError, KeyError):
            continue

    ev = pd.DataFrame(results)
    ev = ev.dropna(subset=["ret_day1"])
    print(f"  Events with price data: {len(ev):,}")

    # Summary stats
    print(f"\n  {'Period':<25} {'Mean':>8} {'Median':>8} {'Win%':>6} {'n':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for col in ["ret_t_minus_1", "ret_t_minus_2", "ret_t_minus_3", "ret_t_minus_5",
                "ret_day1", "ret_day1_2", "ret_day1_3", "ret_announce", "ret_ann_to_day1"]:
        valid = ev[col].dropna()
        if len(valid) < 50:
            continue
        label = col.replace("ret_", "").replace("_", " ")
        print(f"  {label:<25} {valid.mean():>8.2%} {valid.median():>8.2%} {(valid<0).mean():>6.1%} {len(valid):>6}")

    # === ANALYSIS 2: Short strategy simulation ===
    print("\n" + "=" * 70)
    print("分析2: 做空策略模擬")
    print("=" * 70)

    # Strategy: Short at close of day before disposal (T-1), cover after N days
    # Short return = -stock_return (minus costs)
    SHORT_COST = 0.003 + 0.025 / 252  # transaction + borrow cost per day

    for hold_days in [1, 2, 3, 4, 5]:
        col = f"ret_day1_{hold_days}" if hold_days > 1 else "ret_day1"
        if col not in ev.columns:
            continue
        valid = ev[col].dropna()
        if len(valid) < 50:
            continue
        short_ret = -valid - SHORT_COST * hold_days
        print(f"\n  Short {hold_days}d (entry: T-1 close, exit: Day{hold_days} close):")
        print(f"    n={len(valid):,}, avg_short_ret={short_ret.mean():.2%}, win={( short_ret>0).mean():.1%}")

        # Yearly breakdown
        ev_temp = ev.copy()
        ev_temp["short_ret"] = -ev_temp[col] - SHORT_COST * hold_days
        yearly = ev_temp.groupby("year")["short_ret"].agg(["mean", "count", lambda x: (x > 0).mean()])
        yearly.columns = ["avg_ret", "n", "win_rate"]
        yearly = yearly[yearly["n"] >= 10]
        n_pos = (yearly["avg_ret"] > 0).sum()
        print(f"    Years positive: {n_pos}/{len(yearly)}")
        for yr, r in yearly.iterrows():
            print(f"      {yr}: {r['avg_ret']:+.2%} (n={int(r['n'])}, win={r['win_rate']:.1%})")

    # === ANALYSIS 3: Entry timing — announce day vs T-1 vs Day 1 ===
    print("\n" + "=" * 70)
    print("分析3: 最佳做空入場時機")
    print("=" * 70)

    # Option A: Short at announcement close, cover Day 2
    ev["short_A"] = -ev["ret_ann_to_day1"] - SHORT_COST * 2
    # Option B: Short at T-1 close, cover Day 2
    ev["short_B"] = -ev["ret_day1_2"] - SHORT_COST * 2
    # Option C: Short at Day 1 open (gap), cover Day 3
    # Approximate: short from Day 1 close to Day 3 close
    ev["short_C"] = -(ev["ret_day1_3"] - ev["ret_day1"]) - SHORT_COST * 2

    print(f"\n  {'Entry':<30} {'Avg short ret':>14} {'Win%':>6} {'n':>6}")
    print(f"  {'-'*30} {'-'*14} {'-'*6} {'-'*6}")
    for label, col in [("A: 公告日收盤→Day2", "short_A"),
                       ("B: T-1收盤→Day2", "short_B"),
                       ("C: Day1收盤→Day3", "short_C")]:
        valid = ev[col].dropna()
        if len(valid) < 50:
            continue
        print(f"  {label:<30} {valid.mean():>14.2%} {(valid>0).mean():>6.1%} {len(valid):>6}")

    # === ANALYSIS 4: Filter by conditions ===
    print("\n" + "=" * 70)
    print("分析4: 哪些事件前跌最多（做空最佳標的）")
    print("=" * 70)

    # Pre-disposal momentum (stocks that ran up before disposal = more to fall)
    ev["pre_run_up"] = ev["ret_t_minus_5"]
    ev["short_ret_3d"] = -ev["ret_day1_3"] - SHORT_COST * 3

    # Quintile analysis
    ev_valid = ev.dropna(subset=["pre_run_up", "short_ret_3d"]).copy()
    ev_valid["quintile"] = pd.qcut(ev_valid["pre_run_up"], 5, labels=["Q1(跌)", "Q2", "Q3", "Q4", "Q5(漲)"])

    print(f"\n  處置前5日漲幅 vs 做空報酬 (short Day1-3):")
    print(f"  {'Quintile':<12} {'pre_run_up':>10} {'short_ret':>10} {'win%':>6} {'n':>6}")
    for q, grp in ev_valid.groupby("quintile", observed=True):
        print(f"  {str(q):<12} {grp['pre_run_up'].mean():>10.2%} {grp['short_ret_3d'].mean():>10.2%} {(grp['short_ret_3d']>0).mean():>6.1%} {len(grp):>6}")

    # By year for best quintile
    best_q = ev_valid[ev_valid["quintile"] == "Q5(漲)"]
    yearly_q5 = best_q.groupby("year")["short_ret_3d"].agg(["mean", "count", lambda x: (x > 0).mean()])
    yearly_q5.columns = ["avg_ret", "n", "win_rate"]
    yearly_q5 = yearly_q5[yearly_q5["n"] >= 5]
    n_pos = (yearly_q5["avg_ret"] > 0).sum()
    print(f"\n  Q5(處置前大漲) 做空逐年:")
    print(f"  Years positive: {n_pos}/{len(yearly_q5)}")
    for yr, r in yearly_q5.iterrows():
        print(f"    {yr}: {r['avg_ret']:+.2%} (n={int(r['n'])}, win={r['win_rate']:.1%})")

    # Save
    ev.to_csv(OUT / "s4_pre_disposal_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
