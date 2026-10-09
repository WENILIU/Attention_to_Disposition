"""P19i: C-Long 優化空間探索

測試方向:
A. 連續注意偵測 (昨天也被注意 = 明天幾乎確定處置)
B. 出場時機優化 (處置公告 vs 處置開始 vs 固定天數)
C. 部位加權 (按模型信心分配)
D. 持有期優化 (5d vs 7d vs 10d vs 動態)
E. 兩階段交易 (先做空1天→再做多)
F. 特徵增強: 實際連續天數計算
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19i_optimization")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P19i: C-Long 優化空間探索")
    print("=" * 70)

    # Load walk-forward predictions
    print("\nLoading data...")
    pred_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19g_walkforward\p19g_all_predictions.csv")
    att = pd.read_csv(pred_path, parse_dates=["announce"])
    att["stock_id"] = att["stock_id"].astype(str).str.zfill(4)
    att = att[att["oos_prob"].notna()].copy()

    # Load price data
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    dis_raw = data.get("disposal_information")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Build disposal lookup with both announce and start dates
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["dis_start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis_by_stock = {}
    for _, row in dis.iterrows():
        sid = row["stock_id"]
        if sid not in dis_by_stock:
            dis_by_stock[sid] = []
        dis_by_stock[sid].append((row["dis_announce"], row["dis_start"]))

    # === A: CONSECUTIVE ATTENTION DETECTION ===
    print("\n" + "=" * 70)
    print("A: 連續注意偵測 (昨天也被注意?)")
    print("=" * 70)

    # Build attention dates per stock
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    # Check: was this stock also attention'd 1 day ago? 2 days ago?
    att["consec_1d"] = False
    att["consec_2d"] = False
    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann = row["announce"]
        if sid in att_dates_by_stock:
            if (ann - pd.Timedelta(days=1)) in att_dates_by_stock[sid] or \
               (ann - pd.Timedelta(days=2)) in att_dates_by_stock[sid]:  # weekend
                att.loc[idx, "consec_1d"] = True
            if (ann - pd.Timedelta(days=2)) in att_dates_by_stock[sid] or \
               (ann - pd.Timedelta(days=3)) in att_dates_by_stock[sid] or \
               (ann - pd.Timedelta(days=4)) in att_dates_by_stock[sid]:  # weekend+
                att.loc[idx, "consec_2d"] = True

    # Upgrade rate by consecutive status
    print("\n  連續注意 vs 升級率:")
    for label, mask in [("非連續", ~att["consec_1d"]), ("連續(昨日也有)", att["consec_1d"]), ("連續2天+", att["consec_2d"])]:
        subset = att[mask]
        if len(subset) < 10:
            continue
        rate = subset["leads_to_disposal"].mean()
        print(f"    {label}: upgrade={rate:.1%}, n={len(subset):,}")

    # Returns for consecutive events
    valid_trades = att[att["trade_ret"].notna()].copy()
    selected = valid_trades[valid_trades["oos_prob"] >= 0.6].copy()

    print(f"\n  選中事件(prob>=0.6)中的連續效應:")
    for label, mask in [("非連續", ~selected["consec_1d"]), ("連續(昨日也有)", selected["consec_1d"])]:
        subset = selected[mask]
        if len(subset) < 10:
            continue
        avg = subset["trade_ret"].mean()
        win = (subset["trade_ret"] > 0).mean()
        prec = subset["leads_to_disposal"].mean()
        print(f"    {label}: ret={avg:.2%}, win={win:.1%}, prec={prec:.1%}, n={len(subset):,}")

    # === B: EXIT TIMING ===
    print("\n" + "=" * 70)
    print("B: 出場時機優化")
    print("=" * 70)

    # For upgraded events, compare exits at different points
    upg = valid_trades[valid_trades["leads_to_disposal"]].copy()
    upg["ann_idx"] = cal.searchsorted(upg["announce"], side="left")

    exit_results = {"dis_announce_minus1": [], "dis_start_minus1": [], "hold_5d": [], "hold_7d": [], "hold_10d": []}

    for _, row in upg.iterrows():
        sym = row["stock_id"]
        ai = int(row["ann_idx"])
        entry_idx = ai + 1
        if sym not in valid_stocks or entry_idx >= n_cal - 1:
            continue
        try:
            entry_price = open_p.iloc[entry_idx][sym]
        except (IndexError, KeyError):
            continue
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Find disposal dates
        ann_date = row["announce"]
        dis_ann_date = None
        dis_start_date = None
        if sym in dis_by_stock:
            for da, ds in dis_by_stock[sym]:
                delta = (da - ann_date).days
                if 0 < delta <= 30:
                    dis_ann_date = da
                    dis_start_date = ds
                    break

        # Exit at disposal announcement - 1
        if dis_ann_date:
            idx = cal.searchsorted(dis_ann_date, side="left") - 1
            if entry_idx < idx < n_cal:
                try:
                    p = close.iloc[idx][sym]
                    if not np.isnan(p) and p > 0:
                        exit_results["dis_announce_minus1"].append(p / entry_price - 1 - COST_RATE)
                except (IndexError, KeyError):
                    pass

        # Exit at disposal start - 1
        if dis_start_date:
            idx = cal.searchsorted(dis_start_date, side="left") - 1
            if entry_idx < idx < n_cal:
                try:
                    p = close.iloc[idx][sym]
                    if not np.isnan(p) and p > 0:
                        exit_results["dis_start_minus1"].append(p / entry_price - 1 - COST_RATE)
                except (IndexError, KeyError):
                    pass

        # Fixed holds
        for hold, key in [(5, "hold_5d"), (7, "hold_7d"), (10, "hold_10d")]:
            idx = min(entry_idx + hold - 1, n_cal - 1)
            try:
                p = close.iloc[idx][sym]
                if not np.isnan(p) and p > 0:
                    exit_results[key].append(p / entry_price - 1 - COST_RATE)
            except (IndexError, KeyError):
                pass

    print("\n  升級事件的出場時機比較:")
    for key, rets in exit_results.items():
        if len(rets) < 50:
            continue
        r = np.array(rets)
        print(f"    {key}: mean={r.mean():.2%}, median={np.median(r):.2%}, win={(r>0).mean():.1%}, n={len(r):,}")

    # === C: CONFIDENCE-WEIGHTED SIZING ===
    print("\n" + "=" * 70)
    print("C: 模型信心加權部位")
    print("=" * 70)

    # Compare equal-weight vs confidence-weighted
    sel = selected.copy()
    if len(sel) > 100:
        # Equal weight
        eq_ret = sel["trade_ret"].mean()
        # Confidence weighted
        weights = sel["oos_prob"] / sel["oos_prob"].sum()
        cw_ret = (sel["trade_ret"] * weights).sum()
        # Top quartile only (highest confidence)
        top_q = sel[sel["oos_prob"] >= sel["oos_prob"].quantile(0.75)]
        tq_ret = top_q["trade_ret"].mean() if len(top_q) > 20 else np.nan

        print(f"    等權重: {eq_ret:.2%}, n={len(sel):,}")
        print(f"    信心加權: {cw_ret:.2%}")
        print(f"    只做最高25%: {tq_ret:.2%}, n={len(top_q):,}")

    # === D: HOLDING PERIOD ===
    print("\n" + "=" * 70)
    print("D: 持有期優化")
    print("=" * 70)

    # Recompute returns for different max holds
    for max_hold in [3, 5, 7, 10, 15]:
        rets = []
        for _, row in selected.iterrows():
            sym = row["stock_id"]
            ai = cal.searchsorted(row["announce"], side="left")
            entry_idx = ai + 1
            if sym not in valid_stocks or entry_idx >= n_cal - 1:
                continue
            try:
                entry_price = open_p.iloc[entry_idx][sym]
            except (IndexError, KeyError):
                continue
            if np.isnan(entry_price) or entry_price <= 0:
                continue

            # Exit: min(disposal, max_hold)
            exit_idx = min(entry_idx + max_hold - 1, n_cal - 1)
            # But if disposal comes earlier, use that
            ann_date = row["announce"]
            if row["leads_to_disposal"] and sym in dis_by_stock:
                for da, ds in dis_by_stock[sym]:
                    delta = (da - ann_date).days
                    if 0 < delta <= 30:
                        dis_idx = cal.searchsorted(da, side="left") - 1
                        exit_idx = min(exit_idx, dis_idx, n_cal - 1)
                        break

            try:
                exit_price = close.iloc[exit_idx][sym]
                if not np.isnan(exit_price) and exit_price > 0:
                    ret = exit_price / entry_price - 1 - COST_RATE
                    hold = exit_idx - entry_idx + 1
                    rets.append((ret, hold))
            except (IndexError, KeyError):
                continue

        if len(rets) < 50:
            continue
        rets_arr = np.array([r[0] for r in rets])
        holds_arr = np.array([r[1] for r in rets])
        avg_hold = holds_arr.mean()
        # Capital efficiency: return per day held
        eff = rets_arr.mean() / avg_hold if avg_hold > 0 else 0
        print(f"    max_hold={max_hold}d: ret={rets_arr.mean():.2%}, win={(rets_arr>0).mean():.1%}, "
              f"avg_hold={avg_hold:.1f}d, eff/day={eff:.3%}, n={len(rets):,}")

    # === E: TWO-PHASE (SHORT 1D → LONG) ===
    print("\n" + "=" * 70)
    print("E: 兩階段交易概念驗證")
    print("=" * 70)
    print("  注意公告 → Day1做空(跌) → Day2做多(漲向處置)")

    # For upgraded events: what's the Day1 return vs Day2+ return?
    upg_valid = valid_trades[valid_trades["leads_to_disposal"]].copy()
    upg_valid["ann_idx"] = cal.searchsorted(upg_valid["announce"], side="left")

    day1_rets = []
    day2plus_rets = []
    for _, row in upg_valid.iterrows():
        sym = row["stock_id"]
        ai = int(row["ann_idx"])
        entry_idx = ai + 1
        if sym not in valid_stocks or entry_idx >= n_cal - 2:
            continue
        try:
            open1 = open_p.iloc[entry_idx][sym]
            close1 = close.iloc[entry_idx][sym]
            open2 = open_p.iloc[entry_idx + 1][sym]
        except (IndexError, KeyError):
            continue
        if any(np.isnan(x) or x <= 0 for x in [open1, close1, open2]):
            continue

        # Day 1: short from open to close
        day1_short = -(close1 / open1 - 1) - COST_RATE
        # Day 2+: long from open2 to disposal
        day2plus_rets.append(row["trade_ret"])  # approximate
        day1_rets.append(day1_short)

    if day1_rets:
        d1 = np.array(day1_rets)
        print(f"\n    Day1做空(升級事件): mean={d1.mean():.2%}, win={(d1>0).mean():.1%}, n={len(d1):,}")
        print(f"    (如果Day1做空有效,可以先做空再反手做多)")

    # === F: CONSECUTIVE AS FEATURE BOOST ===
    print("\n" + "=" * 70)
    print("F: 連續注意作為額外過濾")
    print("=" * 70)

    # If we ONLY trade consecutive attention events (with model confirmation)
    consec_selected = selected[selected["consec_1d"]].copy()
    if len(consec_selected) > 20:
        print(f"\n  只做連續注意 + model>=0.6:")
        print(f"    avg_ret={consec_selected['trade_ret'].mean():.2%}, win={(consec_selected['trade_ret']>0).mean():.1%}, n={len(consec_selected):,}")
        print(f"    precision={consec_selected['leads_to_disposal'].mean():.1%}")
        for yr in sorted(consec_selected["year"].unique()):
            yr_data = consec_selected[consec_selected["year"] == yr]
            if len(yr_data) < 5:
                continue
            avg = yr_data["trade_ret"].mean()
            sign = "PASS" if avg > 0 else "FAIL"
            print(f"    {yr}: {avg:.2%}, n={len(yr_data)} {sign}")

    # Save
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
