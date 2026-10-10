"""S4c: 入處置做空 — 正確方向（弱市做空）"""
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
    print("S4c: 入處置做空 — 弱市過濾（正確方向）")
    print("=" * 70)

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    dis_raw = data.get("disposal_information")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Market context
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_close = (1 + mkt_ret).cumprod()
    mkt_ma20 = mkt_close.rolling(20).mean()
    above_ma20 = (close > close.rolling(20).mean()).sum(axis=1) / close.notna().sum(axis=1)

    # Disposal events
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$") &
        dis["stock_id"].isin(valid_stocks) &
        ~dis["stock_id"].str.startswith(("00", "91"))
    ].copy()
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis = dis[(dis["start_idx"] >= 6)].copy()

    # Build events
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

            ti = si - 1
            feat = {
                "stock_id": sym, "start_idx": si, "year": row["start"].year,
                "day1_ret": day1_ret,
                "short_ret": -day1_ret - 0.006,
            }
            try:
                feat["mkt_below_ma20"] = 1 if mkt_close.iloc[ti] < mkt_ma20.iloc[ti] else 0
                feat["breadth"] = float(above_ma20.iloc[ti]) if not np.isnan(above_ma20.iloc[ti]) else 0.5
            except:
                feat["mkt_below_ma20"] = 0
                feat["breadth"] = 0.5
            try:
                feat["pre_ret_5d"] = float(c.iloc[ti] / c.iloc[ti - 5] - 1)
                feat["vol_20d"] = float(c.iloc[ti-20:ti].pct_change().dropna().std())
            except:
                feat["pre_ret_5d"] = 0
                feat["vol_20d"] = 0
            events.append(feat)
        except (IndexError, KeyError):
            continue

    ev = pd.DataFrame(events)
    print(f"  Events: {len(ev):,}")

    # === CORRECT FILTER: Short in WEAK markets ===
    print("\n" + "=" * 70)
    print("弱市過濾測試（正確方向）")
    print("=" * 70)

    baseline = ev["short_ret"].mean()
    print(f"\n  Baseline: avg={baseline:.2%}, win={(ev['short_ret']>0).mean():.1%}")

    filters = {
        "大盤MA20下": ev["mkt_below_ma20"] == 1,
        "廣度<40%": ev["breadth"] < 0.40,
        "廣度<50%": ev["breadth"] < 0.50,
        "MA20下+廣度<50%": (ev["mkt_below_ma20"] == 1) & (ev["breadth"] < 0.50),
        "MA20下+廣度<40%": (ev["mkt_below_ma20"] == 1) & (ev["breadth"] < 0.40),
        "高波動>4%": ev["vol_20d"] > 0.04,
        "MA20下+高波動": (ev["mkt_below_ma20"] == 1) & (ev["vol_20d"] > 0.04),
        "廣度<50%+高波動": (ev["breadth"] < 0.50) & (ev["vol_20d"] > 0.04),
    }

    print(f"\n  {'Filter':<25} {'keep_n':>7} {'keep_ret':>8} {'keep_win':>8} {'skip_ret':>8} {'impact':>8}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, mask in filters.items():
        keep = ev[mask]
        skip = ev[~mask]
        if len(keep) < 50 or len(skip) < 50:
            continue
        print(f"  {name:<25} {len(keep):>7} {keep['short_ret'].mean():>8.2%} {(keep['short_ret']>0).mean():>8.1%} {skip['short_ret'].mean():>8.2%} {keep['short_ret'].mean()-baseline:>+8.2%}")

    # === BEST: MA20下 + 廣度<50% ===
    print("\n" + "=" * 70)
    print("最佳過濾逐年驗證: MA20下 + 廣度<50%")
    print("=" * 70)

    best_mask = (ev["mkt_below_ma20"] == 1) & (ev["breadth"] < 0.50)
    filtered = ev[best_mask].copy()
    print(f"\n  Kept: {len(filtered)}, avg={filtered['short_ret'].mean():.2%}, win={(filtered['short_ret']>0).mean():.1%}")

    yearly = filtered.groupby("year")["short_ret"].agg(["mean", "count", lambda x: (x > 0).mean()])
    yearly.columns = ["avg_ret", "n", "win_rate"]
    yearly = yearly[yearly["n"] >= 5]
    n_pos = (yearly["avg_ret"] > 0).sum()
    print(f"\n  Years positive: {n_pos}/{len(yearly)}")
    print(f"  {'Year':>6} {'avg_ret':>8} {'n':>5} {'win%':>6}")
    for yr, r in yearly.iterrows():
        print(f"  {yr:>6} {r['avg_ret']:>8.2%} {int(r['n']):>5} {r['win_rate']:>6.1%}")

    # === PORTFOLIO: 弱市做空 + 每日最多N檔 ===
    print("\n" + "=" * 70)
    print("組合模擬: 弱市入處置做空")
    print("=" * 70)

    for max_pos in [5, 10, 20]:
        filtered_sorted = filtered.sort_values("start_idx")
        daily_short = pd.Series(0.0, index=cal)
        n_trades = 0
        daily_count = pd.Series(0, index=cal)

        for _, t in filtered_sorted.iterrows():
            si = int(t["start_idx"])
            if si >= n_cal:
                continue
            if daily_count.iloc[si] >= max_pos:
                continue
            daily_count.iloc[si] += 1
            n_trades += 1
            daily_short.iloc[si] += t["short_ret"] / max_pos

        daily_short = daily_short.clip(upper=0.05, lower=-0.05)

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
        for yr in sorted(yr_returns.keys()):
            print(f"      {yr}: {yr_returns[yr]:+.1%}")

    # === NO FILTER: 全量做空 ===
    print("\n" + "=" * 70)
    print("無過濾: 所有處置事件做空")
    print("=" * 70)

    for max_pos in [10, 20, 50]:
        ev_sorted = ev.sort_values("start_idx")
        daily_short = pd.Series(0.0, index=cal)
        daily_count = pd.Series(0, index=cal)
        n_trades = 0

        for _, t in ev_sorted.iterrows():
            si = int(t["start_idx"])
            if si >= n_cal:
                continue
            if daily_count.iloc[si] >= max_pos:
                continue
            daily_count.iloc[si] += 1
            n_trades += 1
            daily_short.iloc[si] += t["short_ret"] / max_pos

        daily_short = daily_short.clip(upper=0.05, lower=-0.05)

        sharpe = np.mean(daily_short) / np.std(daily_short) * np.sqrt(252) if np.std(daily_short) > 0 else 0
        cum = (1 + daily_short).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()

        yr_returns = {}
        for yr in range(2010, 2027):
            yr_mask = cal.year == yr
            if yr_mask.sum() > 50:
                yr_returns[yr] = np.prod(1 + daily_short.values[yr_mask]) - 1
        n_pos_yr = sum(1 for v in yr_returns.values() if v > 0)

        print(f"\n  No filter, Max {max_pos}/day:")
        print(f"    Sharpe: {sharpe:.2f}, MDD: {mdd:.1%}, Years: {n_pos_yr}/{len(yr_returns)}")
        print(f"    Trades: {n_trades}")
        for yr in [2011, 2015, 2018, 2020, 2022, 2023, 2024, 2025]:
            if yr in yr_returns:
                print(f"      {yr}: {yr_returns[yr]:+.1%}")

    ev.to_csv(OUT / "s4c_all_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
