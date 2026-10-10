"""P25b: 全策略 100% 資金配置績效"""
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p25_strategy_summary")
OUT.mkdir(parents=True, exist_ok=True)

V20_MAX_POSITIONS = 5
V20_BREADTH_MIN = 0.40
BREADTH_DECLINE_THRESHOLD = -0.15
SHORT_HOLD_DAYS = 3
SHORT_COST = 0.001
S4_DISPOSAL_PRESSURE_MIN = 5
S4_SHORT_BONUS = 0.10


def compute_metrics(daily_ret, cal, label=""):
    cum = (1 + daily_ret).cumprod()
    total_yrs = len(daily_ret[daily_ret.index.year >= 2011]) / 252
    cagr = cum.iloc[-1] ** (1 / max(total_yrs, 1)) - 1 if cum.iloc[-1] > 0 else -1
    sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) > 0 else 0
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    yearly = {}
    for yr in range(2011, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() > 50:
            yearly[yr] = np.prod(1 + daily_ret.values[yr_mask]) - 1
    n_pos = sum(1 for v in yearly.values() if v > 0)

    mid = len(daily_ret) // 2
    s1 = np.mean(daily_ret.iloc[:mid]) / np.std(daily_ret.iloc[:mid]) * np.sqrt(252) if np.std(daily_ret.iloc[:mid]) > 0 else 0
    s2 = np.mean(daily_ret.iloc[mid:]) / np.std(daily_ret.iloc[mid:]) * np.sqrt(252) if np.std(daily_ret.iloc[mid:]) > 0 else 0

    rolling_sharpes = []
    for start in range(0, len(daily_ret) - 756, 252):
        window = daily_ret.iloc[start:start+756]
        if np.std(window) > 0:
            rolling_sharpes.append(np.mean(window) / np.std(window) * np.sqrt(252))
    rolling_min = min(rolling_sharpes) if rolling_sharpes else 0

    return {
        "label": label, "cagr": cagr, "sharpe": sharpe, "mdd": mdd,
        "years_positive": f"{n_pos}/{len(yearly)}", "yearly": yearly,
        "sharpe_first": s1, "sharpe_second": s2, "rolling_min": rolling_min,
    }


def main():
    print("=" * 70)
    print("P25b: 全策略 100% 資金績效")
    print("=" * 70)

    close = data.get("price:收盤價")
    dis_raw = data.get("disposal_information")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Market context
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_close = (1 + mkt_ret).cumprod()
    mkt_ma20 = mkt_close.rolling(20).mean()
    above_ma20 = (close > close.rolling(20).mean()).sum(axis=1) / close.notna().sum(axis=1)
    breadth_decline = above_ma20 - above_ma20.shift(5)
    s1_signal = (breadth_decline < BREADTH_DECLINE_THRESHOLD).rolling(SHORT_HOLD_DAYS, min_periods=1).max()

    # Disposal
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$") &
        dis["stock_id"].isin(valid_stocks) &
        ~dis["stock_id"].str.startswith(("00", "91"))
    ].copy()
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")

    # S4 pressure
    dis_counts = dis.groupby("start_idx").size()
    disposal_pressure = pd.Series(0, index=range(n_cal))
    for idx, count in dis_counts.items():
        if 0 <= idx < n_cal:
            disposal_pressure.iloc[idx] = count
    mkt_weak = pd.Series(mkt_close.values < mkt_ma20.values, index=cal)

    # === V20 100% (no filter) ===
    print("\n[1] V20 100% 無過濾...")
    v20_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p23_v20_longterm\p23_v20_longterm_trades.csv")
    v20_trades = pd.read_csv(v20_path, parse_dates=["entry_date", "exit_date"])

    v20_daily = pd.Series(0.0, index=cal)
    for _, t in v20_trades.iterrows():
        ei = cal.searchsorted(t["entry_date"], side="left")
        xi = cal.searchsorted(t["exit_date"], side="left")
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold * (1.0 / V20_MAX_POSITIONS)  # 100% / 5 = 20% per trade
        for d in range(ei, min(xi + 1, n_cal)):
            v20_daily.iloc[d] += dr

    m_v20 = compute_metrics(v20_daily, cal, "V20 100% 無過濾")

    # === V20 100% with breadth filter ===
    print("[1b] V20 100% + 廣度過濾...")
    v20_trades["entry_idx"] = cal.searchsorted(v20_trades["entry_date"], side="left")
    mask_bad = pd.Series(False, index=v20_trades.index)
    for idx, t in v20_trades.iterrows():
        ei = int(t["entry_idx"])
        if ei < len(above_ma20) and ei < len(mkt_close):
            b = float(above_ma20.iloc[ei]) if not np.isnan(above_ma20.iloc[ei]) else 0.5
            below = bool(mkt_close.iloc[ei] < mkt_ma20.iloc[ei]) if not np.isnan(mkt_ma20.iloc[ei]) else False
            if b < V20_BREADTH_MIN and below:
                mask_bad.iloc[idx] = True
    v20_filtered = v20_trades[~mask_bad].copy()

    v20f_daily = pd.Series(0.0, index=cal)
    for _, t in v20_filtered.iterrows():
        ei = cal.searchsorted(t["entry_date"], side="left")
        xi = cal.searchsorted(t["exit_date"], side="left")
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold * (1.0 / V20_MAX_POSITIONS)
        for d in range(ei, min(xi + 1, n_cal)):
            v20f_daily.iloc[d] += dr

    m_v20f = compute_metrics(v20f_daily, cal, "V20 100% + 廣度過濾")

    # === S1 100% ===
    print("[2] S1 100%...")
    s1_daily = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        if s1_signal.iloc[i]:
            s1_daily.iloc[i] = -mkt_ret.iloc[i] - SHORT_COST
    m_s1 = compute_metrics(s1_daily, cal, "S1 廣度做空 100%")

    # === S1+S4 100% ===
    print("[2+3] S1+S4 100%...")
    short_daily = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        s1_a = bool(s1_signal.iloc[i])
        s4_a = bool(disposal_pressure.iloc[i] >= S4_DISPOSAL_PRESSURE_MIN and mkt_weak.iloc[i])
        if s1_a and s4_a:
            alloc = 1.0 + S4_SHORT_BONUS  # 110% (leverage)
        elif s1_a:
            alloc = 1.0
        elif s4_a:
            alloc = 0.5
        else:
            continue
        short_daily.iloc[i] = -mkt_ret.iloc[i] * alloc - SHORT_COST * alloc
    m_short = compute_metrics(short_daily, cal, "Short S1+S4 100%")

    # === COMBO: V20 100% + Short overlay ===
    print("[COMBO] V20 100% + Short overlay...")
    # V20 gets 100%, short is overlay (additional 30% when active)
    combo = v20f_daily + short_daily * 0.30
    m_combo = compute_metrics(combo, cal, "V20+Short 組合")

    # === V20 100% + Short 50% overlay ===
    combo2 = v20f_daily + short_daily * 0.50
    m_combo2 = compute_metrics(combo2, cal, "V20+Short50% 組合")

    # === PRINT ===
    all_m = [m_v20, m_v20f, m_s1, m_short, m_combo, m_combo2]

    print("\n" + "=" * 70)
    print("策略績效 (100% 資金)")
    print("=" * 70)
    print(f"\n  {'Strategy':<25} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'Years':>8} {'3yrMin':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for m in all_m:
        print(f"  {m['label']:<25} {m['cagr']:>8.1%} {m['sharpe']:>7.2f} {m['mdd']:>8.1%} {m['years_positive']:>8} {m['rolling_min']:>7.2f}")

    # Year-by-year
    print("\n" + "=" * 70)
    print("逐年報酬")
    print("=" * 70)
    years = sorted(m_v20["yearly"].keys())
    print(f"\n  {'Year':>6}", end="")
    for m in all_m:
        print(f" {m['label'][:14]:>14}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in all_m:
        print(f" {'-'*14}", end="")
    print()
    for yr in years:
        print(f"  {yr:>6}", end="")
        for m in all_m:
            val = m["yearly"].get(yr, 0)
            print(f" {val:>14.1%}", end="")
        print()

    # Bear years
    print("\n" + "=" * 70)
    print("熊年")
    print("=" * 70)
    bear_years = [2011, 2015, 2018, 2022]
    print(f"\n  {'Year':>6}", end="")
    for m in all_m:
        print(f" {m['label'][:14]:>14}", end="")
    print()
    for yr in bear_years:
        print(f"  {yr:>6}", end="")
        for m in all_m:
            val = m["yearly"].get(yr, 0)
            print(f" {val:>14.1%}", end="")
        print()

    # Overfitting
    print("\n" + "=" * 70)
    print("Overfitting")
    print("=" * 70)
    print(f"\n  {'Strategy':<25} {'前半':>6} {'後半':>6} {'3yrMin':>7} {'判定':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")
    for m in all_m:
        verdict = "OK" if m["rolling_min"] > 0 and m["sharpe_second"] > 0.5 * m["sharpe"] else "⚠️"
        print(f"  {m['label']:<25} {m['sharpe_first']:>6.2f} {m['sharpe_second']:>6.2f} {m['rolling_min']:>7.2f} {verdict:>10}")

    # Save
    yearly_df = pd.DataFrame({"year": years})
    for m in all_m:
        yearly_df[m["label"]] = [m["yearly"].get(y, 0) for y in years]
    yearly_df.to_csv(OUT / "p25b_100pct_yearly.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
