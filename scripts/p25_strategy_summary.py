"""P25: 全策略績效總覽 + Overfitting 檢測"""
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

ALLOC_V20 = 0.70
ALLOC_SHORT = 0.20
ALLOC_CLONG = 0.10
V20_MAX_POSITIONS = 5
V20_BREADTH_MIN = 0.40
BREADTH_DECLINE_THRESHOLD = -0.15
SHORT_HOLD_DAYS = 3
SHORT_COST = 0.001
S4_DISPOSAL_PRESSURE_MIN = 5
S4_SHORT_BONUS = 0.10


def compute_metrics(daily_ret, cal, label=""):
    cum = (1 + daily_ret).cumprod()
    total_yrs = len(daily_ret[daily_ret.index.year >= 2010]) / 252
    cagr = cum.iloc[-1] ** (1 / max(total_yrs, 1)) - 1 if cum.iloc[-1] > 0 else -1
    sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) > 0 else 0
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    yearly = {}
    for yr in range(2010, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() > 50:
            yearly[yr] = np.prod(1 + daily_ret.values[yr_mask]) - 1
    n_pos = sum(1 for v in yearly.values() if v > 0)

    # Sub-period analysis (overfitting check)
    mid = len(daily_ret) // 2
    first_half = daily_ret.iloc[:mid]
    second_half = daily_ret.iloc[mid:]
    s1 = np.mean(first_half) / np.std(first_half) * np.sqrt(252) if np.std(first_half) > 0 else 0
    s2 = np.mean(second_half) / np.std(second_half) * np.sqrt(252) if np.std(second_half) > 0 else 0

    # Rolling 3-year Sharpe stability
    rolling_sharpes = []
    for start in range(0, len(daily_ret) - 756, 252):
        window = daily_ret.iloc[start:start+756]
        if np.std(window) > 0:
            rolling_sharpes.append(np.mean(window) / np.std(window) * np.sqrt(252))
    rolling_min = min(rolling_sharpes) if rolling_sharpes else 0
    rolling_max = max(rolling_sharpes) if rolling_sharpes else 0

    return {
        "label": label,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "years_positive": f"{n_pos}/{len(yearly)}",
        "yearly": yearly,
        "sharpe_first_half": s1,
        "sharpe_second_half": s2,
        "rolling_3yr_min": rolling_min,
        "rolling_3yr_max": rolling_max,
    }


def main():
    print("=" * 70)
    print("P25: 全策略績效總覽")
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
    breadth_decline = above_ma20 - above_ma20.shift(5)
    s1_signal = (breadth_decline < BREADTH_DECLINE_THRESHOLD).rolling(SHORT_HOLD_DAYS, min_periods=1).max()

    # Disposal
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$") &
        dis["stock_id"].isin(valid_stocks) &
        ~dis["stock_id"].str.startswith(("00", "91"))
    ].copy()
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")

    # S4: disposal pressure
    dis_counts = dis.groupby("start_idx").size()
    disposal_pressure = pd.Series(0, index=range(n_cal))
    for idx, count in dis_counts.items():
        if 0 <= idx < n_cal:
            disposal_pressure.iloc[idx] = count
    mkt_weak = pd.Series(mkt_close.values < mkt_ma20.values, index=cal)

    # === STRATEGY 1: V20 (no filter) ===
    print("\n[1] V20 無過濾...")
    v20_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p23_v20_longterm\p23_v20_longterm_trades.csv")
    v20_trades = pd.read_csv(v20_path, parse_dates=["entry_date", "exit_date"])

    v20_daily_raw = pd.Series(0.0, index=cal)
    for _, t in v20_trades.iterrows():
        ei = cal.searchsorted(t["entry_date"], side="left")
        xi = cal.searchsorted(t["exit_date"], side="left")
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold * (ALLOC_V20 / V20_MAX_POSITIONS)
        for d in range(ei, min(xi + 1, n_cal)):
            v20_daily_raw.iloc[d] += dr

    m_v20_raw = compute_metrics(v20_daily_raw, cal, "V20 無過濾")

    # === STRATEGY 1b: V20 with breadth filter ===
    print("[1b] V20 + 廣度過濾...")
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

    v20_daily = pd.Series(0.0, index=cal)
    for _, t in v20_filtered.iterrows():
        ei = cal.searchsorted(t["entry_date"], side="left")
        xi = cal.searchsorted(t["exit_date"], side="left")
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold * (ALLOC_V20 / V20_MAX_POSITIONS)
        for d in range(ei, min(xi + 1, n_cal)):
            v20_daily.iloc[d] += dr

    m_v20 = compute_metrics(v20_daily, cal, "V20 + 廣度過濾")

    # === STRATEGY 2: S1 廣度做空 ===
    print("[2] S1 廣度做空...")
    s1_daily = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        if s1_signal.iloc[i]:
            s1_daily.iloc[i] = -mkt_ret.iloc[i] * ALLOC_SHORT - SHORT_COST * ALLOC_SHORT
    m_s1 = compute_metrics(s1_daily, cal, "S1 廣度做空")

    # === STRATEGY 3: S4 處置壓力做空 ===
    print("[3] S4 處置壓力做空...")
    s4_daily = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        s1_active = bool(s1_signal.iloc[i])
        s4_active = bool(disposal_pressure.iloc[i] >= S4_DISPOSAL_PRESSURE_MIN and mkt_weak.iloc[i])
        if s4_active and not s1_active:
            s4_daily.iloc[i] = -mkt_ret.iloc[i] * ALLOC_SHORT * 0.5 - SHORT_COST * ALLOC_SHORT * 0.5
    m_s4 = compute_metrics(s4_daily, cal, "S4 處置壓力(獨立)")

    # === STRATEGY 2+3: Combined Short ===
    print("[2+3] 組合做空 (S1+S4)...")
    short_daily = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        s1_a = bool(s1_signal.iloc[i])
        s4_a = bool(disposal_pressure.iloc[i] >= S4_DISPOSAL_PRESSURE_MIN and mkt_weak.iloc[i])
        if s1_a and s4_a:
            alloc = ALLOC_SHORT + S4_SHORT_BONUS
        elif s1_a:
            alloc = ALLOC_SHORT
        elif s4_a:
            alloc = ALLOC_SHORT * 0.5
        else:
            continue
        short_daily.iloc[i] = -mkt_ret.iloc[i] * alloc - SHORT_COST * alloc
    m_short = compute_metrics(short_daily, cal, "Short S1+S4")

    # === COMBINED: V20 + Short ===
    print("[COMBO] V20 + Short...")
    combined = v20_daily + short_daily
    m_combo = compute_metrics(combined, cal, "V20 + Short(S1+S4)")

    # === V20 raw + Short (for comparison) ===
    combined_raw = v20_daily_raw + short_daily
    m_combo_raw = compute_metrics(combined_raw, cal, "V20無過濾 + Short")

    # === PRINT RESULTS ===
    all_metrics = [m_v20_raw, m_v20, m_s1, m_s4, m_short, m_combo]

    print("\n" + "=" * 70)
    print("策略績效總覽")
    print("=" * 70)
    print(f"\n  {'Strategy':<25} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'Years':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    for m in all_metrics:
        print(f"  {m['label']:<25} {m['cagr']:>8.1%} {m['sharpe']:>7.2f} {m['mdd']:>8.1%} {m['years_positive']:>8}")

    # Year-by-year
    print("\n" + "=" * 70)
    print("逐年報酬")
    print("=" * 70)
    years = sorted(m_combo["yearly"].keys())
    print(f"\n  {'Year':>6}", end="")
    for m in all_metrics:
        print(f" {m['label'][:12]:>12}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in all_metrics:
        print(f" {'-'*12}", end="")
    print()

    for yr in years:
        print(f"  {yr:>6}", end="")
        for m in all_metrics:
            val = m["yearly"].get(yr, 0)
            print(f" {val:>12.1%}", end="")
        print()

    # Bear years focus
    print("\n" + "=" * 70)
    print("熊年表現 (大盤下跌年份)")
    print("=" * 70)
    bear_years = []
    for yr in years:
        yr_mask = cal.year == yr
        if yr_mask.sum() > 50:
            yr_indices = np.where(yr_mask)[0]
            start_i = yr_indices[0]
            end_i = yr_indices[-1]
            mkt_yr_ret = mkt_close.iloc[end_i] / mkt_close.iloc[start_i] - 1
            if mkt_yr_ret < -0.05:
                bear_years.append(yr)

    print(f"\n  大盤下跌>5%的年份: {bear_years}")
    print(f"\n  {'Year':>6}", end="")
    for m in all_metrics:
        print(f" {m['label'][:12]:>12}", end="")
    print()
    for yr in bear_years:
        print(f"  {yr:>6}", end="")
        for m in all_metrics:
            val = m["yearly"].get(yr, 0)
            print(f" {val:>12.1%}", end="")
        print()

    # === OVERFITTING ASSESSMENT ===
    print("\n" + "=" * 70)
    print("Overfitting 檢測")
    print("=" * 70)

    print(f"\n  {'Strategy':<25} {'Sharpe前半':>10} {'Sharpe後半':>10} {'3yr最低':>8} {'3yr最高':>8} {'穩定性':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for m in all_metrics:
        stability = "OK" if m["sharpe_second_half"] > 0.5 * m["sharpe"] else "⚠️"
        if m["rolling_3yr_min"] < 0:
            stability = "⚠️ 有負Sharpe窗口"
        print(f"  {m['label']:<25} {m['sharpe_first_half']:>10.2f} {m['sharpe_second_half']:>10.2f} {m['rolling_3yr_min']:>8.2f} {m['rolling_3yr_max']:>8.2f} {stability:>8}")

    # Parameter count
    print(f"\n  參數數量統計:")
    print(f"    V20: 入場日(Day1/Day4), 出場(關末-1), 流動性門檻(20M), Gap(-8%~+4%), 熔斷(>400), 廣度過濾(40%)")
    print(f"      → 6 個參數, 但全部有明確經濟邏輯")
    print(f"    S1: 廣度下降門檻(-15%), 持有期(3天)")
    print(f"      → 2 個參數")
    print(f"    S4: 處置進場門檻(5檔), 市場弱(MA20下)")
    print(f"      → 2 個參數")
    print(f"    總計: 10 個參數, 17年數據 = 每參數1.7年樣本")

    # Sensitivity check
    print(f"\n  參數敏感度 (V20 廣度過濾門檻):")
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
        mask_test = pd.Series(False, index=v20_trades.index)
        for idx, t in v20_trades.iterrows():
            ei = int(t["entry_idx"])
            if ei < len(above_ma20) and ei < len(mkt_close):
                b = float(above_ma20.iloc[ei]) if not np.isnan(above_ma20.iloc[ei]) else 0.5
                below = bool(mkt_close.iloc[ei] < mkt_ma20.iloc[ei]) if not np.isnan(mkt_ma20.iloc[ei]) else False
                if b < threshold and below:
                    mask_test.iloc[idx] = True
        kept = v20_trades[~mask_test]
        avg_ret = kept["ret"].mean()
        print(f"    門檻 {threshold:.0%}: 保留{len(kept)}筆, 平均{avg_ret:.2%}")

    # Save yearly table
    yearly_df = pd.DataFrame({"year": years})
    for m in all_metrics:
        yearly_df[m["label"]] = [m["yearly"].get(y, 0) for y in years]
    yearly_df.to_csv(OUT / "p25_yearly_performance.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
