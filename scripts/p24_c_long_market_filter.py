"""P24: C-Long 大盤濾網測試.

測試各種大盤濾網能否修復空頭年虧損:
  A. 大盤MA20之上才進場
  B. 大盤MA60之上才進場
  C. 大盤5日跌幅 > -X% 才進場
  D. 市場波動率 < X 才進場
  E. 大盤動量(20日) > 0 才進場
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p24_market_filter")
OUT.mkdir(parents=True, exist_ok=True)

MAX_POSITIONS = 5


def simulate(ex, cal, n_cal):
    """Simulate portfolio and return metrics."""
    if len(ex) < 30:
        return None
    daily_returns = pd.Series(0.0, index=cal)
    cap = 1.0 / MAX_POSITIONS
    for _, t in ex.iterrows():
        ei = int(t["ann_idx"]) + 1
        xi = min(int(t["exit_idx"]), n_cal - 1)
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            daily_returns.iloc[d] += dr * cap
    dm = daily_returns.mean()
    ds = daily_returns.std()
    sharpe = dm / ds * np.sqrt(252) if ds > 0 else 0
    cum = (1 + daily_returns).cumprod()
    rm = cum.cummax()
    mdd = ((cum - rm) / rm).min()
    first_idx = int(ex["ann_idx"].min()) + 1
    total_days = n_cal - first_idx
    cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1
    yearly_pass = 0
    yearly_total = 0
    for yr in sorted(ex["year"].unique()):
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 5:
            continue
        yearly_total += 1
        if yr_data["ret"].mean() > 0:
            yearly_pass += 1
    return {"cagr": cagr, "sharpe": sharpe, "mdd": mdd,
            "yearly_pass": yearly_pass, "yearly_total": yearly_total,
            "n": len(ex), "avg_ret": ex["ret"].mean()}


def main():
    print("=" * 70)
    print("P24: C-Long 大盤濾網測試")
    print("=" * 70)

    # Load p22 results
    print("\nLoading p22 walk-forward results...")
    pred_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p22_longterm\p22_longterm_trades.csv")
    ex_base = pd.read_csv(pred_path, parse_dates=["announce"])
    ex_base["stock_id"] = ex_base["stock_id"].astype(str).str.zfill(4)
    print(f"  Base trades: {len(ex_base):,}")

    # Load price data for market indicators
    print("Loading market data...")
    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Market indicators (equal-weighted all stocks as proxy)
    mkt_close = close.mean(axis=1)  # equal-weighted "index"
    mkt_ret = mkt_close.pct_change()
    mkt_ma20 = mkt_close.rolling(20).mean()
    mkt_ma60 = mkt_close.rolling(60).mean()
    mkt_ret_5d = mkt_close.pct_change(5)
    mkt_ret_20d = mkt_close.pct_change(20)
    mkt_vol_20d = mkt_ret.rolling(20).std()

    # Add market indicators to each trade
    ex_base["ann_idx"] = cal.searchsorted(ex_base["announce"], side="left")
    ex_base["entry_idx"] = ex_base["ann_idx"] + 1

    # Get market state at entry time
    ex_base["mkt_above_ma20"] = False
    ex_base["mkt_above_ma60"] = False
    ex_base["mkt_ret_5d"] = np.nan
    ex_base["mkt_ret_20d"] = np.nan
    ex_base["mkt_vol_20d"] = np.nan

    for idx, row in ex_base.iterrows():
        ei = int(row["entry_idx"])
        if ei >= n_cal:
            continue
        try:
            mc = mkt_close.iloc[ei]
            ma20 = mkt_ma20.iloc[ei]
            ma60 = mkt_ma60.iloc[ei]
            r5 = mkt_ret_5d.iloc[ei]
            r20 = mkt_ret_20d.iloc[ei]
            v20 = mkt_vol_20d.iloc[ei]
            ex_base.loc[idx, "mkt_above_ma20"] = mc > ma20 if not np.isnan(ma20) else True
            ex_base.loc[idx, "mkt_above_ma60"] = mc > ma60 if not np.isnan(ma60) else True
            ex_base.loc[idx, "mkt_ret_5d"] = r5
            ex_base.loc[idx, "mkt_ret_20d"] = r20
            ex_base.loc[idx, "mkt_vol_20d"] = v20
        except (IndexError, KeyError):
            pass

    # === TEST FILTERS ===
    print("\n" + "=" * 70)
    print("濾網測試結果")
    print("=" * 70)

    # Baseline
    base_m = simulate(ex_base, cal, n_cal)
    print(f"\n  [Baseline] 無濾網:")
    print(f"    CAGR={base_m['cagr']:.1%}, Sharpe={base_m['sharpe']:.2f}, MDD={base_m['mdd']:.1%}")
    print(f"    Years: {base_m['yearly_pass']}/{base_m['yearly_total']}, n={base_m['n']}")

    # Show baseline bear years
    print(f"\n    空頭年:")
    for yr in [2011, 2015, 2018, 2022]:
        yr_data = ex_base[ex_base["year"] == yr]
        if len(yr_data) < 3:
            continue
        print(f"      {yr}: n={len(yr_data)}, avg={yr_data['ret'].mean():.2%}")

    # Filter A: Market above MA20
    print(f"\n  [A] 大盤 > MA20 才進場:")
    filtered = ex_base[ex_base["mkt_above_ma20"]].copy()
    m = simulate(filtered, cal, n_cal)
    if m:
        print(f"    CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, MDD={m['mdd']:.1%}")
        print(f"    Years: {m['yearly_pass']}/{m['yearly_total']}, n={m['n']}")
        for yr in [2011, 2015, 2018, 2022]:
            yr_data = filtered[filtered["year"] == yr]
            if len(yr_data) < 3:
                print(f"      {yr}: n={len(yr_data)} (filtered out)")
                continue
            print(f"      {yr}: n={len(yr_data)}, avg={yr_data['ret'].mean():.2%}")

    # Filter B: Market above MA60
    print(f"\n  [B] 大盤 > MA60 才進場:")
    filtered = ex_base[ex_base["mkt_above_ma60"]].copy()
    m = simulate(filtered, cal, n_cal)
    if m:
        print(f"    CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, MDD={m['mdd']:.1%}")
        print(f"    Years: {m['yearly_pass']}/{m['yearly_total']}, n={m['n']}")
        for yr in [2011, 2015, 2018, 2022]:
            yr_data = filtered[filtered["year"] == yr]
            if len(yr_data) < 3:
                print(f"      {yr}: n={len(yr_data)} (filtered out)")
                continue
            print(f"      {yr}: n={len(yr_data)}, avg={yr_data['ret'].mean():.2%}")

    # Filter C: Market 5d return > threshold
    for thresh in [-0.03, -0.05, -0.07]:
        print(f"\n  [C] 大盤5日跌幅 > {thresh:.0%} 才進場:")
        filtered = ex_base[ex_base["mkt_ret_5d"] > thresh].copy()
        m = simulate(filtered, cal, n_cal)
        if m:
            print(f"    CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, MDD={m['mdd']:.1%}")
            print(f"    Years: {m['yearly_pass']}/{m['yearly_total']}, n={m['n']}")
            for yr in [2011, 2015, 2018, 2022]:
                yr_data = filtered[filtered["year"] == yr]
                if len(yr_data) < 3:
                    print(f"      {yr}: n={len(yr_data)} (filtered out)")
                    continue
                print(f"      {yr}: n={len(yr_data)}, avg={yr_data['ret'].mean():.2%}")

    # Filter D: Market volatility < threshold
    vol_thresholds = ex_base["mkt_vol_20d"].quantile([0.5, 0.6, 0.7, 0.8]).values
    for vt in vol_thresholds:
        if np.isnan(vt):
            continue
        print(f"\n  [D] 大盤波動率 < {vt:.4f} 才進場:")
        filtered = ex_base[ex_base["mkt_vol_20d"] < vt].copy()
        m = simulate(filtered, cal, n_cal)
        if m:
            print(f"    CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, MDD={m['mdd']:.1%}")
            print(f"    Years: {m['yearly_pass']}/{m['yearly_total']}, n={m['n']}")
            for yr in [2011, 2015, 2018, 2022]:
                yr_data = filtered[filtered["year"] == yr]
                if len(yr_data) < 3:
                    print(f"      {yr}: n={len(yr_data)} (filtered out)")
                    continue
                print(f"      {yr}: n={len(yr_data)}, avg={yr_data['ret'].mean():.2%}")

    # Filter E: Market 20d momentum > 0
    print(f"\n  [E] 大盤20日動量 > 0 才進場:")
    filtered = ex_base[ex_base["mkt_ret_20d"] > 0].copy()
    m = simulate(filtered, cal, n_cal)
    if m:
        print(f"    CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, MDD={m['mdd']:.1%}")
        print(f"    Years: {m['yearly_pass']}/{m['yearly_total']}, n={m['n']}")
        for yr in [2011, 2015, 2018, 2022]:
            yr_data = filtered[filtered["year"] == yr]
            if len(yr_data) < 3:
                print(f"      {yr}: n={len(yr_data)} (filtered out)")
                continue
            print(f"      {yr}: n={len(yr_data)}, avg={yr_data['ret'].mean():.2%}")

    # === BEST COMBINATION ===
    print("\n" + "=" * 70)
    print("最佳組合")
    print("=" * 70)

    # MA20 + 5d return
    print(f"\n  [A+C] MA20 + 5日跌幅>-5%:")
    filtered = ex_base[(ex_base["mkt_above_ma20"]) & (ex_base["mkt_ret_5d"] > -0.05)].copy()
    m = simulate(filtered, cal, n_cal)
    if m:
        print(f"    CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, MDD={m['mdd']:.1%}")
        print(f"    Years: {m['yearly_pass']}/{m['yearly_total']}, n={m['n']}")
        for yr in [2011, 2015, 2018, 2022]:
            yr_data = filtered[filtered["year"] == yr]
            if len(yr_data) < 3:
                print(f"      {yr}: n={len(yr_data)} (filtered out)")
                continue
            print(f"      {yr}: n={len(yr_data)}, avg={yr_data['ret'].mean():.2%}")

    # MA60 + vol
    vt = vol_thresholds[1] if len(vol_thresholds) > 1 else 0.02
    print(f"\n  [B+D] MA60 + 波動率<{vt:.4f}:")
    filtered = ex_base[(ex_base["mkt_above_ma60"]) & (ex_base["mkt_vol_20d"] < vt)].copy()
    m = simulate(filtered, cal, n_cal)
    if m:
        print(f"    CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.2f}, MDD={m['mdd']:.1%}")
        print(f"    Years: {m['yearly_pass']}/{m['yearly_total']}, n={m['n']}")
        for yr in [2011, 2015, 2018, 2022]:
            yr_data = filtered[filtered["year"] == yr]
            if len(yr_data) < 3:
                print(f"      {yr}: n={len(yr_data)} (filtered out)")
                continue
            print(f"      {yr}: n={len(yr_data)}, avg={yr_data['ret'].mean():.2%}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
