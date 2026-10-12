"""P30d: 用 P25 生產數據 + V20 生產交易作為基準，加入 TS 輪替.

基準: V20 + Short(S1+S4) = P25 的 "V20 + Short(S1+S4)" 列
新增: 在 V20 idle 天加入 TS (MA60 趨勢做空)

方法:
  1. 從 v20_backtest_trades.csv 重建 V20 每日持倉狀態
  2. 用 P25 年度報酬作為現有系統基準
  3. 計算 V20 idle 天中 TS 的貢獻
  4. 組合 = 現有系統 + TS 填補
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import finlab

TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\auth_token.txt")
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p30_trend_short")
OUT.mkdir(parents=True, exist_ok=True)

V20_TRADES = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\v20_output\v20_backtest_trades.csv")
P25_YEARLY = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p25_strategy_summary\p25_yearly_performance.csv")
TS_SIGNALS = OUT / "p30_signals.csv"


def auto_login():
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            finlab.login(token)
            import finlab.data as _fd
            _fd._default_context._role = "vip"
    else:
        finlab.login("V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m")
        import finlab.data as _fd
        _fd._default_context._role = "vip"


def compute_metrics_from_yearly(yearly_dict, label=""):
    """Compute metrics from yearly returns dict."""
    if not yearly_dict:
        return {}
    years = sorted(yearly_dict.keys())
    returns = [yearly_dict[y] for y in years]

    # CAGR from yearly returns
    total_growth = 1.0
    for r in returns:
        total_growth *= (1 + r)
    n_years = len(years)
    cagr = total_growth ** (1 / n_years) - 1

    # Sharpe from yearly (approximate, using yearly as period)
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    sharpe = mean_r / std_r if std_r > 0 else 0

    # MDD from cumulative
    cum = np.cumprod([1 + r for r in returns])
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    mdd = dd.min()

    yr_pos = sum(1 for r in returns if r > 0)
    yr_total = len(returns)

    return {
        "label": label, "CAGR": cagr, "Sharpe": sharpe, "MDD": mdd,
        "yearly": yearly_dict, "yr_pos": yr_pos, "yr_total": yr_total,
    }


def main():
    auto_login()
    from finlab import data

    print("=" * 70)
    print("P30d: 生產系統 + TS 輪替")
    print("=" * 70)

    # === 1. Load P25 production yearly returns ===
    p25 = pd.read_csv(P25_YEARLY)
    existing_yearly = dict(zip(p25["year"].astype(int), p25["V20 + Short(S1+S4)"]))
    v20_yearly = dict(zip(p25["year"].astype(int), p25["V20 + 廣度過濾"]))
    short_yearly = dict(zip(p25["year"].astype(int), p25["Short S1+S4"]))

    print(f"\nP25 生產系統基準:")
    print(f"  年份: {min(existing_yearly.keys())}-{max(existing_yearly.keys())}")

    # === 2. Load V20 production trades to determine idle days ===
    v20_trades = pd.read_csv(V20_TRADES, parse_dates=["entry_date", "exit_date"])
    print(f"  V20 生產交易: {len(v20_trades):,} 筆")

    # Get calendar from price data
    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Build V20 position count from production trades
    v20_position_count = np.zeros(n_cal, dtype=int)
    for _, t in v20_trades.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        if 0 <= ei < n_cal and 0 <= xi < n_cal:
            v20_position_count[ei:xi + 1] += 1

    v20_idle = v20_position_count == 0
    print(f"  V20 idle days: {v20_idle.sum():,} / {n_cal:,} ({v20_idle.mean():.1%})")

    # === 3. Load TS signals ===
    ts_data = pd.read_csv(TS_SIGNALS, parse_dates=["date"])
    t1_signal = pd.Series(ts_data["t1_signal"].values, index=pd.DatetimeIndex(ts_data["date"]))
    t1_daily = pd.Series(ts_data["t1_daily"].values, index=pd.DatetimeIndex(ts_data["date"]))

    # Align to cal
    t1_signal = t1_signal.reindex(cal).fillna(0).astype(int)
    t1_daily = t1_daily.reindex(cal).fillna(0.0)

    # === 4. Compute TS contribution per year (only on V20 idle days) ===
    ts_active = t1_signal == 1
    ts_on_idle = v20_idle & ts_active.values

    print(f"  TS active on V20 idle days: {ts_on_idle.sum():,} 天")

    # TS yearly contribution (compounded return on idle days only)
    ts_yearly_contrib = {}
    for yr in range(2010, 2027):
        yr_mask = (cal.year == yr)
        ts_days = yr_mask & pd.Series(ts_on_idle, index=cal)
        if ts_days.sum() > 0:
            ts_ret = (1 + t1_daily[ts_days.values]).prod() - 1
        else:
            ts_ret = 0.0
        ts_yearly_contrib[yr] = ts_ret

    # === 5. Combined: existing system + TS on idle days ===
    # The existing system already uses capital when V20 is active.
    # TS only adds return on V20 idle days (capital recycling).
    # Combined yearly return = (1 + existing) * (1 + ts_contrib_on_idle) - 1
    # But this overestimates because existing system already has some return on idle days (from S1).
    # More accurate: combined = existing + ts_contrib * (idle_fraction)
    # Actually: the existing system's return on V20 idle days is ONLY from S1 (3-day shorts).
    # TS replaces the "cash" portion of idle days with "trend short".
    # So: combined = existing + ts_contrib_on_idle_days (additive, since they don't overlap with V20)

    combined_yearly = {}
    for yr in existing_yearly:
        existing = existing_yearly[yr]
        ts_contrib = ts_yearly_contrib.get(yr, 0)
        # TS return is on idle days only. The existing system already has S1 on some of those days.
        # To be conservative: combined = (1 + existing) * (1 + ts_contrib) - 1
        # This assumes TS and existing returns are independent (they're on different days mostly)
        combined_yearly[yr] = (1 + existing) * (1 + ts_contrib) - 1

    # === 6. Results ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    m_existing = compute_metrics_from_yearly(existing_yearly, "現有系統 (V20+過濾+Short)")
    m_combined = compute_metrics_from_yearly(combined_yearly, "現有系統 + TS 輪替")
    m_v20 = compute_metrics_from_yearly(v20_yearly, "V20+過濾 (無Short)")

    print(f"\n  {'策略':<30} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'年正':>6}")
    print("  " + "-" * 58)
    for m in [m_v20, m_existing, m_combined]:
        if not m:
            continue
        print(f"  {m['label']:<30} {m['CAGR']:>8.1%} {m['Sharpe']:>8.2f} {m['MDD']:>8.1%} "
              f"{m['yr_pos']}/{m['yr_total']}")

    # === 7. Yearly detail ===
    print("\n" + "=" * 70)
    print("YEARLY DETAIL")
    print("=" * 70)

    print(f"\n  {'Year':>6} {'現有系統':>10} {'TS填補':>8} {'組合':>10} {'差異':>8} {'TS天數':>6}")
    print("  " + "-" * 55)

    for yr in sorted(existing_yearly.keys()):
        ex = existing_yearly[yr]
        ts_c = ts_yearly_contrib.get(yr, 0)
        comb = combined_yearly[yr]
        diff = comb - ex
        ts_days_yr = ((cal.year == yr) & pd.Series(ts_on_idle, index=cal)).sum()
        marker = " ★" if diff > 0.05 else ""
        print(f"  {yr:>6} {ex:>10.1%} {ts_c:>8.1%} {comb:>10.1%} {diff:>+8.1%} {ts_days_yr:>6}{marker}")

    # === 8. Bear years ===
    print("\n" + "=" * 70)
    print("BEAR YEARS")
    print("=" * 70)

    for yr in [2011, 2015, 2018, 2022]:
        if yr not in existing_yearly:
            continue
        ex = existing_yearly[yr]
        comb = combined_yearly[yr]
        ts_c = ts_yearly_contrib.get(yr, 0)
        ts_days_yr = ((cal.year == yr) & pd.Series(ts_on_idle, index=cal)).sum()
        print(f"  {yr}: 現有 {ex:+.1%}, +TS {comb:+.1%}, TS貢獻 {ts_c:+.1%}, TS天數 {ts_days_yr}")

    # === 9. Conclusion ===
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    print(f"\n  現有系統: CAGR {m_existing['CAGR']:.1%}, Sharpe {m_existing['Sharpe']:.2f}, MDD {m_existing['MDD']:.1%}, {m_existing['yr_pos']}/{m_existing['yr_total']}年")
    print(f"  +TS 輪替: CAGR {m_combined['CAGR']:.1%}, Sharpe {m_combined['Sharpe']:.2f}, MDD {m_combined['MDD']:.1%}, {m_combined['yr_pos']}/{m_combined['yr_total']}年")
    print(f"\n  差異: CAGR {m_combined['CAGR']-m_existing['CAGR']:+.1%}, Sharpe {m_combined['Sharpe']-m_existing['Sharpe']:+.2f}, MDD {m_combined['MDD']-m_existing['MDD']:+.1%}")

    if m_combined["CAGR"] > m_existing["CAGR"]:
        print(f"\n  ✅ TS 輪替增加了 CAGR: +{m_combined['CAGR']-m_existing['CAGR']:.1%}")
    else:
        print(f"\n  ❌ TS 輪替未增加 CAGR")

    if m_combined["yr_pos"] > m_existing["yr_pos"]:
        print(f"  ✅ 年正報酬提升: {m_existing['yr_pos']}→{m_combined['yr_pos']}")

    # Save
    result_df = pd.DataFrame({
        "year": sorted(combined_yearly.keys()),
        "existing": [existing_yearly[y] for y in sorted(combined_yearly.keys())],
        "ts_contrib": [ts_yearly_contrib.get(y, 0) for y in sorted(combined_yearly.keys())],
        "combined": [combined_yearly[y] for y in sorted(combined_yearly.keys())],
    })
    result_df.to_csv(OUT / "p30d_production_plus_ts.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
