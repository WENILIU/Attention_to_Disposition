"""S1: 大盤空頭訊號探索 — 期貨/反向ETF做空策略.

測試訊號:
  A. 融資餘額急降 (散戶恐慌平倉 = 趨勢反轉)
  B. 投信連續賣超 (法人看空)
  C. 大盤波動率急升 (VIX 概念)
  D. 注意事件數量異常暴增 (市場過熱前兆)
  E. 大盤動量翻負 (MA20 跌破)
  F. 融資/大盤成交額比率異常高 (槓桿過高)

策略: 訊號觸發 → 做空大盤 N 天 (用等權指數模擬期貨)
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\s1_market_short")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001  # 期貨交易成本 (低)


def main():
    print("=" * 70)
    print("S1: 大盤空頭訊號探索")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Equal-weighted market index (proxy for 台指期貨)
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_close = (1 + mkt_ret).cumprod()
    mkt_vol_20d = mkt_ret.rolling(20).std()
    mkt_ma20 = mkt_close.rolling(20).mean()
    mkt_ma60 = mkt_close.rolling(60).mean()
    mkt_ret_5d = mkt_close.pct_change(5)
    mkt_ret_20d = mkt_close.pct_change(20)

    # Market breadth: % stocks above MA20
    above_ma20 = (close > close.rolling(20).mean()).sum(axis=1) / close.notna().sum(axis=1)

    # Total market turnover
    total_turnover = (close * vol).sum(axis=1)
    turnover_ma20 = total_turnover.rolling(20).mean()

    print(f"  Market data: {cal[0].date()} to {cal[-1].date()} ({n_cal} days)")

    # Try to load margin data
    margin_available = False
    try:
        margin_bal = data.get("margin_balance_total")
        margin_available = True
        print(f"  Margin data: available")
    except Exception:
        try:
            # Try alternative
            margin_bal = data.get("margin_loan_balance")
            margin_available = True
            print(f"  Margin data: available (alt)")
        except Exception:
            print(f"  Margin data: NOT available")

    # Try to load investor flow data
    investor_available = False
    try:
        investor_daily = data.get("investor_daily")
        investor_available = True
        print(f"  Investor data: available")
    except Exception:
        try:
            investor_daily = data.get("foreign_institutional")
            investor_available = True
            print(f"  Investor data: available (alt)")
        except Exception:
            print(f"  Investor data: NOT available")

    # Attention event count per day
    print("  Computing attention event counts...")
    att_raw = data.get("trading_attention")
    att_raw["date"] = pd.to_datetime(att_raw["date"]).dt.normalize()
    daily_att_count = att_raw.groupby("date").size()
    att_count_series = daily_att_count.reindex(cal, fill_value=0)
    att_count_ma20 = att_count_series.rolling(20).mean()
    att_count_std = att_count_series.rolling(20).std()

    # === BUILD SIGNALS ===
    print("\nBuilding signals...")

    signals = pd.DataFrame(index=cal)

    # Signal A: 融資餘額急降
    if margin_available:
        try:
            mb = margin_bal.reindex(cal) if hasattr(margin_bal, 'reindex') else None
            if mb is not None:
                mb_change_5d = mb.pct_change(5)
                signals["A_margin_drop"] = mb_change_5d < -0.03  # 融資5日跌>3%
            else:
                signals["A_margin_drop"] = False
        except Exception:
            signals["A_margin_drop"] = False
    else:
        signals["A_margin_drop"] = False

    # Signal B: 投信連續賣超
    if investor_available:
        try:
            signals["B_investor_sell"] = False  # placeholder
        except Exception:
            signals["B_investor_sell"] = False
    else:
        signals["B_investor_sell"] = False

    # Signal C: 波動率急升
    vol_zscore = (mkt_vol_20d - mkt_vol_20d.rolling(60).mean()) / mkt_vol_20d.rolling(60).std()
    signals["C_vol_spike"] = vol_zscore > 1.5  # 波動率 z-score > 1.5

    # Signal D: 注意事件異常暴增
    att_zscore = (att_count_series - att_count_ma20) / att_count_std
    signals["D_attention_spike"] = att_zscore > 2.0  # 注意事件 z-score > 2

    # Signal E: 大盤跌破 MA20
    signals["E_below_ma20"] = mkt_close < mkt_ma20

    # Signal F: 大盤跌破 MA60
    signals["F_below_ma60"] = mkt_close < mkt_ma60

    # Signal G: 市場廣度極低 (< 30% stocks above MA20)
    signals["G_breadth_low"] = above_ma20 < 0.30

    # Signal H: 大盤5日跌幅 > 5%
    signals["H_mkt_drop_5d"] = mkt_ret_5d < -0.05

    # Signal I: 投信賣超 (use market breadth decline as proxy)
    breadth_decline = above_ma20 - above_ma20.shift(5)
    signals["I_breadth_decline"] = breadth_decline < -0.15  # 5天內廣度下降15%

    # === TEST EACH SIGNAL ===
    print("\n" + "=" * 70)
    print("訊號預測力測試 (訊號觸發後 N 天的大盤報酬)")
    print("=" * 70)

    forward_returns = {}
    for horizon in [1, 3, 5, 10, 20]:
        forward_returns[horizon] = mkt_close.shift(-horizon) / mkt_close - 1

    signal_names = [c for c in signals.columns if c != "B_investor_sell"]

    print(f"\n  {'Signal':<20} {'n_fire':>7} {'fwd_1d':>8} {'fwd_5d':>8} {'fwd_10d':>8} {'fwd_20d':>8} {'hit%':>6}")
    print(f"  {'-'*20} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for sig_name in signal_names:
        fired = signals[sig_name].values
        n_fire = fired.sum()
        if n_fire < 20:
            print(f"  {sig_name:<20} {n_fire:>7}  (insufficient)")
            continue

        fwd_1 = forward_returns[1].values[fired]
        fwd_5 = forward_returns[5].values[fired]
        fwd_10 = forward_returns[10].values[fired]
        fwd_20 = forward_returns[20].values[fired]

        # Hit rate: % of times market was negative 5d later
        hit = (fwd_5 < 0).mean()

        print(f"  {sig_name:<20} {n_fire:>7} {np.nanmean(fwd_1):>8.3%} {np.nanmean(fwd_5):>8.3%} "
              f"{np.nanmean(fwd_10):>8.3%} {np.nanmean(fwd_20):>8.3%} {hit:>6.1%}")

    # === STRATEGY SIMULATION ===
    print("\n" + "=" * 70)
    print("做空策略模擬 (訊號觸發 → 做空 N 天)")
    print("=" * 70)

    best_signals = ["C_vol_spike", "D_attention_spike", "E_below_ma20", "F_below_ma60",
                    "G_breadth_low", "H_mkt_drop_5d", "I_breadth_decline"]

    for sig_name in best_signals:
        for hold_days in [3, 5, 10]:
            # Simulate: when signal fires, go short for hold_days
            short_position = signals[sig_name].rolling(hold_days, min_periods=1).max().values.astype(float)
            # Short return = -market return (minus cost)
            strategy_ret = -short_position * mkt_ret.values - short_position * COST_RATE

            # Metrics
            total_ret = np.prod(1 + strategy_ret) - 1
            ann_ret = total_ret * 252 / n_cal
            sharpe = np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252) if np.std(strategy_ret) > 0 else 0

            # Yearly
            yearly_rets = {}
            for yr in range(2010, 2027):
                yr_mask = cal.year == yr
                if yr_mask.sum() > 50:
                    yearly_rets[yr] = np.prod(1 + strategy_ret[yr_mask]) - 1

            n_years = len(yearly_rets)
            n_positive = sum(1 for v in yearly_rets.values() if v > 0)

            print(f"\n  {sig_name} (hold={hold_days}d):")
            print(f"    Ann return: {ann_ret:.1%}, Sharpe: {sharpe:.2f}")
            print(f"    Years positive: {n_positive}/{n_years}")
            # Show bear years
            for yr in [2011, 2015, 2018, 2022]:
                if yr in yearly_rets:
                    print(f"      {yr}: {yearly_rets[yr]:+.1%}")

    # === COMBINED SIGNAL ===
    print("\n" + "=" * 70)
    print("組合訊號 (多個條件同時滿足)")
    print("=" * 70)

    # Combined: below MA20 + vol spike OR breadth low
    combined = (signals["E_below_ma20"] & (signals["C_vol_spike"] | signals["G_breadth_low"]))
    for hold_days in [5, 10]:
        short_pos = combined.rolling(hold_days, min_periods=1).max().values.astype(float)
        strategy_ret = -short_pos * mkt_ret.values - short_pos * COST_RATE
        ann_ret = (np.prod(1 + strategy_ret) - 1) * 252 / n_cal
        sharpe = np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252) if np.std(strategy_ret) > 0 else 0

        yearly_rets = {}
        for yr in range(2010, 2027):
            yr_mask = cal.year == yr
            if yr_mask.sum() > 50:
                yearly_rets[yr] = np.prod(1 + strategy_ret[yr_mask]) - 1
        n_positive = sum(1 for v in yearly_rets.values() if v > 0)

        print(f"\n  Combined (MA20 below + vol/breadth, hold={hold_days}d):")
        print(f"    Ann return: {ann_ret:.1%}, Sharpe: {sharpe:.2f}")
        print(f"    Years positive: {n_positive}/{len(yearly_rets)}")
        for yr in [2011, 2015, 2018, 2022]:
            if yr in yearly_rets:
                print(f"      {yr}: {yearly_rets[yr]:+.1%}")

    # === V20 + SHORT COMBINED ===
    print("\n" + "=" * 70)
    print("V20 + 做空組合效果")
    print("=" * 70)

    # Load V20 daily returns
    v20_trades = pd.read_csv(
        Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p23_v20_longterm\p23_v20_longterm_trades.csv"),
        parse_dates=["entry_date", "exit_date"]
    )

    v20_daily = pd.Series(0.0, index=cal)
    cap = 0.2  # 20% per position
    for _, t in v20_trades.iterrows():
        ei = cal.searchsorted(t["entry_date"], side="left")
        xi = cal.searchsorted(t["exit_date"], side="left")
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold * cap
        for d in range(ei, min(xi + 1, n_cal)):
            v20_daily.iloc[d] += dr

    # Best short signal
    short_sig = signals["E_below_ma20"] & (signals["C_vol_spike"] | signals["G_breadth_low"])
    short_pos = short_sig.rolling(5, min_periods=1).max().values.astype(float)
    short_daily = pd.Series(-short_pos * mkt_ret.values - short_pos * COST_RATE, index=cal)

    # Combined: V20 70% + Short 30%
    combined_ret = v20_daily * 0.7 + short_daily * 0.3

    # Metrics
    for label, ret_series in [("V20 only", v20_daily), ("Short only", short_daily), ("V20 70% + Short 30%", combined_ret)]:
        ann = (np.prod(1 + ret_series.values) - 1) * 252 / n_cal
        sharpe = np.mean(ret_series) / np.std(ret_series) * np.sqrt(252) if np.std(ret_series) > 0 else 0
        cum = (1 + ret_series).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()

        yearly = {}
        for yr in range(2010, 2027):
            yr_mask = cal.year == yr
            if yr_mask.sum() > 50:
                yearly[yr] = np.prod(1 + ret_series.values[yr_mask]) - 1
        n_pos = sum(1 for v in yearly.values() if v > 0)

        print(f"\n  {label}:")
        print(f"    Ann return: {ann:.1%}, Sharpe: {sharpe:.2f}, MDD: {mdd:.1%}")
        print(f"    Years positive: {n_pos}/{len(yearly)}")
        for yr in [2011, 2015, 2018, 2022]:
            if yr in yearly:
                print(f"      {yr}: {yearly[yr]:+.1%}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
