"""S2: 增強做空訊號 — 廣度 + 投信賣超 + 融資餘額.

目標: 提高做空訊號的頻率和準確度
目前: 廣度急降 (一年觸發2-3次, 15/17年正)
目標: 加入投信/融資 → 一年觸發5-8次, 16+/17年正
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\s2_enhanced_short")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001


def main():
    print("=" * 70)
    print("S2: 增強做空訊號")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Market returns
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_close = (1 + mkt_ret).cumprod()

    # === Signal 1: Breadth decline (existing) ===
    print("Computing breadth...")
    ma20 = close.rolling(20).mean()
    above_ma20 = (close > ma20).sum(axis=1) / close.notna().sum(axis=1)
    breadth_decline = above_ma20 - above_ma20.shift(5)
    sig_breadth = breadth_decline < -0.15

    # === Signal 2: Institutional net selling ===
    print("Computing institutional flows...")
    try:
        inst_net = data.get("institutional_investors_trading_all_market_summary:買賣超")
        # This might be a Series or DataFrame
        if isinstance(inst_net, pd.DataFrame):
            inst_net_daily = inst_net.sum(axis=1)
        else:
            inst_net_daily = inst_net
        inst_net_daily = inst_net_daily.reindex(cal).fillna(0)

        # Signal: 5-day cumulative institutional selling > threshold
        inst_5d = inst_net_daily.rolling(5).sum()
        inst_std = inst_net_daily.rolling(60).std()
        inst_zscore = inst_5d / inst_std
        sig_institutional = inst_zscore < -2.0  # 5日累計賣超 z < -2
        print(f"  Institutional data: OK, range {inst_net_daily.index[0].date()} to {inst_net_daily.index[-1].date()}")
    except Exception as e:
        print(f"  Institutional data FAILED: {e}")
        sig_institutional = pd.Series(False, index=cal)

    # === Signal 3: Margin balance decline ===
    print("Computing margin balance...")
    try:
        margin_df = data.get("margin_transactions:融資今日餘額")
        # Sum across all stocks to get total margin balance
        margin_total = margin_df.sum(axis=1)
        margin_total = margin_total.reindex(cal).fillna(0)

        # Signal: 5-day margin decline > 2%
        margin_change_5d = margin_total.pct_change(5)
        sig_margin = margin_change_5d < -0.02  # 融資5日跌>2%
        print(f"  Margin data: OK, range {margin_total.index[0].date()} to {margin_total.index[-1].date()}")
    except Exception as e:
        print(f"  Margin data FAILED: {e}")
        sig_margin = pd.Series(False, index=cal)

    # === Signal 4: Combined (any 2 of 3) ===
    signal_count = sig_breadth.astype(int) + sig_institutional.astype(int) + sig_margin.astype(int)
    sig_any2 = signal_count >= 2
    sig_any1 = signal_count >= 1

    # === TEST EACH SIGNAL ===
    print("\n" + "=" * 70)
    print("訊號預測力測試")
    print("=" * 70)

    forward_5d = mkt_close.shift(-5) / mkt_close - 1
    forward_3d = mkt_close.shift(-3) / mkt_close - 1

    signals = {
        "breadth_decline": sig_breadth,
        "institutional_sell": sig_institutional,
        "margin_drop": sig_margin,
        "any_1_of_3": sig_any1,
        "any_2_of_3": sig_any2,
    }

    print(f"\n  {'Signal':<20} {'n_fire':>7} {'fwd_3d':>8} {'fwd_5d':>8} {'hit%':>6} {'freq/yr':>8}")
    print(f"  {'-'*20} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    n_years = n_cal / 252
    for name, sig in signals.items():
        fired = sig.values
        n_fire = fired.sum()
        if n_fire < 10:
            print(f"  {name:<20} {n_fire:>7}  (insufficient)")
            continue
        fwd_3 = forward_3d.values[fired]
        fwd_5 = forward_5d.values[fired]
        hit = (fwd_3 < 0).mean()
        freq = n_fire / n_years
        print(f"  {name:<20} {n_fire:>7} {np.nanmean(fwd_3):>8.3%} {np.nanmean(fwd_5):>8.3%} {hit:>6.1%} {freq:>8.1f}")

    # === STRATEGY SIMULATION ===
    print("\n" + "=" * 70)
    print("做空策略模擬")
    print("=" * 70)

    for name, sig in signals.items():
        for hold in [3, 5]:
            short_pos = sig.rolling(hold, min_periods=1).max().values.astype(float)
            strategy_ret = -short_pos * mkt_ret.values - short_pos * COST_RATE

            sharpe = np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252) if np.std(strategy_ret) > 0 else 0

            yearly = {}
            for yr in range(2010, 2027):
                yr_mask = cal.year == yr
                if yr_mask.sum() > 50:
                    yearly[yr] = np.prod(1 + strategy_ret[yr_mask]) - 1
            n_pos = sum(1 for v in yearly.values() if v > 0)
            n_total = len(yearly)

            # Count signal fires per year
            fires_per_year = sig.groupby(cal.year).sum()
            avg_fires = fires_per_year.mean()

            print(f"\n  {name} (hold={hold}d): Sharpe={sharpe:.2f}, Years={n_pos}/{n_total}, fires/yr={avg_fires:.1f}")
            for yr in [2011, 2015, 2018, 2022]:
                if yr in yearly:
                    print(f"      {yr}: {yearly[yr]:+.1%}")

    # === V20 + Enhanced Short ===
    print("\n" + "=" * 70)
    print("V20 + 增強做空 組合")
    print("=" * 70)

    # V20 daily returns
    v20_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p23_v20_longterm\p23_v20_longterm_trades.csv")
    v20_trades = pd.read_csv(v20_path, parse_dates=["entry_date", "exit_date"])
    v20_daily = pd.Series(0.0, index=cal)
    for _, t in v20_trades.iterrows():
        ei = cal.searchsorted(t["entry_date"], side="left")
        xi = cal.searchsorted(t["exit_date"], side="left")
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold * 0.14  # 70% / 5 positions
        for d in range(ei, min(xi + 1, n_cal)):
            v20_daily.iloc[d] += dr

    # Best enhanced short
    best_sig = sig_any2  # or whichever performs best
    short_pos = best_sig.rolling(3, min_periods=1).max().values.astype(float)
    short_daily = pd.Series(-short_pos * mkt_ret.values - short_pos * COST_RATE, index=cal)

    combined = v20_daily + short_daily * 0.20

    for label, ret_series in [("V20 only", v20_daily), ("Enhanced Short", short_daily), ("V20+Short combined", combined)]:
        sharpe = np.mean(ret_series) / np.std(ret_series) * np.sqrt(252) if np.std(ret_series) > 0 else 0
        cum = (1 + ret_series).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()

        yearly = {}
        for yr in range(2010, 2027):
            yr_mask = cal.year == yr
            if yr_mask.sum() > 50:
                yearly[yr] = np.prod(1 + ret_series.values[yr_mask]) - 1
        n_pos = sum(1 for v in yearly.values() if v > 0)

        print(f"\n  {label}: Sharpe={sharpe:.2f}, MDD={mdd:.1%}, Years={n_pos}/{len(yearly)}")
        for yr in [2011, 2015, 2018, 2022]:
            if yr in yearly:
                print(f"      {yr}: {yearly[yr]:+.1%}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
