"""P23: V20 (處置期間做多) 長週期回測 (2007-2026).

V20 是純規則策略，不需要 ML 模型，無冷啟動問題。
直接測試所有歷史處置事件。

邏輯:
  舊制 (處置10天): Day 4 進場, Day 9 出場
  新制 (處置5天): Day 1 進場, Day 4 出場
  濾網: 流動性>20M, Gap -8%~+4%
  熔斷: >400家跌>6% → 清倉9天
  部位: 5檔, 每檔20%
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p23_v20_longterm")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
MAX_POSITIONS = 5
MIN_TURNOVER = 20_000_000
GAP_LOW = -0.08
GAP_HIGH = 0.04
PANIC_THRESHOLD = 400
PANIC_COOLDOWN = 9
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")


def main():
    print("=" * 70)
    print("P23: V20 長週期回測 (2007-2026)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    dis_raw = data.get("disposal_information")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)
    print(f"  Price data: {cal[0].date()} to {cal[-1].date()} ({n_cal} days)")

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

    # Calculate indices
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1
    dis["is_new_regime"] = dis["announce"] >= NEW_REGIME_DATE

    # Filter valid events
    valid = dis[
        (dis["start_idx"] >= 0) &
        (dis["end_idx"] >= 0) &
        (dis["end_idx"] < n_cal) &
        (dis["duration"] >= 3)
    ].copy()
    valid["year"] = valid["start"].dt.year

    print(f"  Valid disposal events: {len(valid):,}")
    print(f"  Year range: {valid['year'].min()} to {valid['year'].max()}")
    print(f"  Duration distribution:")
    for d in sorted(valid["duration"].unique()):
        cnt = (valid["duration"] == d).sum()
        if cnt > 10:
            print(f"    {d} days: {cnt:,}")

    # Pre-compute indicators
    print("\nComputing indicators...")
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Circuit breaker
    daily_ret = close.pct_change()
    panic_count = (daily_ret < -0.06).sum(axis=1)
    is_panic = panic_count > PANIC_THRESHOLD
    danger_zone = is_panic.rolling(PANIC_COOLDOWN, min_periods=1).max() > 0
    print(f"  Circuit breaker triggered: {int(is_panic.sum())} times")

    # Generate trades
    print("\nGenerating trades...")
    trades = []
    skip_gap = 0
    skip_liq = 0
    skip_panic = 0

    for _, row in valid.iterrows():
        sym = row["stock_id"]
        si = int(row["start_idx"])
        ei = int(row["end_idx"])
        dur = int(row["duration"])

        # Determine entry/exit based on regime
        if row["is_new_regime"]:
            entry_idx = si  # Day 1
            exit_idx = ei - 1  # Day before end
        else:
            entry_idx = si + 3  # Day 4
            exit_idx = ei - 1  # Day before end

        if entry_idx < 0 or exit_idx >= n_cal or entry_idx >= exit_idx:
            continue

        # Check circuit breaker
        if danger_zone.iloc[entry_idx]:
            skip_panic += 1
            continue

        # Get prices
        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[entry_idx - 1][sym]
            avg_to = avg_turnover_5d.iloc[entry_idx - 1][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            continue

        # Liquidity filter
        if not (np.isnan(avg_to) or avg_to >= MIN_TURNOVER):
            skip_liq += 1
            continue

        # Gap filter
        gap = entry_price / prev_close - 1
        if not (GAP_LOW < gap < GAP_HIGH):
            skip_gap += 1
            continue

        # Exit price
        try:
            exit_price = close.iloc[exit_idx][sym]
            if np.isnan(exit_price) or exit_price <= 0:
                continue
        except (IndexError, KeyError):
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        trades.append({
            "stock_id": sym,
            "entry_date": cal[entry_idx],
            "exit_date": cal[exit_idx],
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "year": cal[entry_idx].year,
            "ret": ret,
            "hold_days": exit_idx - entry_idx + 1,
            "duration": dur,
            "is_new_regime": row["is_new_regime"],
        })

    all_trades = pd.DataFrame(trades)
    print(f"\n  Total trades: {len(all_trades):,}")
    print(f"  Skipped (gap): {skip_gap:,}")
    print(f"  Skipped (liquidity): {skip_liq:,}")
    print(f"  Skipped (circuit breaker): {skip_panic:,}")

    # Portfolio simulation
    print("\n" + "=" * 70)
    print("組合模擬 (5檔)")
    print("=" * 70)

    all_trades = all_trades.sort_values(["entry_date", "ret"], ascending=[True, False])

    active = []
    executed = []
    for _, trade in all_trades.iterrows():
        ei = int(trade["entry_idx"])
        xi = int(trade["exit_idx"])
        active = [(e, s) for e, s in active if e > ei]
        if len(active) >= MAX_POSITIONS:
            continue
        if any(s == trade["stock_id"] for _, s in active):
            continue
        active.append((xi, trade["stock_id"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)
    print(f"\n  Executed: {len(ex):,} trades")
    print(f"  Date range: {ex['entry_date'].min().date()} to {ex['entry_date'].max().date()}")

    # Yearly breakdown
    print(f"\n  逐年表現:")
    print(f"  {'Year':>6} {'n':>5} {'avg_ret':>8} {'win%':>6} {'mkt_ret':>8} {'判定':>6}")

    # Market benchmark
    mkt_daily = close.pct_change().mean(axis=1)
    mkt_yearly = {}
    for yr in range(2007, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() > 0:
            mkt_yearly[yr] = (1 + mkt_daily[yr_mask]).prod() - 1
        else:
            mkt_yearly[yr] = 0

    yearly_pass = 0
    yearly_total = 0
    yearly_results = []
    for yr in sorted(ex["year"].unique()):
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 5:
            continue
        avg = yr_data["ret"].mean()
        win = (yr_data["ret"] > 0).mean()
        mkt = mkt_yearly.get(yr, 0)
        sign = "PASS" if avg > 0 else "FAIL"
        yearly_pass += (1 if avg > 0 else 0)
        yearly_total += 1
        yearly_results.append((yr, avg, win, len(yr_data), mkt))
        print(f"  {yr:>6} {len(yr_data):>5} {avg:>8.2%} {win:>6.1%} {mkt:>8.1%} {sign:>6}")

    print(f"\n  逐年通過: {yearly_pass}/{yearly_total}")

    # Bear market analysis
    print("\n" + "=" * 70)
    print("空頭年份聚焦分析")
    print("=" * 70)

    bear_years = [2008, 2011, 2015, 2018, 2022]
    for yr in bear_years:
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 3:
            print(f"\n  {yr}: 樣本不足 (n={len(yr_data)})")
            continue
        avg = yr_data["ret"].mean()
        win = (yr_data["ret"] > 0).mean()
        mkt = mkt_yearly.get(yr, 0)
        print(f"\n  {yr} (大盤{mkt:+.1%}):")
        print(f"    n={len(yr_data)}, avg={avg:.2%}, win={win:.1%}")
        yr_data = yr_data.copy()
        yr_data["month"] = pd.to_datetime(yr_data["entry_date"]).dt.month
        for m in sorted(yr_data["month"].unique()):
            m_data = yr_data[yr_data["month"] == m]
            if len(m_data) < 3:
                continue
            print(f"      {m}月: n={len(m_data)}, avg={m_data['ret'].mean():.2%}")

    # Portfolio metrics
    print("\n" + "=" * 70)
    print("組合指標")
    print("=" * 70)

    daily_returns = pd.Series(0.0, index=cal)
    cap = 1.0 / MAX_POSITIONS
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
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
    first_idx = int(ex["entry_idx"].min())
    total_days = n_cal - first_idx
    cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

    print(f"\n  CAGR: {cagr:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  MDD: {mdd:.1%}")
    print(f"  Avg return: {ex['ret'].mean():.2%}")
    print(f"  Win rate: {(ex['ret'] > 0).mean():.1%}")
    print(f"  Total trades: {len(ex):,}")
    print(f"  Avg hold: {ex['hold_days'].mean():.1f} days")
    print(f"  Period: {ex['entry_date'].min().date()} to {ex['entry_date'].max().date()}")

    # Save
    ex.to_csv(OUT / "p23_v20_longterm_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_results, columns=["year", "avg_ret", "win", "n", "mkt_ret"]).to_csv(
        OUT / "p23_yearly_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
