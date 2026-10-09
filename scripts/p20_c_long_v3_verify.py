"""P20: C-Long v3 完整驗證 — 逐年 + 新舊制.

v3 配置:
  訊號：注意公告 + 昨日也被注意(連續) + GB model prob >= 0.6
  進場：注意公告次日開盤
  出場：固定15天
  濾網：流動性>20M, Gap -8%~+4%
  止損：無
  部位：5檔，每檔20%
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p20_c_long_v3")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")
MAX_HOLD_DAYS = 15
THRESHOLD = 0.6
MAX_POSITIONS = 5


def main():
    print("=" * 70)
    print("P20: C-Long v3 完整驗證")
    print("=" * 70)

    # Load walk-forward predictions
    print("\nLoading walk-forward predictions...")
    pred_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19g_walkforward\p19g_all_predictions.csv")
    att = pd.read_csv(pred_path, parse_dates=["announce"])
    att["stock_id"] = att["stock_id"].astype(str).str.zfill(4)
    att = att[att["oos_prob"].notna()].copy()
    print(f"  Events with OOS predictions: {len(att):,}")

    # Load price data
    print("Loading price data...")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    dis_raw = data.get("disposal_information")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Build disposal lookup
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis_by_stock = {}
    for _, row in dis.iterrows():
        sid = row["stock_id"]
        if sid not in dis_by_stock:
            dis_by_stock[sid] = []
        dis_by_stock[sid].append(row["dis_announce"])

    # Pre-compute turnover
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # === CONSECUTIVE ATTENTION DETECTION ===
    print("\nComputing consecutive attention...")
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    att["consec_1d"] = False
    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann = row["announce"]
        if sid in att_dates_by_stock:
            for delta in [1, 2, 3]:
                if (ann - pd.Timedelta(days=delta)) in att_dates_by_stock[sid]:
                    att.loc[idx, "consec_1d"] = True
                    break

    print(f"  Consecutive events: {att['consec_1d'].sum():,} / {len(att):,} ({att['consec_1d'].mean():.1%})")

    # === COMPUTE TRADE RETURNS (exit at disposal OR max 15d) ===
    print("\nComputing trade returns (smart exit: disposal OR max 15d)...")
    att["ann_idx"] = cal.searchsorted(att["announce"], side="left")

    trade_returns = []
    hold_days_list = []
    for _, row in att.iterrows():
        sym = row["stock_id"]
        ai = int(row["ann_idx"])
        if sym not in valid_stocks:
            trade_returns.append(np.nan)
            hold_days_list.append(np.nan)
            continue
        entry_idx = ai + 1
        if entry_idx >= n_cal - 1:
            trade_returns.append(np.nan)
            hold_days_list.append(np.nan)
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[ai][sym]
            avg_to = avg_turnover_5d.iloc[ai][sym]
        except (IndexError, KeyError):
            trade_returns.append(np.nan)
            hold_days_list.append(np.nan)
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            trade_returns.append(np.nan)
            hold_days_list.append(np.nan)
            continue

        # Liquidity filter
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            trade_returns.append(np.nan)
            hold_days_list.append(np.nan)
            continue

        # Gap filter
        gap = entry_price / prev_close - 1
        if not (-0.08 < gap < 0.04):
            trade_returns.append(np.nan)
            hold_days_list.append(np.nan)
            continue

        # Exit: disposal announcement day - 1, OR max_hold
        exit_idx = min(entry_idx + MAX_HOLD_DAYS - 1, n_cal - 1)
        ann_date = row["announce"]
        if sym in dis_by_stock:
            for dis_date in dis_by_stock[sym]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    dis_exit_idx = cal.searchsorted(dis_date, side="left") - 1
                    if dis_exit_idx > entry_idx:
                        exit_idx = min(exit_idx, dis_exit_idx)
                    break

        try:
            exit_price = close.iloc[exit_idx][sym]
            if np.isnan(exit_price) or exit_price <= 0:
                trade_returns.append(np.nan)
                hold_days_list.append(np.nan)
                continue
        except (IndexError, KeyError):
            trade_returns.append(np.nan)
            hold_days_list.append(np.nan)
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        trade_returns.append(ret)
        hold_days_list.append(exit_idx - entry_idx + 1)

    att["trade_ret_15d"] = trade_returns
    att["hold_days"] = hold_days_list
    print(f"  Valid trades: {att['trade_ret_15d'].notna().sum():,}")
    print(f"  Avg hold: {att['hold_days'].mean():.1f} days")

    # === V3 SIGNAL: consecutive + prob >= 0.6 ===
    print("\n" + "=" * 70)
    print("V3 訊號生成: consecutive + prob >= 0.6")
    print("=" * 70)

    v3_signal = att[
        (att["consec_1d"]) &
        (att["oos_prob"] >= THRESHOLD) &
        (att["trade_ret_15d"].notna())
    ].copy()
    v3_signal = v3_signal.sort_values(["announce", "oos_prob"], ascending=[True, False])

    print(f"\n  V3 signals: {len(v3_signal):,}")
    print(f"  Upgrade rate: {v3_signal['leads_to_disposal'].mean():.1%}")
    print(f"  Avg return (event-level): {v3_signal['trade_ret_15d'].mean():.2%}")
    print(f"  Win rate (event-level): {(v3_signal['trade_ret_15d'] > 0).mean():.1%}")

    # === YEARLY BREAKDOWN (event-level) ===
    print("\n" + "=" * 70)
    print("逐年表現 (event-level, 無部位限制)")
    print("=" * 70)
    print(f"  {'Year':>6} {'n':>5} {'avg_ret':>8} {'win%':>6} {'prec%':>6} {'判定':>6}")

    yearly_pass = 0
    yearly_total = 0
    for yr in sorted(v3_signal["year"].unique()):
        yr_data = v3_signal[v3_signal["year"] == yr]
        if len(yr_data) < 5:
            continue
        avg = yr_data["trade_ret_15d"].mean()
        win = (yr_data["trade_ret_15d"] > 0).mean()
        prec = yr_data["leads_to_disposal"].mean()
        sign = "PASS" if avg > 0 else "FAIL"
        yearly_pass += (1 if avg > 0 else 0)
        yearly_total += 1
        print(f"  {yr:>6} {len(yr_data):>5} {avg:>8.2%} {win:>6.1%} {prec:>6.1%} {sign:>6}")

    print(f"\n  逐年通過: {yearly_pass}/{yearly_total}")

    # === NEW vs OLD REGIME ===
    print("\n" + "=" * 70)
    print("新制 vs 舊制比較")
    print("=" * 70)

    att["is_new_regime"] = att["announce"] >= NEW_REGIME_DATE
    v3_signal["is_new_regime"] = v3_signal["announce"] >= NEW_REGIME_DATE

    old = v3_signal[~v3_signal["is_new_regime"]]
    new = v3_signal[v3_signal["is_new_regime"]]

    print(f"\n  舊制 (before {NEW_REGIME_DATE.date()}):")
    if len(old) > 10:
        print(f"    n={len(old):,}, avg={old['trade_ret_15d'].mean():.2%}, win={(old['trade_ret_15d']>0).mean():.1%}")
        print(f"    upgrade rate: {old['leads_to_disposal'].mean():.1%}")
    else:
        print(f"    n={len(old):,} (insufficient)")

    print(f"\n  新制 (after {NEW_REGIME_DATE.date()}):")
    if len(new) > 10:
        print(f"    n={len(new):,}, avg={new['trade_ret_15d'].mean():.2%}, win={(new['trade_ret_15d']>0).mean():.1%}")
        print(f"    upgrade rate: {new['leads_to_disposal'].mean():.1%}")
    else:
        print(f"    n={len(new):,} (insufficient, monitoring)")

    # === PORTFOLIO SIMULATION (5 positions) ===
    print("\n" + "=" * 70)
    print("組合模擬 (5檔部位限制, 15d hold)")
    print("=" * 70)

    v3_signal["entry_idx"] = cal.searchsorted(v3_signal["announce"], side="left") + 1
    v3_signal["exit_idx"] = v3_signal["entry_idx"] + v3_signal["hold_days"].astype(int) - 1

    active_positions = []
    executed = []
    for _, trade in v3_signal.iterrows():
        entry_idx = int(trade["entry_idx"])
        exit_idx = min(int(trade["exit_idx"]), n_cal - 1)

        active_positions = [(ei, s) for ei, s in active_positions if ei > entry_idx]
        if len(active_positions) >= MAX_POSITIONS:
            continue
        if any(s == trade["stock_id"] for _, s in active_positions):
            continue
        active_positions.append((exit_idx, trade["stock_id"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)
    print(f"\n  Executed: {len(ex):,} trades")
    print(f"  Avg return: {ex['trade_ret_15d'].mean():.2%}")
    print(f"  Win rate: {(ex['trade_ret_15d'] > 0).mean():.1%}")
    print(f"  Upgrade rate: {ex['leads_to_disposal'].mean():.1%}")
    print(f"  Avg hold: {ex['hold_days'].mean():.1f} days")

    # Daily P&L
    daily_returns = pd.Series(0.0, index=cal)
    capital_per_pos = 1.0 / MAX_POSITIONS
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = min(int(t["exit_idx"]), n_cal - 1)
        hold = xi - ei + 1
        daily_ret = t["trade_ret_15d"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            daily_returns.iloc[d] += daily_ret * capital_per_pos

    # Metrics
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0

    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    mdd = drawdown.min()

    first_trade_idx = int(ex["entry_idx"].min())
    total_days = n_cal - first_trade_idx
    cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

    print(f"\n  CAGR: {cagr:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  MDD: {mdd:.1%}")
    print(f"  Avg hold: {ex['hold_days'].mean():.1f} days")

    # Yearly portfolio
    print(f"\n  逐年 (portfolio-level):")
    print(f"  {'Year':>6} {'n':>5} {'avg_ret':>8} {'win%':>6} {'判定':>6}")
    port_yearly_pass = 0
    port_yearly_total = 0
    for yr in sorted(ex["year"].unique()):
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 3:
            continue
        avg = yr_data["trade_ret_15d"].mean()
        win = (yr_data["trade_ret_15d"] > 0).mean()
        sign = "PASS" if avg > 0 else "FAIL"
        port_yearly_pass += (1 if avg > 0 else 0)
        port_yearly_total += 1
        print(f"  {yr:>6} {len(yr_data):>5} {avg:>8.2%} {win:>6.1%} {sign:>6}")

    print(f"\n  組合逐年通過: {port_yearly_pass}/{port_yearly_total}")

    # === COMPARISON: v3 vs v2 (no consecutive, threshold=0.5, 10d) ===
    print("\n" + "=" * 70)
    print("對照組: v2 配置 (threshold=0.5, 無連續, 10d)")
    print("=" * 70)

    v2_signal = att[
        (att["oos_prob"] >= 0.5) &
        (att["trade_ret"].notna())
    ].copy()
    v2_signal = v2_signal.sort_values(["announce", "oos_prob"], ascending=[True, False])

    v2_active = []
    v2_executed = []
    for _, trade in v2_signal.iterrows():
        entry_idx = int(cal.searchsorted(trade["announce"], side="left") + 1)
        exit_idx = min(entry_idx + 9, n_cal - 1)
        v2_active = [(ei, s) for ei, s in v2_active if ei > entry_idx]
        if len(v2_active) >= MAX_POSITIONS:
            continue
        if any(s == trade["stock_id"] for _, s in v2_active):
            continue
        v2_active.append((exit_idx, trade["stock_id"]))
        v2_executed.append(trade)

    v2_ex = pd.DataFrame(v2_executed)
    print(f"\n  v2 Executed: {len(v2_ex):,} trades")
    print(f"  v2 Avg return: {v2_ex['trade_ret'].mean():.2%}")
    print(f"  v2 Win rate: {(v2_ex['trade_ret'] > 0).mean():.1%}")

    v2_daily = pd.Series(0.0, index=cal)
    for _, t in v2_ex.iterrows():
        ei = int(cal.searchsorted(t["announce"], side="left") + 1)
        xi = min(ei + 9, n_cal - 1)
        hold = xi - ei + 1
        daily_ret = t["trade_ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            v2_daily.iloc[d] += daily_ret * capital_per_pos

    v2_cum = (1 + v2_daily).cumprod()
    v2_rm = v2_cum.cummax()
    v2_mdd = ((v2_cum - v2_rm) / v2_rm).min()
    v2_dm = v2_daily.mean()
    v2_ds = v2_daily.std()
    v2_sharpe = v2_dm / v2_ds * np.sqrt(252) if v2_ds > 0 else 0
    v2_first = int(v2_ex["announce"].apply(lambda x: cal.searchsorted(x, side="left")).min())
    v2_days = n_cal - v2_first
    v2_cagr = v2_cum.iloc[-1] ** (252 / v2_days) - 1 if v2_cum.iloc[-1] > 0 else -1

    print(f"  v2 CAGR: {v2_cagr:.1%}, Sharpe: {v2_sharpe:.2f}, MDD: {v2_mdd:.1%}")

    # === SUMMARY TABLE ===
    print("\n" + "=" * 70)
    print("V3 vs V2 比較總結")
    print("=" * 70)
    print(f"  {'Config':<20} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'AvgRet':>8} {'Win%':>6} {'Years':>6}")
    print(f"  {'v3 (consec+0.6+15d)':<20} {cagr:>8.1%} {sharpe:>8.2f} {mdd:>8.1%} {ex['trade_ret_15d'].mean():>8.2%} {(ex['trade_ret_15d']>0).mean():>6.1%} {port_yearly_pass}/{port_yearly_total}")
    print(f"  {'v2 (0.5+10d)':<20} {v2_cagr:>8.1%} {v2_sharpe:>8.2f} {v2_mdd:>8.1%} {v2_ex['trade_ret'].mean():>8.2%} {(v2_ex['trade_ret']>0).mean():>6.1%}")

    # Save
    ex.to_csv(OUT / "p20_v3_executed_trades.csv", index=False, encoding="utf-8-sig")
    v3_signal.to_csv(OUT / "p20_v3_all_signals.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
