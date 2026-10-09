"""P19d: Optimize stop-loss for C-Long strategy.

Grid search:
- Model threshold: 0.30, 0.35, 0.40, 0.50
- Stop-loss: -1%, -1.5%, -2%, -2.5%, -3%, none
- Stop window: 1d, 2d, 3d
- Max hold: 5d, 7d, 10d, until_disposal

Objective: maximize CAGR with 5 positions.
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19d_stoploss")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P19d: C-Long 止損參數優化")
    print("=" * 70)

    # Load predictions from P19c
    print("\nLoading model predictions...")
    pred_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19c_prediction\p19c_predictions.csv")
    att_model = pd.read_csv(pred_path, parse_dates=["announce"])
    att_model = att_model[att_model["prob_gb"].notna()].copy()  # test period only
    att_model["stock_id"] = att_model["stock_id"].astype(str).str.zfill(4)
    print(f"  Test events: {len(att_model):,}")
    print(f"  Upgrade rate: {att_model['leads_to_disposal'].mean():.1%}")

    # Load price data
    print("Loading price data...")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    low = data.get("price:最低價")
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

    # Pre-compute entry/exit for all events
    print("Pre-computing trade paths...")
    att_model["ann_idx"] = cal.searchsorted(att_model["announce"], side="left")

    trade_data = []
    for _, row in att_model.iterrows():
        sym = row["stock_id"]
        ai = int(row["ann_idx"])
        if sym not in valid_stocks:
            continue
        entry_idx = ai + 1
        if entry_idx >= n_cal - 1:
            continue
        try:
            entry_price = open_p.iloc[entry_idx][sym]
        except (IndexError, KeyError):
            continue
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Find disposal date (for exit target)
        dis_exit_idx = None
        ann_date = row["announce"]
        if row["leads_to_disposal"] and sym in dis_by_stock:
            for dis_date in dis_by_stock[sym]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    dis_exit_idx = cal.searchsorted(dis_date, side="left") - 1
                    break

        # Compute daily prices for stop-loss check
        max_lookahead = min(20, n_cal - entry_idx - 1)
        if max_lookahead < 1:
            continue

        daily_close = []
        daily_low = []
        for d in range(max_lookahead):
            idx = entry_idx + d
            try:
                c = close.iloc[idx][sym]
                l = low.iloc[idx][sym]
                daily_close.append(c if not np.isnan(c) else np.nan)
                daily_low.append(l if not np.isnan(l) else np.nan)
            except (IndexError, KeyError):
                daily_close.append(np.nan)
                daily_low.append(np.nan)

        trade_data.append({
            "idx": row.name,
            "symbol": sym,
            "entry_idx": entry_idx,
            "entry_price": entry_price,
            "year": row["announce"].year,
            "prob_gb": row["prob_gb"],
            "leads_to_disposal": row["leads_to_disposal"],
            "dis_exit_idx": dis_exit_idx,
            "daily_close": daily_close,
            "daily_low": daily_low,
            "max_lookahead": max_lookahead,
        })

    print(f"  Pre-computed: {len(trade_data):,} trades")

    # === GRID SEARCH ===
    print("\n" + "=" * 70)
    print("Grid Search")
    print("=" * 70)

    thresholds = [0.30, 0.35, 0.40, 0.50]
    stop_losses = [0.01, 0.015, 0.02, 0.025, 0.03, None]  # None = no stop
    stop_windows = [1, 2, 3]
    max_holds = [5, 7, 10, None]  # None = until disposal

    results = []

    for threshold, sl, sw, mh in itertools.product(thresholds, stop_losses, stop_windows, max_holds):
        # Filter by model threshold
        trades = [t for t in trade_data if t["prob_gb"] >= threshold]
        if len(trades) < 100:
            continue

        # Simulate each trade
        returns = []
        for t in trades:
            entry_price = t["entry_price"]
            daily_close = t["daily_close"]
            daily_low = t["daily_low"]

            # Determine exit
            exit_ret = None
            exit_reason = "timeout"

            # Check stop-loss within window
            if sl is not None:
                check_days = min(sw, len(daily_close))
                for d in range(check_days):
                    if not np.isnan(daily_low[d]) and daily_low[d] > 0:
                        intraday_ret = daily_low[d] / entry_price - 1
                        if intraday_ret <= -sl:
                            # Stop hit: exit at close of that day (conservative)
                            if not np.isnan(daily_close[d]):
                                exit_ret = daily_close[d] / entry_price - 1 - COST_RATE
                            else:
                                exit_ret = -sl - COST_RATE
                            exit_reason = "stop_loss"
                            break

            if exit_ret is not None:
                returns.append({"ret": exit_ret, "year": t["year"], "reason": exit_reason,
                               "upgraded": t["leads_to_disposal"]})
                continue

            # Determine hold period
            if mh is None and t["dis_exit_idx"] is not None:
                # Hold until disposal
                hold = t["dis_exit_idx"] - t["entry_idx"]
                hold = max(1, min(hold, t["max_lookahead"]))
            elif mh is not None:
                hold = min(mh, t["max_lookahead"])
            else:
                hold = min(10, t["max_lookahead"])  # default 10d

            # Exit at close of hold day
            exit_d = hold - 1
            if exit_d < len(daily_close) and not np.isnan(daily_close[exit_d]) and daily_close[exit_d] > 0:
                exit_ret = daily_close[exit_d] / entry_price - 1 - COST_RATE
                exit_reason = "hold_exit"
            else:
                continue

            returns.append({"ret": exit_ret, "year": t["year"], "reason": exit_reason,
                           "upgraded": t["leads_to_disposal"]})

        if len(returns) < 50:
            continue

        ret_df = pd.DataFrame(returns)
        avg_ret = ret_df["ret"].mean()
        win_rate = (ret_df["ret"] > 0).mean()

        # CAGR estimate: 5 positions, average hold days
        avg_hold = mh if mh is not None else 8  # estimate
        trades_per_year = len(returns) / 3  # test period is 3 years
        capacity = 5 / avg_hold * 252
        actual_trades = min(trades_per_year, capacity)
        annual_ret = actual_trades / 5 * avg_ret

        # Yearly stability
        yearly_ok = 0
        yearly_total = 0
        for yr in ret_df["year"].unique():
            yr_data = ret_df[ret_df["year"] == yr]
            if len(yr_data) >= 10:
                yearly_total += 1
                if yr_data["ret"].mean() > 0:
                    yearly_ok += 1

        results.append({
            "threshold": threshold,
            "stop_loss": sl,
            "stop_window": sw,
            "max_hold": mh,
            "avg_ret": avg_ret,
            "win_rate": win_rate,
            "n_trades": len(returns),
            "trades_per_year": trades_per_year,
            "capacity": capacity,
            "annual_ret": annual_ret,
            "yearly_pass": yearly_ok,
            "yearly_total": yearly_total,
            "stop_hit_rate": (ret_df["reason"] == "stop_loss").mean(),
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("annual_ret", ascending=False)

    # Print top 20
    print("\nTOP 20 CONFIGURATIONS (by estimated annual return):")
    print(f"{'Thresh':>6} {'SL':>5} {'SW':>3} {'Hold':>5} | {'AvgRet':>7} {'Win%':>5} {'AnnRet':>7} {'Yrs':>5} {'StopHit':>7}")
    print("-" * 80)
    for _, r in res_df.head(20).iterrows():
        sl_str = f"{r['stop_loss']:.1%}" if r['stop_loss'] else "none"
        mh_str = str(int(r['max_hold'])) if r['max_hold'] else "dis"
        print(f"{r['threshold']:>6.2f} {sl_str:>5} {int(r['stop_window']):>3} {mh_str:>5} | "
              f"{r['avg_ret']:>7.2%} {r['win_rate']:>5.1%} {r['annual_ret']:>7.1%} "
              f"{int(r['yearly_pass'])}/{int(r['yearly_total'])} {r['stop_hit_rate']:>7.1%}")

    # Best with all years positive
    print("\n\nBEST WITH ALL YEARS POSITIVE:")
    good = res_df[(res_df["yearly_pass"] == res_df["yearly_total"]) & (res_df["yearly_total"] >= 3)]
    for _, r in good.head(10).iterrows():
        sl_str = f"{r['stop_loss']:.1%}" if r['stop_loss'] else "none"
        mh_str = str(int(r['max_hold'])) if r['max_hold'] else "dis"
        print(f"  threshold={r['threshold']:.2f}, SL={sl_str}/{int(r['stop_window'])}d, hold={mh_str} | "
              f"avg={r['avg_ret']:.2%}, annual={r['annual_ret']:.1%}, n={int(r['n_trades'])}")

    # Save full results
    res_df.to_csv(OUT / "p19d_grid_results.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
