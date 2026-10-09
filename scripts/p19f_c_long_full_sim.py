"""P19f: C-Long full strategy simulation with improved model.

Uses P19e predictions (AUC 0.827, precision 67%).
Simulates: 5 positions, 10d hold, no stop-loss.
Calculates: CAGR, Sharpe, MDD, yearly breakdown.
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19f_c_long_sim")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P19f: C-Long 完整策略模擬（改進模型）")
    print("=" * 70)

    # Load improved predictions
    print("\nLoading predictions...")
    pred_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19e_better_pred\p19e_predictions.csv")
    att_model = pd.read_csv(pred_path, parse_dates=["announce"])
    att_model["stock_id"] = att_model["stock_id"].astype(str).str.zfill(4)
    att_model = att_model[att_model["prob_gb"].notna()].copy()
    print(f"  Events with predictions: {len(att_model):,}")
    print(f"  Upgrade rate: {att_model['leads_to_disposal'].mean():.1%}")

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

    # Pre-compute turnover filter
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Compute all trade returns
    print("Computing trade returns...")
    att_model["ann_idx"] = cal.searchsorted(att_model["announce"], side="left")

    trades = []
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
            prev_close = close.iloc[ai][sym]
            avg_to = avg_turnover_5d.iloc[ai][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            continue
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            continue
        gap = entry_price / prev_close - 1
        if not (-0.08 < gap < 0.04):
            continue

        # Determine exit: hold until disposal announcement (if upgraded) or 10d max
        dis_exit_idx = None
        ann_date = row["announce"]
        if row["leads_to_disposal"] and sym in dis_by_stock:
            for dis_date in dis_by_stock[sym]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    dis_exit_idx = cal.searchsorted(dis_date, side="left") - 1
                    break

        # Exit logic: min(disposal_day - 1, entry + 9, n_cal - 1)
        if dis_exit_idx is not None and dis_exit_idx > entry_idx:
            exit_idx = min(dis_exit_idx, entry_idx + 9, n_cal - 1)
        else:
            exit_idx = min(entry_idx + 9, n_cal - 1)

        try:
            exit_price = close.iloc[exit_idx][sym]
            if np.isnan(exit_price) or exit_price <= 0:
                continue
        except (IndexError, KeyError):
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        hold_days = exit_idx - entry_idx + 1

        trades.append({
            "symbol": sym,
            "entry_date": cal[entry_idx],
            "exit_date": cal[exit_idx],
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "year": row["announce"].year,
            "prob_gb": row["prob_gb"],
            "leads_to_disposal": row["leads_to_disposal"],
            "ret": ret,
            "hold_days": hold_days,
        })

    all_trades = pd.DataFrame(trades)
    print(f"  Total trades: {len(all_trades):,}")

    # === SIMULATE WITH POSITION LIMIT ===
    print("\n" + "=" * 70)
    print("策略模擬（5檔部位限制）")
    print("=" * 70)

    for threshold in [0.40, 0.50, 0.60, 0.70]:
        filtered = all_trades[all_trades["prob_gb"] >= threshold].copy()
        if len(filtered) < 100:
            continue

        # Sort by entry date, then by probability (highest first)
        filtered = filtered.sort_values(["entry_date", "prob_gb"], ascending=[True, False])

        # Simulate: max 5 positions at any time
        n_positions = 5
        active_positions = []  # list of (exit_idx, symbol)
        executed = []

        for _, trade in filtered.iterrows():
            entry_idx = trade["entry_idx"]
            exit_idx = trade["exit_idx"]

            # Remove expired positions
            active_positions = [(ei, s) for ei, s in active_positions if ei > entry_idx]

            # Check capacity
            if len(active_positions) >= n_positions:
                continue

            # Check no overlap with same symbol
            if any(s == trade["symbol"] for _, s in active_positions):
                continue

            # Execute trade
            active_positions.append((exit_idx, trade["symbol"]))
            executed.append(trade)

        if len(executed) < 50:
            continue

        ex = pd.DataFrame(executed)

        # Calculate metrics
        total_ret = ex["ret"].sum()
        avg_ret = ex["ret"].mean()
        win_rate = (ex["ret"] > 0).mean()
        n_years = (ex["entry_date"].max().year - ex["entry_date"].min().year + 1)
        n_years = max(n_years, 1)

        # Daily P&L for Sharpe/MDD
        daily_returns = pd.Series(0.0, index=cal)
        capital_per_pos = 1.0 / n_positions
        for _, t in ex.iterrows():
            # Distribute return over holding period
            daily_ret = t["ret"] / t["hold_days"]
            for d in range(t["entry_idx"], min(t["exit_idx"] + 1, len(cal))):
                daily_returns.iloc[d] += daily_ret * capital_per_pos

        # Annualize
        daily_mean = daily_returns.mean()
        daily_std = daily_returns.std()
        sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0

        # Cumulative for MDD
        cum = (1 + daily_returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        mdd = drawdown.min()

        # CAGR
        total_days = len(daily_returns[daily_returns.index >= ex["entry_date"].min()])
        total_days = max(total_days, 1)
        cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

        # Yearly
        yearly_stats = []
        for yr in sorted(ex["year"].unique()):
            yr_trades = ex[ex["year"] == yr]
            yr_ret = yr_trades["ret"].mean()
            yr_win = (yr_trades["ret"] > 0).mean()
            yearly_stats.append((yr, yr_ret, yr_win, len(yr_trades)))

        print(f"\n  Threshold={threshold:.2f} (precision~{filtered['leads_to_disposal'].mean():.0%}):")
        print(f"    Executed: {len(ex):,} trades, {n_years} years")
        print(f"    Avg return: {avg_ret:.2%}, Win rate: {win_rate:.1%}")
        print(f"    CAGR: {cagr:.1%}, Sharpe: {sharpe:.2f}, MDD: {mdd:.1%}")
        print(f"    Avg hold: {ex['hold_days'].mean():.1f} days")
        print(f"    Upgraded in executed: {ex['leads_to_disposal'].mean():.1%}")
        print(f"    Yearly:")
        for yr, yr_ret, yr_win, yr_n in yearly_stats:
            sign = "PASS" if yr_ret > 0 else "FAIL"
            print(f"      {yr}: {yr_ret:.2%}, win={yr_win:.1%}, n={yr_n} {sign}")

    # === BEST CONFIG DETAIL ===
    print("\n" + "=" * 70)
    print("最佳配置詳細分析")
    print("=" * 70)

    # Use threshold=0.50 as primary
    best_threshold = 0.50
    filtered = all_trades[all_trades["prob_gb"] >= best_threshold].copy()
    filtered = filtered.sort_values(["entry_date", "prob_gb"], ascending=[True, False])

    n_positions = 5
    active_positions = []
    executed = []
    for _, trade in filtered.iterrows():
        entry_idx = trade["entry_idx"]
        exit_idx = trade["exit_idx"]
        active_positions = [(ei, s) for ei, s in active_positions if ei > entry_idx]
        if len(active_positions) >= n_positions:
            continue
        if any(s == trade["symbol"] for _, s in active_positions):
            continue
        active_positions.append((exit_idx, trade["symbol"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)

    # Monthly returns
    print("\n  月度報酬:")
    ex["month"] = pd.to_datetime(ex["entry_date"]).dt.to_period("M")
    monthly = ex.groupby("month").agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        total_ret=("ret", "sum"),
        win=("ret", lambda x: (x > 0).mean()),
    )
    for month, row in monthly.iterrows():
        print(f"    {month}: n={int(row['n'])}, avg={row['avg_ret']:.2%}, total={row['total_ret']:.1%}, win={row['win']:.1%}")

    # Save
    ex.to_csv(OUT / "p19f_executed_trades.csv", index=False, encoding="utf-8-sig")
    all_trades.to_csv(OUT / "p19f_all_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
