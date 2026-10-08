"""P16: Position sizing optimization.

Tests different position_limit values:
  0.33 (3 positions) - most concentrated
  0.25 (4 positions)
  0.20 (5 positions) - current V20
  0.15 (6-7 positions)
  0.10 (10 positions)
  0.07 (14 positions)
  0.05 (20 positions) - most diversified

Uses old regime data (2018-2026) for large sample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import finlab
from finlab import data
from finlab.backtest import sim

TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\auth_token.txt")
if TOKEN_FILE.exists():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        finlab.login(f.read().strip())

RESULT_PATH = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p16_position_sizing")
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# Parameters
ENTRY_OFFSET = 0
EXIT_OFFSET = -1
MIN_TURNOVER = 20_000_000
GAP_LOW, GAP_HIGH = -0.08, 0.04
STOP_LOSS = 0.99
FEE_RATIO = 1.425 / 1000 / 3

# Load data
print("Loading data...")
dis_raw = data.get("disposal_information")
close = data.get("price:收盤價")
open_p = data.get("price:開盤價")
vol = data.get("price:成交股數")

dis = pd.DataFrame(dis_raw).copy()
dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()

valid_stocks = set(close.columns)
dis = dis[
    dis["stock_id"].str.match(r"^\d{4}$") &
    dis["stock_id"].isin(valid_stocks) &
    ~dis["stock_id"].str.startswith(("00", "91")) &
    (dis["announce"] >= "2018-01-01") &
    (dis["announce"] < "2026-08-10")
].copy()
print(f"Old regime events: {len(dis):,}")

# Indicators
print("Computing indicators...")
trading_days = close.index
n_days = len(trading_days)
target_stocks = list(set(dis["stock_id"]) & valid_stocks)
filtered_close = close[target_stocks].astype(np.float32)
filtered_vol = vol[target_stocks].astype(np.float32)
turnover = filtered_close * filtered_vol
avg_turnover_5d = turnover.rolling(5).mean()

# Build signals (same for all configs)
print("Building signals...")
signal_list = []
for row in dis.itertuples():
    sym = row.stock_id
    if sym not in target_stocks:
        continue
    start_idx = trading_days.searchsorted(row.start)
    end_idx = trading_days.searchsorted(row.end, side="right") - 1
    if start_idx >= n_days or end_idx >= n_days or end_idx < start_idx:
        continue
    signal_idx = start_idx + ENTRY_OFFSET
    exec_idx = signal_idx + 1
    exit_idx = end_idx + EXIT_OFFSET
    if exec_idx >= n_days or exit_idx < exit_idx:
        continue
    signal_day = trading_days[signal_idx]
    exec_day = trading_days[exec_idx]
    try:
        prev_c = filtered_close.loc[signal_day, sym]
        e_open = open_p.loc[exec_day, sym]
        avg_to = avg_turnover_5d.loc[signal_day, sym]
    except KeyError:
        continue
    if any(pd.isna(x) or x <= 0 for x in [prev_c, e_open]):
        continue
    gap_pct = (e_open - prev_c) / prev_c
    if not (GAP_LOW < gap_pct < GAP_HIGH):
        continue
    if pd.isna(avg_to) or avg_to < MIN_TURNOVER:
        continue
    signal_list.append({
        "sym": sym,
        "exec_day": exec_day,
        "exit_day": trading_days[exit_idx],
    })

print(f"Signals: {len(signal_list)}")

# Build base position (all signals)
base_position = pd.DataFrame(False, index=close.index, columns=close.columns)
for sig in signal_list:
    base_position.loc[sig["exec_day"]:sig["exit_day"], sig["sym"]] = True

# Circuit breaker
panic_count = (close.pct_change() < -0.06).sum(axis=1)
is_panic = panic_count > 400
danger_zone = is_panic.rolling(9, min_periods=1).max() > 0
base_position.loc[danger_zone, :] = False

# === Test different position limits ===
configs = [
    (0.33, "3 檔 (33%)"),
    (0.25, "4 檔 (25%)"),
    (0.20, "5 檔 (20%) ← 現行"),
    (0.15, "6-7 檔 (15%)"),
    (0.10, "10 檔 (10%)"),
    (0.07, "14 檔 (7%)"),
    (0.05, "20 檔 (5%)"),
]

print(f"\n{'='*70}")
print("POSITION SIZING OPTIMIZATION (Old Regime 2018-2026)")
print(f"{'='*70}")
print(f"\n  {'Config':<20} {'CAGR':>8} {'MDD':>8} {'Win':>6} {'Total':>10} {'Trades':>6} {'Sharpe':>7}")
print(f"  {'-'*72}")

results = []
for pos_limit, label in configs:
    report = sim(base_position, trade_at_price="open", fee_ratio=FEE_RATIO,
                 position_limit=pos_limit, stop_loss=STOP_LOSS,
                 upload=False, name=f"pos_{label}")
    stats = report.get_stats()
    trades = report.get_trades()

    # Compute Sharpe from monthly returns
    if len(trades) > 0:
        trades_copy = trades.copy()
        trades_copy["entry_date"] = pd.to_datetime(trades_copy["entry_date"])
        trades_copy["month"] = trades_copy["entry_date"].dt.to_period("M")
        monthly = trades_copy.groupby("month")["return"].mean()
        sharpe = (monthly.mean() / monthly.std() * np.sqrt(12)
                  if len(monthly) > 2 and monthly.std() > 0 else 0)
    else:
        sharpe = 0

    print(f"  {label:<20} {stats['cagr']*100:>7.1f}% {stats['max_drawdown']*100:>7.1f}% "
          f"{stats['win_ratio']*100:>5.1f}% {stats['total_return']*100:>9.1f}% "
          f"{len(trades):>6} {sharpe:>7.2f}")

    results.append({
        "config": label, "pos_limit": pos_limit,
        "cagr": stats["cagr"], "mdd": stats["max_drawdown"],
        "win": stats["win_ratio"], "total": stats["total_return"],
        "n_trades": len(trades), "sharpe": sharpe,
    })

    # Save yearly for best configs
    if pos_limit in [0.10, 0.20]:
        trades_copy = trades.copy()
        trades_copy["entry_date"] = pd.to_datetime(trades_copy["entry_date"])
        trades_copy["year"] = trades_copy["entry_date"].dt.year
        yearly = trades_copy.groupby("year").agg(
            n=("return", "count"),
            mean_ret=("return", "mean"),
            win=("return", lambda x: (x > 0).mean()),
        ).reset_index()
        yearly.to_csv(RESULT_PATH / f"yearly_pos_{int(pos_limit*100)}.csv",
                      index=False, encoding="utf-8-sig")

# Save summary
pd.DataFrame(results).to_csv(RESULT_PATH / "position_sizing_summary.csv",
                              index=False, encoding="utf-8-sig")

# === Analysis ===
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

# Find best by different metrics
best_cagr = max(results, key=lambda x: x["cagr"])
best_sharpe = max(results, key=lambda x: x["sharpe"])
best_mdd = max(results, key=lambda x: x["mdd"])  # least negative

print(f"\n  Best CAGR: {best_cagr['config']} ({best_cagr['cagr']*100:.1f}%)")
print(f"  Best Sharpe: {best_sharpe['config']} (Sharpe {best_sharpe['sharpe']:.2f})")
print(f"  Best MDD: {best_mdd['config']} (MDD {best_mdd['mdd']*100:.1f}%)")

# Risk-adjusted comparison
print(f"\n  Risk-adjusted (CAGR / |MDD|):")
for r in results:
    ratio = r["cagr"] / abs(r["mdd"]) if r["mdd"] != 0 else 0
    print(f"    {r['config']:<20} ratio={ratio:.2f}")

print(f"\nOutput: {RESULT_PATH}")
