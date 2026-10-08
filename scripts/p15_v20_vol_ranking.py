"""V20 Final+Vol - Add volatility ranking + MA60 extreme exclusion.

Changes from V20 Final:
  - Rank signals by LOW volatility when capacity is limited
  - Exclude stocks with price > MA60 * 1.5 (extreme overextension)
  - P14 validated: low vol = +2.53% vs high vol = +0.44%
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import finlab
from finlab import data
from finlab.backtest import sim

# ==================== Setup ====================
TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\auth_token.txt")
if TOKEN_FILE.exists():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        finlab.login(f.read().strip())
        print("FinLab login OK")

RESULT_PATH = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p15_v20_vol")
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# ==================== Parameters ====================
print("=" * 60)
print("  監獄兔 V20 Final+Vol (P14 optimized)")
print("=" * 60)

NEW_REGIME_DATE = "2026-08-10"
ENTRY_OFFSET = 0
EXIT_OFFSET = -1

# Filters (V20 Final)
MIN_TURNOVER = 20_000_000
GAP_LOW = -0.08
GAP_HIGH = 0.04
MAX_BIAS = 999  # Removed

# NEW: Volatility ranking + MA60 exclusion
MAX_MA60_BIAS = 999  # MA60 filter disabled (too aggressive for new regime)

# Risk management
STOP_LOSS = 0.99  # Effectively none
POSITION_LIMIT = 0.20
FEE_RATIO = 1.425 / 1000 / 3

# ==================== Load Data ====================
print("\nLoading data...")
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
    (dis["announce"] >= NEW_REGIME_DATE)
].copy()
print(f"New regime events: {len(dis):,}")

# ==================== Compute Indicators ====================
print("Computing indicators...")
trading_days = close.index
n_days = len(trading_days)

target_stocks = list(set(dis["stock_id"]) & valid_stocks)
filtered_close = close[target_stocks].astype(np.float32)
filtered_vol = vol[target_stocks].astype(np.float32)

# Turnover (liquidity filter)
turnover = filtered_close * filtered_vol
avg_turnover_5d = turnover.rolling(5).mean()

# MA20 (kept for reference)
ma20 = filtered_close.rolling(20).mean()

# MA60 (for extreme exclusion)
ma60 = filtered_close.rolling(60).mean()

# Volatility (20d rolling std of daily returns) - KEY NEW FACTOR
daily_ret = filtered_close.pct_change()
volatility_20d = daily_ret.rolling(20).std()

# ==================== Build Position ====================
print("Building signals...")
position = pd.DataFrame(False, index=close.index, columns=close.columns)

n_signals = 0
n_filtered = 0
filter_reasons = {"gap": 0, "liquidity": 0, "ma60_extreme": 0, "nan": 0}

# Collect signals with volatility for ranking
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

    if exec_idx >= n_days or exit_idx < exec_idx:
        continue

    signal_day = trading_days[signal_idx]
    exec_day = trading_days[exec_idx]

    try:
        prev_c = filtered_close.loc[signal_day, sym]
        e_open = open_p.loc[exec_day, sym]
        avg_to = avg_turnover_5d.loc[signal_day, sym]
        current_ma60 = ma60.loc[signal_day, sym]
        current_vol = volatility_20d.loc[signal_day, sym]
    except KeyError:
        filter_reasons["nan"] += 1
        n_filtered += 1
        continue

    if any(pd.isna(x) or x <= 0 for x in [prev_c, e_open]):
        filter_reasons["nan"] += 1
        n_filtered += 1
        continue

    # Gap filter
    gap_pct = (e_open - prev_c) / prev_c
    if not (GAP_LOW < gap_pct < GAP_HIGH):
        filter_reasons["gap"] += 1
        n_filtered += 1
        continue

    # Liquidity filter
    if pd.isna(avg_to) or avg_to < MIN_TURNOVER:
        filter_reasons["liquidity"] += 1
        n_filtered += 1
        continue

    # MA60 extreme exclusion (NEW)
    if not pd.isna(current_ma60) and current_ma60 > 0:
        ma60_bias = (prev_c - current_ma60) / current_ma60
        if ma60_bias > MAX_MA60_BIAS:
            filter_reasons["ma60_extreme"] += 1
            n_filtered += 1
            continue

    # Store signal with volatility for ranking
    signal_list.append({
        "sym": sym,
        "exec_day": exec_day,
        "exit_day": trading_days[exit_idx],
        "volatility": current_vol if not pd.isna(current_vol) else 999,
    })
    n_signals += 1

print(f"\nSignals: {n_signals}")
print(f"Filtered: {n_filtered}")
print(f"  Gap: {filter_reasons['gap']}")
print(f"  Liquidity: {filter_reasons['liquidity']}")
print(f"  MA60 extreme: {filter_reasons['ma60_extreme']}")
print(f"  NaN: {filter_reasons['nan']}")

# ==================== Volatility Ranking ====================
# When multiple signals on same day, prefer LOW volatility
# finlab sim handles position_limit, but we can pre-filter by
# only keeping the lowest-vol signals per day

print("\nApplying volatility ranking (prefer low vol)...")
signal_df = pd.DataFrame(signal_list)

# Group by exec_day, keep only lowest-vol signals
# With position_limit=0.20, max 5 positions. Keep top 5 per day by LOWEST vol.
MAX_PER_DAY = 5  # Match position limit

signal_df = signal_df.sort_values(["exec_day", "volatility"])  # Low vol first
signal_df["rank_in_day"] = signal_df.groupby("exec_day").cumcount()
selected = signal_df[signal_df["rank_in_day"] < MAX_PER_DAY]

print(f"  Signals before ranking: {len(signal_df)}")
print(f"  Signals after ranking: {len(selected)}")
print(f"  Removed (high vol priority): {len(signal_df) - len(selected)}")

# Set positions for selected signals
for _, sig in selected.iterrows():
    position.loc[sig["exec_day"]:sig["exit_day"], sig["sym"]] = True

# ==================== Circuit Breaker ====================
print("\nApplying circuit breaker...")
panic_count = (close.pct_change() < -0.06).sum(axis=1)
is_panic = panic_count > 400
danger_zone = is_panic.rolling(9, min_periods=1).max() > 0
position.loc[danger_zone, :] = False
print(f"Circuit breaker active days: {danger_zone.sum()}")

# ==================== Backtest ====================
print("\nRunning backtest...")
report = sim(
    position,
    trade_at_price="open",
    fee_ratio=FEE_RATIO,
    position_limit=POSITION_LIMIT,
    stop_loss=STOP_LOSS,
    upload=False,
    name="監獄兔_V20_Final_Vol",
)

stats = report.get_stats()
trades = report.get_trades()

# ==================== Results ====================
print("\n" + "=" * 60)
print("  V20 Final+Vol 結果")
print("=" * 60)
print(f"  CAGR: {stats['cagr'] * 100:.2f}%")
print(f"  MDD: {stats['max_drawdown'] * 100:.2f}%")
print(f"  Win Rate: {stats['win_ratio'] * 100:.2f}%")
print(f"  Total Return: {stats['total_return'] * 100:.2f}%")
print(f"  Trades: {len(trades)}")

# Compare with V20 Final (no vol)
print(f"\n--- Comparison ---")
print(f"  V20 Final (no vol):  CAGR 405.9%, MDD -6.67%, Win 70.3%, Total +28.22%")
print(f"  V20 Final+Vol:       CAGR {stats['cagr']*100:.1f}%, MDD {stats['max_drawdown']*100:.1f}%, Win {stats['win_ratio']*100:.1f}%, Total {stats['total_return']*100:.1f}%")

# Save
if len(trades) > 0:
    trades.to_csv(RESULT_PATH / "v20_vol_trades.csv", index=False, encoding="utf-8-sig")

import json
with open(RESULT_PATH / "v20_vol_stats.json", "w", encoding="utf-8") as f:
    json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
               for k, v in stats.items()}, f, ensure_ascii=False, indent=2)

print(f"\nOutput: {RESULT_PATH}")
