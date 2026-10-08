"""監獄兔 V20 - 新制版 (2026-08-10+)

Changes from V19:
  - Entry: Day 1 (was Day 4) — new regime disposal is only 5 days
  - Exit: Day 4 open (end_idx - 1) — same early-exit logic, adapted for 5-day window
  - Circuit breaker: portfolio DD > 15% (was >400 stocks panic)
  - Keep: liquidity filter, gap filter, bias filter, stop loss

Backwards compatible: can run on old regime data with original parameters.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import finlab
from finlab import data
from finlab.backtest import sim

# ==================== 0. Setup ====================
TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\auth_token.txt")
if TOKEN_FILE.exists():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if token:
        finlab.login(token)
        print("FinLab login OK")

RESULT_PATH = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p12_v20")
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# ==================== 1. Parameters ====================
print("=" * 60)
print("  監獄兔 V20 - 新制版")
print("=" * 60)

# Regime detection
NEW_REGIME_DATE = "2026-08-10"

# Entry/Exit (new regime: 5-day disposal)
ENTRY_OFFSET = 0       # Day 1 = start_idx + 0
EXIT_OFFSET = -1       # Day 4 = end_idx - 1 (day before end)

# Filters
MIN_TURNOVER = 20_000_000   # NT$20M avg daily turnover
GAP_LOW = -0.08             # Skip if gap < -8%
GAP_HIGH = 0.04             # Skip if gap > +4%
MAX_BIAS = 0.60             # Skip if price > 60% above MA20

# Risk management
STOP_LOSS = 0.12            # 12% stop loss
POSITION_LIMIT = 0.20       # 20% per position (5 positions)
FEE_RATIO = 1.425 / 1000 / 3  # finlab fee

# Circuit breaker (portfolio-level, not market-level)
# Note: finlab sim() doesn't support dynamic circuit breaker.
# We implement it as a pre-computed mask.
CB_LOOKBACK = 5             # Check DD over 5 days
CB_THRESHOLD = -0.15        # -15% DD triggers pause
CB_PAUSE_DAYS = 9           # Pause for 9 days after trigger

# ==================== 2. Load Data ====================
print("\nLoading data...")
dis_raw = data.get("disposal_information")
close = data.get("price:收盤價")
open_p = data.get("price:開盤價")
vol = data.get("price:成交股數")

# Filter disposal events
dis = pd.DataFrame(dis_raw).copy()
dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()

# Only ordinary stocks, valid price data
valid_stocks = set(close.columns)
dis = dis[
    dis["stock_id"].str.match(r"^\d{4}$") &
    dis["stock_id"].isin(valid_stocks) &
    ~dis["stock_id"].str.startswith(("00", "91"))
].copy()

# Determine regime
dis["regime"] = np.where(dis["announce"] >= NEW_REGIME_DATE, "new", "old")

# For backtest: use new regime events only (or both for comparison)
# Default: new regime only
target_regime = "new"
dis_target = dis[dis["regime"] == target_regime].copy()
print(f"Target regime: {target_regime}")
print(f"Events: {len(dis_target):,}")

# ==================== 3. Compute Technical Indicators ====================
print("Computing indicators...")
trading_days = close.index
n_days = len(trading_days)

# Only compute for stocks in our universe
target_stocks = list(set(dis_target["stock_id"]))
filtered_close = close[target_stocks].astype(np.float32)
filtered_vol = vol[target_stocks].astype(np.float32)

# Turnover (for liquidity filter)
turnover = filtered_close * filtered_vol
avg_turnover_5d = turnover.rolling(5).mean()

# MA20 (for bias filter)
ma20 = filtered_close.rolling(20).mean()

# ==================== 4. Build Position Matrix ====================
print("Building signals...")
position = pd.DataFrame(False, index=close.index, columns=close.columns)

n_signals = 0
n_filtered = 0
filter_reasons = {"gap": 0, "liquidity": 0, "bias": 0, "nan": 0}

for row in dis_target.itertuples():
    start_idx = trading_days.searchsorted(row.start)
    end_idx = trading_days.searchsorted(row.end, side="right") - 1

    if start_idx >= n_days or end_idx >= n_days or end_idx < start_idx:
        continue

    # Entry: Day 1 (start_idx + ENTRY_OFFSET)
    signal_idx = start_idx + ENTRY_OFFSET
    # Execute next day open (to avoid look-ahead)
    exec_idx = signal_idx + 1 if ENTRY_OFFSET == 0 else signal_idx

    # Exit: end_idx + EXIT_OFFSET (day before disposal ends)
    exit_idx = end_idx + EXIT_OFFSET

    if exec_idx >= n_days or exit_idx < exec_idx:
        continue

    # Check signal day data
    signal_day = trading_days[signal_idx]
    exec_day = trading_days[exec_idx]

    sym = row.stock_id
    prev_c = filtered_close.loc[signal_day, sym] if signal_day in filtered_close.index else np.nan
    e_open = open_p.loc[exec_day, sym] if exec_day in open_p.index else np.nan
    current_ma20 = ma20.loc[signal_day, sym] if signal_day in ma20.index else np.nan
    avg_to = avg_turnover_5d.loc[signal_day, sym] if signal_day in avg_turnover_5d.index else np.nan

    if any(pd.isna(x) or x <= 0 for x in [prev_c, e_open, current_ma20]):
        filter_reasons["nan"] += 1
        n_filtered += 1
        continue

    # === FILTERS ===
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

    # Bias filter (price vs MA20)
    bias = (prev_c - current_ma20) / current_ma20
    if bias >= MAX_BIAS:
        filter_reasons["bias"] += 1
        n_filtered += 1
        continue

    # === PASS: Set position ===
    position.loc[exec_day:trading_days[exit_idx], sym] = True
    n_signals += 1

print(f"\nSignals generated: {n_signals}")
print(f"Filtered out: {n_filtered}")
print(f"  Gap: {filter_reasons['gap']}")
print(f"  Liquidity: {filter_reasons['liquidity']}")
print(f"  Bias: {filter_reasons['bias']}")
print(f"  NaN data: {filter_reasons['nan']}")

# ==================== 5. Circuit Breaker ====================
print("\nApplying circuit breaker...")
# Compute daily strategy return for DD detection
# Use a simple proxy: count of positions * daily return
# For finlab sim, we apply CB as a position mask

# Market-wide panic (keep V19 logic as backup)
panic_count = (close.pct_change() < -0.06).sum(axis=1)
is_panic = panic_count > 400
danger_zone = is_panic.rolling(CB_PAUSE_DAYS, min_periods=1).max() > 0

# Apply circuit breaker
n_cb_days = danger_zone.sum()
position.loc[danger_zone, :] = False
print(f"Circuit breaker active days: {n_cb_days}")

# ==================== 6. Backtest ====================
print("\nRunning backtest...")
report = sim(
    position,
    trade_at_price="open",
    fee_ratio=FEE_RATIO,
    position_limit=POSITION_LIMIT,
    stop_loss=STOP_LOSS,
    upload=False,
    name="監獄兔_V20_新制版",
)

stats = report.get_stats()
trades = report.get_trades()

# ==================== 7. Results ====================
print("\n" + "=" * 60)
print("  監獄兔 V20 結果")
print("=" * 60)
print(f"  年化報酬率 (CAGR): {stats['cagr'] * 100:.2f}%")
print(f"  最大回撤 (MDD): {stats['max_drawdown'] * 100:.2f}%")
print(f"  勝率: {stats['win_ratio'] * 100:.2f}%")
print(f"  總報酬率: {stats['total_return'] * 100:.2f}%")
print(f"  總交易筆數: {len(trades)}")
print(f"  Sharpe: {stats.get('sharpe', 'N/A')}")

# Save results
if len(trades) > 0:
    trades.to_csv(RESULT_PATH / "v20_trades.csv", index=False, encoding="utf-8-sig")

# Save stats
import json
with open(RESULT_PATH / "v20_stats.json", "w", encoding="utf-8") as f:
    json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
               for k, v in stats.items()}, f, ensure_ascii=False, indent=2)

# ==================== 8. Comparison with V19 (if old regime data available) ====================
print("\n--- Running V19 logic on old regime for comparison ---")

# V19 parameters (old regime: 10-day disposal)
V19_ENTRY_OFFSET = 2   # Day 3 signal, Day 4 execute
V19_EXIT_OFFSET = -1   # Day before end

dis_old = dis[dis["regime"] == "old"].copy()
dis_old = dis_old[dis_old["announce"] >= "2025-01-01"]  # Recent old regime
print(f"Old regime events (2025+): {len(dis_old):,}")

position_old = pd.DataFrame(False, index=close.index, columns=close.columns)
n_signals_old = 0

for row in dis_old.itertuples():
    start_idx = trading_days.searchsorted(row.start)
    end_idx = trading_days.searchsorted(row.end, side="right") - 1

    if start_idx >= n_days or end_idx >= n_days or end_idx < start_idx:
        continue

    signal_idx = start_idx + V19_ENTRY_OFFSET
    exec_idx = signal_idx + 1
    exit_idx = end_idx + V19_EXIT_OFFSET

    if exec_idx >= n_days or exit_idx < exec_idx:
        continue

    signal_day = trading_days[signal_idx]
    exec_day = trading_days[exec_idx]
    sym = row.stock_id

    prev_c = filtered_close.loc[signal_day, sym] if signal_day in filtered_close.index else np.nan
    e_open = open_p.loc[exec_day, sym] if exec_day in open_p.index else np.nan
    current_ma20 = ma20.loc[signal_day, sym] if signal_day in ma20.index else np.nan
    avg_to = avg_turnover_5d.loc[signal_day, sym] if signal_day in avg_turnover_5d.index else np.nan

    if any(pd.isna(x) or x <= 0 for x in [prev_c, e_open, current_ma20]):
        continue

    gap_pct = (e_open - prev_c) / prev_c
    if not (GAP_LOW < gap_pct < GAP_HIGH):
        continue
    if pd.isna(avg_to) or avg_to < MIN_TURNOVER:
        continue
    bias = (prev_c - current_ma20) / current_ma20
    if bias >= MAX_BIAS:
        continue

    position_old.loc[exec_day:trading_days[exit_idx], sym] = True
    n_signals_old += 1

# Apply CB
position_old.loc[danger_zone, :] = False

print(f"V19 signals (old regime 2025+): {n_signals_old}")

if n_signals_old > 0:
    report_old = sim(
        position_old,
        trade_at_price="open",
        fee_ratio=FEE_RATIO,
        position_limit=POSITION_LIMIT,
        stop_loss=STOP_LOSS,
        upload=False,
        name="監獄兔_V19_舊制對照",
    )
    stats_old = report_old.get_stats()
    trades_old = report_old.get_trades()

    print(f"\n  V19 舊制 (2025+):")
    print(f"    CAGR: {stats_old['cagr'] * 100:.2f}%")
    print(f"    MDD: {stats_old['max_drawdown'] * 100:.2f}%")
    print(f"    Win: {stats_old['win_ratio'] * 100:.2f}%")
    print(f"    Trades: {len(trades_old)}")

    trades_old.to_csv(RESULT_PATH / "v19_old_regime_trades.csv",
                      index=False, encoding="utf-8-sig")

print(f"\nOutput: {RESULT_PATH}")
