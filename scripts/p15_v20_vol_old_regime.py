"""V20 Final+Vol on OLD REGIME (2018-2026) for proper validation.

Old regime: 10-day disposal.
Entry: Day 1 (start_idx), Exit: end_idx - 1 (day before end).
Same filters: liquidity, gap, volatility ranking.
Compare: with vs without vol ranking.
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

RESULT_PATH = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p15_v20_vol")

# Parameters
ENTRY_OFFSET = 0       # Day 1
EXIT_OFFSET = -1       # Day before end
MIN_TURNOVER = 20_000_000
GAP_LOW, GAP_HIGH = -0.08, 0.04
STOP_LOSS = 0.99
POSITION_LIMIT = 0.20
FEE_RATIO = 1.425 / 1000 / 3
MAX_PER_DAY = 5

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
    (dis["announce"] < "2026-08-10")  # OLD REGIME ONLY
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
daily_ret = filtered_close.pct_change()
volatility_20d = daily_ret.rolling(20).std()

# Build signals
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
    if exec_idx >= n_days or exit_idx < exec_idx:
        continue

    signal_day = trading_days[signal_idx]
    exec_day = trading_days[exec_idx]

    try:
        prev_c = filtered_close.loc[signal_day, sym]
        e_open = open_p.loc[exec_day, sym]
        avg_to = avg_turnover_5d.loc[signal_day, sym]
        current_vol = volatility_20d.loc[signal_day, sym]
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
        "volatility": current_vol if not pd.isna(current_vol) else 999,
    })

print(f"Total signals (pass filters): {len(signal_list)}")

# Circuit breaker
panic_count = (close.pct_change() < -0.06).sum(axis=1)
is_panic = panic_count > 400
danger_zone = is_panic.rolling(9, min_periods=1).max() > 0

# === Run TWO configs: with and without vol ranking ===
configs = [
    ("V20 Final (no vol rank)", False),
    ("V20 Final+Vol (low vol first)", True),
]

results = {}
for name, use_vol_rank in configs:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    signal_df = pd.DataFrame(signal_list)

    if use_vol_rank:
        # Sort by day then LOWEST volatility first
        signal_df = signal_df.sort_values(["exec_day", "volatility"])
        signal_df["rank_in_day"] = signal_df.groupby("exec_day").cumcount()
        selected = signal_df[signal_df["rank_in_day"] < MAX_PER_DAY]
    else:
        # No ranking: take all (finlab position_limit will handle capacity)
        selected = signal_df.copy()

    # Build position
    position = pd.DataFrame(False, index=close.index, columns=close.columns)
    for _, sig in selected.iterrows():
        position.loc[sig["exec_day"]:sig["exit_day"], sig["sym"]] = True

    # Apply circuit breaker
    position.loc[danger_zone, :] = False

    # Backtest
    report = sim(position, trade_at_price="open", fee_ratio=FEE_RATIO,
                 position_limit=POSITION_LIMIT, stop_loss=STOP_LOSS,
                 upload=False, name=name)

    stats = report.get_stats()
    trades = report.get_trades()

    print(f"  CAGR: {stats['cagr'] * 100:.2f}%")
    print(f"  MDD: {stats['max_drawdown'] * 100:.2f}%")
    print(f"  Win Rate: {stats['win_ratio'] * 100:.2f}%")
    print(f"  Total Return: {stats['total_return'] * 100:.2f}%")
    print(f"  Trades: {len(trades)}")

    results[name] = {"stats": stats, "trades": trades}

    if len(trades) > 0:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        trades.to_csv(RESULT_PATH / f"old_regime_{safe_name}.csv",
                      index=False, encoding="utf-8-sig")

# === Comparison ===
print(f"\n{'='*60}")
print("  COMPARISON (Old Regime 2018-2026)")
print(f"{'='*60}")
print(f"  {'Config':<35} {'CAGR':>8} {'MDD':>8} {'Win':>6} {'Total':>8} {'Trades':>6}")
print(f"  {'-'*75}")
for name, r in results.items():
    s = r["stats"]
    print(f"  {name:<35} {s['cagr']*100:>7.1f}% {s['max_drawdown']*100:>7.1f}% "
          f"{s['win_ratio']*100:>5.1f}% {s['total_return']*100:>7.1f}% "
          f"{len(r['trades']):>6}")

# Yearly comparison
print(f"\n--- Yearly Win Rate ---")
for name, r in results.items():
    trades = r["trades"].copy()
    if len(trades) == 0:
        continue
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["year"] = trades["entry_date"].dt.year
    yearly = trades.groupby("year").agg(
        n=("return", "count"),
        win=("return", lambda x: (x > 0).mean()),
        mean_ret=("return", "mean"),
    ).reset_index()
    print(f"\n  {name}:")
    print(f"    {'Year':<6} {'n':>5} {'Win':>6} {'Mean':>8}")
    for _, row in yearly.iterrows():
        print(f"    {int(row['year']):<6} {int(row['n']):>5} {row['win']:>6.1%} {row['mean_ret']:>8.2%}")
