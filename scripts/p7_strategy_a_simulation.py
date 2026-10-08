"""P7: Strategy A simulation - disposal period long with realistic constraints.

Strategy A: Buy on Day 3 of disposal, hold until exit day close.
Tests:
  1. Tradability: Can you actually buy on Day 3? (not at limit-up)
  2. Exit feasibility: Can you sell on exit day? (not at limit-down)
  3. Capital lockup: trading days from entry to exit
  4. Tail risk: full return distribution, worst cases
  5. Portfolio simulation: sequential capital deployment
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p7_strategy_a"
)
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003  # commission + tax
LIMIT_UP = 1.10   # +10% daily limit
LIMIT_DOWN = 0.90  # -10% daily limit
# Conservative: if open >= 9.5% above prev close, consider "hard to buy"
NEAR_LIMIT_THRESHOLD = 0.095


def main():
    # Load disposal events
    dis = pd.DataFrame(data.get("disposal_information"))
    dis["stock_key"] = dis["symbol"].astype(str).str.lstrip("0")
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis = dis[dis["symbol"].str.match(r"^\d{4}$")].copy()
    dis = dis[dis["announce"] >= "2018-01-01"].copy()
    dis["year"] = dis["announce"].dt.year
    dis["regime"] = np.where(dis["announce"] >= "2026-08-10", "new", "old")

    # Load prices
    close_p = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    high_p = data.get("price:最高價")
    low_p = data.get("price:最低價")
    vol_p = data.get("price:成交股數")
    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)

    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration_days"] = dis["end_idx"] - dis["start_idx"] + 1

    # Filter valid events (old regime only for main analysis)
    dis = dis[(dis["start_idx"] >= 0) & (dis["end_idx"] >= 0) &
              (dis["end_idx"] < n_cal) & (dis["duration_days"] >= 3)].copy()
    old = dis[dis["regime"] == "old"].copy()
    print(f"Old regime events: {len(old):,}")

    # === Per-event detailed simulation ===
    results = []
    stocks = set(close_p.columns)

    for _, row in old.iterrows():
        sym = row["symbol"]
        if sym not in stocks:
            continue
        si = row["start_idx"]
        ei = row["end_idx"]
        exit_idx = ei + 1
        if exit_idx >= n_cal:
            continue

        # Day 3 entry
        entry_idx = si + 2  # 0-indexed: day 1=si, day 2=si+1, day 3=si+2
        if entry_idx > ei:
            continue

        try:
            # Entry day prices
            entry_open = open_p.iloc[entry_idx][sym]
            entry_high = high_p.iloc[entry_idx][sym]
            entry_low = low_p.iloc[entry_idx][sym]
            entry_close = close_p.iloc[entry_idx][sym]
            entry_vol = vol_p.iloc[entry_idx][sym]

            # Previous day close (for limit calculation)
            prev_close = close_p.iloc[entry_idx - 1][sym]

            # Exit day prices
            exit_open = open_p.iloc[exit_idx][sym]
            exit_high = high_p.iloc[exit_idx][sym]
            exit_low = low_p.iloc[exit_idx][sym]
            exit_close = close_p.iloc[exit_idx][sym]
            exit_vol = vol_p.iloc[exit_idx][sym]

            # Day before exit (for exit limit check)
            prev_exit_close = close_p.iloc[exit_idx - 1][sym]

            # Minimum price during holding (for max drawdown)
            hold_slice = low_p.iloc[entry_idx:exit_idx + 1][sym]
            min_price = hold_slice.min()
        except (IndexError, KeyError):
            continue

        # Skip if any critical price is NaN
        critical_prices = [entry_open, prev_close, exit_close, prev_exit_close]
        if any(np.isnan(x) or x <= 0 for x in critical_prices):
            continue

        # === Tradability checks ===
        # Can we buy on Day 3?
        limit_up_price = prev_close * LIMIT_UP
        near_limit_up = (entry_open / prev_close - 1) >= NEAR_LIMIT_THRESHOLD
        # If open is at limit up AND high == open (stuck all day), truly unbuyable
        stuck_at_limit_up = near_limit_up and (entry_high <= entry_open * 1.001)
        can_buy = not stuck_at_limit_up

        # Can we sell on exit day?
        limit_down_price = prev_exit_close * LIMIT_DOWN
        near_limit_down = (exit_close / prev_exit_close - 1) <= -NEAR_LIMIT_THRESHOLD
        stuck_at_limit_down = near_limit_down and (exit_low >= exit_close * 0.999)
        can_sell = not stuck_at_limit_down

        # Volume check: need at least 1000 shares tradeable (minimum lot)
        # Use median volume as proxy for liquidity
        vol_ok = (not np.isnan(entry_vol)) and entry_vol >= 1000

        # === Return calculation ===
        gross_return = exit_close / entry_open - 1
        net_return = gross_return - COST_RATE

        # Capital lockup: trading days from entry to exit
        holding_days = exit_idx - entry_idx + 1  # inclusive

        # Max drawdown during holding (from entry price)
        if not np.isnan(min_price) and min_price > 0:
            max_dd = min_price / entry_open - 1
        else:
            max_dd = np.nan

        # Annualized return (for capital efficiency)
        if holding_days > 0 and net_return > -1:
            ann_return = (1 + net_return) ** (250 / holding_days) - 1
        else:
            ann_return = np.nan

        results.append({
            "symbol": sym,
            "entry_date": cal[entry_idx],
            "exit_date": cal[exit_idx],
            "year": row["year"],
            "condition": row.get("處置條件", ""),
            "duration": row["duration_days"],
            "holding_days": holding_days,
            "gross_return": gross_return,
            "net_return": net_return,
            "max_drawdown": max_dd,
            "ann_return": ann_return,
            "can_buy": can_buy,
            "can_sell": can_sell,
            "vol_ok": vol_ok,
            "executable": can_buy and can_sell and vol_ok,
            "entry_vol": entry_vol,
            "near_limit_up": near_limit_up,
            "near_limit_down": near_limit_down,
        })

    res = pd.DataFrame(results)
    res.to_csv(OUT / "p7_all_trades.csv", index=False, encoding="utf-8-sig")

    # === Analysis ===
    print("\n" + "=" * 70)
    print("P7 STRATEGY A: DISPOSAL PERIOD LONG SIMULATION")
    print("=" * 70)

    # 1. Tradability summary
    print("\n--- 1. TRADABILITY AUDIT ---")
    n_total = len(res)
    n_can_buy = res["can_buy"].sum()
    n_can_sell = res["can_sell"].sum()
    n_vol_ok = res["vol_ok"].sum()
    n_exec = res["executable"].sum()
    print(f"  Total events: {n_total:,}")
    print(f"  Can buy (not stuck at limit-up): {n_can_buy:,} ({n_can_buy/n_total:.1%})")
    print(f"  Can sell (not stuck at limit-down): {n_can_sell:,} ({n_can_sell/n_total:.1%})")
    print(f"  Volume OK (>=1000 shares): {n_vol_ok:,} ({n_vol_ok/n_total:.1%})")
    print(f"  Fully executable: {n_exec:,} ({n_exec/n_total:.1%})")
    print(f"  Near limit-up on entry: {res['near_limit_up'].sum():,} ({res['near_limit_up'].mean():.1%})")
    print(f"  Near limit-down on exit: {res['near_limit_down'].sum():,} ({res['near_limit_down'].mean():.1%})")

    # 2. Returns: executable vs non-executable
    print("\n--- 2. RETURNS (executable trades only) ---")
    exec_trades = res[res["executable"]].copy()
    non_exec = res[~res["executable"]].copy()
    print(f"  Executable: n={len(exec_trades):,}")
    print(f"  Non-executable: n={len(non_exec):,}")
    if len(non_exec) > 0:
        print(f"  Non-exec avg return: {non_exec['net_return'].mean():.4%}")
        print(f"  (If these were included, avg would be: {res['net_return'].mean():.4%})")

    net = exec_trades["net_return"]
    print(f"\n  Net return distribution:")
    print(f"    Mean: {net.mean():.4%}")
    print(f"    Median: {net.median():.4%}")
    print(f"    Std: {net.std():.4%}")
    print(f"    Win rate: {(net>0).mean():.1%}")
    print(f"    P1: {net.quantile(0.01):.4%}")
    print(f"    P5: {net.quantile(0.05):.4%}")
    print(f"    P10: {net.quantile(0.10):.4%}")
    print(f"    P25: {net.quantile(0.25):.4%}")
    print(f"    P75: {net.quantile(0.75):.4%}")
    print(f"    P90: {net.quantile(0.90):.4%}")
    print(f"    P95: {net.quantile(0.95):.4%}")
    print(f"    P99: {net.quantile(0.99):.4%}")
    print(f"    Min: {net.min():.4%}")
    print(f"    Max: {net.max():.4%}")

    # 3. Tail risk
    print("\n--- 3. TAIL RISK ---")
    for threshold in [-0.05, -0.10, -0.15, -0.20, -0.30]:
        pct = (net < threshold).mean()
        n = (net < threshold).sum()
        print(f"  Loss > {abs(threshold):.0%}: {n:,} trades ({pct:.1%})")

    # Max drawdown during holding
    dd = exec_trades["max_drawdown"].dropna()
    print(f"\n  Max drawdown during holding:")
    print(f"    Mean: {dd.mean():.4%}")
    print(f"    Median: {dd.median():.4%}")
    print(f"    P5 (worst): {dd.quantile(0.05):.4%}")
    print(f"    P1 (worst): {dd.quantile(0.01):.4%}")
    print(f"    Min: {dd.min():.4%}")

    # 4. Capital lockup
    print("\n--- 4. CAPITAL LOCKUP ---")
    hd = exec_trades["holding_days"]
    print(f"  Holding days distribution:")
    print(f"    Mean: {hd.mean():.1f} trading days")
    print(f"    Median: {hd.median():.0f} trading days")
    print(f"    P25: {hd.quantile(0.25):.0f}")
    print(f"    P75: {hd.quantile(0.75):.0f}")
    print(f"    P95: {hd.quantile(0.95):.0f}")
    print(f"    Max: {hd.max():.0f}")

    # Annualized return per trade
    ann = exec_trades["ann_return"].dropna()
    print(f"\n  Annualized return per trade:")
    print(f"    Median: {ann.median():.1%}")
    print(f"    Mean: {ann.mean():.1%}")

    # 5. Yearly stability (executable only)
    print("\n--- 5. YEARLY STABILITY (executable only) ---")
    yearly = exec_trades.groupby("year").agg(
        n=("net_return", "count"),
        mean_net=("net_return", "mean"),
        median_net=("net_return", "median"),
        win_rate=("net_return", lambda x: (x > 0).mean()),
        avg_holding=("holding_days", "mean"),
        p5_net=("net_return", lambda x: x.quantile(0.05)),
    ).reset_index()
    yearly = yearly[yearly["n"] >= 10]
    print(yearly.to_string(index=False))
    yearly.to_csv(OUT / "p7_yearly.csv", index=False, encoding="utf-8-sig")

    # 6. Portfolio simulation (sequential, 1 position at a time)
    print("\n--- 6. PORTFOLIO SIMULATION (sequential, 1 position) ---")
    trades_sorted = exec_trades.sort_values("entry_date").reset_index(drop=True)

    # Simulate: start with 1.0 capital, each trade uses 100% of capital
    capital = 1.0
    peak = 1.0
    max_dd_portfolio = 0.0
    trade_log = []
    current_exit = pd.Timestamp("2000-01-01")

    for _, t in trades_sorted.iterrows():
        if t["entry_date"] <= current_exit:
            continue  # still in previous trade
        capital *= (1 + t["net_return"])
        peak = max(peak, capital)
        dd_now = (capital - peak) / peak
        max_dd_portfolio = min(max_dd_portfolio, dd_now)
        current_exit = t["exit_date"]
        trade_log.append({
            "entry": t["entry_date"], "exit": t["exit_date"],
            "capital": capital, "return": t["net_return"],
        })

    n_trades_executed = len(trade_log)
    total_return = capital - 1
    # Annualized over the full period
    if trade_log:
        first_entry = pd.Timestamp(trade_log[0]["entry"])
        last_exit = pd.Timestamp(trade_log[-1]["exit"])
        years_elapsed = (last_exit - first_entry).days / 365.25
        ann_growth = (capital) ** (1 / years_elapsed) - 1 if years_elapsed > 0 else np.nan
    else:
        ann_growth = np.nan
        years_elapsed = 0

    print(f"  Sequential trades executed: {n_trades_executed:,}")
    print(f"  Total return: {total_return:.1%}")
    print(f"  Years elapsed: {years_elapsed:.1f}")
    print(f"  Annualized growth: {ann_growth:.1%}")
    print(f"  Max portfolio drawdown: {max_dd_portfolio:.1%}")
    print(f"  Avg return per trade: {exec_trades['net_return'].mean():.4%}")
    print(f"  Avg holding: {exec_trades['holding_days'].mean():.1f} days")
    print(f"  Capital utilization: {sum(exec_trades['holding_days']) / (years_elapsed * 250) * 100:.0f}%")

    # 7. Capital efficiency comparison
    print("\n--- 7. CAPITAL EFFICIENCY ---")
    avg_hold = exec_trades["holding_days"].mean()
    avg_ret = exec_trades["net_return"].mean()
    # If we could run N positions in parallel
    for n_pos in [1, 3, 5, 10]:
        # Rough estimate: annual return = (avg_ret / avg_hold) * 250 / n_overlap
        # More realistic: use the sequential simulation result
        pass
    # Simple metric: return per capital-day
    ret_per_day = avg_ret / avg_hold
    print(f"  Return per capital-day: {ret_per_day:.5%}")
    print(f"  If 1 position at a time: annualized ~{ann_growth:.1%}")
    print(f"  If 5 parallel positions: annualized ~{ann_growth/5:.1%} (conservative)")

    # Save portfolio log
    pd.DataFrame(trade_log).to_csv(OUT / "p7_portfolio_log.csv",
                                    index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
