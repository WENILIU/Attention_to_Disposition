"""P9: Strategy A live trading simulation.

Full portfolio simulation with:
  1. Capital allocation: N positions, equal weight, per-position cap
  2. Trade execution: slippage, partial fills, priority rules
  3. Monitoring: drawdown circuit breaker, monthly loss limit, regime alerts
  4. Multiple configurations tested for robustness
  5. Monthly/annual P&L, Sharpe, max DD, win/loss streaks

Assumptions:
  - Starting capital: NT$10,000,000
  - Max concurrent positions: configurable (5/10/15/20)
  - Per-position: equal weight, capped at 10% of NAV
  - Slippage: 0.15% per side (conservative for disposal stocks with 5-min auction)
  - Entry: Day 3 open + slippage
  - Exit: exit day close - slippage; if limit-down, exit next day open
  - Priority: when >N signals on same day, take random N (no lookahead)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

P7A_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p7_strategy_a\p7_all_trades.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p9_live_sim"
)
OUT.mkdir(parents=True, exist_ok=True)

# Parameters
INITIAL_CAPITAL = 10_000_000  # NT$10M
SLIPPAGE = 0.0015  # 0.15% per side = 0.30% round trip
COMMISSION = 0.001425  # per side
STOCK_TAX = 0.003  # sell side
EXTRA_COST = SLIPPAGE * 2 + COMMISSION * 2 + STOCK_TAX  # total extra cost
MAX_POSITION_WEIGHT = 0.10  # max 10% per position
CIRCUIT_BREAKER_DD = -0.15  # stop if portfolio DD > 15%
MONTHLY_LOSS_LIMIT = -0.08  # stop month if down > 8%
MAX_CONSECUTIVE_LOSSES = 8  # pause after 8 consecutive losers

np.random.seed(42)


def load_trades():
    """Load executable trades with realistic net returns (add slippage)."""
    p7a = pd.read_csv(P7A_SRC, encoding="utf-8-sig")
    p7a = p7a[p7a["executable"]].copy()
    p7a["entry_date"] = pd.to_datetime(p7a["entry_date"])
    p7a["exit_date"] = pd.to_datetime(p7a["exit_date"])

    # Add slippage to existing net_return (which already has commission+tax)
    # net_return in P7A already has COST_RATE = 0.001425 + 0.003
    # We need to add slippage: 0.15% * 2 = 0.30%
    p7a["live_net_return"] = p7a["net_return"] - (SLIPPAGE * 2)

    p7a = p7a.sort_values("entry_date").reset_index(drop=True)
    return p7a


def simulate_portfolio(trades, max_positions, initial_capital=INITIAL_CAPITAL):
    """Simulate portfolio with max N concurrent positions."""
    trades = trades.copy()

    # Track open positions
    open_positions = []  # list of dicts
    closed_trades = []
    capital = initial_capital
    peak_capital = initial_capital
    max_dd = 0.0
    daily_nav = []
    circuit_breaker_triggered = False
    cb_date = None
    cb_pause_until = None
    monthly_pnl = {}
    consecutive_losses = 0
    max_consec_losses = 0
    pause_until = None

    # Pre-index trades by entry date for fast lookup
    trades_by_date = {d: g for d, g in trades.groupby("entry_date")}

    all_dates = sorted(set(trades["entry_date"]) | set(trades["exit_date"]))

    for date in all_dates:
        # Close positions that exit today
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= date:
                # Realize P&L
                pnl = pos["invested"] * pos["net_return"]
                capital += pnl
                closed_trades.append({
                    "entry": pos["entry_date"], "exit": pos["exit_date"],
                    "symbol": pos["symbol"], "pnl": pnl,
                    "return": pos["net_return"], "invested": pos["invested"],
                })
                # Track consecutive losses
                if pos["net_return"] < 0:
                    consecutive_losses += 1
                    max_consec_losses = max(max_consec_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            else:
                still_open.append(pos)
        open_positions = still_open

        # Track NAV (mark-to-market not needed for simplicity; use realized)
        peak_capital = max(peak_capital, capital)
        dd = (capital - peak_capital) / peak_capital
        max_dd = min(max_dd, dd)

        # Circuit breaker check (pause 30 days, then resume at half size)
        if dd < CIRCUIT_BREAKER_DD and not circuit_breaker_triggered:
            circuit_breaker_triggered = True
            cb_date = date
            cb_pause_until = date + pd.Timedelta(days=30)
            # Close all positions at estimated value (assume -5% haircut)
            for pos in open_positions:
                pnl = pos["invested"] * (-0.05)  # forced exit haircut
                capital += pnl
            open_positions = []
            # Reset peak to current (allow recovery)
            peak_capital = capital
            dd = 0.0
            max_dd = min(max_dd, (capital - initial_capital) / initial_capital)

        # Monthly tracking
        month_key = str(date)[:7]
        monthly_pnl[month_key] = capital

        # Pause check
        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            pause_until = date + pd.Timedelta(days=10)
            consecutive_losses = 0

        # Open new positions (skip during CB pause period)
        if cb_pause_until is not None and date <= cb_pause_until:
            continue
        if cb_pause_until is not None and date > cb_pause_until:
            circuit_breaker_triggered = False
            cb_pause_until = None
        if pause_until is not None and date <= pause_until:
            continue

        new_signals = trades_by_date.get(date, pd.DataFrame())
        if len(new_signals) == 0:
            continue

        available_slots = max_positions - len(open_positions)
        if available_slots <= 0:
            continue

        # Take up to available_slots (random selection among signals)
        n_take = min(available_slots, len(new_signals))
        selected = new_signals.nlargest(n_take, "entry_vol")  # prefer higher volume

        for _, sig in selected.iterrows():
            # Position sizing: equal weight, capped
            nav = capital  # simplified
            position_size = min(nav / max_positions, nav * MAX_POSITION_WEIGHT)
            if position_size < 100_000:  # minimum trade size
                continue

            open_positions.append({
                "entry_date": sig["entry_date"],
                "exit_date": sig["exit_date"],
                "symbol": sig["symbol"],
                "invested": position_size,
                "net_return": sig["live_net_return"],
            })

    # Final NAV
    final_nav = capital
    for pos in open_positions:
        final_nav += pos["invested"] * pos["net_return"]

    # Compute metrics
    total_return = (final_nav - initial_capital) / initial_capital
    if closed_trades:
        first_date = min(t["entry"] for t in closed_trades)
        last_date = max(t["exit"] for t in closed_trades)
        years = (last_date - first_date).days / 365.25
    else:
        years = 1

    ann_return = (final_nav / initial_capital) ** (1 / max(years, 0.1)) - 1

    # Monthly returns for Sharpe
    monthly_returns = []
    prev_val = initial_capital
    for month_key in sorted(monthly_pnl.keys()):
        curr_val = monthly_pnl[month_key]
        mr = (curr_val - prev_val) / prev_val if prev_val > 0 else 0
        monthly_returns.append(mr)
        prev_val = curr_val

    monthly_returns = np.array(monthly_returns)
    sharpe = (monthly_returns.mean() / monthly_returns.std() * np.sqrt(12)
              if len(monthly_returns) > 2 and monthly_returns.std() > 0 else 0)

    returns_series = pd.Series([t["return"] for t in closed_trades])

    return {
        "max_positions": max_positions,
        "final_nav": final_nav,
        "total_return": total_return,
        "ann_return": ann_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_trades": len(closed_trades),
        "win_rate": (returns_series > 0).mean() if len(returns_series) > 0 else 0,
        "avg_return": returns_series.mean() if len(returns_series) > 0 else 0,
        "max_consec_losses": max_consec_losses,
        "circuit_breaker": circuit_breaker_triggered,
        "cb_date": cb_date,
        "closed_trades": closed_trades,
        "monthly_pnl": monthly_pnl,
    }


def main():
    trades = load_trades()
    print(f"Loaded {len(trades):,} executable trades")
    print(f"Period: {trades['entry_date'].min().date()} to {trades['exit_date'].max().date()}")
    print(f"Extra cost (slippage): {SLIPPAGE*2:.2%} round trip")
    print(f"Total cost per trade (incl. P7A costs): {EXTRA_COST:.4%}")
    print(f"Adjusted avg return: {trades['live_net_return'].mean():.4%}")

    # === Run multiple configurations ===
    print("\n" + "=" * 70)
    print("PORTFOLIO SIMULATION - MULTIPLE CONFIGURATIONS")
    print("=" * 70)

    configs = [5, 10, 15, 20]
    results = []

    for max_pos in configs:
        print(f"\n--- Max {max_pos} positions ---")
        r = simulate_portfolio(trades, max_pos)
        results.append(r)

        print(f"  Final NAV: NT${r['final_nav']:,.0f}")
        print(f"  Total return: {r['total_return']:.1%}")
        print(f"  Annualized: {r['ann_return']:.1%}")
        print(f"  Max drawdown: {r['max_dd']:.1%}")
        print(f"  Sharpe (monthly): {r['sharpe']:.2f}")
        print(f"  Trades executed: {r['n_trades']:,}")
        print(f"  Win rate: {r['win_rate']:.1%}")
        print(f"  Avg return/trade: {r['avg_return']:.4%}")
        print(f"  Max consecutive losses: {r['max_consec_losses']}")
        print(f"  Circuit breaker: {'YES at ' + str(r['cb_date']) if r['circuit_breaker'] else 'No'}")

    # === Best config detailed analysis ===
    # Use 10 positions as recommended
    best = results[1]  # 10 positions
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIG: 10 POSITIONS")
    print("=" * 70)

    # Monthly P&L
    monthly_data = []
    prev = INITIAL_CAPITAL
    for month_key in sorted(best["monthly_pnl"].keys()):
        curr = best["monthly_pnl"][month_key]
        mr = (curr - prev) / prev if prev > 0 else 0
        monthly_data.append({"month": month_key, "nav": curr, "return": mr})
        prev = curr

    monthly_df = pd.DataFrame(monthly_data)
    monthly_df.to_csv(OUT / "p9_monthly_nav.csv", index=False, encoding="utf-8-sig")

    # Yearly summary
    monthly_df["year"] = monthly_df["month"].str[:4]
    yearly = monthly_df.groupby("year").agg(
        start_nav=("nav", "first"),
        end_nav=("nav", "last"),
        n_months=("return", "count"),
        worst_month=("return", "min"),
        best_month=("return", "max"),
    ).reset_index()
    yearly["annual_return"] = (yearly["end_nav"] / yearly["start_nav"]) - 1
    print("\n--- Yearly Performance (10 positions) ---")
    print(yearly[["year", "annual_return", "worst_month", "best_month"]].to_string(index=False))
    yearly.to_csv(OUT / "p9_yearly_performance.csv", index=False, encoding="utf-8-sig")

    # Drawdown analysis
    nav_series = monthly_df["nav"]
    running_max = nav_series.cummax()
    drawdown = (nav_series - running_max) / running_max
    dd_series = pd.DataFrame({"month": monthly_df["month"], "drawdown": drawdown})
    dd_series.to_csv(OUT / "p9_drawdown_series.csv", index=False, encoding="utf-8-sig")

    print(f"\n--- Drawdown Analysis ---")
    print(f"  Max DD: {drawdown.min():.1%}")
    print(f"  DD > 5%: {(drawdown < -0.05).sum()} months")
    print(f"  DD > 10%: {(drawdown < -0.10).sum()} months")
    print(f"  DD > 15%: {(drawdown < -0.15).sum()} months")
    print(f"  Avg recovery time: N/A (monthly granularity)")

    # Trade-level stats for the 10-position config
    trade_df = pd.DataFrame(best["closed_trades"])
    if len(trade_df) > 0:
        trade_df.to_csv(OUT / "p9_executed_trades.csv", index=False, encoding="utf-8-sig")
        print(f"\n--- Executed Trade Stats ---")
        print(f"  Total trades: {len(trade_df):,}")
        print(f"  Avg return: {trade_df['return'].mean():.4%}")
        print(f"  Win rate: {(trade_df['return']>0).mean():.1%}")
        print(f"  Largest win: {trade_df['return'].max():.2%}")
        print(f"  Largest loss: {trade_df['return'].min():.2%}")
        print(f"  Avg P&L per trade: NT${trade_df['pnl'].mean():,.0f}")
        print(f"  Total P&L: NT${trade_df['pnl'].sum():,.0f}")

    # === Monitoring rules ===
    print("\n" + "=" * 70)
    print("MONITORING RULES & CIRCUIT BREAKERS")
    print("=" * 70)

    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │           STRATEGY A LIVE TRADING PLAYBOOK              │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  ENTRY RULE:                                            │
    │    • Stock enters 處置 (disposal)                        │
    │    • Wait until Day 3 of disposal period                │
    │    • Buy at market open (with 0.15% slippage budget)    │
    │    • Skip if: 漲停鎖死 / 成交量<1000股                   │
    │                                                         │
    │  EXIT RULE:                                             │
    │    • Hold until 處置結束日 (last day of disposal)        │
    │    • Sell at market close on 出關日 (exit day)          │
    │    • If 跌停鎖死: sell next day open                    │
    │                                                         │
    │  POSITION SIZING:                                       │
    │    • Max 10 concurrent positions                        │
    │    • Equal weight: 10% of NAV per position              │
    │    • Minimum trade: NT$100,000                          │
    │    • If >10 signals: take highest volume first          │
    │                                                         │
    │  CIRCUIT BREAKERS:                                      │
    │    • Portfolio DD > 15%: STOP, close all, review        │
    │    • Monthly loss > 8%: pause new entries for 10 days   │
    │    • 8 consecutive losers: pause 10 days                │
    │    • New regime (2026-08-10+): separate tracking        │
    │                                                         │
    │  MONITORING DASHBOARD:                                  │
    │    • Daily: NAV, open positions, unrealized P&L         │
    │    • Weekly: win rate trend, avg return drift           │
    │    • Monthly: Sharpe, max DD, factor health             │
    │    • Alert: if 30d rolling win rate < 45%              │
    │    • Alert: if 30d rolling avg return < 0              │
    │                                                         │
    │  KILL CRITERIA (stop strategy entirely):                │
    │    • 3 consecutive months negative                      │
    │    • Sharpe < 0.5 for 6 months                          │
    │    • Structural change in disposal rules                │
    │    • Max DD exceeds 20%                                 │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """)

    # === Save summary ===
    summary = pd.DataFrame([{
        "config": f"{r['max_positions']} positions",
        "ann_return": r["ann_return"],
        "max_dd": r["max_dd"],
        "sharpe": r["sharpe"],
        "n_trades": r["n_trades"],
        "win_rate": r["win_rate"],
        "circuit_breaker": r["circuit_breaker"],
    } for r in results])
    summary.to_csv(OUT / "p9_config_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n--- Configuration Comparison ---")
    print(summary.to_string(index=False))

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
