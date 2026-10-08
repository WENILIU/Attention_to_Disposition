"""P7: Strategy B simulation - attention-to-disposal short with realistic constraints.

Strategy B: Short high A4 strength + high momentum stocks at attention announcement.
Hold 3-5 days, forced cover at 5 days.

Constraints modeled:
  1. 融券 availability: quota > 0 AND not suspended (74% coverage)
  2. 融券 cost: 2.5%/year, prorated daily
  3. 5-day forced cover: exit at day 5 regardless
  4. Entry feasibility: can sell short (not at limit-down)
  5. Exit feasibility: can buy back (not at limit-up)
  6. Tail risk: short losses are unbounded on upside
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

ENRICHED_SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_followup\p3_enriched_features.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p7_strategy_b"
)
OUT.mkdir(parents=True, exist_ok=True)

# Cost parameters
COMMISSION = 0.001425  # one-way
STOCK_TAX = 0.003      # sell-side only (for short: charged on initial sell)
SECURITIES_LENDING_RATE = 0.025  # 2.5% annual
TRADING_DAYS_PER_YEAR = 250
FORCED_COVER_DAYS = 5  # must cover within 5 trading days
NEAR_LIMIT_THRESHOLD = 0.095


def main():
    # Load enriched features (A4 + A5 + returns)
    enriched = pd.read_csv(ENRICHED_SRC, dtype={"stock_key": str},
                           encoding="utf-8-sig", low_memory=False)
    enriched["attention_date"] = pd.to_datetime(enriched["attention_date"])
    enriched["year"] = enriched["attention_date"].dt.year

    # Filter test period (2023+)
    test = enriched[enriched["year"] >= 2023].copy()
    print(f"Test period events (2023+): {len(test):,}")

    # === Load 融券 availability data ===
    print("Loading 融券 data...")
    quota = data.get("margin_transactions:融券限額")
    suspended = data.get("margin_short_sell_mark:暫停融券賣出")
    quota_dates = pd.DatetimeIndex(quota.index).normalize().unique().sort_values()

    # === Load price data for feasibility checks ===
    close_p = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    high_p = data.get("price:最高價")
    low_p = data.get("price:最低價")
    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # === Build short signals ===
    # Signal: high A4 (margin_to_threshold) + high momentum (ret5) = bearish
    valid_signal = test["margin_to_threshold"].notna() & test["ret5"].notna()
    test = test[valid_signal].copy()
    test["short_score"] = (test["margin_to_threshold"].rank(pct=True) +
                           test["ret5"].rank(pct=True)) / 2
    test["short_q"] = pd.qcut(test["short_score"], 5, labels=[1,2,3,4,5],
                              duplicates="drop")

    # Q5 = most bearish = short candidates
    short_candidates = test[test["short_q"] == 5].copy()
    print(f"Short candidates (Q5): {len(short_candidates):,}")

    # === Per-event simulation ===
    results = []
    stocks_in_quota = set(quota.columns)
    stocks_in_close = set(close_p.columns)

    for _, row in short_candidates.iterrows():
        stock_key = row["stock_key"]
        sym = stock_key.zfill(4)
        att_date = row["attention_date"]

        # Entry: next trading day after attention announcement
        entry_idx = cal.searchsorted(att_date, side="right")
        if entry_idx >= n_cal:
            continue

        # Exit: min(entry + FORCED_COVER_DAYS, available)
        exit_idx = min(entry_idx + FORCED_COVER_DAYS, n_cal - 1)
        actual_hold = exit_idx - entry_idx

        if actual_hold < 1:
            continue

        # === Check 融券 availability ===
        can_short = False
        quota_val = np.nan
        is_suspended = False

        q_idx = quota_dates.searchsorted(att_date, side="left")
        if q_idx < len(quota_dates):
            if sym in stocks_in_quota:
                try:
                    quota_val = quota.iloc[q_idx][sym]
                    if not np.isnan(quota_val) and quota_val > 0:
                        can_short = True
                except (IndexError, KeyError):
                    pass
            if sym in set(suspended.columns):
                try:
                    susp_val = suspended.iloc[q_idx][sym]
                    if str(susp_val).lower() == "true":
                        can_short = False
                        is_suspended = True
                except (IndexError, KeyError):
                    pass

        # === Check price feasibility ===
        if sym not in stocks_in_close:
            continue

        try:
            entry_open = open_p.iloc[entry_idx][sym]
            entry_high = high_p.iloc[entry_idx][sym]
            entry_low = low_p.iloc[entry_idx][sym]
            prev_close = close_p.iloc[entry_idx - 1][sym]
            exit_close = close_p.iloc[exit_idx][sym]
            exit_open = open_p.iloc[exit_idx][sym]
            exit_high = high_p.iloc[exit_idx][sym]
            prev_exit_close = close_p.iloc[exit_idx - 1][sym]

            # Max price during holding (worst case for short)
            hold_slice = high_p.iloc[entry_idx:exit_idx + 1][sym]
            max_price = hold_slice.max()
        except (IndexError, KeyError):
            continue

        critical = [entry_open, prev_close, exit_close, prev_exit_close]
        if any(np.isnan(x) or x <= 0 for x in critical):
            continue

        # Can we short on entry? (not stuck at limit-down)
        near_limit_down = (entry_open / prev_close - 1) <= -NEAR_LIMIT_THRESHOLD
        stuck_at_limit_down = near_limit_down and (entry_low >= entry_open * 0.999)
        can_enter_short = not stuck_at_limit_down

        # Can we buy back on exit? (not stuck at limit-up)
        near_limit_up = (exit_close / prev_exit_close - 1) >= NEAR_LIMIT_THRESHOLD
        stuck_at_limit_up = near_limit_up and (exit_high <= exit_close * 1.001)
        can_exit_short = not stuck_at_limit_up

        # === Return calculation (short perspective) ===
        # Short profit = sell high, buy low = -(price change)
        gross_short_return = -(exit_close / entry_open - 1)

        # Costs:
        # 1. Commission: both ways
        commission_cost = COMMISSION * 2
        # 2. Stock tax: 0.3% on sell (short sell counts as sell)
        tax_cost = STOCK_TAX
        # 3. Securities lending fee: 2.5%/year * days/250
        lending_cost = SECURITIES_LENDING_RATE * actual_hold / TRADING_DAYS_PER_YEAR

        total_cost = commission_cost + tax_cost + lending_cost
        net_short_return = gross_short_return - total_cost

        # Max adverse excursion (worst point during holding for short)
        if not np.isnan(max_price) and max_price > 0:
            max_adverse = -(max_price / entry_open - 1)  # negative = adverse
        else:
            max_adverse = np.nan

        # Was forced cover binding? (would have wanted to hold longer)
        # We can check if 3d return was better than 5d
        try:
            close_3d_idx = min(entry_idx + 3, n_cal - 1)
            close_3d = close_p.iloc[close_3d_idx][sym]
            short_3d = -(close_3d / entry_open - 1) if not np.isnan(close_3d) else np.nan
        except (IndexError, KeyError):
            short_3d = np.nan

        executable = can_short and can_enter_short and can_exit_short

        results.append({
            "symbol": sym,
            "stock_key": stock_key,
            "attention_date": att_date,
            "entry_date": cal[entry_idx],
            "exit_date": cal[exit_idx],
            "year": row["year"],
            "short_score": row["short_score"],
            "holding_days": actual_hold,
            "can_short": can_short,
            "is_suspended": is_suspended,
            "can_enter_short": can_enter_short,
            "can_exit_short": can_exit_short,
            "executable": executable,
            "gross_short_return": gross_short_return,
            "lending_cost": lending_cost,
            "total_cost": total_cost,
            "net_short_return": net_short_return,
            "max_adverse": max_adverse,
            "short_3d": short_3d,
            "near_limit_down": near_limit_down,
            "near_limit_up": near_limit_up,
        })

    res = pd.DataFrame(results)
    res.to_csv(OUT / "p7b_all_trades.csv", index=False, encoding="utf-8-sig")

    # === Analysis ===
    print("\n" + "=" * 70)
    print("P7 STRATEGY B: SHORT SIMULATION (with 融券 constraints)")
    print("=" * 70)

    # 1. Feasibility funnel
    print("\n--- 1. FEASIBILITY FUNNEL ---")
    n_total = len(res)
    n_can_short = res["can_short"].sum()
    n_can_enter = res["can_enter_short"].sum()
    n_can_exit = res["can_exit_short"].sum()
    n_exec = res["executable"].sum()
    print(f"  Short candidates (Q5): {n_total:,}")
    print(f"  融券 available (quota>0, not suspended): {n_can_short:,} ({n_can_short/n_total:.1%})")
    print(f"  Can enter (not stuck at limit-down): {n_can_enter:,} ({n_can_enter/n_total:.1%})")
    print(f"  Can exit (not stuck at limit-up): {n_can_exit:,} ({n_can_exit/n_total:.1%})")
    print(f"  FULLY EXECUTABLE: {n_exec:,} ({n_exec/n_total:.1%})")

    # 2. Returns comparison
    print("\n--- 2. RETURNS ---")
    exec_t = res[res["executable"]].copy()
    non_exec = res[~res["executable"]].copy()

    print(f"  All candidates avg gross short: {res['gross_short_return'].mean():.4%}")
    print(f"  Executable avg gross short: {exec_t['gross_short_return'].mean():.4%}")
    print(f"  Non-executable avg gross short: {non_exec['gross_short_return'].mean():.4%}")

    net = exec_t["net_short_return"]
    print(f"\n  NET SHORT RETURN (after all costs):")
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

    # 3. Cost breakdown
    print("\n--- 3. COST BREAKDOWN ---")
    print(f"  Commission (round-trip): {COMMISSION*2:.4%}")
    print(f"  Stock tax (sell side): {STOCK_TAX:.4%}")
    print(f"  Lending cost (5 days): {SECURITIES_LENDING_RATE * 5 / TRADING_DAYS_PER_YEAR:.4%}")
    print(f"  Total cost per trade: {exec_t['total_cost'].mean():.4%}")
    print(f"  Gross alpha before costs: {exec_t['gross_short_return'].mean():.4%}")
    print(f"  Net alpha after costs: {net.mean():.4%}")
    print(f"  Cost drag: {exec_t['total_cost'].mean():.4%}")

    # 4. Tail risk (short-specific: losses when price goes UP)
    print("\n--- 4. TAIL RISK (SHORT-SPECIFIC) ---")
    for threshold in [-0.05, -0.10, -0.15, -0.20, -0.30]:
        pct = (net < threshold).mean()
        n = (net < threshold).sum()
        print(f"  Loss > {abs(threshold):.0%}: {n:,} trades ({pct:.1%})")

    adverse = exec_t["max_adverse"].dropna()
    print(f"\n  Max adverse excursion during holding:")
    print(f"    Mean: {adverse.mean():.4%}")
    print(f"    Median: {adverse.median():.4%}")
    print(f"    P5 (worst): {adverse.quantile(0.05):.4%}")
    print(f"    P1 (worst): {adverse.quantile(0.01):.4%}")
    print(f"    Min: {adverse.min():.4%}")

    # 5. Yearly stability
    print("\n--- 5. YEARLY STABILITY (executable only) ---")
    yearly = exec_t.groupby("year").agg(
        n=("net_short_return", "count"),
        mean_net=("net_short_return", "mean"),
        median_net=("net_short_return", "median"),
        win_rate=("net_short_return", lambda x: (x > 0).mean()),
        p5_net=("net_short_return", lambda x: x.quantile(0.05)),
    ).reset_index()
    yearly = yearly[yearly["n"] >= 10]
    print(yearly.to_string(index=False))
    yearly.to_csv(OUT / "p7b_yearly.csv", index=False, encoding="utf-8-sig")

    # 6. Forced cover impact
    print("\n--- 6. FORCED COVER IMPACT ---")
    # Compare 5d (forced) vs 3d (optimal?)
    both_valid = exec_t["net_short_return"].notna() & exec_t["short_3d"].notna()
    if both_valid.sum() > 50:
        vd = exec_t[both_valid]
        net_3d = vd["short_3d"] - (COMMISSION*2 + STOCK_TAX + SECURITIES_LENDING_RATE*3/TRADING_DAYS_PER_YEAR)
        net_5d = vd["net_short_return"]
        print(f"  3d hold (net): mean={net_3d.mean():.4%} win={(net_3d>0).mean():.1%}")
        print(f"  5d hold (net): mean={net_5d.mean():.4%} win={(net_5d>0).mean():.1%}")
        print(f"  Forced cover cost: {net_5d.mean() - net_3d.mean():.4%}")
        print(f"  3d better in {(net_3d > net_5d).mean():.1%} of cases")

    # 7. Signal frequency & capital usage
    print("\n--- 7. SIGNAL FREQUENCY ---")
    exec_t["month"] = pd.to_datetime(exec_t["entry_date"]).dt.to_period("M")
    monthly = exec_t.groupby("month").size()
    print(f"  Monthly avg signals: {monthly.mean():.1f}")
    print(f"  Monthly median: {monthly.median():.0f}")
    print(f"  Max month: {monthly.max()}")

    # Concurrent positions
    exec_t["entry_dt"] = pd.to_datetime(exec_t["entry_date"])
    exec_t["exit_dt"] = pd.to_datetime(exec_t["exit_date"])
    all_dates = pd.date_range(exec_t["entry_dt"].min(), exec_t["exit_dt"].max(), freq="B")
    concurrent = []
    for d in all_dates:
        n = ((exec_t["entry_dt"] <= d) & (exec_t["exit_dt"] >= d)).sum()
        concurrent.append(n)
    concurrent = pd.Series(concurrent)
    print(f"  Mean concurrent positions: {concurrent.mean():.1f}")
    print(f"  Median: {concurrent.median():.0f}")
    print(f"  P95: {concurrent.quantile(0.95):.0f}")
    print(f"  Max: {concurrent.max()}")

    # 8. Capital efficiency
    print("\n--- 8. CAPITAL EFFICIENCY ---")
    avg_hold = exec_t["holding_days"].mean()
    avg_ret = net.mean()
    ret_per_day = avg_ret / avg_hold
    print(f"  Avg holding: {avg_hold:.1f} days")
    print(f"  Avg net return: {avg_ret:.4%}")
    print(f"  Return per capital-day: {ret_per_day:.5%}")

    # Compare with Strategy A
    print(f"\n  Strategy A comparison:")
    print(f"    A: +4.76%/trade, 9 days, +0.53%/day")
    print(f"    B: {avg_ret:.2%}/trade, {avg_hold:.0f} days, {ret_per_day:.3%}/day")

    # 9. Portfolio simulation
    print("\n--- 9. PORTFOLIO SIMULATION (sequential) ---")
    trades_sorted = exec_t.sort_values("entry_date").reset_index(drop=True)
    capital = 1.0
    peak = 1.0
    max_dd = 0.0
    current_exit = pd.Timestamp("2000-01-01")
    n_trades = 0

    for _, t in trades_sorted.iterrows():
        if t["entry_dt"] <= current_exit:
            continue
        capital *= (1 + t["net_short_return"])
        peak = max(peak, capital)
        dd_now = (capital - peak) / peak
        max_dd = min(max_dd, dd_now)
        current_exit = t["exit_dt"]
        n_trades += 1

    total_ret = capital - 1
    first_entry = trades_sorted.iloc[0]["entry_dt"]
    last_exit = trades_sorted.iloc[-1]["exit_dt"]
    years = (last_exit - first_entry).days / 365.25
    ann_growth = capital ** (1/years) - 1 if years > 0 else np.nan

    print(f"  Sequential trades: {n_trades}")
    print(f"  Total return: {total_ret:.1%}")
    print(f"  Years: {years:.1f}")
    print(f"  Annualized: {ann_growth:.1%}")
    print(f"  Max drawdown: {max_dd:.1%}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
