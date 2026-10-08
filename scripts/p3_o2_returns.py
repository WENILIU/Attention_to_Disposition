"""P3 Step 1: Compute O2 returns for all attention events.

For each eligible attention event, compute:
  - Raw returns: 1/3/5/10 trading days from next-day open
  - Market-adjusted returns (subtract equal-weight market return)
  - Cost-adjusted returns (subtract 0.1425% commission + 0.3% tax)
  - Tradability layer assignment (all / tradable / not_tradable)

Output: p3_o2_returns.csv (one row per event with all return columns)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\a00b_tradability\a00b_attention_master_with_tradability.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_o2"
)
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 3, 5, 10)
COST_RATE = 0.001425 + 0.003  # commission + securities transaction tax
SLIPPAGE = 0.005  # 0.5% slippage for sensitivity


def main():
    df = pd.read_csv(
        SRC, dtype={"stock_key": str, "raw_id": str, "stock_id": str},
        encoding="utf-8-sig", low_memory=False,
    )
    df["attention_date"] = pd.to_datetime(df["attention_date"])
    eligible = df["eligible_risk_set"].astype(str).str.lower().eq("true")
    sub = df.loc[eligible].reset_index(drop=True).copy()
    sub["t_idx"] = sub["t_idx"].astype(int)
    print(f"Eligible events: {len(sub):,}")

    # Load prices
    open_p = data.get("price:開盤價")
    close_p = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Compute equal-weight market return (for market adjustment)
    # Use a simple approach: mean of all stock returns per day
    # To save memory, compute daily market return from close-to-close
    ret_close = close_p.pct_change(fill_method=None)
    mkt_ret = ret_close.mean(axis=1).values  # equal-weight daily market return
    print(f"Market return computed: {len(mkt_ret)} days")

    # For each event, compute forward returns from next-day open
    # Entry: next-day open (t_idx + 1)
    # Exit: close at t_idx + 1 + k - 1 (k days after entry)
    # Return = (exit_close / entry_open) - 1

    n = len(sub)
    entry_idx = sub["t_idx"].to_numpy() + 1  # next trading day

    # Initialize return arrays
    for k in HORIZONS:
        sub[f"raw_ret_{k}d"] = np.nan
        sub[f"mkt_adj_ret_{k}d"] = np.nan
        sub[f"net_ret_{k}d"] = np.nan  # after cost
        sub[f"net_ret_{k}d_slip"] = np.nan  # after cost + slippage

    # Process by stock for efficiency
    stocks_in_data = set(open_p.columns) & set(sub["stock_key"])
    print(f"Stocks with price data: {len(stocks_in_data):,}")

    for key in stocks_in_data:
        mask = (sub["stock_key"] == key).to_numpy()
        idx = np.where(mask)[0]
        e_idx = entry_idx[idx]
        valid = (e_idx >= 0) & (e_idx < n_cal)
        if not valid.any():
            continue
        valid_idx = idx[valid]
        valid_e = e_idx[valid]

        # Entry price: next-day open
        entry_prices = open_p.iloc[valid_e][key].to_numpy()
        valid_entry = ~np.isnan(entry_prices) & (entry_prices > 0)
        if not valid_entry.any():
            continue
        final_idx = valid_idx[valid_entry]
        final_e = valid_e[valid_entry]
        ep = entry_prices[valid_entry]

        for k in HORIZONS:
            exit_idx = final_e + k - 1  # k days from entry (entry is day 1)
            in_range = exit_idx < n_cal
            if not in_range.any():
                continue
            fi = final_idx[in_range]
            ei = exit_idx[in_range]
            exit_prices = close_p.iloc[ei][key].to_numpy()
            valid_exit = ~np.isnan(exit_prices) & (exit_prices > 0)
            if not valid_exit.any():
                continue
            fi2 = fi[valid_exit]
            ep2 = ep[in_range][valid_exit]
            xp = exit_prices[valid_exit]

            raw = (xp / ep2) - 1
            # Market-adjusted: subtract cumulative market return over same period
            mkt_cum = np.array([
                np.sum(mkt_ret[final_e[in_range][valid_exit][j]:
                              min(ei[j] + 1, n_cal)])
                for j in range(len(fi2))
            ])
            mkt_adj = raw - mkt_cum
            net = raw - COST_RATE
            net_slip = raw - COST_RATE - SLIPPAGE

            sub.loc[fi2, f"raw_ret_{k}d"] = raw
            sub.loc[fi2, f"mkt_adj_ret_{k}d"] = mkt_adj
            sub.loc[fi2, f"net_ret_{k}d"] = net
            sub.loc[fi2, f"net_ret_{k}d_slip"] = net_slip

    # Tradability layer
    sub["not_tradable"] = sub["not_tradable"].astype(str).str.lower().eq("true")
    sub["can_buy"] = sub["can_buy"].astype(str).str.lower().eq("true")
    sub["layer"] = np.where(
        ~sub["can_buy"], "not_tradable", "tradable"
    )

    # Summary
    print(f"\n=== O2 RETURNS SUMMARY ===")
    for k in HORIZONS:
        col = f"raw_ret_{k}d"
        net_col = f"net_ret_{k}d"
        has = sub[col].notna()
        tradable = sub["layer"] == "tradable"
        print(f"\n{k}d horizon: {int(has.sum()):,} events with returns")
        print(f"  All: mean={sub.loc[has, col].mean():.4%} "
              f"median={sub.loc[has, col].median():.4%}")
        print(f"  Net (after cost): mean={sub.loc[has, net_col].mean():.4%}")
        t_mask = has & tradable
        if t_mask.any():
            print(f"  Tradable only: mean={sub.loc[t_mask, col].mean():.4%} "
                  f"net={sub.loc[t_mask, net_col].mean():.4%}")
        nt_mask = has & ~tradable
        if nt_mask.any():
            print(f"  Not tradable: mean={sub.loc[nt_mask, col].mean():.4%}")

    # Save
    out_cols = [
        "stock_key", "attention_date", "t_idx", "streak", "c10", "c30",
        "disposed_within_1d", "disposed_within_5d", "disposed_within_10d",
        "disposed_within_20d", "layer", "can_buy", "not_tradable",
        "one_word_limit_up", "one_word_limit_down", "low_volume",
    ]
    for k in HORIZONS:
        out_cols += [f"raw_ret_{k}d", f"mkt_adj_ret_{k}d",
                     f"net_ret_{k}d", f"net_ret_{k}d_slip"]
    # Include any other useful columns
    for c in ["注意交易資訊", "收盤價"]:
        if c in sub.columns:
            out_cols.append(c)

    sub[out_cols].to_csv(OUT / "p3_o2_returns.csv",
                         index=False, encoding="utf-8-sig")

    # Save summary stats
    summary_rows = []
    for k in HORIZONS:
        for layer in ["all", "tradable", "not_tradable"]:
            mask = sub[f"raw_ret_{k}d"].notna()
            if layer == "tradable":
                mask &= (sub["layer"] == "tradable")
            elif layer == "not_tradable":
                mask &= (sub["layer"] == "not_tradable")
            if not mask.any():
                continue
            g = sub.loc[mask]
            summary_rows.append({
                "horizon": k, "layer": layer,
                "n": int(mask.sum()),
                "mean_raw": g[f"raw_ret_{k}d"].mean(),
                "median_raw": g[f"raw_ret_{k}d"].median(),
                "mean_net": g[f"net_ret_{k}d"].mean(),
                "mean_net_slip": g[f"net_ret_{k}d_slip"].mean(),
                "win_rate_net": (g[f"net_ret_{k}d"] > 0).mean(),
                "p5": g[f"net_ret_{k}d"].quantile(0.05),
                "p95": g[f"net_ret_{k}d"].quantile(0.95),
            })
    pd.DataFrame(summary_rows).to_csv(
        OUT / "p3_o2_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
