"""Add tradability flags to the a00 v2 attention master table.

For each attention event, compute whether the NEXT trading day is:
  - one_word_limit_up (一字漲停): open == high == low == close at upper limit
  - one_word_limit_down (一字跌停): open == high == low == close at lower limit
  - low_volume: 成交股數 below a configurable threshold

Output: a00b_attention_master_with_tradability.csv (same rows + new columns)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

SRC = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\a00_attention_master_v2\a00_v2_attention_master.csv"
)
OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\a00b_tradability"
)
OUT.mkdir(parents=True, exist_ok=True)

MIN_VOLUME_SHARES = 5000  # below this, mark as low_volume


def main():
    df = pd.read_csv(
        SRC, dtype={"stock_key": str, "raw_id": str, "stock_id": str},
        encoding="utf-8-sig", low_memory=False,
    )
    df["attention_date"] = pd.to_datetime(df["attention_date"])
    print(f"Master table: {len(df):,} rows, "
          f"{df['stock_key'].nunique():,} stocks")

    # Get price data
    open_p = data.get("price:開盤價")
    high_p = data.get("price:最高價")
    low_p = data.get("price:最低價")
    close_p = data.get("price:收盤價")
    vol_p = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close_p.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # For each eligible event, look at the NEXT trading day (t_idx + 1)
    eligible = df["eligible_risk_set"].astype(str).str.lower().eq("true")
    sub = df.loc[eligible].copy()
    sub = sub.reset_index(drop=True)
    sub["t_idx"] = sub["t_idx"].astype(int)
    sub["next_idx"] = sub["t_idx"] + 1
    sub["next_date"] = np.where(
        sub["next_idx"] < n_cal,
        cal[sub["next_idx"].clip(upper=n_cal - 1).to_numpy()],
        pd.NaT,
    )
    sub["next_date"] = pd.to_datetime(sub["next_date"])

    # Fetch next-day prices for each event
    # Build lookup: for each (stock_key, next_date), get OHLCV
    # Use vectorized approach: map stock_key to column, next_date to row
    stock_cols = {k: k for k in sub["stock_key"].unique() if k in close_p.columns}
    missing_stocks = set(sub["stock_key"]) - set(stock_cols)
    if missing_stocks:
        print(f"Stocks not in price data: {len(missing_stocks)} "
              f"(e.g. {list(missing_stocks)[:5]})")

    # Initialize flags
    n = len(sub)
    next_open = np.full(n, np.nan)
    next_high = np.full(n, np.nan)
    next_low = np.full(n, np.nan)
    next_close = np.full(n, np.nan)
    next_vol = np.full(n, np.nan)
    prev_close = np.full(n, np.nan)

    # Process by stock to avoid memory issues
    for key, grp in sub.groupby("stock_key", sort=False):
        if key not in stock_cols:
            continue
        idx = grp.index.to_numpy()
        dates = grp["next_date"].to_numpy()
        valid = ~pd.isna(dates)
        if not valid.any():
            continue
        # Get prices for next day
        valid_dates = pd.DatetimeIndex(dates[valid])
        row_idx = cal.get_indexer(valid_dates)
        ok = row_idx >= 0
        if not ok.any():
            continue
        actual_idx = idx[valid][ok]
        actual_rows = row_idx[ok]
        col = key
        next_open[actual_idx] = open_p.iloc[actual_rows][col].to_numpy()
        next_high[actual_idx] = high_p.iloc[actual_rows][col].to_numpy()
        next_low[actual_idx] = low_p.iloc[actual_rows][col].to_numpy()
        next_close[actual_idx] = close_p.iloc[actual_rows][col].to_numpy()
        next_vol[actual_idx] = vol_p.iloc[actual_rows][col].to_numpy()
        # Previous close (the attention day itself)
        prev_rows = grp.loc[ok, "t_idx"].to_numpy()
        prev_close[actual_idx] = close_p.iloc[prev_rows][col].to_numpy()

    sub["next_open"] = next_open
    sub["next_high"] = next_high
    sub["next_low"] = next_low
    sub["next_close"] = next_close
    sub["next_volume"] = next_vol
    sub["prev_close"] = prev_close

    # Compute price limits (注意股 uses ±10%, 處置股 uses ±5%)
    # Since these are attention stocks (not yet in disposal), use ±10%
    upper_limit = prev_close * 1.10
    lower_limit = prev_close * 0.90

    # One-word limit up: open == high == low == close AND close >= upper_limit * 0.999
    has_price = ~np.isnan(next_open) & ~np.isnan(prev_close) & (prev_close > 0)
    tol = 0.001  # floating point tolerance

    sub["one_word_limit_up"] = (
        has_price
        & (np.abs(next_open - next_high) < tol)
        & (np.abs(next_high - next_low) < tol)
        & (np.abs(next_low - next_close) < tol)
        & (next_close >= upper_limit * (1 - tol))
    )
    sub["one_word_limit_down"] = (
        has_price
        & (np.abs(next_open - next_high) < tol)
        & (np.abs(next_high - next_low) < tol)
        & (np.abs(next_low - next_close) < tol)
        & (next_close <= lower_limit * (1 + tol))
    )
    sub["low_volume"] = (
        ~np.isnan(next_vol) & (next_vol < MIN_VOLUME_SHARES)
    )
    sub["no_price_data"] = np.isnan(next_close) & has_price  # stock exists but no data
    sub["not_tradable"] = (
        sub["one_word_limit_up"] | sub["one_word_limit_down"]
        | sub["low_volume"] | sub["no_price_data"]
    )
    sub["can_buy"] = has_price & ~sub["one_word_limit_up"] & ~sub["low_volume"] & ~sub["no_price_data"]
    sub["can_sell"] = has_price & ~sub["one_word_limit_down"] & ~sub["low_volume"] & ~sub["no_price_data"]

    # Summary
    total = len(sub)
    print(f"\n=== TRADABILITY SUMMARY (eligible events: {total:,}) ===")
    print(f"Has next-day price data: {int(has_price.sum()):,} ({has_price.mean():.1%})")
    print(f"One-word limit UP (cannot buy): {int(sub['one_word_limit_up'].sum()):,} "
          f"({sub['one_word_limit_up'].mean():.2%})")
    print(f"One-word limit DOWN (cannot sell): {int(sub['one_word_limit_down'].sum()):,} "
          f"({sub['one_word_limit_down'].mean():.2%})")
    print(f"Low volume (<{MIN_VOLUME_SHARES}): {int(sub['low_volume'].sum()):,} "
          f"({sub['low_volume'].mean():.2%})")
    print(f"Not tradable (any reason): {int(sub['not_tradable'].sum()):,} "
          f"({sub['not_tradable'].mean():.2%})")
    print(f"Can buy: {int(sub['can_buy'].sum()):,} ({sub['can_buy'].mean():.2%})")
    print(f"Can sell: {int(sub['can_sell'].sum()):,} ({sub['can_sell'].mean():.2%})")

    # Breakdown by armed state (from a03)
    # Merge with state table to get armed flag
    state_path = Path(
        r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
        r"\a01_clause_parser\a03_state_table.csv"
    )
    if state_path.exists():
        st = pd.read_csv(state_path, usecols=["stock_key", "date", "armed"],
                         dtype={"stock_key": str}, encoding="utf-8-sig")
        st["date"] = pd.to_datetime(st["date"]).dt.normalize()
        st["armed"] = st["armed"].astype(str).str.lower().eq("true")
        merged = sub.merge(
            st, left_on=["stock_key", "attention_date"],
            right_on=["stock_key", "date"], how="left",
        )
        armed_mask = merged["armed"].fillna(False).astype(bool)
        print(f"\n=== BY ARMED STATE ===")
        for label, mask in [("Armed", armed_mask), ("Not armed", ~armed_mask)]:
            g = merged.loc[mask]
            if len(g) == 0:
                continue
            print(f"{label}: n={len(g):,} "
                  f"limit_up={g['one_word_limit_up'].mean():.2%} "
                  f"not_tradable={g['not_tradable'].mean():.2%} "
                  f"can_buy={g['can_buy'].mean():.2%}")

    # Save
    out_cols = [c for c in df.columns] + [
        "next_date", "next_open", "next_high", "next_low", "next_close",
        "next_volume", "prev_close", "one_word_limit_up", "one_word_limit_down",
        "low_volume", "no_price_data", "not_tradable", "can_buy", "can_sell",
    ]
    # Merge back to full master
    full = df.merge(
        sub[["stock_key", "attention_date"] + [
            c for c in out_cols if c not in df.columns
        ]],
        on=["stock_key", "attention_date"], how="left",
    )
    full.to_csv(OUT / "a00b_attention_master_with_tradability.csv",
                index=False, encoding="utf-8-sig")

    # Save summary table
    summary = pd.DataFrame([
        {"metric": "total_eligible", "value": total},
        {"metric": "has_price_data", "value": int(has_price.sum())},
        {"metric": "one_word_limit_up", "value": int(sub["one_word_limit_up"].sum())},
        {"metric": "one_word_limit_down", "value": int(sub["one_word_limit_down"].sum())},
        {"metric": "low_volume", "value": int(sub["low_volume"].sum())},
        {"metric": "not_tradable", "value": int(sub["not_tradable"].sum())},
        {"metric": "can_buy", "value": int(sub["can_buy"].sum())},
        {"metric": "can_sell", "value": int(sub["can_sell"].sum())},
    ])
    summary.to_csv(OUT / "a00b_tradability_summary.csv",
                   index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
