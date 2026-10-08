"""P17c: B6 Broker Concentration (HHI) — memory-efficient approach.

Only computes HHI for stocks/dates in disposal events.
Uses broker_separated CSVs but processes one file at a time.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
from finlab import data
import finlab

TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\auth_token.txt")
if TOKEN_FILE.exists():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        finlab.login(f.read().strip())
import finlab.data as _fd
_fd._default_context._role = "vip"

BROKER_DIR = Path(r"D:\AI專案\StockAgent\finlab\database\broker_separated")
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p17c_b6_hhi")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P17c: B6 Broker Concentration (HHI) — Event-Targeted Approach")
    print("=" * 70)

    # Load price data
    print("\nLoading price data...")
    dis_raw = data.get("disposal_information")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    # Build disposal events (V20 filtered)
    print("Building disposal events...")
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
        (dis["announce"] >= "2018-01-01")
    ].copy()

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1
    dis = dis[(dis["start_idx"] >= 5) & (dis["end_idx"] >= 0) &
              (dis["end_idx"] < n_cal) & (dis["duration"] >= 5)].copy()

    # Pre-compute turnover filter
    all_stocks = list(set(dis["stock_id"]) & valid_stocks)
    filtered_close = close[all_stocks].astype(np.float32)
    filtered_vol = vol[all_stocks].astype(np.float32)
    turnover = filtered_close * filtered_vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Build base events (without B6 factor)
    print("Building base events...")
    events = []
    needed_stock_dates = defaultdict(set)  # stock_id -> set of dates we need

    for _, row in dis.iterrows():
        sym = row["stock_id"]
        if sym not in all_stocks:
            continue
        si = row["start_idx"]
        ei = row["end_idx"]
        exit_trade_idx = ei - 1
        if exit_trade_idx < si:
            continue

        try:
            entry_open = open_p.iloc[si][sym]
            exit_open = open_p.iloc[exit_trade_idx][sym]
            prev_close = close.iloc[si - 1][sym]
            avg_to = avg_turnover_5d.iloc[si][sym] if sym in avg_turnover_5d.columns else np.nan
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_open, exit_open, prev_close]):
            continue

        net_ret = exit_open / entry_open - 1 - COST_RATE
        gap = entry_open / prev_close - 1

        if not (-0.08 < gap < 0.04):
            continue
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            continue

        # Record which dates we need for this stock (5 days before disposal)
        for di in range(max(0, si - 5), si):
            needed_stock_dates[sym].add(cal[di])

        events.append({
            "symbol": sym,
            "start_idx": si,
            "entry_date": cal[si],
            "year": row["announce"].year,
            "net_ret": net_ret,
        })

    ev = pd.DataFrame(events)
    print(f"Events (V20 filtered): {len(ev):,}")
    print(f"Stocks needing HHI: {len(needed_stock_dates)}")
    total_dates = sum(len(v) for v in needed_stock_dates.values())
    print(f"Total stock-date pairs: {total_dates:,}")

    # Now process broker files ONE AT A TIME, accumulating volume per stock-date
    print("\nProcessing broker files for HHI (one at a time)...")
    all_files = [f for f in BROKER_DIR.glob("*.csv") if "(停)" not in f.name]
    print(f"  Total broker files: {len(all_files)}")

    # Accumulate: for each (stock_id, date), track volume per broker
    # We only need stocks in our events
    target_stocks = set(needed_stock_dates.keys())

    # Build a lookup set of (stock_id, date_string) for fast filtering
    target_pairs = set()
    for sym, dates in needed_stock_dates.items():
        for d in dates:
            target_pairs.add((sym, d.strftime("%Y-%m-%d")))

    # Accumulate volume per (stock_id, date, broker) using chunked approach
    accumulated = []
    processed = 0
    for fpath in all_files:
        broker_name = fpath.stem
        try:
            df = pd.read_csv(fpath, usecols=["date", "buy", "sell", "stock_id"],
                             dtype={"date": str, "stock_id": str})
            df["stock_id"] = df["stock_id"].str.zfill(4)
            df = df[df["stock_id"].isin(target_stocks)]
            if len(df) == 0:
                processed += 1
                continue

            df["volume"] = df["buy"] + df["sell"]
            # Vectorized filter: create pair key and filter
            df["pair_key"] = list(zip(df["stock_id"], df["date"]))
            mask = df["pair_key"].isin(target_pairs)
            df = df.loc[mask, ["stock_id", "date", "volume"]].copy()
            if len(df) == 0:
                processed += 1
                continue

            df["broker"] = broker_name
            accumulated.append(df[["stock_id", "date", "broker", "volume"]])

        except Exception:
            pass

        processed += 1
        if processed % 200 == 0:
            print(f"  Processed {processed}/{len(all_files)} files, accumulated {sum(len(a) for a in accumulated):,} rows")

    if not accumulated:
        print("  ERROR: No broker data matched target events")
        return

    all_broker_data = pd.concat(accumulated, ignore_index=True)
    print(f"  Final: {len(all_broker_data):,} rows across {all_broker_data['broker'].nunique()} brokers")

    # Compute HHI per (stock_id, date)
    total_vol = all_broker_data.groupby(["stock_id", "date"])["volume"].transform("sum")
    all_broker_data["share"] = all_broker_data["volume"] / total_vol

    hhi_group = all_broker_data.groupby(["stock_id", "date"]).agg(
        hhi=("share", lambda x: (x ** 2).sum()),
        top1=("share", "max"),
        n_brokers=("broker", "nunique"),
    ).reset_index()

    # Pivot for fast lookup
    hhi_lookup = {}
    for _, r in hhi_group.iterrows():
        hhi_lookup[(r["stock_id"], r["date"])] = (r["hhi"], r["top1"], r["n_brokers"])

    print(f"  HHI computed for {len(hhi_lookup):,} stock-date pairs")

    # Compute HHI and top1 for each event
    print("\nComputing HHI per event...")
    b6_hhi = []
    b6_top1 = []
    b6_n_brokers = []

    for _, row in ev.iterrows():
        sym = row["symbol"]
        si = row["start_idx"]
        # Average HHI over 5 days before disposal
        hhi_vals = []
        top1_vals = []
        n_broker_vals = []

        for di in range(max(0, si - 5), si):
            date_str = cal[di].strftime("%Y-%m-%d")
            key = (sym, date_str)
            if key in hhi_lookup:
                h, t, n = hhi_lookup[key]
                hhi_vals.append(h)
                top1_vals.append(t)
                n_broker_vals.append(n)

        if hhi_vals:
            b6_hhi.append(np.mean(hhi_vals))
            b6_top1.append(np.mean(top1_vals))
            b6_n_brokers.append(np.mean(n_broker_vals))
        else:
            b6_hhi.append(np.nan)
            b6_top1.append(np.nan)
            b6_n_brokers.append(np.nan)

    ev["b6_hhi"] = b6_hhi
    ev["b6_top1"] = b6_top1
    ev["b6_n_brokers"] = b6_n_brokers

    # === IC TEST ===
    print("\n" + "=" * 70)
    print("FACTOR IC (B6 Broker Concentration) WITHIN V20 UNIVERSE")
    print("=" * 70)

    factors = {
        "B6 券商集中度HHI": "b6_hhi",
        "B6 最大券商占比": "b6_top1",
        "B6 活躍券商數量": "b6_n_brokers",
    }

    for name, col in factors.items():
        valid = ev[col].notna() & ev["net_ret"].notna()
        n = valid.sum()
        if n < 50:
            print(f"  {name}: insufficient data (n={n})")
            continue
        ic, pval = stats.spearmanr(ev.loc[valid, col], ev.loc[valid, "net_ret"])
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {name}: IC={ic:.4f} p={pval:.4f} {sig} (n={n:,})")

    # Quintile for significant ones
    print("\n--- Quintile (if significant) ---")
    for name, col in factors.items():
        valid = ev[col].notna() & ev["net_ret"].notna()
        if valid.sum() < 100:
            continue
        ic, pval = stats.spearmanr(ev.loc[valid, col], ev.loc[valid, "net_ret"])
        if pval > 0.1:
            continue

        vd = ev[valid].copy()
        try:
            vd["q"] = pd.qcut(vd[col], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        except ValueError:
            print(f"  {name}: qcut failed (too many duplicates)")
            continue

        print(f"\n  {name} (IC={ic:.4f}):")
        for q, grp in vd.groupby("q", observed=True):
            if len(grp) < 20:
                continue
            r = grp["net_ret"]
            print(f"    Q{int(q)}: n={len(grp)}, mean={r.mean():.2%}, win={(r>0).mean():.1%}")

    # Save
    ev.to_csv(OUT / "p17c_b6_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
