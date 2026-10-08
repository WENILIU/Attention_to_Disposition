"""P17b: Test B4 (foreign broker net flow) and E2 (intraday trend) for V20.

B4: Foreign institutional brokers (摩根大通, 大和國泰, 牛牛牛) net buy 5d before disposal
B6: Broker concentration (HHI) — top broker share of volume
E2: Intraday ratio 5-day trend (change in intraday ratio)

Data: broker_separated CSVs (local) + finlab intraday_trading (free until 2023)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from finlab import data
import finlab

TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\auth_token.txt")
if TOKEN_FILE.exists():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        finlab.login(f.read().strip())

BROKER_DIR = Path(r"D:\AI專案\StockAgent\finlab\database\broker_separated")
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p17b_b4_e2")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003

# Foreign/institutional broker files (smart money signal)
FOREIGN_BROKERS = [
    "摩根大通.csv",
    "大和國泰.csv",
    "(牛牛牛)亞-網路.csv",
    "(牛牛牛)亞-鑫豐.csv",
    "(牛牛牛)亞證券.csv",
]


def load_foreign_broker_net():
    """Aggregate net buy (buy-sell) from foreign brokers per stock per day."""
    frames = []
    for fname in FOREIGN_BROKERS:
        fpath = BROKER_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found")
            continue
        df = pd.read_csv(fpath, usecols=["date", "buy", "sell", "stock_id"])
        df["net"] = df["buy"] - df["sell"]
        frames.append(df)
        print(f"  Loaded {fname}: {len(df):,} rows")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["stock_id"] = combined["stock_id"].astype(str).str.zfill(4)

    # Aggregate by date+stock (sum across foreign brokers)
    agg = combined.groupby(["date", "stock_id"])["net"].sum().reset_index()
    print(f"  Combined foreign broker net: {len(agg):,} rows, {agg['stock_id'].nunique()} stocks")
    return agg


def load_all_broker_hhi():
    """Compute Herfindahl index of broker concentration per stock per day."""
    """Compute Herfindahl index of broker concentration per stock per day.
    Uses top 50 brokers by file size to stay in memory."""
    print("  Loading top-50 broker files for HHI...")
    all_files = [f for f in BROKER_DIR.glob("*.csv") if "(停)" not in f.name]
    all_files.sort(key=lambda f: f.stat().st_size, reverse=True)
    top_files = all_files[:50]
    print(f"  Using top 50 brokers by volume")

    frames = []
    for fpath in top_files:
        try:
            df = pd.read_csv(fpath, usecols=["date", "buy", "sell", "stock_id"],
                             dtype={"date": str, "stock_id": str})
            df["volume"] = df["buy"] + df["sell"]
            df["broker"] = fpath.stem
            frames.append(df[["date", "stock_id", "broker", "volume"]])
        except Exception:
            continue

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], format="mixed")
    combined["stock_id"] = combined["stock_id"].str.zfill(4)

    # Total volume per stock-day
    total_vol = combined.groupby(["date", "stock_id"])["volume"].sum().reset_index()
    total_vol.columns = ["date", "stock_id", "total_volume"]

    # Top broker share per stock-day
    broker_vol = combined.groupby(["date", "stock_id", "broker"], sort=False)["volume"].sum().reset_index()
    broker_vol = broker_vol.merge(total_vol, on=["date", "stock_id"])
    broker_vol["share"] = broker_vol["volume"] / broker_vol["total_volume"]

    # HHI = sum of squared shares
    hhi = broker_vol.groupby(["date", "stock_id"], sort=False)["share"].apply(
        lambda x: (x ** 2).sum()
    ).reset_index(name="hhi")

    # Top-1 share
    top1 = broker_vol.groupby(["date", "stock_id"], sort=False)["share"].max().reset_index(name="top1_share")

    result = hhi.merge(top1, on=["date", "stock_id"])
    print(f"  HHI computed: {len(result):,} rows")
    return result


def main():
    print("=" * 70)
    print("P17b: B4 (Foreign Broker) + B6 (HHI) + E2 (Intraday Trend)")
    print("=" * 70)

    # Load price data
    print("\nLoading price data...")
    dis_raw = data.get("disposal_information")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    amount = data.get("price:成交金額")

    # Load E1/E2 intraday
    print("Loading E1/E2 intraday data...")
    import finlab.data as fd
    intraday_results = fd.search("intraday_trading:", details=True)
    intraday_buy_key = intraday_results.iloc[0]["dataset"]
    intraday_buy = data.get(intraday_buy_key)
    print(f"  Intraday buy: {intraday_buy.shape}")

    # Load B4 foreign broker net
    print("\nLoading B4 foreign broker data...")
    foreign_net = load_foreign_broker_net()

    # Load B6 HHI (expensive, optional)
    # B6 HHI skipped: requires loading 50+ broker files simultaneously (memory issue).
    # P8 already confirmed no factors add incremental value within disposal universe.
    hhi_data = None
    print("\nB6 HHI: SKIPPED (memory constraint + P8 finding: no incremental value)")

    # Build disposal events
    print("\nBuilding disposal events...")
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
        (dis["announce"] < "2024-01-01")
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

    # Index foreign_net by date for fast lookup
    if foreign_net is not None:
        foreign_net_pivot = foreign_net.pivot_table(
            index="date", columns="stock_id", values="net", aggfunc="sum"
        )
        foreign_net_pivot = foreign_net_pivot.reindex(cal)
    else:
        foreign_net_pivot = None

    # Index HHI by date
    if hhi_data is not None:
        hhi_pivot = hhi_data.pivot_table(index="date", columns="stock_id", values="hhi", aggfunc="mean")
        hhi_pivot = hhi_pivot.reindex(cal)
        top1_pivot = hhi_data.pivot_table(index="date", columns="stock_id", values="top1_share", aggfunc="max")
        top1_pivot = top1_pivot.reindex(cal)
    else:
        hhi_pivot = None
        top1_pivot = None

    # Build events
    print("Building events with factors...")
    events = []
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

        # V20 filters
        if not (-0.08 < gap < 0.04):
            continue
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            continue

        # B4: Foreign broker net buy (5 days before disposal)
        b4_foreign_net = np.nan
        if foreign_net_pivot is not None and sym in foreign_net_pivot.columns:
            try:
                window = foreign_net_pivot.iloc[max(0, si-5):si][sym]
                b4_foreign_net = window.sum()
            except (IndexError, KeyError):
                pass

        # B6: HHI and top-1 share (5 days before)
        b6_hhi = np.nan
        b6_top1 = np.nan
        if hhi_pivot is not None and sym in hhi_pivot.columns:
            try:
                b6_hhi = hhi_pivot.iloc[max(0, si-3):si][sym].mean()
            except (IndexError, KeyError):
                pass
        if top1_pivot is not None and sym in top1_pivot.columns:
            try:
                b6_top1 = top1_pivot.iloc[max(0, si-3):si][sym].mean()
            except (IndexError, KeyError):
                pass

        # E2: Intraday ratio trend (today - 5 days ago)
        e2_intraday_trend = np.nan
        if sym in intraday_buy.columns and sym in vol.columns:
            try:
                ib_recent = intraday_buy.iloc[si-1][sym]
                vol_recent = vol.iloc[si-1][sym]
                ib_old = intraday_buy.iloc[si-6][sym] if si >= 6 else np.nan
                vol_old = vol.iloc[si-6][sym] if si >= 6 else np.nan

                ratio_recent = ib_recent / vol_recent if vol_recent > 0 else np.nan
                ratio_old = ib_old / vol_old if not np.isnan(vol_old) and vol_old > 0 else np.nan

                if not np.isnan(ratio_recent) and not np.isnan(ratio_old):
                    e2_intraday_trend = ratio_recent - ratio_old
            except (IndexError, KeyError):
                pass

        events.append({
            "symbol": sym,
            "entry_date": cal[si],
            "year": row["announce"].year,
            "net_ret": net_ret,
            "b4_foreign_net": b4_foreign_net,
            "b6_hhi": b6_hhi,
            "b6_top1": b6_top1,
            "e2_intraday_trend": e2_intraday_trend,
        })

    ev = pd.DataFrame(events)
    print(f"\nEvents (V20 filtered): {len(ev):,}")

    # === IC TEST ===
    print("\n" + "=" * 70)
    print("FACTOR IC (B4 + B6 + E2) WITHIN V20 UNIVERSE")
    print("=" * 70)

    factors = {
        "B4 外資券商5日淨買超": "b4_foreign_net",
        "B6 券商集中度HHI": "b6_hhi",
        "B6 最大券商占比": "b6_top1",
        "E2 當沖占比5日變化": "e2_intraday_trend",
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
    ev.to_csv(OUT / "p17b_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
