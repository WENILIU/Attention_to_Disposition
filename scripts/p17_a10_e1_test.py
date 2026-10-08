"""P17: Test A10 (institutional) and E1 (intraday) factors for V20.

Data available until 2023-12-31 (free tier).
Tests IC within V20 filtered universe (Day 1 entry, old regime 2018-2023).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from finlab import data
import finlab
from pathlib import Path as P

TOKEN_FILE = P(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\auth_token.txt")
if TOKEN_FILE.exists():
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        finlab.login(f.read().strip())

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p17_a10_e1")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003

def main():
    # Load base data
    print("Loading data...")
    dis_raw = data.get("disposal_information")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    amount = data.get("price:成交金額")

    # Load A10: institutional investors
    print("Loading A10 institutional data...")
    # Get exact field names from search
    import finlab.data as fd
    inst_results = fd.search("institutional_investors_trading_summary:", details=True)
    stock_level = inst_results[
        inst_results["dataset"].str.contains("institutional_investors_trading_summary:") &
        ~inst_results["dataset"].str.contains("futures") &
        ~inst_results["dataset"].str.contains("all_market")
    ]

    # Get foreign net buy (外資)
    foreign_keys = stock_level[stock_level["dataset"].str.contains("外資")]
    trust_keys = stock_level[stock_level["dataset"].str.contains("投信")]
    dealer_keys = stock_level[stock_level["dataset"].str.contains("自營商")]

    # Use the first available for each
    foreign_key = foreign_keys.iloc[0]["dataset"] if len(foreign_keys) > 0 else None
    trust_key = trust_keys.iloc[0]["dataset"] if len(trust_keys) > 0 else None
    dealer_key = dealer_keys.iloc[0]["dataset"] if len(dealer_keys) > 0 else None

    foreign_net = data.get(foreign_key) if foreign_key else None
    trust_net = data.get(trust_key) if trust_key else None
    dealer_net = data.get(dealer_key) if dealer_key else None

    print(f"  Foreign: {foreign_net.shape if foreign_net is not None else 'N/A'}")
    print(f"  Trust: {trust_net.shape if trust_net is not None else 'N/A'}")
    print(f"  Dealer: {dealer_net.shape if dealer_net is not None else 'N/A'}")

    # Load E1: intraday (當沖)
    print("Loading E1 intraday data...")
    intraday_results = fd.search("intraday_trading:", details=True)
    # Get 當沖買進成交股數
    intraday_buy_key = intraday_results.iloc[0]["dataset"]
    intraday_buy = data.get(intraday_buy_key)
    print(f"  Intraday buy: {intraday_buy.shape}")

    # Also try to get 當沖占比
    ratio_results = fd.search("intraday_trading:", details=True)
    ratio_keys = ratio_results[ratio_results["dataset"].str.contains("占比")]
    if len(ratio_keys) > 0:
        intraday_ratio = data.get(ratio_keys.iloc[0]["dataset"])
        print(f"  Intraday ratio: {intraday_ratio.shape}")
    else:
        intraday_ratio = None

    # Build disposal events
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
        (dis["announce"] < "2024-01-01")  # Limited by free data cutoff
    ].copy()

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1
    dis = dis[(dis["start_idx"] >= 2) & (dis["end_idx"] >= 0) &
              (dis["end_idx"] < n_cal) & (dis["duration"] >= 5)].copy()

    # Compute indicators
    all_stocks = list(set(dis["stock_id"]) & valid_stocks)
    filtered_close = close[all_stocks].astype(np.float32)
    filtered_vol = vol[all_stocks].astype(np.float32)
    turnover = filtered_close * filtered_vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Build events with factors
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

        # Apply V20 filters
        if not (-0.08 < gap < 0.04):
            continue
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            continue

        # A10: Institutional net buy (5 days before disposal)
        a10_foreign = np.nan
        a10_trust = np.nan
        a10_dealer = np.nan
        a10_total = np.nan

        if foreign_net is not None and sym in foreign_net.columns:
            try:
                a10_foreign = foreign_net.iloc[max(0,si-5):si][sym].sum()
            except (IndexError, KeyError):
                pass
        if trust_net is not None and sym in trust_net.columns:
            try:
                a10_trust = trust_net.iloc[max(0,si-5):si][sym].sum()
            except (IndexError, KeyError):
                pass
        if dealer_net is not None and sym in dealer_net.columns:
            try:
                a10_dealer = dealer_net.iloc[max(0,si-5):si][sym].sum()
            except (IndexError, KeyError):
                pass

        if not (np.isnan(a10_foreign) and np.isnan(a10_trust) and np.isnan(a10_dealer)):
            a10_total = (0 if np.isnan(a10_foreign) else a10_foreign) + \
                        (0 if np.isnan(a10_trust) else a10_trust) + \
                        (0 if np.isnan(a10_dealer) else a10_dealer)

        # E1: Intraday ratio (5 days before)
        e1_intraday = np.nan
        if intraday_ratio is not None and sym in intraday_ratio.columns:
            try:
                e1_intraday = intraday_ratio.iloc[max(0,si-5):si][sym].mean()
            except (IndexError, KeyError):
                pass
        elif intraday_buy is not None and sym in intraday_buy.columns:
            try:
                ib = intraday_buy.iloc[max(0,si-5):si][sym].sum()
                total_vol = vol.iloc[max(0,si-5):si][sym].sum() if sym in vol.columns else np.nan
                e1_intraday = ib / total_vol if not np.isnan(total_vol) and total_vol > 0 else np.nan
            except (IndexError, KeyError):
                pass

        events.append({
            "symbol": sym,
            "entry_date": cal[si],
            "year": row["announce"].year,
            "net_ret": net_ret,
            "a10_foreign": a10_foreign,
            "a10_trust": a10_trust,
            "a10_dealer": a10_dealer,
            "a10_total": a10_total,
            "e1_intraday": e1_intraday,
        })

    ev = pd.DataFrame(events)
    print(f"Events (V20 filtered): {len(ev):,}")

    # === IC TEST ===
    print("\n" + "=" * 70)
    print("FACTOR IC (A10 + E1) WITHIN V20 UNIVERSE")
    print("=" * 70)

    factors = {
        "A10 外資5日買超": "a10_foreign",
        "A10 投信5日買超": "a10_trust",
        "A10 自營商5日買超": "a10_dealer",
        "A10 三大法人合計": "a10_total",
        "E1 當沖占比": "e1_intraday",
    }

    for name, col in factors.items():
        valid = ev[col].notna() & ev["net_ret"].notna()
        n = valid.sum()
        if n < 50:
            print(f"  {name}: insufficient (n={n})")
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
            vd["q"] = pd.qcut(vd[col], 5, labels=[1,2,3,4,5], duplicates="drop")
        except ValueError:
            continue

        print(f"\n  {name} (IC={ic:.4f}):")
        for q, grp in vd.groupby("q", observed=True):
            if len(grp) < 20:
                continue
            r = grp["net_ret"]
            print(f"    Q{int(q)}: n={len(grp)}, mean={r.mean():.2%}, win={(r>0).mean():.1%}")

    ev.to_csv(OUT / "p17_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
