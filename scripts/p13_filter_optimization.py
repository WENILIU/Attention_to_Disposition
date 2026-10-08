"""P13: Filter threshold optimization for V20.

Scans different thresholds for:
  1. Liquidity (min turnover): 5M, 10M, 20M, 30M, 50M
  2. Gap filter: various ranges
  3. Bias filter (MA20 deviation): 30%, 40%, 50%, 60%, 70%, 80%, no filter
  4. Stop loss: 8%, 10%, 12%, 15%, 20%, none

Uses old regime data (2018+) for optimization (larger sample),
then validates on new regime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p13_filter_opt"
)
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003  # commission + tax + slippage


def main():
    # Load all data
    print("Loading data...")
    dis_raw = data.get("disposal_information")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    high_p = data.get("price:最高價")
    low_p = data.get("price:最低價")
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
        (dis["announce"] >= "2018-01-01")
    ].copy()

    # Compute trading day indices
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1

    # Filter valid (duration >= 5 for old regime)
    dis = dis[(dis["start_idx"] >= 0) & (dis["end_idx"] >= 0) &
              (dis["end_idx"] < n_cal) & (dis["duration"] >= 5)].copy()

    # Compute MA20 and turnover
    print("Computing indicators...")
    all_stocks = list(set(dis["stock_id"]) & valid_stocks)
    filtered_close = close[all_stocks].astype(np.float32)
    filtered_vol = vol[all_stocks].astype(np.float32)
    turnover = filtered_close * filtered_vol
    avg_turnover_5d = turnover.rolling(5).mean()
    ma20 = filtered_close.rolling(20).mean()

    # Build per-event data with filter values
    print("Building event features...")
    events = []
    for _, row in dis.iterrows():
        sym = row["stock_id"]
        if sym not in all_stocks:
            continue
        si = row["start_idx"]
        ei = row["end_idx"]
        exit_idx = ei + 1
        if exit_idx >= n_cal:
            continue

        # Entry: Day 1 (for new regime optimization)
        entry_idx = si
        # Exit: end_idx - 1 (day before end, V19 style)
        exit_trade_idx = ei - 1
        if exit_trade_idx < entry_idx:
            continue

        try:
            entry_open = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[entry_idx - 1][sym] if entry_idx > 0 else np.nan
            exit_close = close.iloc[exit_trade_idx][sym]
            exit_open = open_p.iloc[exit_trade_idx][sym]
            signal_close = close.iloc[entry_idx][sym]
            current_ma20 = ma20.iloc[entry_idx][sym]
            avg_to = avg_turnover_5d.iloc[entry_idx][sym]

            # Max adverse during holding
            hold_low = low_p.iloc[entry_idx:exit_trade_idx + 1][sym]
            min_price = hold_low.min()
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_open, exit_close, exit_open]):
            continue

        # Compute returns
        gross_ret = exit_open / entry_open - 1  # sell at open of exit day
        net_ret = gross_ret - COST_RATE

        # Gap
        gap = (entry_open / prev_close - 1) if not np.isnan(prev_close) and prev_close > 0 else np.nan

        # Bias
        bias = (signal_close / current_ma20 - 1) if not np.isnan(current_ma20) and current_ma20 > 0 else np.nan

        # MAE (max adverse excursion)
        mae = (min_price / entry_open - 1) if not np.isnan(min_price) and min_price > 0 else np.nan

        events.append({
            "symbol": sym,
            "entry_date": cal[entry_idx],
            "year": row["announce"].year,
            "duration": row["duration"],
            "condition": row.get("處置條件", ""),
            "net_ret": net_ret,
            "gross_ret": gross_ret,
            "gap": gap,
            "bias": bias,
            "avg_turnover": avg_to,
            "mae": mae,
        })

    ev = pd.DataFrame(events)
    ev["entry_date"] = pd.to_datetime(ev["entry_date"])
    print(f"Events with features: {len(ev):,}")

    # === OPTIMIZATION ===
    print("\n" + "=" * 70)
    print("FILTER THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # Baseline (no filters)
    baseline = ev["net_ret"]
    print(f"\nBaseline (no filters): n={len(ev):,} mean={baseline.mean():.4%} "
          f"win={(baseline>0).mean():.1%} std={baseline.std():.4%}")

    # --- 1. Liquidity threshold ---
    print("\n--- 1. LIQUIDITY THRESHOLD ---")
    print(f"  {'Threshold':<12} {'n':>6} {'Mean':>8} {'Win':>6} {'Std':>8} {'Keep%':>6}")
    liq_results = []
    for threshold in [0, 5e6, 10e6, 20e6, 30e6, 50e6, 100e6]:
        mask = ev["avg_turnover"].isna() | (ev["avg_turnover"] >= threshold)
        sub = ev[mask]
        if len(sub) < 50:
            continue
        r = sub["net_ret"]
        liq_results.append({
            "threshold": threshold, "n": len(sub),
            "mean": r.mean(), "win": (r > 0).mean(),
            "std": r.std(), "keep_pct": len(sub) / len(ev),
        })
        label = f"{threshold/1e6:.0f}M" if threshold > 0 else "None"
        print(f"  {label:<12} {len(sub):>6} {r.mean():>8.2%} {(r>0).mean():>6.1%} "
              f"{r.std():>8.2%} {len(sub)/len(ev):>6.1%}")

    liq_df = pd.DataFrame(liq_results)
    liq_df.to_csv(OUT / "p13_liquidity_scan.csv", index=False, encoding="utf-8-sig")

    # --- 2. Gap filter ---
    print("\n--- 2. GAP FILTER ---")
    print(f"  {'Range':<16} {'n':>6} {'Mean':>8} {'Win':>6} {'Std':>8} {'Keep%':>6}")
    gap_results = []
    gap_configs = [
        (-999, 999, "None"),
        (-0.15, 0.10, "-15%~+10%"),
        (-0.10, 0.06, "-10%~+6%"),
        (-0.08, 0.04, "-8%~+4%"),
        (-0.06, 0.03, "-6%~+3%"),
        (-0.05, 0.02, "-5%~+2%"),
        (-0.04, 0.01, "-4%~+1%"),
    ]
    for low, high, label in gap_configs:
        mask = ev["gap"].isna() | ((ev["gap"] > low) & (ev["gap"] < high))
        sub = ev[mask]
        if len(sub) < 50:
            continue
        r = sub["net_ret"]
        gap_results.append({
            "config": label, "n": len(sub),
            "mean": r.mean(), "win": (r > 0).mean(),
            "std": r.std(), "keep_pct": len(sub) / len(ev),
        })
        print(f"  {label:<16} {len(sub):>6} {r.mean():>8.2%} {(r>0).mean():>6.1%} "
              f"{r.std():>8.2%} {len(sub)/len(ev):>6.1%}")

    gap_df = pd.DataFrame(gap_results)
    gap_df.to_csv(OUT / "p13_gap_scan.csv", index=False, encoding="utf-8-sig")

    # --- 3. Bias filter ---
    print("\n--- 3. BIAS FILTER (price vs MA20) ---")
    print(f"  {'Max Bias':<10} {'n':>6} {'Mean':>8} {'Win':>6} {'Std':>8} {'Keep%':>6}")
    bias_results = []
    for max_bias in [999, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
        mask = ev["bias"].isna() | (ev["bias"] < max_bias)
        sub = ev[mask]
        if len(sub) < 50:
            continue
        r = sub["net_ret"]
        label = "None" if max_bias >= 999 else f"<{max_bias:.0%}"
        bias_results.append({
            "max_bias": max_bias, "n": len(sub),
            "mean": r.mean(), "win": (r > 0).mean(),
            "std": r.std(), "keep_pct": len(sub) / len(ev),
        })
        print(f"  {label:<10} {len(sub):>6} {r.mean():>8.2%} {(r>0).mean():>6.1%} "
              f"{r.std():>8.2%} {len(sub)/len(ev):>6.1%}")

    bias_df = pd.DataFrame(bias_results)
    bias_df.to_csv(OUT / "p13_bias_scan.csv", index=False, encoding="utf-8-sig")

    # --- 4. Stop loss (MAE-based) ---
    print("\n--- 4. STOP LOSS (simulated via MAE) ---")
    print(f"  {'Stop Loss':<10} {'n':>6} {'Mean':>8} {'Win':>6} {'Std':>8} {'Triggered':>10}")
    sl_results = []
    for sl in [999, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05]:
        # Simulate: if MAE < -sl, assume stopped out at -sl
        # Otherwise, keep original return
        valid = ev["mae"].notna() & ev["net_ret"].notna()
        vd = ev[valid].copy()
        if sl >= 999:
            vd["sl_ret"] = vd["net_ret"]
            triggered = 0
        else:
            stop_mask = vd["mae"] < -sl
            vd["sl_ret"] = np.where(stop_mask, -sl - COST_RATE, vd["net_ret"])
            triggered = stop_mask.sum()

        r = vd["sl_ret"]
        label = "None" if sl >= 999 else f"-{sl:.0%}"
        sl_results.append({
            "stop_loss": sl, "n": len(vd),
            "mean": r.mean(), "win": (r > 0).mean(),
            "std": r.std(), "triggered": triggered,
            "trigger_pct": triggered / len(vd),
        })
        print(f"  {label:<10} {len(vd):>6} {r.mean():>8.2%} {(r>0).mean():>6.1%} "
              f"{r.std():>8.2%} {triggered:>6} ({triggered/len(vd):.1%})")

    sl_df = pd.DataFrame(sl_results)
    sl_df.to_csv(OUT / "p13_stoploss_scan.csv", index=False, encoding="utf-8-sig")

    # --- 5. Combined optimal ---
    print("\n" + "=" * 70)
    print("COMBINED OPTIMAL CONFIGURATION")
    print("=" * 70)

    # Test a few combined configs
    configs = [
        ("V19 原版", 20e6, -0.08, 0.04, 0.6, 0.12),
        ("放寬流動性", 10e6, -0.08, 0.04, 0.6, 0.12),
        ("收緊流動性", 50e6, -0.08, 0.04, 0.6, 0.12),
        ("放寬Gap", 20e6, -0.15, 0.10, 0.6, 0.12),
        ("收緊Gap", 20e6, -0.05, 0.02, 0.6, 0.12),
        ("無Bias", 20e6, -0.08, 0.04, 999, 0.12),
        ("收緊Bias", 20e6, -0.08, 0.04, 0.3, 0.12),
        ("無止損", 20e6, -0.08, 0.04, 0.6, 999),
        ("收緊止損", 20e6, -0.08, 0.04, 0.6, 0.08),
        ("全放寬", 10e6, -0.15, 0.10, 999, 999),
        ("全收緊", 50e6, -0.05, 0.02, 0.3, 0.08),
    ]

    print(f"\n  {'Config':<12} {'n':>6} {'Mean':>8} {'Win':>6} {'Std':>8} {'Keep%':>6}")
    print(f"  {'-'*55}")

    for name, liq, gap_lo, gap_hi, max_bias, sl in configs:
        mask = (
            (ev["avg_turnover"].isna() | (ev["avg_turnover"] >= liq)) &
            (ev["gap"].isna() | ((ev["gap"] > gap_lo) & (ev["gap"] < gap_hi))) &
            (ev["bias"].isna() | (ev["bias"] < max_bias))
        )
        sub = ev[mask].copy()
        if len(sub) < 50:
            continue

        # Apply stop loss
        if sl < 999:
            stop_mask = sub["mae"] < -sl
            sub["final_ret"] = np.where(stop_mask, -sl - COST_RATE, sub["net_ret"])
        else:
            sub["final_ret"] = sub["net_ret"]

        r = sub["final_ret"]
        print(f"  {name:<12} {len(sub):>6} {r.mean():>8.2%} {(r>0).mean():>6.1%} "
              f"{r.std():>8.2%} {len(sub)/len(ev):>6.1%}")

    # Yearly stability of best config
    print("\n--- Yearly stability: V19 原版 vs 全放寬 ---")
    for name, liq, gap_lo, gap_hi, max_bias, sl in [
        ("V19原版", 20e6, -0.08, 0.04, 0.6, 0.12),
        ("全放寬", 10e6, -0.15, 0.10, 999, 999),
    ]:
        mask = (
            (ev["avg_turnover"].isna() | (ev["avg_turnover"] >= liq)) &
            (ev["gap"].isna() | ((ev["gap"] > gap_lo) & (ev["gap"] < gap_hi))) &
            (ev["bias"].isna() | (ev["bias"] < max_bias))
        )
        sub = ev[mask].copy()
        if sl < 999:
            stop_mask = sub["mae"] < -sl
            sub["final_ret"] = np.where(stop_mask, -sl - COST_RATE, sub["net_ret"])
        else:
            sub["final_ret"] = sub["net_ret"]

        yearly = sub.groupby("year").agg(
            n=("final_ret", "count"),
            mean=("final_ret", "mean"),
            win=("final_ret", lambda x: (x > 0).mean()),
        ).reset_index()
        pos_years = (yearly["mean"] > 0).sum()
        print(f"\n  {name}: {pos_years}/{len(yearly)} years positive")
        print(f"  {'Year':<6} {'n':>5} {'Mean':>8} {'Win':>6}")
        for _, r in yearly.iterrows():
            print(f"  {int(r['year']):<6} {int(r['n']):>5} {r['mean']:>8.2%} {r['win']:>6.1%}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
