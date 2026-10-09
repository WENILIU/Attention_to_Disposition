"""P20b: C-Long 參數掃描 — 找最佳組合.

Grid: threshold × consecutive × max_hold
Metric: portfolio CAGR, Sharpe, yearly pass rate
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p20_c_long_v3")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")
MAX_POSITIONS = 5


def simulate_portfolio(signals, cal, n_cal, max_positions=5):
    """Simulate portfolio with position limits. Returns metrics dict."""
    signals = signals.sort_values(["announce", "prob"], ascending=[True, False]).copy()
    signals["entry_idx"] = cal.searchsorted(signals["announce"], side="left") + 1

    active_positions = []
    executed = []
    for _, trade in signals.iterrows():
        entry_idx = int(trade["entry_idx"])
        exit_idx = min(int(trade["exit_idx"]), n_cal - 1)
        active_positions = [(ei, s) for ei, s in active_positions if ei > entry_idx]
        if len(active_positions) >= max_positions:
            continue
        if any(s == trade["stock_id"] for _, s in active_positions):
            continue
        active_positions.append((exit_idx, trade["stock_id"]))
        executed.append(trade)

    if len(executed) < 30:
        return None

    ex = pd.DataFrame(executed)

    # Daily P&L
    daily_returns = pd.Series(0.0, index=cal)
    cap = 1.0 / max_positions
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = min(int(t["exit_idx"]), n_cal - 1)
        hold = max(xi - ei + 1, 1)
        daily_ret = t["ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            daily_returns.iloc[d] += daily_ret * cap

    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0

    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    mdd = ((cum - running_max) / running_max).min()

    first_idx = int(ex["entry_idx"].min())
    total_days = n_cal - first_idx
    cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

    # Yearly
    yearly_pass = 0
    yearly_total = 0
    for yr in sorted(ex["year"].unique()):
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 5:
            continue
        yearly_total += 1
        if yr_data["ret"].mean() > 0:
            yearly_pass += 1

    return {
        "cagr": cagr, "sharpe": sharpe, "mdd": mdd,
        "avg_ret": ex["ret"].mean(), "win": (ex["ret"] > 0).mean(),
        "n_trades": len(ex), "yearly_pass": yearly_pass, "yearly_total": yearly_total,
        "avg_hold": ex["hold_days"].mean(),
    }


def main():
    print("=" * 70)
    print("P20b: C-Long 參數掃描")
    print("=" * 70)

    # Load walk-forward predictions
    print("\nLoading data...")
    pred_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19g_walkforward\p19g_all_predictions.csv")
    att = pd.read_csv(pred_path, parse_dates=["announce"])
    att["stock_id"] = att["stock_id"].astype(str).str.zfill(4)
    att = att[att["oos_prob"].notna()].copy()
    att["prob"] = att["oos_prob"]

    # Load price data
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    dis_raw = data.get("disposal_information")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Disposal lookup
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis_by_stock = {}
    for _, row in dis.iterrows():
        sid = row["stock_id"]
        if sid not in dis_by_stock:
            dis_by_stock[sid] = []
        dis_by_stock[sid].append(row["dis_announce"])

    # Turnover
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Consecutive
    print("Computing consecutive...")
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    att["consec_1d"] = False
    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann = row["announce"]
        if sid in att_dates_by_stock:
            for delta in [1, 2, 3]:
                if (ann - pd.Timedelta(days=delta)) in att_dates_by_stock[sid]:
                    att.loc[idx, "consec_1d"] = True
                    break

    # Compute returns for multiple max_hold values
    print("Computing returns for multiple hold periods...")
    att["ann_idx"] = cal.searchsorted(att["announce"], side="left")

    for max_hold in [7, 10, 12, 15]:
        rets = []
        exits = []
        holds = []
        for _, row in att.iterrows():
            sym = row["stock_id"]
            ai = int(row["ann_idx"])
            if sym not in valid_stocks:
                rets.append(np.nan); exits.append(np.nan); holds.append(np.nan); continue
            entry_idx = ai + 1
            if entry_idx >= n_cal - 1:
                rets.append(np.nan); exits.append(np.nan); holds.append(np.nan); continue
            try:
                entry_price = open_p.iloc[entry_idx][sym]
                prev_close = close.iloc[ai][sym]
                avg_to = avg_turnover_5d.iloc[ai][sym]
            except (IndexError, KeyError):
                rets.append(np.nan); exits.append(np.nan); holds.append(np.nan); continue
            if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
                rets.append(np.nan); exits.append(np.nan); holds.append(np.nan); continue
            if not (np.isnan(avg_to) or avg_to >= 20_000_000):
                rets.append(np.nan); exits.append(np.nan); holds.append(np.nan); continue
            gap = entry_price / prev_close - 1
            if not (-0.08 < gap < 0.04):
                rets.append(np.nan); exits.append(np.nan); holds.append(np.nan); continue

            exit_idx = min(entry_idx + max_hold - 1, n_cal - 1)
            ann_date = row["announce"]
            if sym in dis_by_stock:
                for dis_date in dis_by_stock[sym]:
                    delta = (dis_date - ann_date).days
                    if 0 < delta <= 30:
                        dis_exit_idx = cal.searchsorted(dis_date, side="left") - 1
                        if dis_exit_idx > entry_idx:
                            exit_idx = min(exit_idx, dis_exit_idx)
                        break

            try:
                exit_price = close.iloc[exit_idx][sym]
                if np.isnan(exit_price) or exit_price <= 0:
                    rets.append(np.nan); exits.append(np.nan); holds.append(np.nan); continue
            except (IndexError, KeyError):
                rets.append(np.nan); exits.append(np.nan); holds.append(np.nan); continue

            ret = exit_price / entry_price - 1 - COST_RATE
            rets.append(ret)
            exits.append(exit_idx)
            holds.append(exit_idx - entry_idx + 1)

        att[f"ret_{max_hold}d"] = rets
        att[f"exit_{max_hold}d"] = exits
        att[f"hold_{max_hold}d"] = holds

    # === PARAMETER SWEEP ===
    print("\n" + "=" * 70)
    print("參數掃描結果")
    print("=" * 70)
    print(f"  {'Threshold':>9} {'Consec':>7} {'MaxHold':>8} {'CAGR':>8} {'Sharpe':>7} {'MDD':>7} {'AvgRet':>7} {'Win%':>6} {'Years':>6} {'N':>5}")

    results = []
    for threshold in [0.4, 0.5, 0.55, 0.6]:
        for use_consec in [False, True]:
            for max_hold in [7, 10, 12, 15]:
                mask = (att["prob"] >= threshold) & (att[f"ret_{max_hold}d"].notna())
                if use_consec:
                    mask &= att["consec_1d"]

                signals = att[mask].copy()
                if len(signals) < 50:
                    continue

                signals["ret"] = signals[f"ret_{max_hold}d"]
                signals["exit_idx"] = signals[f"exit_{max_hold}d"]
                signals["hold_days"] = signals[f"hold_{max_hold}d"]

                m = simulate_portfolio(signals, cal, n_cal, MAX_POSITIONS)
                if m is None:
                    continue

                results.append({
                    "threshold": threshold, "consec": use_consec, "max_hold": max_hold, **m
                })

                consec_str = "YES" if use_consec else "NO"
                print(f"  {threshold:>9.2f} {consec_str:>7} {max_hold:>8} {m['cagr']:>8.1%} {m['sharpe']:>7.2f} {m['mdd']:>7.1%} {m['avg_ret']:>7.2%} {m['win']:>6.1%} {m['yearly_pass']}/{m['yearly_total']:<4} {m['n_trades']:>5}")

    # === TOP 5 BY SHARPE ===
    print("\n" + "=" * 70)
    print("TOP 5 by Sharpe (yearly_pass >= 5)")
    print("=" * 70)

    df = pd.DataFrame(results)
    if len(df) > 0:
        good = df[df["yearly_pass"] >= 5].sort_values("sharpe", ascending=False)
        if len(good) > 0:
            for _, r in good.head(5).iterrows():
                print(f"  t={r['threshold']:.2f}, consec={'Y' if r['consec'] else 'N'}, hold={r['max_hold']}d: "
                      f"CAGR={r['cagr']:.1%}, Sharpe={r['sharpe']:.2f}, MDD={r['mdd']:.1%}, "
                      f"years={r['yearly_pass']}/{r['yearly_total']}, n={r['n_trades']}")
        else:
            print("  No configs with 5+ yearly passes")
            best = df.sort_values("sharpe", ascending=False).head(5)
            for _, r in best.iterrows():
                print(f"  t={r['threshold']:.2f}, consec={'Y' if r['consec'] else 'N'}, hold={r['max_hold']}d: "
                      f"CAGR={r['cagr']:.1%}, Sharpe={r['sharpe']:.2f}, MDD={r['mdd']:.1%}, "
                      f"years={r['yearly_pass']}/{r['yearly_total']}, n={r['n_trades']}")

    # === TOP 5 by CAGR ===
    print("\n" + "=" * 70)
    print("TOP 5 by CAGR (yearly_pass >= 5)")
    print("=" * 70)
    if len(df) > 0:
        good = df[df["yearly_pass"] >= 5].sort_values("cagr", ascending=False)
        if len(good) > 0:
            for _, r in good.head(5).iterrows():
                print(f"  t={r['threshold']:.2f}, consec={'Y' if r['consec'] else 'N'}, hold={r['max_hold']}d: "
                      f"CAGR={r['cagr']:.1%}, Sharpe={r['sharpe']:.2f}, MDD={r['mdd']:.1%}, "
                      f"years={r['yearly_pass']}/{r['yearly_total']}, n={r['n_trades']}")

    # Save
    df.to_csv(OUT / "p20b_param_sweep.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
