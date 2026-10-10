"""P28: V20 + B1 (庫藏股) 組合回測.

生成兩個策略的日報酬序列，計算相關性、組合績效。

V20: 處置期間做多 (Day 4 進場, Day 9 出場, 5檔)
B1:  庫藏股公告後做多 (Day 1 open 進場, 持有 20 天, 5檔)

組合測試:
  - 相關性
  - 不同權重的組合績效
  - 逐年比較
  - 熊年表現
  - 滾動相關性

使用方式:
  python p28_v20_b1_combined.py
"""
from __future__ import annotations

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import finlab

TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\auth_token.txt")
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p28_combined")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
V20_MAX_POS = 5
B1_MAX_POS = 5
B1_HOLD_DAYS = 20
MIN_TURNOVER = 20_000_000
GAP_LOW = -0.08
GAP_HIGH = 0.04
PANIC_THRESHOLD = 400
PANIC_COOLDOWN = 9
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")


def auto_login():
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            finlab.login(token)
            import finlab.data as _fd
            _fd._default_context._role = "vip"
    else:
        finlab.login("V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m")
        import finlab.data as _fd
        _fd._default_context._role = "vip"


def generate_v20_daily_returns(close, open_p, vol, cal, n_cal, valid_stocks):
    """Generate V20 daily return series."""
    from finlab import data

    dis_raw = pd.DataFrame(data.get("disposal_information"))
    dis = dis_raw.copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$")
        & dis["stock_id"].isin(valid_stocks)
        & ~dis["stock_id"].str.startswith(("00", "91"))
    ].copy()

    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1
    dis["is_new_regime"] = dis["announce"] >= NEW_REGIME_DATE

    valid = dis[
        (dis["start_idx"] >= 0)
        & (dis["end_idx"] >= 0)
        & (dis["end_idx"] < n_cal)
        & (dis["duration"] >= 3)
    ].copy()

    # Indicators
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()
    daily_ret = close.pct_change()
    panic_count = (daily_ret < -0.06).sum(axis=1)
    is_panic = panic_count > PANIC_THRESHOLD
    danger_zone = is_panic.rolling(PANIC_COOLDOWN, min_periods=1).max() > 0

    # Breadth filter (V20 production version)
    ma20 = close.rolling(20).mean()
    above_ma20 = (close > ma20).sum(axis=1) / close.notna().sum(axis=1)
    breadth_below_40 = above_ma20 < 0.40
    market_close = close.sum(axis=1)
    market_ma20 = market_close.rolling(20).mean()
    market_below_ma20 = market_close < market_ma20
    breadth_filter_block = breadth_below_40 & market_below_ma20

    # Generate trades
    trades = []
    for _, row in valid.iterrows():
        sym = row["stock_id"]
        si = int(row["start_idx"])
        ei = int(row["end_idx"])

        if row["is_new_regime"]:
            entry_idx = si
            exit_idx = ei - 1
        else:
            entry_idx = si + 3
            exit_idx = ei - 1

        if entry_idx < 1 or exit_idx >= n_cal or entry_idx >= exit_idx:
            continue
        if danger_zone.iloc[entry_idx]:
            continue
        if breadth_filter_block.iloc[entry_idx]:
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[entry_idx - 1][sym]
            avg_to = avg_turnover_5d.iloc[entry_idx - 1][sym]
            exit_price = close.iloc[exit_idx][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close, exit_price]):
            continue
        if not (np.isnan(avg_to) or avg_to >= MIN_TURNOVER):
            continue
        gap = entry_price / prev_close - 1
        if not (GAP_LOW < gap < GAP_HIGH):
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "ret": ret,
            "stock_id": sym,
        })

    # Portfolio simulation (5 positions max)
    all_trades = pd.DataFrame(trades).sort_values("entry_idx")
    active = []
    executed = []
    for _, trade in all_trades.iterrows():
        ei = int(trade["entry_idx"])
        xi = int(trade["exit_idx"])
        active = [(e, s) for e, s in active if e > ei]
        if len(active) >= V20_MAX_POS:
            continue
        if any(s == trade["stock_id"] for _, s in active):
            continue
        active.append((xi, trade["stock_id"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)

    # Build daily returns
    daily_returns = pd.Series(0.0, index=cal)
    cap = 1.0 / V20_MAX_POS
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            daily_returns.iloc[d] += dr * cap

    print(f"  V20: {len(ex):,} trades executed")
    return daily_returns, ex


def generate_b1_daily_returns(close, open_p, vol, cal, n_cal, valid_stocks):
    """Generate B1 (buyback) daily return series."""
    from finlab import data

    purpose_df = data.get("treasury_stock:買回目的")

    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Extract events
    events = []
    for date in purpose_df.index:
        row = purpose_df.loc[date]
        active = row[row.notna()]
        if len(active) == 0:
            continue
        date_norm = pd.Timestamp(date).normalize()
        if date_norm < pd.Timestamp("2010-01-01"):
            continue
        for stock_id in active.index:
            if stock_id not in valid_stocks:
                continue
            if not stock_id.isdigit() or len(stock_id) != 4:
                continue
            if stock_id.startswith(("00", "91")):
                continue
            events.append({"stock_id": stock_id, "announce_date": date_norm})

    ev = pd.DataFrame(events)
    ev["announce_idx"] = cal.searchsorted(ev["announce_date"], side="left")
    ev = ev[(ev["announce_idx"] >= 5) & (ev["announce_idx"] < n_cal - 1)].copy()

    # Liquidity filter
    filtered = []
    for _, row in ev.iterrows():
        sym = row["stock_id"]
        idx = int(row["announce_idx"])
        try:
            tv = avg_turnover_5d.iloc[idx][sym]
            filtered.append(pd.notna(tv) and tv >= MIN_TURNOVER)
        except (IndexError, KeyError):
            filtered.append(False)
    ev["liquid"] = filtered
    ev = ev[ev["liquid"]].copy()

    # Entry: Day 1 open (next trading day after announcement)
    trades = []
    for _, row in ev.iterrows():
        sym = row["stock_id"]
        ai = int(row["announce_idx"])
        entry_idx = ai + 1
        exit_idx = entry_idx + B1_HOLD_DAYS - 1

        if entry_idx >= n_cal or exit_idx >= n_cal:
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            exit_price = close.iloc[exit_idx][sym]
            if any(np.isnan(x) or x <= 0 for x in [entry_price, exit_price]):
                continue
        except (IndexError, KeyError):
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "ret": ret,
            "stock_id": sym,
        })

    # Portfolio simulation (5 positions max)
    all_trades = pd.DataFrame(trades).sort_values("entry_idx")
    active = []
    executed = []
    for _, trade in all_trades.iterrows():
        ei = int(trade["entry_idx"])
        xi = int(trade["exit_idx"])
        active = [(e, s) for e, s in active if e > ei]
        if len(active) >= B1_MAX_POS:
            continue
        if any(s == trade["stock_id"] for _, s in active):
            continue
        active.append((xi, trade["stock_id"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)

    # Build daily returns
    daily_returns = pd.Series(0.0, index=cal)
    cap = 1.0 / B1_MAX_POS
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            daily_returns.iloc[d] += dr * cap

    print(f"  B1: {len(ex):,} trades executed")
    return daily_returns, ex


def compute_metrics(daily_returns, cal, label=""):
    """Compute portfolio metrics from daily returns."""
    # Only use period where strategy is active
    active_mask = daily_returns != 0
    if not active_mask.any():
        return {}
    first_active = active_mask.idxmax()
    dr = daily_returns[daily_returns.index >= first_active]

    dm = dr.mean()
    ds = dr.std()
    sharpe = dm / ds * np.sqrt(252) if ds > 0 else 0
    cum = (1 + dr).cumprod()
    rm = cum.cummax()
    mdd = ((cum - rm) / rm).min()
    total_days = len(dr)
    cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

    # Yearly returns
    yearly = {}
    for yr in sorted(set(dr.index.year)):
        yr_dr = dr[dr.index.year == yr]
        if len(yr_dr) > 50:
            yearly[yr] = (1 + yr_dr).prod() - 1

    return {
        "label": label,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MDD": mdd,
        "daily_mean": dm,
        "daily_std": ds,
        "yearly": yearly,
        "cum": cum,
        "dr": dr,
    }


def main():
    auto_login()
    from finlab import data

    print("=" * 70)
    print("P28: V20 + B1 組合回測")
    print("=" * 70)

    print("\nLoading data...")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)
    print(f"  Calendar: {cal[0].date()} to {cal[-1].date()} ({n_cal} days)")

    # Generate daily returns for both strategies
    print("\nGenerating V20 returns...")
    v20_dr, v20_trades = generate_v20_daily_returns(close, open_p, vol, cal, n_cal, valid_stocks)

    print("Generating B1 returns...")
    b1_dr, b1_trades = generate_b1_daily_returns(close, open_p, vol, cal, n_cal, valid_stocks)

    # === Correlation Analysis ===
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Only compare days where at least one strategy is active
    both_active = (v20_dr != 0) | (b1_dr != 0)
    v20_active = v20_dr != 0
    b1_active = b1_dr != 0
    overlap = v20_active & b1_active

    print(f"\n  V20 active days: {v20_active.sum():,}")
    print(f"  B1 active days: {b1_active.sum():,}")
    print(f"  Overlap days (both active): {overlap.sum():,}")
    print(f"  V20-only days: {(v20_active & ~b1_active).sum():,}")
    print(f"  B1-only days: {(~v20_active & b1_active).sum():,}")

    # Correlation on overlap days
    if overlap.sum() > 50:
        corr_overlap = v20_dr[overlap].corr(b1_dr[overlap])
        print(f"\n  Correlation (overlap days): {corr_overlap:.4f}")

    # Correlation on all days (including zeros)
    corr_all = v20_dr.corr(b1_dr)
    print(f"  Correlation (all days): {corr_all:.4f}")

    # Rolling correlation (60-day)
    rolling_corr = v20_dr.rolling(60, min_periods=30).corr(b1_dr)
    rc_valid = rolling_corr.dropna()
    if len(rc_valid) > 100:
        print(f"  Rolling 60d correlation: mean={rc_valid.mean():.4f}, "
              f"min={rc_valid.min():.4f}, max={rc_valid.max():.4f}")

    # === Individual Strategy Metrics ===
    print("\n" + "=" * 70)
    print("INDIVIDUAL STRATEGY METRICS")
    print("=" * 70)

    v20_metrics = compute_metrics(v20_dr, cal, "V20")
    b1_metrics = compute_metrics(b1_dr, cal, "B1")

    for m in [v20_metrics, b1_metrics]:
        if not m:
            continue
        print(f"\n  {m['label']}:")
        print(f"    CAGR: {m['CAGR']:.1%}")
        print(f"    Sharpe: {m['Sharpe']:.2f}")
        print(f"    MDD: {m['MDD']:.1%}")

    # === Combined Portfolio ===
    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO")
    print("=" * 70)

    allocations = [
        (1.0, 0.0, "V20 100%"),
        (0.7, 0.3, "V20 70% + B1 30%"),
        (0.5, 0.5, "V20 50% + B1 50%"),
        (0.3, 0.7, "V20 30% + B1 70%"),
        (0.0, 1.0, "B1 100%"),
    ]

    print(f"\n  {'配置':<22} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'年正':>6}")
    print("  " + "-" * 55)

    results = []
    for w_v20, w_b1, label in allocations:
        combined_dr = v20_dr * w_v20 + b1_dr * w_b1
        m = compute_metrics(combined_dr, cal, label)
        if not m:
            continue
        yr_pos = sum(1 for v in m["yearly"].values() if v > 0)
        yr_total = len(m["yearly"])
        print(f"  {label:<22} {m['CAGR']:>8.1%} {m['Sharpe']:>8.2f} {m['MDD']:>8.1%} {yr_pos}/{yr_total}")
        results.append(m)

    # === Yearly Comparison ===
    print("\n" + "=" * 70)
    print("YEARLY COMPARISON")
    print("=" * 70)

    # Market benchmark
    mkt_daily = close.pct_change().mean(axis=1)
    mkt_dr = pd.Series(mkt_daily.values, index=cal)

    combined_5050 = v20_dr * 0.5 + b1_dr * 0.5
    combined_7030 = v20_dr * 0.7 + b1_dr * 0.3

    print(f"\n  {'Year':>6} {'V20':>8} {'B1':>8} {'50/50':>8} {'70/30':>8} {'大盤':>8}")
    print("  " + "-" * 50)

    yearly_table = []
    for yr in range(2010, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() < 50:
            continue

        v20_yr = (1 + v20_dr[yr_mask]).prod() - 1
        b1_yr = (1 + b1_dr[yr_mask]).prod() - 1
        c5050_yr = (1 + combined_5050[yr_mask]).prod() - 1
        c7030_yr = (1 + combined_7030[yr_mask]).prod() - 1
        mkt_yr = (1 + mkt_dr[yr_mask]).prod() - 1

        yearly_table.append({
            "year": yr, "v20": v20_yr, "b1": b1_yr,
            "combined_5050": c5050_yr, "combined_7030": c7030_yr, "market": mkt_yr,
        })
        print(f"  {yr:>6} {v20_yr:>8.1%} {b1_yr:>8.1%} {c5050_yr:>8.1%} {c7030_yr:>8.1%} {mkt_yr:>8.1%}")

    yearly_df = pd.DataFrame(yearly_table)
    yearly_df.to_csv(OUT / "p28_yearly_comparison.csv", index=False, encoding="utf-8-sig")

    # === Bear Year Analysis ===
    print("\n" + "=" * 70)
    print("BEAR YEAR ANALYSIS (大盤跌 > 5%)")
    print("=" * 70)

    bear_years = [r["year"] for r in yearly_table if r["market"] < -0.05]
    print(f"\n  Bear years: {bear_years}")
    print(f"\n  {'Year':>6} {'大盤':>8} {'V20':>8} {'B1':>8} {'50/50':>8}")
    print("  " + "-" * 42)
    for r in yearly_table:
        if r["year"] in bear_years:
            print(f"  {r['year']:>6} {r['market']:>8.1%} {r['v20']:>8.1%} {r['b1']:>8.1%} {r['combined_5050']:>8.1%}")

    # === Rolling Sharpe ===
    print("\n" + "=" * 70)
    print("ROLLING 3-YEAR SHARPE")
    print("=" * 70)

    for label, dr in [("V20", v20_dr), ("B1", b1_dr), ("50/50", combined_5050)]:
        roll_sharpe = dr.rolling(756, min_periods=504).mean() / dr.rolling(756, min_periods=504).std() * np.sqrt(252)
        rs_valid = roll_sharpe.dropna()
        if len(rs_valid) > 100:
            print(f"  {label}: min={rs_valid.min():.2f}, max={rs_valid.max():.2f}, "
                  f"median={rs_valid.median():.2f}")

    # === Complementarity in different regimes ===
    print("\n" + "=" * 70)
    print("REGIME-BASED COMPLEMENTARITY")
    print("=" * 70)

    market_close = close.sum(axis=1)
    market_ma20 = market_close.rolling(20).mean()
    is_bull = (market_close > market_ma20).values

    # Align with cal
    bull_mask = pd.Series(is_bull, index=cal)[:n_cal]

    for regime, mask in [("多頭 (MA20上)", bull_mask), ("空頭 (MA20下)", ~bull_mask)]:
        v20_r = v20_dr[mask].mean() * 252
        b1_r = b1_dr[mask].mean() * 252
        v20_s = v20_dr[mask].std()
        b1_s = b1_dr[mask].std()
        v20_sharpe = v20_r / (v20_s * np.sqrt(252)) if v20_s > 0 else 0
        b1_sharpe = b1_r / (b1_s * np.sqrt(252)) if b1_s > 0 else 0
        print(f"\n  {regime}:")
        print(f"    V20 annualized: {v20_r:.1%}, Sharpe: {v20_sharpe:.2f}")
        print(f"    B1  annualized: {b1_r:.1%}, Sharpe: {b1_sharpe:.2f}")

    # === Save daily returns ===
    combined_df = pd.DataFrame({
        "date": cal,
        "v20_daily": v20_dr.values,
        "b1_daily": b1_dr.values,
        "combined_5050": combined_5050.values,
        "combined_7030": combined_7030.values,
    })
    combined_df.to_csv(OUT / "p28_daily_returns.csv", index=False, encoding="utf-8-sig")

    # === Conclusion ===
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Find best allocation
    best = max(results, key=lambda m: m["Sharpe"])
    print(f"\n  Best Sharpe allocation: {best['label']}")
    print(f"    CAGR: {best['CAGR']:.1%}, Sharpe: {best['Sharpe']:.2f}, MDD: {best['MDD']:.1%}")

    # Improvement over V20 alone
    if v20_metrics and best["Sharpe"] > v20_metrics["Sharpe"]:
        improvement = (best["Sharpe"] - v20_metrics["Sharpe"]) / v20_metrics["Sharpe"] * 100
        print(f"\n  ✅ B1 組合提升 Sharpe: +{improvement:.0f}%")
    elif v20_metrics:
        print(f"\n  ⚠️ B1 組合未提升 Sharpe（V20 已極強）")

    print(f"\n  Correlation: {corr_all:.4f} (all days)")
    print(f"  相關性 < 0.1 = 極佳互補")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
