"""P29: V20 操作環境深度分析 — 流動性真空的觸發條件與空窗特徵.

核心問題:
  1. V20 在什麼市場環境下最強？（流動性真空最大）
  2. V20 在什麼環境下無訊號？（空窗期的市場特徵）
  3. 空窗期的市場是什麼狀態？→ 這決定了什麼策略適合填補

分析維度:
  - 月度/季度分佈
  - 市場廣度分層
  - 波動率分層
  - 處置事件密度
  - 大盤趨勢狀態
  - 空窗期特徵描繪

使用方式:
  python p29_v20_regime_analysis.py
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
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p29_regime")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
V20_MAX_POS = 5
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


def main():
    auto_login()
    from finlab import data

    print("=" * 70)
    print("P29: V20 流動性真空 — 操作環境分析")
    print("=" * 70)

    print("\nLoading data...")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # === Build V20 trades (same as production) ===
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
    daily_ret_all = close.pct_change()
    panic_count = (daily_ret_all < -0.06).sum(axis=1)
    is_panic = panic_count > PANIC_THRESHOLD
    danger_zone = is_panic.rolling(PANIC_COOLDOWN, min_periods=1).max() > 0

    # Market indicators
    ma20 = close.rolling(20).mean()
    above_ma20 = (close > ma20).sum(axis=1) / close.notna().sum(axis=1)
    breadth_below_40 = above_ma20 < 0.40
    market_close = close.sum(axis=1)
    market_ma20 = market_close.rolling(20).mean()
    market_below_ma20 = market_close < market_ma20
    breadth_filter_block = breadth_below_40 & market_below_ma20

    # Market volatility (20d realized vol of equal-weight index)
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_vol_20d = mkt_ret.rolling(20).std() * np.sqrt(252)

    # Market trend (MA20 slope)
    mkt_ma20_slope = market_ma20.pct_change(5)

    # === Generate V20 trades ===
    v20_trades = []
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
        v20_trades.append({
            "entry_idx": entry_idx, "exit_idx": exit_idx,
            "entry_date": cal[entry_idx], "exit_date": cal[exit_idx],
            "ret": ret, "stock_id": sym,
            "year": cal[entry_idx].year, "month": cal[entry_idx].month,
            "duration": int(row["duration"]),
        })

    # Portfolio simulation
    v20_all = pd.DataFrame(v20_trades).sort_values("entry_idx")
    active = []
    executed = []
    for _, trade in v20_all.iterrows():
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
    print(f"V20 executed trades: {len(ex):,}")

    # === Build daily state ===
    # Position count per day
    v20_position_count = np.zeros(n_cal, dtype=int)
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        for d in range(ei, min(xi + 1, n_cal)):
            v20_position_count[d] += 1

    # Daily returns
    v20_dr = pd.Series(0.0, index=cal)
    cap = 1.0 / V20_MAX_POS
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            v20_dr.iloc[d] += dr * cap

    # === 1. Monthly distribution ===
    print("\n" + "=" * 70)
    print("1. MONTHLY DISTRIBUTION (V20 activity & performance)")
    print("=" * 70)

    monthly = ex.groupby(["year", "month"]).agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        win=("ret", lambda x: (x > 0).mean()),
    ).reset_index()

    # Aggregate by month (across all years)
    month_agg = ex.groupby("month").agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        win=("ret", lambda x: (x > 0).mean()),
        total_ret=("ret", "sum"),
    ).reset_index()

    # Count idle days per month
    idle_by_month = pd.DataFrame({"month": cal.month, "idle": (v20_position_count == 0).astype(int)})
    idle_monthly = idle_by_month.groupby("month").agg(
        idle_days=("idle", "sum"),
        total_days=("idle", "count"),
    ).reset_index()
    idle_monthly["idle_pct"] = idle_monthly["idle_days"] / idle_monthly["total_days"]

    month_agg = month_agg.merge(idle_monthly[["month", "idle_pct"]], on="month")

    print(f"\n  {'Month':>5} {'Trades':>7} {'Avg Ret':>8} {'Win%':>6} {'Idle%':>7} {'Active%':>8}")
    print("  " + "-" * 45)
    for _, r in month_agg.iterrows():
        active_pct = 1 - r["idle_pct"]
        print(f"  {int(r['month']):>5} {int(r['n']):>7} {r['avg_ret']:>8.2%} {r['win']:>6.1%} "
              f"{r['idle_pct']:>7.1%} {active_pct:>8.1%}")

    month_agg.to_csv(OUT / "p29_monthly.csv", index=False, encoding="utf-8-sig")

    # === 2. Market breadth layers ===
    print("\n" + "=" * 70)
    print("2. BY MARKET BREADTH (MA20上股票占比)")
    print("=" * 70)

    breadth_vals = above_ma20.values
    breadth_bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    breadth_labels = ["<30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", ">80%"]

    # For each trade, get breadth at entry
    ex_with_breadth = ex.copy()
    ex_with_breadth["breadth_at_entry"] = [breadth_vals[int(t)] for t in ex["entry_idx"]]
    ex_with_breadth["breadth_bin"] = pd.cut(ex_with_breadth["breadth_at_entry"], bins=breadth_bins, labels=breadth_labels)

    by_breadth = ex_with_breadth.groupby("breadth_bin", observed=True).agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        win=("ret", lambda x: (x > 0).mean()),
    ).reset_index()

    # Also: idle % by breadth
    breadth_series = pd.Series(breadth_vals, index=cal)
    breadth_df = pd.DataFrame({"breadth": breadth_series.values, "idle": (v20_position_count == 0).astype(int)})
    breadth_df["bin"] = pd.cut(breadth_df["breadth"], bins=breadth_bins, labels=breadth_labels)
    idle_by_breadth = breadth_df.groupby("bin", observed=True).agg(
        idle_pct=("idle", "mean"),
        days=("idle", "count"),
    ).reset_index()

    by_breadth = by_breadth.merge(idle_by_breadth, left_on="breadth_bin", right_on="bin")

    print(f"\n  {'廣度':>8} {'Trades':>7} {'Avg Ret':>8} {'Win%':>6} {'Idle%':>7} {'Days':>6}")
    print("  " + "-" * 48)
    for _, r in by_breadth.iterrows():
        print(f"  {str(r['breadth_bin']):>8} {int(r['n']):>7} {r['avg_ret']:>8.2%} {r['win']:>6.1%} "
              f"{r['idle_pct']:>7.1%} {int(r['days']):>6}")

    by_breadth.to_csv(OUT / "p29_by_breadth.csv", index=False, encoding="utf-8-sig")

    # === 3. Market volatility layers ===
    print("\n" + "=" * 70)
    print("3. BY MARKET VOLATILITY (20d annualized)")
    print("=" * 70)

    vol_vals = mkt_vol_20d.values
    vol_bins = [0, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    vol_labels = ["<10%", "10-15%", "15-20%", "20-30%", "30-50%", ">50%"]

    ex_with_vol = ex.copy()
    ex_with_vol["vol_at_entry"] = [vol_vals[int(t)] if not np.isnan(vol_vals[int(t)]) else np.nan for t in ex["entry_idx"]]
    ex_with_vol = ex_with_vol[ex_with_vol["vol_at_entry"].notna()]
    ex_with_vol["vol_bin"] = pd.cut(ex_with_vol["vol_at_entry"], bins=vol_bins, labels=vol_labels)

    by_vol = ex_with_vol.groupby("vol_bin", observed=True).agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        win=("ret", lambda x: (x > 0).mean()),
    ).reset_index()

    # Idle % by vol
    vol_df = pd.DataFrame({"vol": vol_vals, "idle": (v20_position_count == 0).astype(int)})
    vol_df = vol_df[vol_df["vol"].notna()]
    vol_df["bin"] = pd.cut(vol_df["vol"], bins=vol_bins, labels=vol_labels)
    idle_by_vol = vol_df.groupby("bin", observed=True).agg(
        idle_pct=("idle", "mean"),
        days=("idle", "count"),
    ).reset_index()

    by_vol = by_vol.merge(idle_by_vol, left_on="vol_bin", right_on="bin")

    print(f"\n  {'波動率':>8} {'Trades':>7} {'Avg Ret':>8} {'Win%':>6} {'Idle%':>7} {'Days':>6}")
    print("  " + "-" * 48)
    for _, r in by_vol.iterrows():
        print(f"  {str(r['vol_bin']):>8} {int(r['n']):>7} {r['avg_ret']:>8.2%} {r['win']:>6.1%} "
              f"{r['idle_pct']:>7.1%} {int(r['days']):>6}")

    by_vol.to_csv(OUT / "p29_by_volatility.csv", index=False, encoding="utf-8-sig")

    # === 4. Disposal event density ===
    print("\n" + "=" * 70)
    print("4. BY DISPOSAL EVENT DENSITY (events per month)")
    print("=" * 70)

    # Count disposal events per month (all, not just executed)
    valid["month_key"] = valid["start"].dt.to_period("M")
    monthly_events = valid.groupby("month_key").size().reset_index(name="n_events")

    # For executed trades, get the monthly event count
    ex["month_key"] = pd.to_datetime(ex["entry_date"]).dt.to_period("M")
    ex_with_density = ex.merge(monthly_events, on="month_key", how="left")

    density_bins = [0, 50, 100, 200, 400, 1000]
    density_labels = ["<50", "50-100", "100-200", "200-400", ">400"]
    ex_with_density["density_bin"] = pd.cut(ex_with_density["n_events"], bins=density_bins, labels=density_labels)

    by_density = ex_with_density.groupby("density_bin", observed=True).agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        win=("ret", lambda x: (x > 0).mean()),
    ).reset_index()

    print(f"\n  {'月事件數':>10} {'Trades':>7} {'Avg Ret':>8} {'Win%':>6}")
    print("  " + "-" * 35)
    for _, r in by_density.iterrows():
        print(f"  {str(r['density_bin']):>10} {int(r['n']):>7} {r['avg_ret']:>8.2%} {r['win']:>6.1%}")

    # === 5. Market trend (MA20 slope) ===
    print("\n" + "=" * 70)
    print("5. BY MARKET TREND (MA20 5d slope)")
    print("=" * 70)

    slope_vals = mkt_ma20_slope.values
    slope_bins = [-1, -0.03, -0.01, 0, 0.01, 0.03, 1]
    slope_labels = ["<-3%", "-3~-1%", "-1~0%", "0~1%", "1~3%", ">3%"]

    ex_with_slope = ex.copy()
    ex_with_slope["slope_at_entry"] = [slope_vals[int(t)] if not np.isnan(slope_vals[int(t)]) else np.nan for t in ex["entry_idx"]]
    ex_with_slope = ex_with_slope[ex_with_slope["slope_at_entry"].notna()]
    ex_with_slope["slope_bin"] = pd.cut(ex_with_slope["slope_at_entry"], bins=slope_bins, labels=slope_labels)

    by_slope = ex_with_slope.groupby("slope_bin", observed=True).agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        win=("ret", lambda x: (x > 0).mean()),
    ).reset_index()

    # Idle by slope
    slope_df = pd.DataFrame({"slope": slope_vals, "idle": (v20_position_count == 0).astype(int)})
    slope_df = slope_df[slope_df["slope"].notna()]
    slope_df["bin"] = pd.cut(slope_df["slope"], bins=slope_bins, labels=slope_labels)
    idle_by_slope = slope_df.groupby("bin", observed=True).agg(
        idle_pct=("idle", "mean"),
        days=("idle", "count"),
    ).reset_index()

    by_slope = by_slope.merge(idle_by_slope, left_on="slope_bin", right_on="bin")

    print(f"\n  {'MA20斜率':>10} {'Trades':>7} {'Avg Ret':>8} {'Win%':>6} {'Idle%':>7} {'Days':>6}")
    print("  " + "-" * 50)
    for _, r in by_slope.iterrows():
        print(f"  {str(r['slope_bin']):>10} {int(r['n']):>7} {r['avg_ret']:>8.2%} {r['win']:>6.1%} "
              f"{r['idle_pct']:>7.1%} {int(r['days']):>6}")

    by_slope.to_csv(OUT / "p29_by_trend.csv", index=False, encoding="utf-8-sig")

    # === 6. IDLE PERIOD CHARACTERIZATION ===
    print("\n" + "=" * 70)
    print("6. IDLE PERIOD CHARACTERIZATION (V20 無持倉時的市場狀態)")
    print("=" * 70)

    idle_mask = v20_position_count == 0
    active_mask = v20_position_count > 0

    # Market conditions during idle vs active
    conditions = {
        "breadth": above_ma20.values,
        "market_vol": mkt_vol_20d.values,
        "ma20_slope": mkt_ma20_slope.values,
        "market_return_5d": mkt_ret.rolling(5).sum().values,
        "market_return_20d": mkt_ret.rolling(20).sum().values,
    }

    print(f"\n  {'指標':>20} {'Idle 均值':>12} {'Active 均值':>12} {'差異':>10}")
    print("  " + "-" * 58)
    for name, vals in conditions.items():
        idle_vals = vals[idle_mask]
        active_vals = vals[active_mask]
        idle_valid = idle_vals[~np.isnan(idle_vals)]
        active_valid = active_vals[~np.isnan(active_vals)]
        if len(idle_valid) > 50 and len(active_valid) > 50:
            im = np.mean(idle_valid)
            am = np.mean(active_valid)
            print(f"  {name:>20} {im:>12.4f} {am:>12.4f} {am - im:>+10.4f}")

    # Market returns during idle periods
    idle_returns = mkt_ret.values[idle_mask]
    active_returns = mkt_ret.values[active_mask]
    idle_valid_r = idle_returns[~np.isnan(idle_returns)]
    active_valid_r = active_returns[~np.isnan(active_returns)]

    print(f"\n  大盤日報酬:")
    print(f"    Idle 期間: mean={np.mean(idle_valid_r):.4%}, vol={np.std(idle_valid_r):.4%}")
    print(f"    Active 期間: mean={np.mean(active_valid_r):.4%}, vol={np.std(active_valid_r):.4%}")

    # Idle period duration
    idle_periods = []
    current_start = None
    for i in range(n_cal):
        if idle_mask[i] and current_start is None:
            current_start = i
        elif not idle_mask[i] and current_start is not None:
            idle_periods.append(i - current_start)
            current_start = None
    if current_start is not None:
        idle_periods.append(n_cal - current_start)

    idle_periods = np.array(idle_periods)
    print(f"\n  空窗期持續時間:")
    print(f"    次數: {len(idle_periods)}")
    print(f"    平均: {idle_periods.mean():.1f} 天")
    print(f"    中位數: {np.median(idle_periods):.0f} 天")
    print(f"    最長: {idle_periods.max()} 天")
    print(f"    >30天的空窗: {(idle_periods > 30).sum()} 次")
    print(f"    >60天的空窗: {(idle_periods > 60).sum()} 次")

    # === 7. The key question: what does the market look like when V20 is idle? ===
    print("\n" + "=" * 70)
    print("7. IDLE PERIOD MARKET STATE (what to trade when V20 is off)")
    print("=" * 70)

    # Classify idle periods by market state
    idle_indices = np.where(idle_mask)[0]
    idle_breadth = above_ma20.values[idle_indices]
    idle_vol = mkt_vol_20d.values[idle_indices]
    idle_slope = mkt_ma20_slope.values[idle_indices]

    # Valid only
    valid_idle = ~np.isnan(idle_breadth) & ~np.isnan(idle_vol) & ~np.isnan(idle_slope)
    idle_breadth = idle_breadth[valid_idle]
    idle_vol = idle_vol[valid_idle]
    idle_slope = idle_slope[valid_idle]

    # Classify
    bull_low_vol = (idle_breadth > 0.6) & (idle_vol < 0.15)
    bull_high_vol = (idle_breadth > 0.6) & (idle_vol >= 0.15)
    bear_low_vol = (idle_breadth < 0.4) & (idle_vol < 0.20)
    bear_high_vol = (idle_breadth < 0.4) & (idle_vol >= 0.20)
    neutral = ~bull_low_vol & ~bull_high_vol & ~bear_low_vol & ~bear_high_vol

    total_idle = len(idle_breadth)
    print(f"\n  空窗期市場狀態分類 (total idle days: {total_idle:,}):")
    print(f"    多頭+低波動 (廣度>60%, vol<15%): {bull_low_vol.sum():>5} ({bull_low_vol.mean():.1%})")
    print(f"    多頭+高波動 (廣度>60%, vol>15%): {bull_high_vol.sum():>5} ({bull_high_vol.mean():.1%})")
    print(f"    空頭+低波動 (廣度<40%, vol<20%): {bear_low_vol.sum():>5} ({bear_low_vol.mean():.1%})")
    print(f"    空頭+高波動 (廣度<40%, vol>20%): {bear_high_vol.sum():>5} ({bear_high_vol.mean():.1%})")
    print(f"    中性/混合:                       {neutral.sum():>5} ({neutral.mean():.1%})")

    # Market returns in each idle sub-state
    idle_mkt_ret = mkt_ret.values[idle_indices][valid_idle]
    print(f"\n  各狀態下大盤報酬:")
    for label, mask in [("多頭低波動", bull_low_vol), ("多頭高波動", bull_high_vol),
                        ("空頭低波動", bear_low_vol), ("空頭高波動", bear_high_vol),
                        ("中性", neutral)]:
        if mask.sum() > 50:
            rets = idle_mkt_ret[mask]
            rets = rets[~np.isnan(rets)]
            if len(rets) > 50:
                ann_ret = np.mean(rets) * 252
                ann_vol = np.std(rets) * np.sqrt(252)
                print(f"    {label}: 大盤年化 {ann_ret:+.1%}, vol {ann_vol:.1%}")

    # === 8. V20 performance decomposition ===
    print("\n" + "=" * 70)
    print("8. V20 ALPHA DECOMPOSITION (what creates the return)")
    print("=" * 70)

    # Is it mean-reversion or liquidity vacuum?
    # Test: does the return correlate with how much the stock DROPPED before entry?
    # If mean-reversion: bigger drops → bigger bounces (positive correlation)
    # If liquidity vacuum: return is independent of prior drop (no correlation)

    pre_returns = []
    for _, t in ex.iterrows():
        sym = t["stock_id"]
        ei = int(t["entry_idx"])
        try:
            pre_5d = close.iloc[ei][sym] / close.iloc[max(ei - 5, 0)][sym] - 1
            pre_returns.append(pre_5d)
        except (IndexError, KeyError):
            pre_returns.append(np.nan)

    ex["pre_5d_return"] = pre_returns
    valid_pre = ex[ex["pre_5d_return"].notna()]

    corr_pre = valid_pre["pre_5d_return"].corr(valid_pre["ret"])
    print(f"\n  入場前5日報酬 vs 策略報酬 相關性: {corr_pre:.4f}")

    # Group by pre-return
    valid_pre["pre_bin"] = pd.qcut(valid_pre["pre_5d_return"], 5, labels=["Q1(跌最多)", "Q2", "Q3", "Q4", "Q5(漲最多)"], duplicates="drop")
    by_pre = valid_pre.groupby("pre_bin", observed=True).agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        win=("ret", lambda x: (x > 0).mean()),
        avg_pre=("pre_5d_return", "mean"),
    ).reset_index()

    print(f"\n  入場前5日表現 vs 策略報酬:")
    print(f"  {'分位':>12} {'入場前':>8} {'策略報酬':>8} {'勝率':>6} {'n':>6}")
    print("  " + "-" * 45)
    for _, r in by_pre.iterrows():
        print(f"  {str(r['pre_bin']):>12} {r['avg_pre']:>8.2%} {r['avg_ret']:>8.2%} {r['win']:>6.1%} {int(r['n']):>6}")

    # Interpretation
    print(f"\n  解讀:")
    if abs(corr_pre) < 0.1:
        print("    相關性 ≈ 0 → 確認是流動性真空，不是均值回歸。")
        print("    報酬不取決於入場前跌了多少，而取決於處置機制本身。")
    elif corr_pre < -0.1:
        print("    負相關 → 跌越多反彈越大 → 有均值回歸成分。")
    else:
        print("    正相關 → 漲越多繼續漲 → 動量成分。")

    # === 9. Summary: V20's operating regime ===
    print("\n" + "=" * 70)
    print("9. V20 OPERATING REGIME SUMMARY")
    print("=" * 70)

    print("""
  V20 的 alpha 來源: 流動性真空（非均值回歸）
  
  觸發條件: 處置機制 → 融資戶強制平倉 → 賣壓 → 價格被壓低
  Alpha 實現: 處置期間內，限制逐步鬆綁 → 價格恢復
  
  最佳環境: (從上面數據填充)
  最差環境/空窗: (從上面數據填充)
  空窗期市場特徵: (從上面數據填充)
""")

    # Save daily state for further analysis
    daily_state = pd.DataFrame({
        "date": cal,
        "v20_positions": v20_position_count,
        "v20_daily_ret": v20_dr.values,
        "breadth": above_ma20.values,
        "market_vol_20d": mkt_vol_20d.values,
        "ma20_slope": mkt_ma20_slope.values,
        "market_ret_5d": mkt_ret.rolling(5).sum().values,
        "market_ret_20d": mkt_ret.rolling(20).sum().values,
    })
    daily_state.to_csv(OUT / "p29_daily_state.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
