"""P30: 趨勢跟隨做空 — 填補 V20 空窗的熊市策略.

設計理念:
  V20 空窗 = 熊市 + 高波動 + 大盤下跌 (55% of idle time)
  趨勢做空 = 大盤下跌時自動觸發 → 完美填補

策略邏輯:
  當大盤確認下跌趨勢 → 做空大盤 (00649R / 期貨)
  趨勢結束 → 平倉
  與 V20 互斥: V20 活躍時趨勢做空自然不觸發

測試多種訊號:
  T1: MA 交叉 (跌破 MA60 做空, 回到 MA60 上平倉)
  T2: 廣度持續低檔 (廣度<30% 超過5天 → 做空, 廣度>50% 平倉)
  T3: 波動率突破做空 (vol 突破 + 大盤在 MA20 下)
  T4: 組合訊號 (T1 + T2 任一觸發)

使用方式:
  python p30_trend_short.py
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
OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p30_trend_short")
OUT.mkdir(parents=True, exist_ok=True)

COST_SHORT = 0.001  # 期貨/ETF 交易成本 (單邊)


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


def compute_metrics(dr, label=""):
    """Compute portfolio metrics."""
    active_mask = dr != 0
    if not active_mask.any():
        return {}
    first_active = active_mask.idxmax()
    dr_active = dr[dr.index >= first_active]

    dm = dr_active.mean()
    ds = dr_active.std()
    sharpe = dm / ds * np.sqrt(252) if ds > 0 else 0
    cum = (1 + dr_active).cumprod()
    rm = cum.cummax()
    mdd = ((cum - rm) / rm).min()
    total_days = len(dr_active)
    cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

    yearly = {}
    for yr in sorted(set(dr_active.index.year)):
        yr_dr = dr_active[dr_active.index.year == yr]
        if len(yr_dr) > 50:
            yearly[yr] = (1 + yr_dr).prod() - 1

    yr_pos = sum(1 for v in yearly.values() if v > 0)
    yr_total = len(yearly)

    # Rolling 3yr Sharpe
    roll = dr_active.rolling(756, min_periods=504).mean() / dr_active.rolling(756, min_periods=504).std() * np.sqrt(252)
    roll_valid = roll.dropna()
    min_roll = roll_valid.min() if len(roll_valid) > 100 else np.nan

    return {
        "label": label, "CAGR": cagr, "Sharpe": sharpe, "MDD": mdd,
        "yearly": yearly, "yr_pos": yr_pos, "yr_total": yr_total,
        "min_roll_sharpe": min_roll, "active_days": int(active_mask.sum()),
    }


def main():
    auto_login()
    from finlab import data

    print("=" * 70)
    print("P30: 趨勢跟隨做空策略")
    print("=" * 70)

    print("\nLoading data...")
    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Market equal-weight index
    mkt_ret = close.pct_change().mean(axis=1)  # daily equal-weight return
    market_close = close.sum(axis=1)  # proxy for market level

    # Indicators
    ma20 = market_close.rolling(20).mean()
    ma60 = market_close.rolling(60).mean()
    ma120 = market_close.rolling(120).mean()

    # Breadth
    stock_ma20 = close.rolling(20).mean()
    above_ma20 = (close > stock_ma20).sum(axis=1) / close.notna().sum(axis=1)

    # Volatility
    vol_20d = mkt_ret.rolling(20).std() * np.sqrt(252)
    vol_60d = mkt_ret.rolling(60).std() * np.sqrt(252)

    # === V20 position tracking (to measure overlap) ===
    # Quick V20 position count (simplified)
    dis_raw = pd.DataFrame(data.get("disposal_information"))
    valid_stocks = set(close.columns)
    dis = dis_raw.copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
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
    dis = dis[(dis["start_idx"] >= 0) & (dis["end_idx"] >= 0) & (dis["duration"] >= 3)]

    # Approximate V20 active (simplified: disposal period = V20 active)
    v20_approx_active = np.zeros(n_cal, dtype=bool)
    for _, row in dis.iterrows():
        si = max(int(row["start_idx"]), 0)
        ei = min(int(row["end_idx"]), n_cal - 1)
        v20_approx_active[si:ei + 1] = True

    v20_idle = ~v20_approx_active
    print(f"V20 approximate idle days: {v20_idle.sum():,} / {n_cal:,} ({v20_idle.mean():.1%})")

    # === Strategy T1: MA60 Crossover Short ===
    print("\n" + "=" * 70)
    print("T1: MA60 趨勢做空")
    print("=" * 70)
    print("  訊號: 大盤 < MA60 → 做空, 大盤 > MA60 → 平倉")

    below_ma60 = (market_close < ma60).values
    # Short signal: 1 when below MA60, 0 when above
    # Add hysteresis: enter when below MA60, exit when above MA60 by 1%
    ma60_val = ma60.values
    mkt_val = market_close.values

    t1_signal = np.zeros(n_cal, dtype=int)
    in_short = False
    for i in range(120, n_cal):
        if np.isnan(ma60_val[i]):
            continue
        if not in_short and mkt_val[i] < ma60_val[i]:
            in_short = True
        elif in_short and mkt_val[i] > ma60_val[i] * 1.01:  # exit with 1% buffer
            in_short = False
        t1_signal[i] = 1 if in_short else 0

    # Returns: short the market = negative market return
    t1_dr = pd.Series(0.0, index=cal)
    mkt_ret_vals = mkt_ret.values
    for i in range(n_cal):
        if t1_signal[i] == 1 and not np.isnan(mkt_ret_vals[i]):
            t1_dr.iloc[i] = -mkt_ret_vals[i] - COST_SHORT / 252  # daily cost amortized

    # Count trades (signal changes)
    t1_trades = np.sum(np.diff(t1_signal) != 0)
    t1_active = (t1_signal == 1).sum()
    print(f"  訊號切換次數: {t1_trades}")
    print(f"  做空天數: {t1_active:,} ({t1_active/n_cal:.1%})")
    print(f"  與 V20 idle 重疊: {((t1_signal == 1) & v20_idle).sum():,} / {(t1_signal == 1).sum():,} ({((t1_signal == 1) & v20_idle).sum() / max((t1_signal == 1).sum(), 1):.1%})")

    t1_m = compute_metrics(t1_dr, "T1: MA60 Short")
    print(f"  CAGR: {t1_m['CAGR']:.1%}, Sharpe: {t1_m['Sharpe']:.2f}, MDD: {t1_m['MDD']:.1%}")
    print(f"  年正報酬: {t1_m['yr_pos']}/{t1_m['yr_total']}")

    # === Strategy T2: Breadth Sustained Low ===
    print("\n" + "=" * 70)
    print("T2: 廣度持續低檔做空")
    print("=" * 70)
    print("  訊號: 廣度<30% 持續>5天 → 做空, 廣度>50% → 平倉")

    breadth_vals = above_ma20.values
    t2_signal = np.zeros(n_cal, dtype=int)
    in_short = False
    low_breadth_count = 0

    for i in range(20, n_cal):
        if np.isnan(breadth_vals[i]):
            continue
        if breadth_vals[i] < 0.30:
            low_breadth_count += 1
        else:
            low_breadth_count = 0

        if not in_short and low_breadth_count >= 5:
            in_short = True
        elif in_short and breadth_vals[i] > 0.50:
            in_short = False
        t2_signal[i] = 1 if in_short else 0

    t2_dr = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        if t2_signal[i] == 1 and not np.isnan(mkt_ret_vals[i]):
            t2_dr.iloc[i] = -mkt_ret_vals[i] - COST_SHORT / 252

    t2_trades = np.sum(np.diff(t2_signal) != 0)
    t2_active = (t2_signal == 1).sum()
    print(f"  訊號切換次數: {t2_trades}")
    print(f"  做空天數: {t2_active:,} ({t2_active/n_cal:.1%})")
    print(f"  與 V20 idle 重疊: {((t2_signal == 1) & v20_idle).sum():,} / {(t2_signal == 1).sum():,} ({((t2_signal == 1) & v20_idle).sum() / max((t2_signal == 1).sum(), 1):.1%})")

    t2_m = compute_metrics(t2_dr, "T2: Breadth Short")
    print(f"  CAGR: {t2_m['CAGR']:.1%}, Sharpe: {t2_m['Sharpe']:.2f}, MDD: {t2_m['MDD']:.1%}")
    print(f"  年正報酬: {t2_m['yr_pos']}/{t2_m['yr_total']}")

    # === Strategy T3: Volatility Breakdown Short ===
    print("\n" + "=" * 70)
    print("T3: 波動率突破 + MA20下 做空")
    print("=" * 70)
    print("  訊號: vol_20d > vol_60d * 1.5 AND 大盤 < MA20 → 做空")
    print("  平倉: vol 回到正常 OR 大盤回到 MA20 上")

    below_ma20 = (market_close < ma20).values
    vol_spike = np.zeros(n_cal, dtype=bool)
    vol_20_vals = vol_20d.values
    vol_60_vals = vol_60d.values

    for i in range(60, n_cal):
        if np.isnan(vol_20_vals[i]) or np.isnan(vol_60_vals[i]):
            continue
        if vol_20_vals[i] > vol_60_vals[i] * 1.5 and below_ma20[i]:
            vol_spike[i] = True

    # Smooth: hold signal for 5 days after trigger
    t3_signal = vol_spike.astype(int)
    for i in range(5, n_cal):
        if t3_signal[max(0, i-5):i].sum() > 0 and below_ma20[i]:
            t3_signal[i] = 1

    t3_dr = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        if t3_signal[i] == 1 and not np.isnan(mkt_ret_vals[i]):
            t3_dr.iloc[i] = -mkt_ret_vals[i] - COST_SHORT / 252

    t3_active = (t3_signal == 1).sum()
    print(f"  做空天數: {t3_active:,} ({t3_active/n_cal:.1%})")
    print(f"  與 V20 idle 重疊: {((t3_signal == 1) & v20_idle).sum():,} / {(t3_signal == 1).sum():,} ({((t3_signal == 1) & v20_idle).sum() / max((t3_signal == 1).sum(), 1):.1%})")

    t3_m = compute_metrics(t3_dr, "T3: Vol Breakdown")
    print(f"  CAGR: {t3_m['CAGR']:.1%}, Sharpe: {t3_m['Sharpe']:.2f}, MDD: {t3_m['MDD']:.1%}")
    print(f"  年正報酬: {t3_m['yr_pos']}/{t3_m['yr_total']}")

    # === Strategy T4: Combined (T1 OR T2) ===
    print("\n" + "=" * 70)
    print("T4: 組合訊號 (T1 OR T2)")
    print("=" * 70)

    t4_signal = np.maximum(t1_signal, t2_signal)
    t4_dr = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        if t4_signal[i] == 1 and not np.isnan(mkt_ret_vals[i]):
            t4_dr.iloc[i] = -mkt_ret_vals[i] - COST_SHORT / 252

    t4_active = (t4_signal == 1).sum()
    print(f"  做空天數: {t4_active:,} ({t4_active/n_cal:.1%})")
    print(f"  與 V20 idle 重疊: {((t4_signal == 1) & v20_idle).sum():,} / {(t4_signal == 1).sum():,} ({((t4_signal == 1) & v20_idle).sum() / max((t4_signal == 1).sum(), 1):.1%})")

    t4_m = compute_metrics(t4_dr, "T4: Combined")
    print(f"  CAGR: {t4_m['CAGR']:.1%}, Sharpe: {t4_m['Sharpe']:.2f}, MDD: {t4_m['MDD']:.1%}")
    print(f"  年正報酬: {t4_m['yr_pos']}/{t4_m['yr_total']}")

    # === Comparison Table ===
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    print(f"\n  {'策略':<20} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'年正':>6} {'3yr最低':>8} {'Active%':>8}")
    print("  " + "-" * 72)
    for m, sig in [(t1_m, t1_signal), (t2_m, t2_signal), (t3_m, t3_signal), (t4_m, t4_signal)]:
        mrs = f"{m['min_roll_sharpe']:.2f}" if not np.isnan(m.get('min_roll_sharpe', np.nan)) else "N/A"
        active_pct = (sig == 1).mean()
        print(f"  {m['label']:<20} {m['CAGR']:>8.1%} {m['Sharpe']:>8.2f} {m['MDD']:>8.1%} "
              f"{m['yr_pos']}/{m['yr_total']:<4} {mrs:>8} {active_pct:>8.1%}")

    # === Yearly detail for best strategy ===
    best_m = max([t1_m, t2_m, t3_m, t4_m], key=lambda m: m["Sharpe"])
    print(f"\n" + "=" * 70)
    print(f"BEST STRATEGY YEARLY: {best_m['label']}")
    print("=" * 70)

    mkt_yearly = {}
    for yr in range(2010, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() > 50:
            mkt_yearly[yr] = (1 + mkt_ret[yr_mask]).prod() - 1

    print(f"\n  {'Year':>6} {'策略':>8} {'大盤':>8} {'判定':>6}")
    print("  " + "-" * 35)
    for yr in sorted(best_m["yearly"].keys()):
        ret = best_m["yearly"][yr]
        mkt = mkt_yearly.get(yr, 0)
        sign = "PASS" if ret > 0 else "FAIL"
        print(f"  {yr:>6} {ret:>8.1%} {mkt:>8.1%} {sign:>6}")

    # === V20 + Best Trend Short combined ===
    print("\n" + "=" * 70)
    print("V20 + TREND SHORT COMBINED")
    print("=" * 70)

    # Load V20 daily returns from P29
    v20_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p29_regime\p29_daily_state.csv")
    if v20_path.exists():
        v20_state = pd.read_csv(v20_path, parse_dates=["date"])
        v20_dr = pd.Series(v20_state["v20_daily_ret"].values, index=pd.DatetimeIndex(v20_state["date"]))
    else:
        print("  V20 daily state not found, skipping combined.")
        v20_dr = None

    if v20_dr is not None:
        # Align
        v20_dr = v20_dr.reindex(cal)
        v20_dr = v20_dr.fillna(0)

        # Combined: V20 70% + Trend Short 30%
        for w_v20, w_ts, label in [(1.0, 0.0, "V20 only"), (0.7, 0.3, "V20 70% + TS 30%"),
                                    (0.6, 0.4, "V20 60% + TS 40%"), (0.5, 0.5, "V20 50% + TS 50%")]:
            combined = v20_dr * w_v20 + t4_dr * w_ts
            cm = compute_metrics(combined, label)
            if cm:
                mrs = f"{cm['min_roll_sharpe']:.2f}" if not np.isnan(cm.get('min_roll_sharpe', np.nan)) else "N/A"
                print(f"  {label:<22} CAGR={cm['CAGR']:>7.1%} Sharpe={cm['Sharpe']:>5.2f} MDD={cm['MDD']:>7.1%} "
                      f"年正={cm['yr_pos']}/{cm['yr_total']} 3yr={mrs}")

        # Correlation
        corr = v20_dr.corr(t4_dr)
        print(f"\n  V20 vs T4 correlation: {corr:.4f}")

        # Bear year check
        print(f"\n  BEAR YEAR CHECK (V20 70% + T4 30%):")
        combined_7030 = v20_dr * 0.7 + t4_dr * 0.3
        print(f"  {'Year':>6} {'大盤':>8} {'V20':>8} {'TS':>8} {'組合':>8}")
        print("  " + "-" * 42)
        for yr in range(2010, 2027):
            yr_mask = cal.year == yr
            if yr_mask.sum() < 50:
                continue
            mkt_yr = mkt_yearly.get(yr, 0)
            if mkt_yr < -0.05:
                v20_yr = (1 + v20_dr[yr_mask]).prod() - 1
                ts_yr = (1 + t4_dr[yr_mask]).prod() - 1
                comb_yr = (1 + combined_7030[yr_mask]).prod() - 1
                print(f"  {yr:>6} {mkt_yr:>8.1%} {v20_yr:>8.1%} {ts_yr:>8.1%} {comb_yr:>8.1%}")

    # Save
    pd.DataFrame({
        "date": cal,
        "t1_signal": t1_signal,
        "t2_signal": t2_signal,
        "t3_signal": t3_signal,
        "t4_signal": t4_signal,
        "t1_daily": t1_dr.values,
        "t2_daily": t2_dr.values,
        "t3_daily": t3_dr.values,
        "t4_daily": t4_dr.values,
    }).to_csv(OUT / "p30_signals.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
