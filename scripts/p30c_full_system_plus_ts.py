"""P30c: 現有系統 (V20+過濾+Short) + TS 輪替.

完整系統邏輯:
  V20 有持倉 → V20 70% + S1 Short 30%（現有系統）
  V20 無持倉 + TS 訊號 → TS 100%（資金輪替）
  V20 無持倉 + 無 TS → 現金

S1 訊號: 廣度5日下降>15% → 做空3天
TS 訊號: 大盤 < MA60 → 持續做空直到回到 MA60 上
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

COST_SHORT = 0.001


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

    roll = dr_active.rolling(756, min_periods=504).mean() / dr_active.rolling(756, min_periods=504).std() * np.sqrt(252)
    roll_valid = roll.dropna()
    min_roll = roll_valid.min() if len(roll_valid) > 100 else np.nan

    return {
        "label": label, "CAGR": cagr, "Sharpe": sharpe, "MDD": mdd,
        "yearly": yearly, "yr_pos": yr_pos, "yr_total": yr_total,
        "min_roll_sharpe": min_roll,
    }


def main():
    auto_login()
    from finlab import data

    print("=" * 70)
    print("P30c: 現有系統 + TS 輪替")
    print("=" * 70)

    print("\nLoading data...")
    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Market data
    mkt_ret = close.pct_change().mean(axis=1)
    market_close = close.sum(axis=1)
    ma60 = market_close.rolling(60).mean()

    # Breadth
    stock_ma20 = close.rolling(20).mean()
    above_ma20 = (close > stock_ma20).sum(axis=1) / close.notna().sum(axis=1)
    breadth_decline_5d = above_ma20 - above_ma20.shift(5)

    # === Load V20 (with breadth filter) from P29 ===
    v20_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p29_regime\p29_daily_state.csv")
    v20_state = pd.read_csv(v20_path, parse_dates=["date"])
    v20_dr = pd.Series(v20_state["v20_daily_ret"].values, index=pd.DatetimeIndex(v20_state["date"]))
    v20_positions = pd.Series(v20_state["v20_positions"].values, index=pd.DatetimeIndex(v20_state["date"]))

    # Align to cal
    v20_dr = v20_dr.reindex(cal).fillna(0)
    v20_positions = v20_positions.reindex(cal).fillna(0).astype(int)

    # === Generate S1 Short signal (breadth decline > 15% in 5d, hold 3 days) ===
    s1_signal = (breadth_decline_5d < -0.15).astype(int)
    # Hold for 3 days after signal
    s1_hold = s1_signal.rolling(3, min_periods=1).max()

    # S1 returns: short market for 3 days
    s1_dr = pd.Series(0.0, index=cal)
    mkt_ret_aligned = mkt_ret.reindex(cal).fillna(0)
    for i in range(n_cal):
        if s1_hold.iloc[i] > 0:
            s1_dr.iloc[i] = -mkt_ret_aligned.iloc[i] - COST_SHORT / 252

    # === Generate TS signal (MA60 trend short) ===
    ts_path = OUT / "p30_signals.csv"
    ts_data = pd.read_csv(ts_path, parse_dates=["date"])
    t1_signal = pd.Series(ts_data["t1_signal"].values, index=pd.DatetimeIndex(ts_data["date"]))
    t1_daily = pd.Series(ts_data["t1_daily"].values, index=pd.DatetimeIndex(ts_data["date"]))
    t1_signal = t1_signal.reindex(cal).fillna(0).astype(int)
    t1_daily = t1_daily.reindex(cal).fillna(0.0)

    # === Build systems ===

    # System A: V20 + Short 30% (existing production)
    # V20 70% + S1 30% when S1 active, V20 100% when S1 not active
    sys_a = pd.Series(0.0, index=cal)
    s1_active = s1_hold > 0
    v20_active = v20_positions > 0

    # When V20 active: V20 70% + S1 30% (if S1 triggered) or V20 100%
    v20_and_s1 = v20_active & s1_active
    v20_only = v20_active & ~s1_active
    sys_a[v20_and_s1] = (v20_dr[v20_and_s1] * 0.7 + s1_dr[v20_and_s1] * 0.3).values
    sys_a[v20_only] = v20_dr[v20_only].values
    # When V20 idle + S1 active: S1 100%
    idle_s1 = ~v20_active & s1_active
    sys_a[idle_s1] = s1_dr[idle_s1].values

    # System B: V20 + Short 30% + TS 輪替 (new)
    # Same as A, but when V20 idle + TS signal → TS 100%
    sys_b = sys_a.copy()
    v20_idle = ~v20_active
    ts_active = t1_signal == 1

    # Override: V20 idle + TS active → use TS (full capital)
    # But don't override S1 (S1 is short-term, TS is sustained)
    # Priority: if S1 is active (3-day window), keep S1. Otherwise use TS.
    ts_override = v20_idle & ts_active & ~s1_active
    sys_b[ts_override] = t1_daily[ts_override].values

    # System C: V20 + Short 30% + TS (TS takes priority over S1 when both active)
    sys_c = pd.Series(0.0, index=cal)
    # V20 active: same as A
    sys_c[v20_and_s1] = (v20_dr[v20_and_s1] * 0.7 + s1_dr[v20_and_s1] * 0.3).values
    sys_c[v20_only] = v20_dr[v20_only].values
    # V20 idle: TS takes priority, S1 only when TS not active
    idle_ts = v20_idle & ts_active
    idle_s1_only = v20_idle & ~ts_active & s1_active
    sys_c[idle_ts] = t1_daily[idle_ts].values
    sys_c[idle_s1_only] = s1_dr[idle_s1_only].values

    # === Results ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    configs = [
        (v20_dr, "V20 單獨"),
        (sys_a, "A: V20+Short (現有系統)"),
        (sys_b, "B: V20+Short+TS (S1優先)"),
        (sys_c, "C: V20+Short+TS (TS優先)"),
    ]

    print(f"\n  {'策略':<28} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'年正':>6} {'3yr最低':>8}")
    print("  " + "-" * 65)

    results = []
    for dr, label in configs:
        m = compute_metrics(dr, label)
        if not m:
            continue
        results.append(m)
        mrs = f"{m['min_roll_sharpe']:.2f}" if not np.isnan(m.get('min_roll_sharpe', np.nan)) else "N/A"
        print(f"  {label:<28} {m['CAGR']:>8.1%} {m['Sharpe']:>8.2f} {m['MDD']:>8.1%} "
              f"{m['yr_pos']}/{m['yr_total']:<4} {mrs:>8}")

    # === Yearly comparison ===
    print("\n" + "=" * 70)
    print("YEARLY: A (現有) vs C (新增TS)")
    print("=" * 70)

    mkt_dr = pd.Series(mkt_ret_aligned.values, index=cal)

    print(f"\n  {'Year':>6} {'A(現有)':>8} {'C(+TS)':>8} {'差異':>8} {'大盤':>8}")
    print("  " + "-" * 42)

    for yr in range(2010, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() < 50:
            continue
        a_yr = (1 + sys_a[yr_mask]).prod() - 1
        c_yr = (1 + sys_c[yr_mask]).prod() - 1
        mkt_yr = (1 + mkt_dr[yr_mask]).prod() - 1
        diff = c_yr - a_yr
        marker = " ★" if abs(diff) > 0.05 else ""
        print(f"  {yr:>6} {a_yr:>8.1%} {c_yr:>8.1%} {diff:>+8.1%} {mkt_yr:>8.1%}{marker}")

    # === Bear years ===
    print("\n" + "=" * 70)
    print("BEAR YEARS")
    print("=" * 70)

    for yr in [2011, 2015, 2018, 2022]:
        yr_mask = cal.year == yr
        if yr_mask.sum() < 50:
            continue
        a_yr = (1 + sys_a[yr_mask]).prod() - 1
        c_yr = (1 + sys_c[yr_mask]).prod() - 1
        mkt_yr = (1 + mkt_dr[yr_mask]).prod() - 1
        ts_days = (yr_mask & v20_idle & ts_active).sum()
        print(f"  {yr}: 大盤 {mkt_yr:+.1%}, A {a_yr:+.1%}, C {c_yr:+.1%}, TS填補 {ts_days} 天")

    # === Conclusion ===
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(results) >= 3:
        a = results[1]  # existing system
        c = results[3]  # with TS
        print(f"\n  現有系統 A: CAGR {a['CAGR']:.1%}, Sharpe {a['Sharpe']:.2f}, MDD {a['MDD']:.1%}, {a['yr_pos']}/{a['yr_total']}年")
        print(f"  加入 TS C: CAGR {c['CAGR']:.1%}, Sharpe {c['Sharpe']:.2f}, MDD {c['MDD']:.1%}, {c['yr_pos']}/{c['yr_total']}年")
        print(f"\n  差異: CAGR {c['CAGR']-a['CAGR']:+.1%}, Sharpe {c['Sharpe']-a['Sharpe']:+.2f}, MDD {c['MDD']-a['MDD']:+.1%}")

        if c["CAGR"] > a["CAGR"] and c["yr_pos"] >= a["yr_pos"]:
            print(f"\n  ✅ TS 輪替在現有系統之上進一步增加了 CAGR 和一致性")
        elif c["CAGR"] > a["CAGR"]:
            print(f"\n  ⚠️ TS 增加了 CAGR 但未提升一致性")
        else:
            print(f"\n  ❌ TS 未增加 CAGR（現有系統已覆蓋）")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
