"""P30b: V20 + TS 正確條件式配置 — 資金輪替，非同時分倉.

正確邏輯:
  V20 有持倉 → 100% 資金給 V20，TS 不啟用
  V20 無持倉 + TS 訊號 → 100% 資金給 TS
  V20 無持倉 + TS 無訊號 → 現金

這是「資金輪替」不是「資金分割」。
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
    print("P30b: V20 + TS 資金輪替（正確條件式）")
    print("=" * 70)

    # Load V20 daily state from P29
    v20_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p29_regime\p29_daily_state.csv")
    v20_state = pd.read_csv(v20_path, parse_dates=["date"])
    v20_dr = pd.Series(v20_state["v20_daily_ret"].values, index=pd.DatetimeIndex(v20_state["date"]))
    v20_positions = pd.Series(v20_state["v20_positions"].values, index=pd.DatetimeIndex(v20_state["date"]))

    cal = v20_dr.index
    n_cal = len(cal)

    # Load TS signals from P30
    ts_path = OUT / "p30_signals.csv"
    ts_data = pd.read_csv(ts_path, parse_dates=["date"])
    t1_signal = pd.Series(ts_data["t1_signal"].values, index=pd.DatetimeIndex(ts_data["date"]))
    t1_daily = pd.Series(ts_data["t1_daily"].values, index=pd.DatetimeIndex(ts_data["date"]))

    # Align
    t1_signal = t1_signal.reindex(cal).fillna(0).astype(int)
    t1_daily = t1_daily.reindex(cal).fillna(0.0)

    # === Conditional allocation ===
    # V20 active → V20 return only
    # V20 idle + TS signal → TS return (full capital)
    # V20 idle + no TS → cash (0)

    v20_active = v20_positions > 0
    ts_active = t1_signal == 1

    # Strategy A: Pure conditional (V20 priority)
    combined_cond = pd.Series(0.0, index=cal)
    combined_cond[v20_active] = v20_dr[v20_active].values  # V20 gets full capital
    ts_only = (~v20_active) & ts_active
    combined_cond[ts_only] = t1_daily[ts_only].values  # TS gets full capital when V20 idle

    # Strategy B: V20 + TS overlay (TS active regardless, but V20 takes priority)
    # When both active: V20 70% + TS 30%
    # When only V20: V20 100%
    # When only TS: TS 100%
    combined_overlay = pd.Series(0.0, index=cal)
    both_active = v20_active & ts_active
    v20_only = v20_active & ~ts_active
    ts_only_mask = ~v20_active & ts_active

    combined_overlay[both_active] = (v20_dr[both_active] * 0.7 + t1_daily[both_active] * 0.3).values
    combined_overlay[v20_only] = v20_dr[v20_only].values
    combined_overlay[ts_only_mask] = t1_daily[ts_only_mask].values

    # === Results ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    configs = [
        (v20_dr, "V20 單獨 (基準)"),
        (combined_cond, "V20 + TS 資金輪替"),
        (combined_overlay, "V20 + TS 重疊70/30"),
        (t1_daily, "TS 單獨"),
    ]

    print(f"\n  {'策略':<24} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'年正':>6} {'3yr最低':>8}")
    print("  " + "-" * 65)

    results = []
    for dr, label in configs:
        m = compute_metrics(dr, label)
        if not m:
            continue
        results.append(m)
        mrs = f"{m['min_roll_sharpe']:.2f}" if not np.isnan(m.get('min_roll_sharpe', np.nan)) else "N/A"
        print(f"  {label:<24} {m['CAGR']:>8.1%} {m['Sharpe']:>8.2f} {m['MDD']:>8.1%} "
              f"{m['yr_pos']}/{m['yr_total']:<4} {mrs:>8}")

    # === Time allocation breakdown ===
    print("\n" + "=" * 70)
    print("TIME ALLOCATION BREAKDOWN")
    print("=" * 70)

    total = n_cal
    v20_days = v20_active.sum()
    ts_only_days = ts_only.sum()
    both_days = (v20_active & ts_active).sum()
    cash_days = (~v20_active & ~ts_active).sum()

    print(f"\n  V20 活躍: {v20_days:,} 天 ({v20_days/total:.1%})")
    print(f"  TS 單獨 (V20 idle): {ts_only_days:,} 天 ({ts_only_days/total:.1%})")
    print(f"  兩者同時: {both_days:,} 天 ({both_days/total:.1%})")
    print(f"  現金等待: {cash_days:,} 天 ({cash_days/total:.1%})")
    print(f"\n  資金使用率: {(v20_days + ts_only_days + both_days) / total:.1%}")

    # === Yearly ===
    print("\n" + "=" * 70)
    print("YEARLY: V20 + TS 資金輪替")
    print("=" * 70)

    close = data.get("price:收盤價")
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_dr = pd.Series(mkt_ret.values, index=pd.DatetimeIndex(close.index).normalize().unique().sort_values())
    mkt_dr = mkt_dr.reindex(cal).fillna(0)

    print(f"\n  {'Year':>6} {'V20':>8} {'TS填補':>8} {'組合':>8} {'大盤':>8} {'TS貢獻':>8}")
    print("  " + "-" * 50)

    for yr in range(2010, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() < 50:
            continue
        v20_yr = (1 + v20_dr[yr_mask]).prod() - 1
        comb_yr = (1 + combined_cond[yr_mask]).prod() - 1
        mkt_yr = (1 + mkt_dr[yr_mask]).prod() - 1
        ts_contrib = comb_yr - v20_yr

        # TS contribution in this year
        ts_yr_mask = yr_mask & ts_only
        if ts_yr_mask.sum() > 0:
            ts_yr = (1 + t1_daily[ts_yr_mask]).prod() - 1
        else:
            ts_yr = 0

        print(f"  {yr:>6} {v20_yr:>8.1%} {ts_yr:>8.1%} {comb_yr:>8.1%} {mkt_yr:>8.1%} {ts_contrib:>+8.1%}")

    # === Bear years ===
    print("\n" + "=" * 70)
    print("BEAR YEARS")
    print("=" * 70)

    for yr in [2011, 2015, 2018, 2022]:
        yr_mask = cal.year == yr
        if yr_mask.sum() < 50:
            continue
        v20_yr = (1 + v20_dr[yr_mask]).prod() - 1
        comb_yr = (1 + combined_cond[yr_mask]).prod() - 1
        mkt_yr = (1 + mkt_dr[yr_mask]).prod() - 1
        ts_days_yr = (yr_mask & ts_only).sum()
        print(f"  {yr}: 大盤 {mkt_yr:+.1%}, V20 {v20_yr:+.1%}, 組合 {comb_yr:+.1%}, TS填補 {ts_days_yr} 天")

    # === Conclusion ===
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(results) >= 2:
        base = results[0]
        cond = results[1]
        print(f"\n  V20 基準: CAGR {base['CAGR']:.1%}, Sharpe {base['Sharpe']:.2f}, MDD {base['MDD']:.1%}")
        print(f"  + TS 輪替: CAGR {cond['CAGR']:.1%}, Sharpe {cond['Sharpe']:.2f}, MDD {cond['MDD']:.1%}")
        print(f"\n  CAGR 差異: {cond['CAGR'] - base['CAGR']:+.1%}")
        print(f"  Sharpe 差異: {cond['Sharpe'] - base['Sharpe']:+.2f}")
        print(f"  MDD 差異: {cond['MDD'] - base['MDD']:+.1%}")

        if cond["CAGR"] > base["CAGR"]:
            print(f"\n  ✅ TS 輪替增加了 CAGR（不稀釋 V20）")
        if cond["MDD"] > base["MDD"]:  # less negative = better
            print(f"  ✅ MDD 降低了")
        if cond["yr_pos"] > base["yr_pos"]:
            print(f"  ✅ 年正報酬提升: {base['yr_pos']}/{base['yr_total']} → {cond['yr_pos']}/{cond['yr_total']}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
