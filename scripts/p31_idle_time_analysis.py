"""P31: 空窗期年度分佈 — 處置事件密度 vs 報酬的因果關係.

核心問題: 早期處置股少 → 空窗多 → CAGR低？
驗證: 逐年處置事件數、空窗天數、實際報酬的對應關係.
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
import finlab
from finlab import data

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p31_idle_analysis")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("P31: 空窗期年度分佈 — 處置事件密度 vs 報酬")
    print("=" * 70)

    # Load daily state from P29
    daily = pd.read_csv(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p29_regime\p29_daily_state.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year

    # Load disposal events count per year
    dis_raw = pd.DataFrame(data.get("disposal_information"))
    dis = dis_raw.copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis = dis[dis["stock_id"].str.match(r"^\d{4}$") & ~dis["stock_id"].str.startswith(("00", "91"))].copy()
    dis["year"] = dis["start"].dt.year

    events_per_year = dis.groupby("year").size()

    # === Year-by-year analysis ===
    print("\n" + "=" * 70)
    print("逐年分析: 處置事件數 → 空窗率 → 報酬")
    print("=" * 70)

    rows = []
    for yr in sorted(daily["year"].unique()):
        yr_data = daily[daily["year"] == yr]
        if len(yr_data) < 50:
            continue

        total_days = len(yr_data)
        idle_days = (yr_data["v20_positions"] == 0).sum()
        active_days = total_days - idle_days
        idle_pct = idle_days / total_days

        # Calendar return (what you actually get)
        cal_ret = (1 + yr_data["v20_daily_ret"].fillna(0)).prod() - 1

        # Active-only return (strategy efficiency)
        active_ret_data = yr_data[yr_data["v20_positions"] > 0]["v20_daily_ret"]
        if len(active_ret_data) > 0:
            active_ret = (1 + active_ret_data.fillna(0)).prod() - 1
            active_ann = (1 + active_ret) ** (252 / len(active_ret_data)) - 1
        else:
            active_ret = 0
            active_ann = 0

        n_events = events_per_year.get(yr, 0)

        rows.append({
            "year": yr,
            "events": n_events,
            "total_days": total_days,
            "idle_days": idle_days,
            "idle_pct": idle_pct,
            "active_days": active_days,
            "cal_return": cal_ret,
            "active_return": active_ret,
            "active_annualized": active_ann,
        })

    df = pd.DataFrame(rows)

    print(f"\n{'年份':>6} {'事件數':>7} {'總日':>5} {'空窗':>5} {'空窗%':>7} {'日曆報酬':>9} {'活躍報酬':>9} {'活躍年化':>9}")
    print("-" * 80)
    for _, r in df.iterrows():
        print(f"{int(r['year']):>6} {int(r['events']):>7} {int(r['total_days']):>5} {int(r['idle_days']):>5} {r['idle_pct']:>7.1%} {r['cal_return']:>9.1%} {r['active_return']:>9.1%} {r['active_annualized']:>9.1%}")

    # === Correlation analysis ===
    print("\n" + "=" * 70)
    print("相關性分析")
    print("=" * 70)

    corr_events_idle = df["events"].corr(df["idle_pct"])
    corr_events_calret = df["events"].corr(df["cal_return"])
    corr_idle_calret = df["idle_pct"].corr(df["cal_return"])
    corr_events_activeann = df["events"].corr(df["active_annualized"])

    print(f"\n  事件數 vs 空窗率: {corr_events_idle:.4f}")
    print(f"  事件數 vs 日曆報酬: {corr_events_calret:.4f}")
    print(f"  空窗率 vs 日曆報酬: {corr_idle_calret:.4f}")
    print(f"  事件數 vs 活躍年化: {corr_events_activeann:.4f}")

    # === Period comparison ===
    print("\n" + "=" * 70)
    print("時期對比")
    print("=" * 70)

    periods = [
        ("2010-2013", 2010, 2013),
        ("2014-2016", 2014, 2016),
        ("2017-2019", 2017, 2019),
        ("2020-2022", 2020, 2022),
        ("2023-2025", 2023, 2025),
        ("2026", 2026, 2026),
    ]

    print(f"\n{'時期':>12} {'年均事件':>9} {'平均空窗%':>10} {'平均日曆報酬':>12} {'平均活躍年化':>12}")
    print("-" * 70)
    for label, y1, y2 in periods:
        mask = (df["year"] >= y1) & (df["year"] <= y2)
        sub = df[mask]
        if len(sub) == 0:
            continue
        print(f"{label:>12} {sub['events'].mean():>9.0f} {sub['idle_pct'].mean():>10.1%} {sub['cal_return'].mean():>12.1%} {sub['active_annualized'].mean():>12.1%}")

    # === Key insight: What happens going forward ===
    print("\n" + "=" * 70)
    print("前瞻推算: 未來空窗率趨勢")
    print("=" * 70)

    recent = df[df["year"] >= 2022]
    print(f"\n  2022-2026 平均空窗率: {recent['idle_pct'].mean():.1%}")
    print(f"  2024-2026 平均空窗率: {df[df['year']>=2024]['idle_pct'].mean():.1%}")
    print(f"  2025-2026 平均空窗率: {df[df['year']>=2025]['idle_pct'].mean():.1%}")

    # If idle → 0, what would calendar CAGR be?
    full_period = df[df["year"] >= 2011]
    hist_cal_cagr = (np.prod(1 + full_period["cal_return"]) ** (1/len(full_period))) - 1
    hist_active_cagr = (np.prod(1 + full_period["active_annualized"]) ** (1/len(full_period))) - 1

    recent_period = df[df["year"] >= 2020]
    recent_cal_cagr = (np.prod(1 + recent_period["cal_return"]) ** (1/len(recent_period))) - 1
    recent_active_cagr = (np.prod(1 + recent_period["active_annualized"]) ** (1/len(recent_period))) - 1

    print(f"\n  全期(2011-2026) 日曆CAGR: {hist_cal_cagr:.1%}")
    print(f"  全期(2011-2026) 活躍CAGR: {hist_active_cagr:.1%}")
    print(f"  近7年(2020-2026) 日曆CAGR: {recent_cal_cagr:.1%}")
    print(f"  近7年(2020-2026) 活躍CAGR: {recent_active_cagr:.1%}")

    # Gap analysis
    print(f"\n  差距分析:")
    print(f"    全期: 日曆 vs 活躍差距 = {hist_active_cagr - hist_cal_cagr:.1%}")
    print(f"    近7年: 日曆 vs 活躍差距 = {recent_active_cagr - recent_cal_cagr:.1%}")

    # === Monthly idle distribution ===
    print("\n" + "=" * 70)
    print("月度空窗分佈 (確認是否集中在特定月份)")
    print("=" * 70)

    daily["month"] = daily["date"].dt.month
    monthly_idle = daily.groupby("month").apply(lambda g: (g["v20_positions"]==0).mean())
    print(f"\n  {'月份':>4} {'空窗率':>8}")
    for m, v in monthly_idle.items():
        bar = "█" * int(v * 40)
        print(f"  {int(m):>4} {v:>8.1%} {bar}")

    # === Save ===
    df.to_csv(OUT / "p31_yearly_idle.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT / 'p31_yearly_idle.csv'}")

if __name__ == "__main__":
    main()
