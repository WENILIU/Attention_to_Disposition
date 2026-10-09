"""P19h: Diagnose 2025 failure and test fixes.

Hypotheses for 2025 failure:
1. Low market volatility → more false positives
2. Lower model confidence → should filter by confidence
3. Different event mix (less severe triggers)
4. Market breadth low → fewer real opportunities

Fixes to test:
A. Market volatility filter (only trade when market vol > threshold)
B. Minimum model confidence (raise threshold in low-confidence periods)
C. Event severity filter (require pct_change > X)
D. Market breadth filter (require N+ attention events this week)
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19h_fix2025")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P19h: 2025年失敗診斷與改善")
    print("=" * 70)

    # Load walk-forward results
    print("\nLoading walk-forward predictions...")
    pred_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19g_walkforward\p19g_all_predictions.csv")
    att = pd.read_csv(pred_path, parse_dates=["announce"])
    att["stock_id"] = att["stock_id"].astype(str).str.zfill(4)
    att = att[att["oos_prob"].notna() & att["trade_ret"].notna()].copy()
    print(f"  Events with predictions + returns: {len(att):,}")

    # Load market data for diagnostics
    print("Loading market data...")
    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()

    # Market volatility proxy: average absolute return of all stocks
    returns = close.pct_change()
    market_vol = returns.abs().mean(axis=1)  # cross-sectional avg abs return
    market_vol_20d = market_vol.rolling(20).mean()

    # Market breadth: count of attention events per week
    att["week"] = att["announce"].dt.isocalendar().week.astype(int)
    att["year_week"] = att["announce"].dt.year * 100 + att["week"]
    weekly_count = att.groupby("year_week").size()
    att["weekly_attention_count"] = att["year_week"].map(weekly_count)

    # Add market vol to each event
    att["ann_idx"] = cal.searchsorted(att["announce"], side="left")
    att["market_vol"] = market_vol_20d.reindex(cal).iloc[att["ann_idx"].clip(upper=len(cal)-1)].values

    # === DIAGNOSIS ===
    print("\n" + "=" * 70)
    print("診斷: 2025 vs 其他年份")
    print("=" * 70)

    selected = att[att["oos_prob"] >= 0.5].copy()
    print(f"\n  Selected events (prob>=0.5): {len(selected):,}")

    print("\n  年份對比:")
    print(f"  {'Year':>6} {'n':>5} {'avg_ret':>8} {'win%':>6} {'prec%':>6} {'mkt_vol':>8} {'wk_count':>8} {'avg_prob':>8}")
    for yr in sorted(selected["year"].unique()):
        yr_data = selected[selected["year"] == yr]
        if len(yr_data) < 10:
            continue
        avg_ret = yr_data["trade_ret"].mean()
        win = (yr_data["trade_ret"] > 0).mean()
        prec = yr_data["leads_to_disposal"].mean()
        mkt_vol = yr_data["market_vol"].mean()
        wk_cnt = yr_data["weekly_attention_count"].mean()
        avg_prob = yr_data["oos_prob"].mean()
        print(f"  {yr:>6} {len(yr_data):>5} {avg_ret:>8.2%} {win:>6.1%} {prec:>6.1%} {mkt_vol:>8.4f} {wk_cnt:>8.0f} {avg_prob:>8.3f}")

    # === TEST FIXES ===
    print("\n" + "=" * 70)
    print("測試改善方案")
    print("=" * 70)

    # Baseline
    print("\n  [Baseline] threshold=0.5, no filter:")
    for yr in sorted(selected["year"].unique()):
        yr_data = selected[selected["year"] == yr]
        if len(yr_data) < 10:
            continue
        avg = yr_data["trade_ret"].mean()
        sign = "PASS" if avg > 0 else "FAIL"
        print(f"    {yr}: {avg:.2%}, n={len(yr_data)} {sign}")

    # FIX A: Market volatility filter
    print("\n  [Fix A] 市場波動率過濾 (只交易 market_vol > X):")
    for vol_thresh in [0.015, 0.018, 0.02, 0.022, 0.025]:
        filtered = selected[selected["market_vol"] >= vol_thresh]
        yearly_results = []
        for yr in sorted(filtered["year"].unique()):
            yr_data = filtered[filtered["year"] == yr]
            if len(yr_data) < 10:
                yearly_results.append((yr, np.nan, 0))
                continue
            yearly_results.append((yr, yr_data["trade_ret"].mean(), len(yr_data)))
        n_pass = sum(1 for _, r, n in yearly_results if not np.isnan(r) and r > 0 and n >= 10)
        n_total = sum(1 for _, r, n in yearly_results if not np.isnan(r) and n >= 10)
        overall = filtered["trade_ret"].mean()
        print(f"    vol>={vol_thresh:.3f}: overall={overall:.2%}, years_pass={n_pass}/{n_total}")
        if vol_thresh == 0.02:  # Show detail for best candidate
            for yr, avg, n in yearly_results:
                if n >= 10:
                    sign = "PASS" if avg > 0 else "FAIL"
                    print(f"      {yr}: {avg:.2%}, n={n} {sign}")

    # FIX B: Higher model confidence
    print("\n  [Fix B] 提高模型信心門檻:")
    for prob_thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
        filtered = att[att["oos_prob"] >= prob_thresh].copy()
        yearly_results = []
        for yr in sorted(filtered["year"].unique()):
            yr_data = filtered[filtered["year"] == yr]
            if len(yr_data) < 10:
                continue
            yearly_results.append((yr, yr_data["trade_ret"].mean(), len(yr_data)))
        n_pass = sum(1 for _, r, n in yearly_results if r > 0 and n >= 10)
        n_total = len(yearly_results)
        overall = filtered["trade_ret"].mean()
        print(f"    prob>={prob_thresh:.2f}: overall={overall:.2%}, years_pass={n_pass}/{n_total}")
        if prob_thresh == 0.6:
            for yr, avg, n in yearly_results:
                sign = "PASS" if avg > 0 else "FAIL"
                print(f"      {yr}: {avg:.2%}, n={n} {sign}")

    # FIX C: Event severity (pct_change from reason text)
    print("\n  [Fix C] 事件嚴重度過濾 (需要高漲幅):")
    # Need to re-extract pct_change from the predictions
    # Load from P19e predictions which have the features
    p19e_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19e_better_pred\p19e_predictions.csv")
    if p19e_path.exists():
        p19e = pd.read_csv(p19e_path, parse_dates=["announce"])
        p19e["stock_id"] = p19e["stock_id"].astype(str).str.zfill(4)
        # Merge pct_change into selected
        selected_merged = selected.merge(
            p19e[["stock_id", "announce", "pct_change"]],
            on=["stock_id", "announce"], how="left"
        )
        for pct_thresh in [0, 35, 50, 70]:
            filtered = selected_merged[selected_merged["pct_change"] >= pct_thresh]
            yearly_results = []
            for yr in sorted(filtered["year"].unique()):
                yr_data = filtered[filtered["year"] == yr]
                if len(yr_data) < 10:
                    continue
                yearly_results.append((yr, yr_data["trade_ret"].mean(), len(yr_data)))
            n_pass = sum(1 for _, r, n in yearly_results if r > 0)
            n_total = len(yearly_results)
            overall = filtered["trade_ret"].mean() if len(filtered) > 0 else 0
            print(f"    pct>={pct_thresh}%: overall={overall:.2%}, years_pass={n_pass}/{n_total}, n={len(filtered)}")
    else:
        print("    P19e predictions not found, skipping")

    # FIX D: Market breadth (weekly attention count)
    print("\n  [Fix D] 市場廣度過濾 (每週注意事件數 > X):")
    for breadth_thresh in [50, 80, 100, 120, 150]:
        filtered = selected[selected["weekly_attention_count"] >= breadth_thresh]
        yearly_results = []
        for yr in sorted(filtered["year"].unique()):
            yr_data = filtered[filtered["year"] == yr]
            if len(yr_data) < 10:
                continue
            yearly_results.append((yr, yr_data["trade_ret"].mean(), len(yr_data)))
        n_pass = sum(1 for _, r, n in yearly_results if r > 0)
        n_total = len(yearly_results)
        overall = filtered["trade_ret"].mean() if len(filtered) > 0 else 0
        print(f"    weekly>={breadth_thresh}: overall={overall:.2%}, years_pass={n_pass}/{n_total}, n={len(filtered)}")
        if breadth_thresh == 100:
            for yr, avg, n in yearly_results:
                sign = "PASS" if avg > 0 else "FAIL"
                print(f"      {yr}: {avg:.2%}, n={n} {sign}")

    # === COMBINED BEST FIX ===
    print("\n" + "=" * 70)
    print("組合最佳方案")
    print("=" * 70)

    # Try: prob>=0.5 + market_vol>=0.02
    combo1 = selected[selected["market_vol"] >= 0.02]
    print("\n  [A+B] prob>=0.5 + market_vol>=0.02:")
    for yr in sorted(combo1["year"].unique()):
        yr_data = combo1[combo1["year"] == yr]
        if len(yr_data) < 10:
            continue
        avg = yr_data["trade_ret"].mean()
        sign = "PASS" if avg > 0 else "FAIL"
        print(f"    {yr}: {avg:.2%}, n={len(yr_data)} {sign}")

    # Try: prob>=0.5 + weekly_breadth>=100
    combo2 = selected[selected["weekly_attention_count"] >= 100]
    print("\n  [A+D] prob>=0.5 + weekly_breadth>=100:")
    for yr in sorted(combo2["year"].unique()):
        yr_data = combo2[combo2["year"] == yr]
        if len(yr_data) < 10:
            continue
        avg = yr_data["trade_ret"].mean()
        sign = "PASS" if avg > 0 else "FAIL"
        print(f"    {yr}: {avg:.2%}, n={len(yr_data)} {sign}")

    # Try: prob>=0.6 + market_vol>=0.018
    combo3 = att[(att["oos_prob"] >= 0.6) & (att["market_vol"] >= 0.018)]
    print("\n  [B+A] prob>=0.6 + market_vol>=0.018:")
    for yr in sorted(combo3["year"].unique()):
        yr_data = combo3[combo3["year"] == yr]
        if len(yr_data) < 10:
            continue
        avg = yr_data["trade_ret"].mean()
        sign = "PASS" if avg > 0 else "FAIL"
        print(f"    {yr}: {avg:.2%}, n={len(yr_data)} {sign}")

    # Save
    selected.to_csv(OUT / "p19h_selected_events.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
