"""P18b: 停券 (Short Sale Suspension) event strategy feasibility.

停券 = 融券放空暫停. When a stock is too volatile, exchange suspends
new short positions. Mechanism similar to 處置: forced position unwinding.

Data: margin_short_sale_suspension (39,305 events, 2014-2026)
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p18b_short_susp")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P18b: 停券 (Short Sale Suspension) Event Strategy")
    print("=" * 70)

    # Load suspension data
    print("\nLoading margin_short_sale_suspension...")
    mss = data.get('margin_short_sale_suspension')
    print(f"  Shape: {mss.shape}")
    print(f"  Columns: {list(mss.columns)}")
    print(f"  Date range: {mss['停券起日(最後回補日)'].min()} to {mss['停券起日(最後回補日)'].max()}")
    print(f"  Reasons: {mss['原因'].value_counts().head(10).to_string()}")

    # Filter: only stock events (not ETFs), 2018+, exclude 除息 reasons
    mss['stock_id'] = mss['stock_id'].astype(str).str.zfill(4)
    mss['start_date'] = pd.to_datetime(mss['停券起日(最後回補日)'])
    mss['end_date'] = pd.to_datetime(mss['停券迄日'])

    mss = mss[
        mss['stock_id'].str.match(r'^\d{4}$') &
        ~mss['stock_id'].str.startswith(('00', '91')) &
        (mss['start_date'] >= '2018-01-01')
    ].copy()

    # Exclude non-volatility reasons (除息 is routine)
    non_vol_reasons = ['除息', '分配收益', '公開承銷', '現金增資']
    mss_vol = mss[~mss['原因'].isin(non_vol_reasons)].copy()
    print(f"\n  After filtering (stocks, 2018+, volatility-related): {len(mss_vol):,}")
    print(f"  Reasons breakdown:")
    print(mss_vol['原因'].value_counts().head(10).to_string())

    # Load price data
    print("\nLoading price data...")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    valid_stocks = set(close.columns)

    # Pre-compute turnover
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # For each suspension event, compute returns
    # Strategy: buy at open on the day AFTER suspension starts (停券日)
    # The idea: 停券 means shorts can't add, existing shorts must cover → upward pressure
    print("\nComputing post-suspension returns...")

    results = []
    for _, row in mss_vol.iterrows():
        sym = row['stock_id']
        start_date = row['start_date']
        end_date = row['end_date']

        if sym not in valid_stocks:
            continue

        # Find the suspension start date in calendar
        susp_idx = cal.searchsorted(start_date, side='left')
        if susp_idx >= len(cal) or susp_idx < 1:
            continue
        # Entry: next day after suspension announcement
        entry_idx = susp_idx + 1
        if entry_idx >= len(cal) - 10:
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[entry_idx - 1][sym]
            avg_to = avg_turnover_5d.iloc[entry_idx][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            continue

        # Liquidity filter (same as V20)
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            continue

        # Gap filter
        gap = entry_price / prev_close - 1
        if not (-0.08 < gap < 0.04):
            continue

        # Compute returns for different holding periods
        rets = {}
        for hold in [1, 2, 3, 5, 10]:
            exit_idx = entry_idx + hold - 1
            if exit_idx >= len(cal):
                rets[f'ret_{hold}d'] = np.nan
                continue
            try:
                exit_price = close.iloc[exit_idx][sym]
                if np.isnan(exit_price) or exit_price <= 0:
                    rets[f'ret_{hold}d'] = np.nan
                else:
                    rets[f'ret_{hold}d'] = exit_price / entry_price - 1 - COST_RATE
            except (IndexError, KeyError):
                rets[f'ret_{hold}d'] = np.nan

        # Suspension duration
        susp_duration = (end_date - start_date).days if not pd.isna(end_date) else np.nan

        results.append({
            'symbol': sym,
            'susp_start': start_date,
            'susp_end': end_date,
            'entry_date': cal[entry_idx],
            'year': start_date.year,
            'reason': row['原因'],
            'susp_duration_days': susp_duration,
            'gap': gap,
            'avg_turnover': avg_to,
            **rets,
        })

    res = pd.DataFrame(results)
    print(f"  Valid events (V20 filters): {len(res):,}")

    if len(res) < 50:
        print("  Too few events. Trying without liquidity filter...")
        # Retry without liquidity filter
        results2 = []
        for _, row in mss_vol.iterrows():
            sym = row['stock_id']
            start_date = row['start_date']
            if sym not in valid_stocks:
                continue
            susp_idx = cal.searchsorted(start_date, side='left')
            if susp_idx >= len(cal) or susp_idx < 1:
                continue
            entry_idx = susp_idx + 1
            if entry_idx >= len(cal) - 5:
                continue
            try:
                entry_price = open_p.iloc[entry_idx][sym]
                prev_close = close.iloc[entry_idx - 1][sym]
            except (IndexError, KeyError):
                continue
            if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
                continue
            gap = entry_price / prev_close - 1
            if not (-0.08 < gap < 0.04):
                continue
            rets = {}
            for hold in [1, 3, 5]:
                exit_idx = entry_idx + hold - 1
                if exit_idx >= len(cal):
                    rets[f'ret_{hold}d'] = np.nan
                    continue
                try:
                    exit_price = close.iloc[exit_idx][sym]
                    rets[f'ret_{hold}d'] = exit_price / entry_price - 1 - COST_RATE if not np.isnan(exit_price) and exit_price > 0 else np.nan
                except (IndexError, KeyError):
                    rets[f'ret_{hold}d'] = np.nan
            results2.append({
                'symbol': sym, 'entry_date': cal[entry_idx],
                'year': start_date.year, 'reason': row['原因'],
                'gap': gap, **rets,
            })
        res = pd.DataFrame(results2)
        print(f"  Without liquidity filter: {len(res):,} events")

    if len(res) < 50:
        print("  Still too few. Exiting.")
        return

    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("POST-停券 RETURNS (all events)")
    print("=" * 70)

    for hold in [1, 2, 3, 5, 10]:
        col = f'ret_{hold}d'
        if col not in res.columns:
            continue
        valid = res[col].notna()
        if valid.sum() < 50:
            continue
        r = res.loc[valid, col]
        t_stat, p_val = stats.ttest_1samp(r, 0)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {hold}d: mean={r.mean():.2%}, median={r.median():.2%}, win={(r>0).mean():.1%}, t={t_stat:.2f} p={p_val:.4f} {sig} (n={valid.sum():,})")

    # === BY REASON ===
    print("\n" + "=" * 70)
    print("BY REASON (top reasons)")
    print("=" * 70)

    top_reasons = res['reason'].value_counts().head(5).index
    for reason in top_reasons:
        subset = res[res['reason'] == reason]
        col = 'ret_3d'
        if col not in subset.columns:
            col = 'ret_1d'
        valid = subset[col].notna()
        if valid.sum() < 30:
            continue
        r = subset.loc[valid, col]
        t_stat, p_val = stats.ttest_1samp(r, 0)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {reason}: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={valid.sum()} {sig}")

    # === YEARLY STABILITY ===
    print("\n" + "=" * 70)
    print("YEARLY STABILITY (3d hold)")
    print("=" * 70)

    col = 'ret_3d' if 'ret_3d' in res.columns else 'ret_1d'
    for year in sorted(res['year'].unique()):
        yr_data = res[res['year'] == year]
        valid = yr_data[col].notna()
        if valid.sum() < 20:
            continue
        r = yr_data.loc[valid, col]
        print(f"  {year}: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={valid.sum()}")

    # Save
    res.to_csv(OUT / "p18b_events.csv", index=False, encoding='utf-8-sig')
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
