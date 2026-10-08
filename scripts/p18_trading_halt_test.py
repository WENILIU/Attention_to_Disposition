"""P18: Trading Halt (停損/停利) event strategy feasibility test.

Hypothesis: Stocks that hit 跌停 (price limit down) and get halted
experience forced selling → oversold → bounce after resumption.

Test: After halt/resume, what is the return pattern?
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p18_trading_halt")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P18: Trading Halt (跌停暫停) Event Strategy")
    print("=" * 70)

    # Load halt data
    print("\nLoading trading_halt data...")
    halt_stock = data.get('trading_halt:暫停交易')  # wide: date × symbol, value = stock code
    halt_time = data.get('trading_halt:暫停交易時間')
    resume_time = data.get('trading_halt:恢復交易時間')
    category = data.get('trading_halt:有價證券類別')
    market = data.get('trading_halt:市場別')

    print(f"  Halt data shape: {halt_stock.shape}")
    print(f"  Dates with halts: {halt_stock.shape[0]}")

    # Load price data
    print("Loading price data...")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    high = data.get("price:最高價")
    low = data.get("price:最低價")
    vol = data.get("price:成交股數")
    amount = data.get("price:成交金額")

    # Extract halt events: find non-NaN cells in halt_stock
    # The value in halt_stock is the stock code when halted
    print("\nExtracting halt events...")
    events = []
    for date in halt_stock.index:
        row = halt_stock.loc[date]
        halted = row.dropna()
        if len(halted) == 0:
            continue

        # Filter to stock securities only (not options/warrants)
        cat_row = category.loc[date].dropna() if date in category.index else pd.Series(dtype=str)
        mkt_row = market.loc[date].dropna() if date in market.index else pd.Series(dtype=str)

        for symbol in halted.index:
            # Check category
            sec_cat = cat_row.get(symbol, '')
            sec_mkt = mkt_row.get(symbol, '')
            if sec_cat not in ('上市證券', '上櫃股票'):
                continue
            if sec_mkt not in ('上市', '上櫃'):
                continue

            # Get halt/resume times
            ht = halt_time.loc[date].get(symbol, '') if date in halt_time.index else ''
            rt = resume_time.loc[date].get(symbol, '') if date in resume_time.index else ''

            events.append({
                'date': date,
                'symbol': str(symbol).zfill(4),
                'halt_time': str(ht),
                'resume_time': str(rt),
            })

    ev = pd.DataFrame(events)
    print(f"  Total halt events (stocks only): {len(ev):,}")
    print(f"  Unique stocks: {ev['symbol'].nunique()}")
    print(f"  Date range: {ev['date'].min()} to {ev['date'].max()}")

    # Filter to 2018+ and valid stock codes
    ev = ev[
        ev['symbol'].str.match(r'^\d{4}$') &
        ~ev['symbol'].str.startswith(('00', '91')) &
        (ev['date'] >= '2018-01-01')
    ].copy()
    print(f"  After filtering (2018+, valid stocks): {len(ev):,}")

    # For each event, compute returns after the halt day
    # The halt happens intraday, so the "event day" close is already at limit
    # We test: buy at next day open, sell at various future points
    print("\nComputing post-halt returns...")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    valid_stocks = set(close.columns)

    # Pre-compute turnover
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    results = []
    for _, row in ev.iterrows():
        sym = row['symbol']
        event_date = row['date']
        if sym not in valid_stocks:
            continue

        # Find event day index in calendar
        event_idx = cal.searchsorted(event_date, side='right') - 1
        if event_idx < 0 or event_idx >= len(cal) - 5:
            continue

        # Entry: next day open (day after halt)
        entry_idx = event_idx + 1
        if entry_idx >= len(cal):
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            event_close = close.iloc[event_idx][sym]
            prev_close = close.iloc[event_idx - 1][sym] if event_idx > 0 else np.nan
            avg_to = avg_turnover_5d.iloc[event_idx][sym]
        except (IndexError, KeyError):
            continue

        if any(np.isnan(x) or x <= 0 for x in [entry_price, event_close, prev_close]):
            continue

        # Liquidity filter
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
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

        # Also: gap from event close to next day open
        gap = entry_price / event_close - 1

        # Was it a 跌停 (limit down) or 涨停 (limit up)?
        day_return = event_close / prev_close - 1 if not np.isnan(prev_close) else np.nan

        results.append({
            'symbol': sym,
            'event_date': event_date,
            'entry_date': cal[entry_idx],
            'year': event_date.year,
            'day_return': day_return,
            'gap': gap,
            'avg_turnover': avg_to,
            **rets,
        })

    res = pd.DataFrame(results)
    print(f"  Valid events with returns: {len(res):,}")

    if len(res) == 0:
        print("  No valid events. Exiting.")
        return

    # Classify: 跌停 vs 涨停
    res['direction'] = np.where(res['day_return'] < -0.05, '跌停',
                       np.where(res['day_return'] > 0.05, '涨停', '其他'))

    print(f"\n  Direction breakdown:")
    print(res['direction'].value_counts().to_string())

    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("POST-HALT RETURNS (all events)")
    print("=" * 70)

    for hold in [1, 2, 3, 5, 10]:
        col = f'ret_{hold}d'
        valid = res[col].notna()
        if valid.sum() < 50:
            continue
        r = res.loc[valid, col]
        print(f"  {hold}d: mean={r.mean():.2%}, median={r.median():.2%}, win={(r>0).mean():.1%}, n={valid.sum():,}")

    # === BY DIRECTION ===
    print("\n" + "=" * 70)
    print("BY DIRECTION (跌停 vs 涨停)")
    print("=" * 70)

    for direction in ['跌停', '涨停']:
        subset = res[res['direction'] == direction]
        if len(subset) < 50:
            print(f"\n  {direction}: n={len(subset)} (too few)")
            continue
        print(f"\n  {direction} (n={len(subset):,}):")
        for hold in [1, 2, 3, 5, 10]:
            col = f'ret_{hold}d'
            valid = subset[col].notna()
            if valid.sum() < 30:
                continue
            r = subset.loc[valid, col]
            t_stat, p_val = stats.ttest_1samp(r, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"    {hold}d: mean={r.mean():.2%}, median={r.median():.2%}, win={(r>0).mean():.1%}, t={t_stat:.2f} p={p_val:.4f} {sig}")

    # === YEARLY STABILITY (跌停 only) ===
    print("\n" + "=" * 70)
    print("YEARLY STABILITY (跌停, 3d hold)")
    print("=" * 70)

    dl = res[res['direction'] == '跌停'].copy()
    if len(dl) > 100:
        for year in sorted(dl['year'].unique()):
            yr_data = dl[dl['year'] == year]
            col = 'ret_3d'
            valid = yr_data[col].notna()
            if valid.sum() < 20:
                continue
            r = yr_data.loc[valid, col]
            print(f"  {year}: mean={r.mean():.2%}, win={(r>0).mean():.1%}, n={valid.sum()}")

    # === GAP ANALYSIS ===
    print("\n" + "=" * 70)
    print("GAP ANALYSIS (跌停, next day open vs event close)")
    print("=" * 70)
    if len(dl) > 50:
        gap_valid = dl['gap'].notna()
        if gap_valid.sum() > 50:
            g = dl.loc[gap_valid, 'gap']
            print(f"  Gap: mean={g.mean():.2%}, median={g.median():.2%}, win={(g>0).mean():.1%}")
            print(f"  (Positive gap = price opens higher than limit-down close)")

    # Save
    res.to_csv(OUT / "p18_halt_events.csv", index=False, encoding='utf-8-sig')
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
