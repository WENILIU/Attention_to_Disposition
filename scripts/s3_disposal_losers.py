"""S3: 處置期間的輸家 vs 輸家 — 哪些處置股會跌？

目標: 找到過濾條件，避開虧損交易，提高 V20 勝率
數據: P23 的 1,537 筆交易 + 入場時的股票/市場特徵
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

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\s3_disposal_losers")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("S3: 處置期間輸家分析")
    print("=" * 70)

    # Load V20 trades
    print("\nLoading V20 trades...")
    trades = pd.read_csv(
        Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p23_v20_longterm\p23_v20_longterm_trades.csv"),
        parse_dates=["entry_date", "exit_date"]
    )
    trades["stock_id"] = trades["stock_id"].astype(str).str.zfill(4)
    print(f"  Trades: {len(trades):,}")
    print(f"  Winners: {(trades['ret'] > 0).sum():,} ({(trades['ret'] > 0).mean():.1%})")
    print(f"  Losers: {(trades['ret'] <= 0).sum():,} ({(trades['ret'] <= 0).mean():.1%})")
    print(f"  Avg winner: {trades[trades['ret'] > 0]['ret'].mean():.2%}")
    print(f"  Avg loser: {trades[trades['ret'] <= 0]['ret'].mean():.2%}")

    # Load price data for features
    print("\nLoading features...")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    dis_raw = data.get("disposal_information")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # Market features
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_close = (1 + mkt_ret).cumprod()
    mkt_ma20 = mkt_close.rolling(20).mean()
    mkt_ret_5d = mkt_close.pct_change(5)
    mkt_ret_20d = mkt_close.pct_change(20)
    mkt_vol_20d = mkt_ret.rolling(20).std()
    above_ma20 = (close > close.rolling(20).mean()).sum(axis=1) / close.notna().sum(axis=1)

    # Disposal info
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis["condition"] = dis.get("處置條件", "").astype(str)
    dis["duration"] = (cal.searchsorted(dis["end"], side="right") - 1) - cal.searchsorted(dis["start"], side="left") + 1

    # Match trades to disposal info
    trades["entry_idx"] = cal.searchsorted(trades["entry_date"], side="left")

    # Build features for each trade
    print("Building features...")
    features = []
    for _, t in trades.iterrows():
        sym = t["stock_id"]
        ei = int(t["entry_idx"])
        feat = {}

        # Market environment at entry
        try:
            feat["mkt_above_ma20"] = 1 if mkt_close.iloc[ei] > mkt_ma20.iloc[ei] else 0
            feat["mkt_ret_5d"] = float(mkt_ret_5d.iloc[ei]) if not np.isnan(mkt_ret_5d.iloc[ei]) else 0
            feat["mkt_ret_20d"] = float(mkt_ret_20d.iloc[ei]) if not np.isnan(mkt_ret_20d.iloc[ei]) else 0
            feat["mkt_vol"] = float(mkt_vol_20d.iloc[ei]) if not np.isnan(mkt_vol_20d.iloc[ei]) else 0
            feat["breadth"] = float(above_ma20.iloc[ei]) if not np.isnan(above_ma20.iloc[ei]) else 0.5
        except (IndexError, KeyError):
            for k in ["mkt_above_ma20","mkt_ret_5d","mkt_ret_20d","mkt_vol","breadth"]:
                feat[k] = 0

        # Stock characteristics at entry
        try:
            if sym in close.columns and ei >= 20:
                c = close.iloc[ei-20:ei+1][sym]
                v = vol.iloc[ei-20:ei+1][sym]
                if len(c.dropna()) >= 10:
                    feat["stock_mom_5d"] = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else 0
                    feat["stock_mom_20d"] = float(c.iloc[-1] / c.iloc[0] - 1)
                    feat["stock_vol_20d"] = float(c.pct_change().dropna().std())
                    feat["stock_bias_20d"] = float(c.iloc[-1] / c.mean() - 1)
                    feat["stock_turnover"] = float(c.iloc[-1] * v.iloc[-1])
                    feat["stock_drop_before"] = float(c.iloc[-1] / c.iloc[0] - 1)  # 處置前跌幅
                else:
                    for k in ["stock_mom_5d","stock_mom_20d","stock_vol_20d","stock_bias_20d","stock_turnover","stock_drop_before"]:
                        feat[k] = 0
            else:
                for k in ["stock_mom_5d","stock_mom_20d","stock_vol_20d","stock_bias_20d","stock_turnover","stock_drop_before"]:
                    feat[k] = 0
        except (IndexError, KeyError):
            for k in ["stock_mom_5d","stock_mom_20d","stock_vol_20d","stock_bias_20d","stock_turnover","stock_drop_before"]:
                feat[k] = 0

        # Disposal characteristics
        matching_dis = dis[(dis["stock_id"] == sym) & (dis["start"] <= t["entry_date"]) & (dis["end"] >= t["entry_date"])]
        if len(matching_dis) > 0:
            d_row = matching_dis.iloc[-1]
            feat["dis_duration"] = int(d_row["duration"])
            feat["dis_condition"] = d_row["condition"]
            feat["is_repeat"] = 1 if len(dis[(dis["stock_id"] == sym) & (dis["end"] < t["entry_date"]) & (dis["end"] > t["entry_date"] - pd.Timedelta(days=90))]) > 0 else 0
        else:
            feat["dis_duration"] = 10
            feat["dis_condition"] = ""
            feat["is_repeat"] = 0

        features.append(feat)

    feat_df = pd.DataFrame(features, index=trades.index)
    trades_full = pd.concat([trades[["stock_id", "entry_date", "exit_date", "year", "ret", "hold_days"]], feat_df], axis=1)
    trades_full["is_winner"] = (trades_full["ret"] > 0).astype(int)

    # === ANALYSIS: Winners vs Losers ===
    print("\n" + "=" * 70)
    print("輸家 vs 贏家特徵對比")
    print("=" * 70)

    winners = trades_full[trades_full["is_winner"] == 1]
    losers = trades_full[trades_full["is_winner"] == 0]

    numeric_cols = [c for c in feat_df.columns if c not in ["dis_condition"]]

    print(f"\n  {'Feature':<25} {'Winners':>10} {'Losers':>10} {'Diff':>10} {'Win%':>6}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    for col in numeric_cols:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        diff = w_mean - l_mean
        # Win rate when feature > median
        median_val = trades_full[col].median()
        above = trades_full[trades_full[col] > median_val]
        win_above = above["is_winner"].mean() if len(above) > 50 else 0
        print(f"  {col:<25} {w_mean:>10.4f} {l_mean:>10.4f} {diff:>+10.4f} {win_above:>6.1%}")

    # === CONDITION ANALYSIS ===
    print("\n" + "=" * 70)
    print("處置條件 vs 勝率")
    print("=" * 70)

    cond_groups = trades_full.groupby("dis_condition").agg(
        n=("ret", "count"),
        avg_ret=("ret", "mean"),
        win_rate=("is_winner", "mean"),
    ).sort_values("win_rate")

    print(f"\n  {'Condition':<30} {'n':>5} {'avg_ret':>8} {'win%':>6}")
    for cond, row in cond_groups.iterrows():
        if row["n"] < 10:
            continue
        cond_str = str(cond)[:30]
        print(f"  {cond_str:<30} {int(row['n']):>5} {row['avg_ret']:>8.2%} {row['win_rate']:>6.1%}")

    # === FILTER TESTS ===
    print("\n" + "=" * 70)
    print("過濾條件測試 (避開輸家)")
    print("=" * 70)

    baseline_win = trades_full["is_winner"].mean()
    baseline_ret = trades_full["ret"].mean()
    print(f"\n  Baseline: win={baseline_win:.1%}, avg_ret={baseline_ret:.2%}, n={len(trades_full)}")

    # Test various filters
    filters = {
        "大盤在MA20下": trades_full["mkt_above_ma20"] == 0,
        "大盤5日跌>3%": trades_full["mkt_ret_5d"] < -0.03,
        "股票20日跌>20%": trades_full["stock_mom_20d"] < -0.20,
        "股票乖離<-15%": trades_full["stock_bias_20d"] < -0.15,
        "波動率>5%": trades_full["stock_vol_20d"] > 0.05,
        "廣度<30%": trades_full["breadth"] < 0.30,
        "90天內重複處置": trades_full["is_repeat"] == 1,
        "大盤MA20下+廣度<40%": (trades_full["mkt_above_ma20"] == 0) & (trades_full["breadth"] < 0.40),
        "股票20日跌>15%+MA20下": (trades_full["stock_mom_20d"] < -0.15) & (trades_full["mkt_above_ma20"] == 0),
    }

    print(f"\n  {'Filter':<30} {'skip_n':>7} {'keep_win':>8} {'keep_ret':>8} {'skip_ret':>8} {'impact':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, mask in filters.items():
        skip = trades_full[mask]
        keep = trades_full[~mask]
        if len(skip) < 20 or len(keep) < 100:
            continue
        keep_win = keep["is_winner"].mean()
        keep_ret = keep["ret"].mean()
        skip_ret = skip["ret"].mean()
        impact = keep_ret - baseline_ret
        print(f"  {name:<30} {len(skip):>7} {keep_win:>8.1%} {keep_ret:>8.2%} {skip_ret:>8.2%} {impact:>+8.2%}")

    # === BEST FILTER SIMULATION ===
    print("\n" + "=" * 70)
    print("最佳過濾 + V20 組合回測")
    print("=" * 70)

    # Test: skip when market below MA20 AND breadth < 40%
    best_filter = (trades_full["mkt_above_ma20"] == 0) & (trades_full["breadth"] < 0.40)
    filtered_trades = trades_full[~best_filter]

    print(f"\n  Filter: 大盤MA20下 + 廣度<40%")
    print(f"  Skipped: {best_filter.sum()} trades (avg ret: {trades_full[best_filter]['ret'].mean():.2%})")
    print(f"  Kept: {len(filtered_trades)} trades (avg ret: {filtered_trades['ret'].mean():.2%}, win: {filtered_trades['is_winner'].mean():.1%})")

    # Portfolio simulation with filter
    filtered_trades = filtered_trades.sort_values(["entry_date", "ret"], ascending=[True, False])
    active = []
    executed = []
    for _, trade in filtered_trades.iterrows():
        ei = cal.searchsorted(trade["entry_date"], side="left")
        xi = cal.searchsorted(trade["exit_date"], side="left")
        active = [(e, s) for e, s in active if e > ei]
        if len(active) >= 5:
            continue
        if any(s == trade["stock_id"] for _, s in active):
            continue
        active.append((xi, trade["stock_id"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)

    # Metrics
    daily_returns = pd.Series(0.0, index=cal)
    cap = 0.14  # 70% / 5
    for _, t in ex.iterrows():
        ei = cal.searchsorted(t["entry_date"], side="left")
        xi = cal.searchsorted(t["exit_date"], side="left")
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold * cap
        for d in range(ei, min(xi + 1, n_cal)):
            daily_returns.iloc[d] += dr

    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    cum = (1 + daily_returns).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    yearly = {}
    for yr in range(2010, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() > 50:
            yearly[yr] = np.prod(1 + daily_returns.values[yr_mask]) - 1
    n_pos = sum(1 for v in yearly.values() if v > 0)

    print(f"\n  V20 + filter: Sharpe={sharpe:.2f}, MDD={mdd:.1%}, Years={n_pos}/{len(yearly)}")
    print(f"  Trades: {len(ex)}, Win: {(ex['ret']>0).mean():.1%}, Avg: {ex['ret'].mean():.2%}")
    for yr in [2011, 2015, 2018, 2022]:
        if yr in yearly:
            print(f"    {yr}: {yearly[yr]:+.1%}")

    # Save
    trades_full.to_csv(OUT / "s3_trades_with_features.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
