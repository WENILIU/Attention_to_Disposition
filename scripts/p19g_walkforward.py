"""P19g: C-Long walk-forward backtest (2018-2026) + new/old regime comparison.

Walk-forward:
  Train 2018-2020 → Test 2021
  Train 2018-2021 → Test 2022
  ...
  Train 2018-2025 → Test 2026

Also: compare old regime (before 2026-08-10) vs new regime.
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, roc_auc_score
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19g_walkforward")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")


def extract_features_from_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return {}
    feat = {}
    kuai = re.search(r'[﹝（(]第([一二三四五六七八九十]+)款[﹞）)]', text)
    kuai_map = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'十一':11,'十二':12}
    feat["kuai"] = kuai_map.get(kuai.group(1), 0) if kuai else 0
    feat["is_supervisory"] = 1 if '督導會報' in text else 0
    pct = re.search(r'[涨跌漲跌]幅達(\d+\.?\d*)%', text)
    feat["pct_change"] = float(pct.group(1)) if pct else 0
    amp = re.search(r'振幅達(\d+\.?\d*)%', text)
    feat["amplitude"] = float(amp.group(1)) if amp else 0
    vol_ratio = re.search(r'放大為\s*(\d+\.?\d*)倍', text)
    feat["volume_ratio_text"] = float(vol_ratio.group(1)) if vol_ratio else 0
    feat["is_price_trigger"] = 1 if ('漲幅' in text or '跌幅' in text or '振幅' in text) else 0
    feat["is_volume_trigger"] = 1 if ('成交量' in text or '成交值' in text) else 0
    feat["is_lending"] = 1 if '借券' in text else 0
    feat["is_intraday"] = 1 if '當沖' in text else 0
    feat["text_len"] = len(text)
    if feat["pct_change"] > 0:
        thresholds = {1: 35, 2: 50, 4: 100}
        thr = thresholds.get(feat["kuai"], 35)
        feat["excess_over_threshold"] = feat["pct_change"] - thr
    else:
        feat["excess_over_threshold"] = 0
    return feat


def main():
    print("=" * 70)
    print("P19g: C-Long Walk-Forward 回測 (2018-2026)")
    print("=" * 70)

    # Load all data
    print("\nLoading data...")
    att_raw = data.get("trading_attention")
    dis_raw = data.get("disposal_information")
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Build attention events
    att = pd.DataFrame(att_raw).copy()
    att["stock_id"] = att["symbol"].astype(str).str.zfill(4)
    att["announce"] = pd.to_datetime(att["date"]).dt.normalize()
    att["reason_text"] = att.get("注意交易資訊", "").astype(str)
    att = att[
        att["stock_id"].str.match(r"^\d{4}$") &
        att["stock_id"].isin(valid_stocks) &
        ~att["stock_id"].str.startswith(("00", "91")) &
        (att["announce"] >= "2018-01-01")
    ].copy()
    att["ann_idx"] = cal.searchsorted(att["announce"], side="left")
    att = att[(att["ann_idx"] >= 20) & (att["ann_idx"] < n_cal - 1)].copy()

    # Match to disposal
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis_by_stock = {}
    for _, row in dis.iterrows():
        sid = row["stock_id"]
        if sid not in dis_by_stock:
            dis_by_stock[sid] = []
        dis_by_stock[sid].append(row["dis_announce"])

    att["leads_to_disposal"] = False
    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann_date = row["announce"]
        if sid in dis_by_stock:
            for dis_date in dis_by_stock[sid]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    att.loc[idx, "leads_to_disposal"] = True
                    break

    print(f"  Events: {len(att):,}, Upgraded: {att['leads_to_disposal'].sum():,} ({att['leads_to_disposal'].mean():.1%})")

    # Build features
    print("Building features...")
    returns = close.pct_change()
    vol_20d = returns.rolling(20).std()
    ma20 = close.rolling(20).mean()
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    dis_dates_by_stock = {sid: sorted(dates) for sid, dates in dis_by_stock.items()}
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = sorted(group["announce"].tolist())

    # Text features
    text_feats = att["reason_text"].apply(extract_features_from_text).apply(pd.Series)

    # Price features
    price_feats = []
    for _, row in att.iterrows():
        sym = row["stock_id"]
        ai = row["ann_idx"]
        ann_date = row["announce"]
        feat = {}
        try:
            if sym in close.columns:
                c_now = close.iloc[ai][sym]
                c_5 = close.iloc[ai-5][sym] if ai >= 5 else np.nan
                c_10 = close.iloc[ai-10][sym] if ai >= 10 else np.nan
                c_20 = close.iloc[ai-20][sym] if ai >= 20 else np.nan
                feat["mom_5d"] = (c_now/c_5-1) if not np.isnan(c_5) and c_5>0 else 0
                feat["mom_10d"] = (c_now/c_10-1) if not np.isnan(c_10) and c_10>0 else 0
                feat["mom_20d"] = (c_now/c_20-1) if not np.isnan(c_20) and c_20>0 else 0
                feat["vol_20d"] = vol_20d.iloc[ai][sym] if sym in vol_20d.columns and not np.isnan(vol_20d.iloc[ai][sym]) else 0
                v5 = vol_ma5.iloc[ai][sym] if sym in vol_ma5.columns else np.nan
                v20 = vol_ma20.iloc[ai][sym] if sym in vol_ma20.columns else np.nan
                feat["vol_ratio"] = v5/v20 if not np.isnan(v20) and v20>0 else 1
                feat["turnover"] = avg_turnover_5d.iloc[ai][sym] if sym in avg_turnover_5d.columns and not np.isnan(avg_turnover_5d.iloc[ai][sym]) else 0
                ma20_val = ma20.iloc[ai][sym] if sym in ma20.columns else np.nan
                feat["bias_20d"] = (c_now/ma20_val-1) if not np.isnan(ma20_val) and ma20_val>0 else 0
            else:
                for k in ["mom_5d","mom_10d","mom_20d","vol_20d","vol_ratio","turnover","bias_20d"]:
                    feat[k] = 0
        except (IndexError, KeyError):
            for k in ["mom_5d","mom_10d","mom_20d","vol_20d","vol_ratio","turnover","bias_20d"]:
                feat[k] = 0
        hist_count = 0
        if sym in dis_dates_by_stock:
            cutoff = ann_date - pd.Timedelta(days=365)
            hist_count = sum(1 for d in dis_dates_by_stock[sym] if cutoff <= d < ann_date)
        feat["hist_disposal_count"] = hist_count
        days_since = 999
        if sym in att_dates_by_stock:
            prev = [d for d in att_dates_by_stock[sym] if d < ann_date]
            if prev:
                days_since = (ann_date - prev[-1]).days
        feat["days_since_last_att"] = min(days_since, 999)
        att_count = 0
        if sym in att_dates_by_stock:
            cutoff = ann_date - pd.Timedelta(days=30)
            att_count = sum(1 for d in att_dates_by_stock[sym] if cutoff <= d < ann_date)
        feat["att_count_30d"] = att_count
        price_feats.append(feat)

    price_df = pd.DataFrame(price_feats, index=att.index)
    all_features = pd.concat([text_feats, price_df], axis=1)
    feature_cols = list(all_features.columns)

    att["year"] = att["announce"].dt.year
    att["is_new_regime"] = att["announce"] >= NEW_REGIME_DATE

    # Compute trade returns for all events
    print("Computing trade returns...")
    trade_returns = []
    for _, row in att.iterrows():
        sym = row["stock_id"]
        ai = row["ann_idx"]
        if sym not in valid_stocks:
            trade_returns.append(np.nan)
            continue
        entry_idx = ai + 1
        if entry_idx >= n_cal - 1:
            trade_returns.append(np.nan)
            continue
        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[ai][sym]
            avg_to = avg_turnover_5d.iloc[ai][sym]
        except (IndexError, KeyError):
            trade_returns.append(np.nan)
            continue
        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            trade_returns.append(np.nan)
            continue
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            trade_returns.append(np.nan)
            continue
        gap = entry_price / prev_close - 1
        if not (-0.08 < gap < 0.04):
            trade_returns.append(np.nan)
            continue

        # Exit: disposal day - 1 or 10d max
        dis_exit_idx = None
        ann_date = row["announce"]
        if row["leads_to_disposal"] and sym in dis_by_stock:
            for dis_date in dis_by_stock[sym]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    dis_exit_idx = cal.searchsorted(dis_date, side="left") - 1
                    break

        if dis_exit_idx is not None and dis_exit_idx > entry_idx:
            exit_idx = min(dis_exit_idx, entry_idx + 9, n_cal - 1)
        else:
            exit_idx = min(entry_idx + 9, n_cal - 1)

        try:
            exit_price = close.iloc[exit_idx][sym]
            if np.isnan(exit_price) or exit_price <= 0:
                trade_returns.append(np.nan)
                continue
        except (IndexError, KeyError):
            trade_returns.append(np.nan)
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        trade_returns.append(ret)

    att["trade_ret"] = trade_returns
    print(f"  Valid trades: {att['trade_ret'].notna().sum():,}")

    # === WALK-FORWARD ===
    print("\n" + "=" * 70)
    print("Walk-Forward 回測")
    print("=" * 70)

    X_all = all_features.values.astype(float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = att["leads_to_disposal"].astype(int).values

    all_oos_probs = np.full(len(att), np.nan)

    for test_year in range(2021, 2027):
        train_mask = (att["year"] < test_year).values
        test_mask = (att["year"] == test_year).values

        if train_mask.sum() < 1000 or test_mask.sum() < 50:
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test = X_all[test_mask]

        gb = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42, min_samples_leaf=20
        )
        gb.fit(X_train, y_train)
        probs = gb.predict_proba(X_test)[:, 1]
        all_oos_probs[test_mask] = probs

        # Quick stats
        y_test = y_all[test_mask]
        valid_trade = att.loc[test_mask, "trade_ret"].notna().values
        if valid_trade.sum() > 20:
            # Apply threshold 0.5
            high_prob = probs >= 0.5
            selected = high_prob & valid_trade
            if selected.sum() > 10:
                rets = att.loc[test_mask].loc[selected, "trade_ret"].values
                prec = y_test[high_prob].mean() if high_prob.sum() > 0 else 0
                print(f"  {test_year}: n_selected={selected.sum()}, avg_ret={rets.mean():.2%}, "
                      f"win={(rets>0).mean():.1%}, precision={prec:.1%}")

    att["oos_prob"] = all_oos_probs

    # === FULL WALK-FORWARD STRATEGY ===
    print("\n" + "=" * 70)
    print("完整 Walk-Forward 策略模擬 (threshold=0.50, 5檔)")
    print("=" * 70)

    threshold = 0.50
    n_positions = 5

    # Get all OOS trades above threshold
    oos_trades = att[
        (att["oos_prob"] >= threshold) &
        (att["trade_ret"].notna())
    ].copy()
    oos_trades = oos_trades.sort_values(["announce", "oos_prob"], ascending=[True, False])

    # Simulate with position limit
    oos_trades["entry_idx"] = cal.searchsorted(oos_trades["announce"], side="left") + 1

    active_positions = []
    executed = []
    for _, trade in oos_trades.iterrows():
        entry_idx = int(trade["entry_idx"])
        # Estimate exit_idx from trade_ret (approximate hold as 9 days)
        exit_idx = min(entry_idx + 9, n_cal - 1)

        active_positions = [(ei, s) for ei, s in active_positions if ei > entry_idx]
        if len(active_positions) >= n_positions:
            continue
        if any(s == trade["stock_id"] for _, s in active_positions):
            continue
        active_positions.append((exit_idx, trade["stock_id"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)
    print(f"\n  Total executed: {len(ex):,} trades")
    print(f"  Date range: {ex['announce'].min().date()} to {ex['announce'].max().date()}")
    print(f"  Avg return: {ex['trade_ret'].mean():.2%}")
    print(f"  Win rate: {(ex['trade_ret']>0).mean():.1%}")
    print(f"  Upgraded in executed: {ex['leads_to_disposal'].mean():.1%}")

    # Yearly
    print(f"\n  逐年:")
    for yr in sorted(ex["year"].unique()):
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 5:
            continue
        avg = yr_data["trade_ret"].mean()
        win = (yr_data["trade_ret"] > 0).mean()
        sign = "PASS" if avg > 0 else "FAIL"
        print(f"    {yr}: avg={avg:.2%}, win={win:.1%}, n={len(yr_data)} {sign}")

    # === NEW vs OLD REGIME ===
    print("\n" + "=" * 70)
    print("新制 vs 舊制比較")
    print("=" * 70)

    old_regime = ex[~ex["is_new_regime"]]
    new_regime = ex[ex["is_new_regime"]]

    print(f"\n  舊制 (before 2026-08-10): n={len(old_regime):,}")
    if len(old_regime) > 10:
        print(f"    avg={old_regime['trade_ret'].mean():.2%}, win={(old_regime['trade_ret']>0).mean():.1%}")
        print(f"    upgraded rate: {old_regime['leads_to_disposal'].mean():.1%}")

    print(f"\n  新制 (after 2026-08-10): n={len(new_regime):,}")
    if len(new_regime) > 10:
        print(f"    avg={new_regime['trade_ret'].mean():.2%}, win={(new_regime['trade_ret']>0).mean():.1%}")
        print(f"    upgraded rate: {new_regime['leads_to_disposal'].mean():.1%}")

    # Check if model precision differs
    if len(old_regime) > 50 and len(new_regime) > 20:
        old_prec = old_regime["leads_to_disposal"].mean()
        new_prec = new_regime["leads_to_disposal"].mean()
        print(f"\n  模型精確度: 舊制={old_prec:.1%}, 新制={new_prec:.1%}")

    # Check upgrade rate difference
    old_all = att[(~att["is_new_regime"]) & att["oos_prob"].notna()]
    new_all = att[(att["is_new_regime"]) & att["oos_prob"].notna()]
    if len(new_all) > 50:
        print(f"\n  全部注意升級率: 舊制={old_all['leads_to_disposal'].mean():.1%}, 新制={new_all['leads_to_disposal'].mean():.1%}")

    # Save
    ex.to_csv(OUT / "p19g_walkforward_trades.csv", index=False, encoding="utf-8-sig")
    att[["stock_id","announce","year","is_new_regime","leads_to_disposal","oos_prob","trade_ret"]].to_csv(
        OUT / "p19g_all_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
