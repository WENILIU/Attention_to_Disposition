"""P19j: Predict disposal date for dynamic exit.

Goal: Instead of fixed 10d hold, predict WHEN disposal will come.
Exit at predicted_disposal_date - 1.

Approach:
1. For upgraded events, compute actual days_to_disposal
2. Train regression model to predict days_to_disposal
3. Use prediction to set dynamic exit
4. Compare: fixed 10d vs dynamic exit vs oracle
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19j_date_pred")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003


def main():
    print("=" * 70)
    print("P19j: 預測處置日期 → 動態出場")
    print("=" * 70)

    # Load data
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

    # Build disposal with dates
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis_by_stock = {}
    for _, row in dis.iterrows():
        sid = row["stock_id"]
        if sid not in dis_by_stock:
            dis_by_stock[sid] = []
        dis_by_stock[sid].append(row["dis_announce"])

    # Match and compute days_to_disposal
    att["leads_to_disposal"] = False
    att["days_to_disposal"] = np.nan
    att["dis_ann_date"] = pd.NaT

    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann_date = row["announce"]
        if sid in dis_by_stock:
            for dis_date in dis_by_stock[sid]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    att.loc[idx, "leads_to_disposal"] = True
                    att.loc[idx, "days_to_disposal"] = delta
                    att.loc[idx, "dis_ann_date"] = dis_date
                    break

    # Consecutive detection
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    att["consec_1d"] = False
    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann = row["announce"]
        if sid in att_dates_by_stock:
            for d in [1, 2, 3]:  # account for weekends
                if (ann - pd.Timedelta(days=d)) in att_dates_by_stock[sid]:
                    att.loc[idx, "consec_1d"] = True
                    break

    # Text features
    def extract_feat(text):
        if pd.isna(text) or not isinstance(text, str):
            return {"kuai": 0, "pct_change": 0, "is_supervisory": 0, "volume_ratio_text": 0}
        feat = {}
        kuai = re.search(r'[﹝（(]第([一二三四五六七八九十]+)款[﹞）)]', text)
        kuai_map = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'十一':11,'十二':12}
        feat["kuai"] = kuai_map.get(kuai.group(1), 0) if kuai else 0
        feat["is_supervisory"] = 1 if '督導會報' in text else 0
        pct = re.search(r'[涨跌漲跌]幅達(\d+\.?\d*)%', text)
        feat["pct_change"] = float(pct.group(1)) if pct else 0
        vol_ratio = re.search(r'放大為\s*(\d+\.?\d*)倍', text)
        feat["volume_ratio_text"] = float(vol_ratio.group(1)) if vol_ratio else 0
        return feat

    text_feats = att["reason_text"].apply(extract_feat).apply(pd.Series)

    # Price features (simplified for speed)
    returns = close.pct_change()
    vol_20d = returns.rolling(20).std()
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()

    price_feats = []
    for _, row in att.iterrows():
        sym = row["stock_id"]
        ai = row["ann_idx"]
        feat = {}
        try:
            if sym in close.columns:
                c_now = close.iloc[ai][sym]
                c_5 = close.iloc[ai-5][sym] if ai >= 5 else np.nan
                feat["mom_5d"] = (c_now/c_5-1) if not np.isnan(c_5) and c_5>0 else 0
                feat["vol_20d"] = vol_20d.iloc[ai][sym] if sym in vol_20d.columns and not np.isnan(vol_20d.iloc[ai][sym]) else 0
                v5 = vol_ma5.iloc[ai][sym] if sym in vol_ma5.columns else np.nan
                v20 = vol_ma20.iloc[ai][sym] if sym in vol_ma20.columns else np.nan
                feat["vol_ratio"] = v5/v20 if not np.isnan(v20) and v20>0 else 1
            else:
                feat["mom_5d"] = 0; feat["vol_20d"] = 0; feat["vol_ratio"] = 1
        except (IndexError, KeyError):
            feat["mom_5d"] = 0; feat["vol_20d"] = 0; feat["vol_ratio"] = 1
        price_feats.append(feat)

    price_df = pd.DataFrame(price_feats, index=att.index)

    # Combine features
    att["consec_1d_int"] = att["consec_1d"].astype(int)
    feature_df = pd.concat([text_feats, price_df, att[["consec_1d_int"]]], axis=1)
    feature_cols = list(feature_df.columns)

    att["year"] = att["announce"].dt.year

    # === ANALYZE DAYS_TO_DISPOSAL DISTRIBUTION ===
    print("\n" + "=" * 70)
    print("處置日期分佈")
    print("=" * 70)

    upg = att[att["leads_to_disposal"]].copy()
    print(f"\n  升級事件: {len(upg):,}")
    print(f"  Days to disposal distribution:")
    for d in [1, 2, 3, 5, 7, 10, 14, 20, 30]:
        pct = (upg["days_to_disposal"] <= d).mean()
        print(f"    <= {d}d: {pct:.1%}")

    print(f"\n  連續注意 vs days_to_disposal:")
    for label, mask in [("非連續", ~upg["consec_1d"]), ("連續", upg["consec_1d"])]:
        subset = upg[mask]
        if len(subset) < 10:
            continue
        median = subset["days_to_disposal"].median()
        p25 = subset["days_to_disposal"].quantile(0.25)
        p75 = subset["days_to_disposal"].quantile(0.75)
        print(f"    {label}: median={median:.0f}d, p25={p25:.0f}d, p75={p75:.0f}d, n={len(subset):,}")

    # === TRAIN DAYS-TO-DISPOSAL PREDICTOR ===
    print("\n" + "=" * 70)
    print("訓練處置日期預測模型")
    print("=" * 70)

    # Only train on upgraded events (to predict timing)
    upg_mask = att["leads_to_disposal"].values
    X_all = feature_df.values.astype(float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Train on 2018-2023, test on 2024-2026
    train_mask = upg_mask & (att["year"] <= 2023).values
    test_mask = (att["year"] >= 2024).values

    y_days = att["days_to_disposal"].values

    X_train = X_all[train_mask]
    y_train = y_days[train_mask]

    # Filter valid training targets
    valid_train = ~np.isnan(y_train)
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]

    print(f"  Training samples (upgraded, 2018-2023): {len(X_train):,}")

    reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
    reg.fit(X_train, y_train)

    # Predict for all test events
    X_test = X_all[test_mask]
    pred_days = reg.predict(X_test)
    pred_days = np.clip(pred_days, 1, 20)  # cap at 1-20 days

    # Feature importance
    print("\n  Feature importance (days predictor):")
    for name, imp in sorted(zip(feature_cols, reg.feature_importances_), key=lambda x: x[1], reverse=True)[:8]:
        print(f"    {name}: {imp:.4f}")

    # === SIMULATE DYNAMIC EXIT ===
    print("\n" + "=" * 70)
    print("動態出場模擬")
    print("=" * 70)

    # Load walk-forward model predictions for filtering
    wf_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19g_walkforward\p19g_all_predictions.csv")
    wf = pd.read_csv(wf_path, parse_dates=["announce"])
    wf["stock_id"] = wf["stock_id"].astype(str).str.zfill(4)

    # Merge predicted days into test events
    test_att = att[test_mask].copy()
    test_att["pred_days"] = pred_days

    # Merge with walk-forward predictions
    test_att = test_att.merge(
        wf[["stock_id", "announce", "oos_prob"]],
        on=["stock_id", "announce"], how="left"
    )

    # Filter: model >= 0.6 + consecutive
    selected = test_att[
        (test_att["oos_prob"] >= 0.6) &
        (test_att["consec_1d"])
    ].copy()
    print(f"\n  Selected (model>=0.6 + consecutive): {len(selected):,}")

    # Compute returns for different exit strategies
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    results = {"fixed_10d": [], "fixed_15d": [], "dynamic_pred": [], "oracle": []}

    for _, row in selected.iterrows():
        sym = row["stock_id"]
        ai = int(row["ann_idx"])
        entry_idx = ai + 1
        if sym not in valid_stocks or entry_idx >= n_cal - 1:
            continue
        try:
            entry_price = open_p.iloc[entry_idx][sym]
            avg_to = avg_turnover_5d.iloc[ai][sym]
        except (IndexError, KeyError):
            continue
        if np.isnan(entry_price) or entry_price <= 0:
            continue
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            continue

        # Oracle exit (if upgraded)
        actual_exit = None
        if row["leads_to_disposal"] and not pd.isna(row.get("dis_ann_date", pd.NaT)):
            dis_idx = cal.searchsorted(row["dis_ann_date"], side="left") - 1
            if dis_idx > entry_idx:
                actual_exit = dis_idx

        # Fixed 10d
        exit_10 = min(entry_idx + 9, n_cal - 1)
        # Fixed 15d
        exit_15 = min(entry_idx + 14, n_cal - 1)
        # Dynamic: use predicted days
        pred_exit = min(entry_idx + int(row["pred_days"]) - 1, n_cal - 1)
        # Cap dynamic at 15d max
        pred_exit = min(pred_exit, entry_idx + 14)

        for exit_idx, key in [(exit_10, "fixed_10d"), (exit_15, "fixed_15d"),
                               (pred_exit, "dynamic_pred")]:
            if exit_idx >= n_cal:
                continue
            try:
                p = close.iloc[exit_idx][sym]
                if not np.isnan(p) and p > 0:
                    ret = p / entry_price - 1 - COST_RATE
                    results[key].append({"ret": ret, "year": row["year"],
                                         "upgraded": row["leads_to_disposal"]})
            except (IndexError, KeyError):
                continue

        # Oracle
        if actual_exit is not None and actual_exit < n_cal:
            try:
                p = close.iloc[actual_exit][sym]
                if not np.isnan(p) and p > 0:
                    ret = p / entry_price - 1 - COST_RATE
                    results["oracle"].append({"ret": ret, "year": row["year"],
                                              "upgraded": True})
            except (IndexError, KeyError):
                continue

    print(f"\n  出場策略比較:")
    print(f"  {'Strategy':>15} {'AvgRet':>8} {'Win%':>6} {'n':>6} {'Yearly':>10}")
    for key, rets in results.items():
        if len(rets) < 30:
            continue
        df = pd.DataFrame(rets)
        avg = df["ret"].mean()
        win = (df["ret"] > 0).mean()
        # Yearly check
        yr_pass = 0
        yr_total = 0
        for yr in sorted(df["year"].unique()):
            yr_data = df[df["year"] == yr]
            if len(yr_data) >= 10:
                yr_total += 1
                if yr_data["ret"].mean() > 0:
                    yr_pass += 1
        print(f"  {key:>15} {avg:>8.2%} {win:>6.1%} {len(df):>6} {yr_pass}/{yr_total}")

    # Detail for dynamic
    if len(results["dynamic_pred"]) > 50:
        dyn = pd.DataFrame(results["dynamic_pred"])
        print(f"\n  動態出場逐年:")
        for yr in sorted(dyn["year"].unique()):
            yr_data = dyn[dyn["year"] == yr]
            if len(yr_data) < 10:
                continue
            avg = yr_data["ret"].mean()
            win = (yr_data["ret"] > 0).mean()
            sign = "PASS" if avg > 0 else "FAIL"
            print(f"    {yr}: {avg:.2%}, win={win:.1%}, n={len(yr_data)} {sign}")

    # === PREDICTION ACCURACY ===
    print("\n" + "=" * 70)
    print("預測準確度")
    print("=" * 70)

    upg_test = selected[selected["leads_to_disposal"]].copy()
    if len(upg_test) > 50:
        actual = upg_test["days_to_disposal"].values
        pred = upg_test["pred_days"].values
        mae = np.abs(actual - pred).mean()
        within_2d = (np.abs(actual - pred) <= 2).mean()
        within_5d = (np.abs(actual - pred) <= 5).mean()
        print(f"\n  MAE: {mae:.1f} days")
        print(f"  Within 2 days: {within_2d:.1%}")
        print(f"  Within 5 days: {within_5d:.1%}")
        print(f"  n={len(upg_test):,}")

    # Save
    pd.DataFrame(results).to_csv(OUT / "p19j_results.csv", index=False)
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
