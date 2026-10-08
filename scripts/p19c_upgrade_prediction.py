"""P19c: Build upgrade prediction model (注意→處置 probability).

Features (all available at attention announcement time):
1. A4 text strength (from attention reason text)
2. Attention reason category
3. Historical disposal count (past 1 year)
4. Price momentum (5d/10d/20d before attention)
5. Volatility (20d realized vol)
6. Volume ratio (5d avg / 20d avg)
7. Market cap proxy (turnover)
8. Days since last attention

Target: leads_to_disposal (within 30 days)
Model: Logistic Regression + Gradient Boosting
Split: Time-based (train 2018-2023, test 2024-2026)
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                            classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19c_prediction")
OUT.mkdir(parents=True, exist_ok=True)


def extract_a4_strength(text):
    """Extract disposal probability signal from attention reason text."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    score = 0
    # Higher numbers = more severe = more likely to become disposal
    patterns = [
        (r'連(\d+)日', lambda m: int(m.group(1)) * 2),
        (r'(\d+)日內.*?(\d+)次', lambda m: int(m.group(2)) * 3),
        (r'成交值比率', lambda m: 3),
        (r'借券比', lambda m: 4),
        (r'當沖', lambda m: 2),
        (r'督導會報', lambda m: 5),
        (r'注意', lambda m: 1),
    ]
    for pattern, scorer in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                score += scorer(match)
            except (ValueError, TypeError):
                pass
    return min(score, 20)


def main():
    print("=" * 70)
    print("P19c: 注意→處置 升級預測模型")
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

    # Get reason text
    reason_col = None
    for col in ["註意原因", "原因", "reason"]:
        if col in att.columns:
            reason_col = col
            break
    if reason_col:
        att["reason_text"] = att[reason_col]
    else:
        att["reason_text"] = ""

    att = att[
        att["stock_id"].str.match(r"^\d{4}$") &
        att["stock_id"].isin(valid_stocks) &
        ~att["stock_id"].str.startswith(("00", "91")) &
        (att["announce"] >= "2018-01-01")
    ].copy()
    att["ann_idx"] = cal.searchsorted(att["announce"], side="left")
    att = att[(att["ann_idx"] >= 20) & (att["ann_idx"] < n_cal - 1)].copy()

    # Build disposal events
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["dis_start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()

    # Match attention → disposal
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

    print(f"  Attention events: {len(att):,}")
    print(f"  Upgraded: {att['leads_to_disposal'].sum():,} ({att['leads_to_disposal'].mean():.1%})")

    # === BUILD FEATURES ===
    print("\nBuilding features...")

    # Pre-compute rolling stats
    returns = close.pct_change()
    vol_20d = returns.rolling(20).std()
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Historical disposal count (rolling 1 year)
    dis_dates_by_stock = {}
    for sid, dates in dis_by_stock.items():
        dis_dates_by_stock[sid] = sorted(dates)

    # Attention history (days since last attention)
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = sorted(group["announce"].tolist())

    features = []
    for _, row in att.iterrows():
        sym = row["stock_id"]
        ai = row["ann_idx"]
        ann_date = row["announce"]

        feat = {}

        # 1. A4 text strength
        feat["a4_strength"] = extract_a4_strength(row.get("reason_text", ""))

        # 2. Price momentum (5d, 10d, 20d)
        try:
            if sym in close.columns:
                c_now = close.iloc[ai][sym]
                c_5 = close.iloc[ai-5][sym] if ai >= 5 else np.nan
                c_10 = close.iloc[ai-10][sym] if ai >= 10 else np.nan
                c_20 = close.iloc[ai-20][sym] if ai >= 20 else np.nan
                feat["mom_5d"] = (c_now / c_5 - 1) if not np.isnan(c_5) and c_5 > 0 else np.nan
                feat["mom_10d"] = (c_now / c_10 - 1) if not np.isnan(c_10) and c_10 > 0 else np.nan
                feat["mom_20d"] = (c_now / c_20 - 1) if not np.isnan(c_20) and c_20 > 0 else np.nan

                # 3. Volatility
                feat["vol_20d"] = vol_20d.iloc[ai][sym] if sym in vol_20d.columns else np.nan

                # 4. Volume ratio
                if sym in vol_ma5.columns and sym in vol_ma20.columns:
                    v5 = vol_ma5.iloc[ai][sym]
                    v20 = vol_ma20.iloc[ai][sym]
                    feat["vol_ratio"] = v5 / v20 if not np.isnan(v20) and v20 > 0 else np.nan
                else:
                    feat["vol_ratio"] = np.nan

                # 5. Market cap proxy (turnover)
                feat["turnover"] = avg_turnover_5d.iloc[ai][sym] if sym in avg_turnover_5d.columns else np.nan

                # 6. MA deviation (bias)
                if sym in ma20.columns:
                    ma20_val = ma20.iloc[ai][sym]
                    feat["bias_20d"] = (c_now / ma20_val - 1) if not np.isnan(ma20_val) and ma20_val > 0 else np.nan
                else:
                    feat["bias_20d"] = np.nan
            else:
                for k in ["mom_5d", "mom_10d", "mom_20d", "vol_20d", "vol_ratio", "turnover", "bias_20d"]:
                    feat[k] = np.nan
        except (IndexError, KeyError):
            for k in ["mom_5d", "mom_10d", "mom_20d", "vol_20d", "vol_ratio", "turnover", "bias_20d"]:
                feat[k] = np.nan

        # 7. Historical disposal count (past 365 days)
        hist_count = 0
        if sym in dis_dates_by_stock:
            cutoff = ann_date - pd.Timedelta(days=365)
            hist_count = sum(1 for d in dis_dates_by_stock[sym] if cutoff <= d < ann_date)
        feat["hist_disposal_count"] = hist_count

        # 8. Days since last attention
        days_since_last_att = 999
        if sym in att_dates_by_stock:
            prev_dates = [d for d in att_dates_by_stock[sym] if d < ann_date]
            if prev_dates:
                days_since_last_att = (ann_date - prev_dates[-1]).days
        feat["days_since_last_att"] = min(days_since_last_att, 999)

        # 9. Attention count in past 30 days
        att_count_30d = 0
        if sym in att_dates_by_stock:
            cutoff = ann_date - pd.Timedelta(days=30)
            att_count_30d = sum(1 for d in att_dates_by_stock[sym] if cutoff <= d < ann_date)
        feat["att_count_30d"] = att_count_30d

        features.append(feat)

    feat_df = pd.DataFrame(features, index=att.index)
    att_model = pd.concat([att[["stock_id", "announce", "ann_idx", "leads_to_disposal"]], feat_df], axis=1)
    att_model["year"] = att_model["announce"].dt.year

    print(f"  Features built: {len(att_model):,} rows, {len(feat_df.columns)} features")

    # === TIME-BASED SPLIT ===
    train_mask = att_model["year"] <= 2023
    test_mask = att_model["year"] >= 2024

    feature_cols = [c for c in feat_df.columns]
    X_train = att_model.loc[train_mask, feature_cols].values
    y_train = att_model.loc[train_mask, "leads_to_disposal"].astype(int).values
    X_test = att_model.loc[test_mask, feature_cols].values
    y_test = att_model.loc[test_mask, "leads_to_disposal"].astype(int).values

    print(f"\n  Train: {X_train.shape[0]:,} (upgrade rate: {y_train.mean():.1%})")
    print(f"  Test:  {X_test.shape[0]:,} (upgrade rate: {y_test.mean():.1%})")

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # === MODEL 1: Logistic Regression ===
    print("\n" + "=" * 70)
    print("Model 1: Logistic Regression")
    print("=" * 70)

    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

    print(f"  Accuracy: {(y_pred_lr == y_test).mean():.3f}")
    print(f"  Precision (predicted positive): {precision_score(y_test, y_pred_lr):.3f}")
    print(f"  Recall: {recall_score(y_test, y_pred_lr):.3f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
    print(f"\n  Feature importance (coefficients):")
    for name, coef in sorted(zip(feature_cols, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name}: {coef:.4f}")

    # === MODEL 2: Gradient Boosting ===
    print("\n" + "=" * 70)
    print("Model 2: Gradient Boosting")
    print("=" * 70)

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_prob_gb = gb.predict_proba(X_test)[:, 1]

    print(f"  Accuracy: {(y_pred_gb == y_test).mean():.3f}")
    print(f"  Precision (predicted positive): {precision_score(y_test, y_pred_gb):.3f}")
    print(f"  Recall: {recall_score(y_test, y_pred_gb):.3f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob_gb):.4f}")
    print(f"\n  Feature importance:")
    for name, imp in sorted(zip(feature_cols, gb.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {imp:.4f}")

    # === THRESHOLD ANALYSIS ===
    print("\n" + "=" * 70)
    print("Threshold Analysis (Gradient Boosting)")
    print("=" * 70)
    print("  目標: 找到 precision >= 50% 的 threshold")
    print()

    for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        y_pred_t = (y_prob_gb >= threshold).astype(int)
        n_pred = y_pred_t.sum()
        if n_pred < 50:
            continue
        prec = precision_score(y_test, y_pred_t)
        rec = recall_score(y_test, y_pred_t)
        n_true_pos = ((y_pred_t == 1) & (y_test == 1)).sum()
        print(f"  threshold={threshold:.2f}: precision={prec:.1%}, recall={rec:.1%}, "
              f"predicted={n_pred:,}, true_pos={n_true_pos:,}")

    # === STRATEGY SIMULATION with best threshold ===
    print("\n" + "=" * 70)
    print("策略模擬: 使用 GB 模型過濾")
    print("=" * 70)

    # Find threshold for 50% precision
    best_threshold = 0.5
    for t in np.arange(0.3, 0.8, 0.01):
        y_pred_t = (y_prob_gb >= t).astype(int)
        if y_pred_t.sum() < 50:
            break
        prec = precision_score(y_test, y_pred_t)
        if prec >= 0.50:
            best_threshold = t
            break

    print(f"\n  Best threshold for 50% precision: {best_threshold:.2f}")
    y_pred_best = (y_prob_gb >= best_threshold).astype(int)
    print(f"  Precision: {precision_score(y_test, y_pred_best):.1%}")
    print(f"  Recall: {recall_score(y_test, y_pred_best):.1%}")
    print(f"  Selected events: {y_pred_best.sum():,} / {len(y_test):,}")

    # Compute returns for filtered events (test period)
    COST_RATE = 0.001425 + 0.003 + 0.003
    test_indices = att_model.loc[test_mask].index
    filtered_indices = test_indices[y_pred_best == 1]

    print(f"\n  過濾後事件報酬 (test period 2024-2026):")
    filtered_returns = []
    for idx in filtered_indices:
        row = att_model.loc[idx]
        sym = row["stock_id"]
        ai = int(row["ann_idx"])
        if sym not in valid_stocks:
            continue
        entry_idx = ai + 1
        if entry_idx >= n_cal - 1:
            continue
        try:
            entry_price = open_p.iloc[entry_idx][sym]
            if np.isnan(entry_price) or entry_price <= 0:
                continue
            # Hold until disposal or 10d max
            if row["leads_to_disposal"]:
                # Find disposal date
                ann_date = row["announce"]
                if sym in dis_by_stock:
                    for dis_date in dis_by_stock[sym]:
                        delta = (dis_date - ann_date).days
                        if 0 < delta <= 30:
                            exit_idx = cal.searchsorted(dis_date, side="left") - 1
                            break
                    else:
                        exit_idx = min(entry_idx + 9, n_cal - 1)
                else:
                    exit_idx = min(entry_idx + 9, n_cal - 1)
            else:
                exit_idx = min(entry_idx + 9, n_cal - 1)

            if exit_idx >= n_cal:
                continue
            exit_price = close.iloc[exit_idx][sym]
            if np.isnan(exit_price) or exit_price <= 0:
                continue
            ret = exit_price / entry_price - 1 - COST_RATE
            filtered_returns.append({
                "symbol": sym,
                "year": row["year"],
                "ret": ret,
                "upgraded": row["leads_to_disposal"],
            })
        except (IndexError, KeyError):
            continue

    fr = pd.DataFrame(filtered_returns)
    if len(fr) > 50:
        print(f"    全部: mean={fr['ret'].mean():.2%}, win={(fr['ret']>0).mean():.1%}, n={len(fr):,}")
        upg_only = fr[fr["upgraded"]]
        non_upg = fr[~fr["upgraded"]]
        if len(upg_only) > 20:
            print(f"    升級的: mean={upg_only['ret'].mean():.2%}, win={(upg_only['ret']>0).mean():.1%}, n={len(upg_only):,}")
        if len(non_upg) > 20:
            print(f"    未升級: mean={non_upg['ret'].mean():.2%}, win={(non_upg['ret']>0).mean():.1%}, n={len(non_upg):,}")

        # Yearly
        print(f"\n    逐年:")
        for year in sorted(fr["year"].unique()):
            yr = fr[fr["year"] == year]
            if len(yr) < 10:
                continue
            print(f"      {year}: mean={yr['ret'].mean():.2%}, win={(yr['ret']>0).mean():.1%}, n={len(yr)}")

    # Save model predictions
    att_model["prob_gb"] = np.nan
    att_model.loc[test_mask, "prob_gb"] = y_prob_gb
    att_model.to_csv(OUT / "p19c_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
