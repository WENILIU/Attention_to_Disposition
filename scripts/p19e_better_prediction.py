"""P19e: Improved upgrade prediction using regulatory threshold proximity.

KEY INSIGHT: The attention reason text contains EXACT information about
how close the stock is to the disposal threshold:
- "連續第N日" → disposal at N=3 (or N=5 for 連5日)
- "10日內第N次" → disposal at N=6
- "督導會報" → disposal is certain
- "借券比" → specific threshold

This should give near-perfect prediction for many cases!
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p19e_better_pred")
OUT.mkdir(parents=True, exist_ok=True)


def extract_regulatory_features(text):
    """Extract EXACT regulatory threshold proximity from attention reason."""
    if pd.isna(text) or not isinstance(text, str):
        return {}

    feat = {}

    # Parse the 第X款 criterion type (uses full-width brackets ﹝﹞)
    kuai = re.search(r'[﹝（(]第([一二三四五六七八九十]+)款[﹞）)]', text)
    kuai_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12}
    feat["kuai"] = kuai_map.get(kuai.group(1), 0) if kuai else 0

    # 督導會報 (第十二款) → disposal is CERTAIN
    feat["is_supervisory"] = 1 if '督導會報' in text else 0

    # Extract numeric severity
    # "漲幅達XX%" or "跌幅達XX%"
    pct = re.search(r'[涨跌漲跌]幅達(\d+\.?\d*)%', text)
    feat["pct_change"] = float(pct.group(1)) if pct else 0

    # "振幅達XX%"
    amp = re.search(r'振幅達(\d+\.?\d*)%', text)
    feat["amplitude"] = float(amp.group(1)) if amp else 0

    # "放大為X.XX倍" (volume ratio)
    vol_ratio = re.search(r'放大為\s*(\d+\.?\d*)倍', text)
    feat["volume_ratio_text"] = float(vol_ratio.group(1)) if vol_ratio else 0

    # "成交值比率達XX%"
    val_ratio = re.search(r'成交值比率達(\d+\.?\d*)%', text)
    feat["value_ratio"] = float(val_ratio.group(1)) if val_ratio else 0

    # "借券比" related
    feat["is_lending"] = 1 if '借券' in text else 0

    # "當沖" related
    feat["is_intraday"] = 1 if '當沖' in text else 0

    # Price-based vs volume-based trigger
    feat["is_price_trigger"] = 1 if ('漲幅' in text or '跌幅' in text or '振幅' in text) else 0
    feat["is_volume_trigger"] = 1 if ('成交量' in text or '成交值' in text) else 0

    # Severity: how far above the threshold (higher = more likely to repeat)
    # 第一款 threshold: 35% for 6-day, 第二款: 50% for 5-day, 第四款: 100% for 30-day
    if feat["pct_change"] > 0:
        if feat["kuai"] == 1:
            feat["excess_over_threshold"] = feat["pct_change"] - 35  # 第一款 threshold ~35%
        elif feat["kuai"] == 2:
            feat["excess_over_threshold"] = feat["pct_change"] - 50  # 第二款 threshold ~50%
        elif feat["kuai"] == 4:
            feat["excess_over_threshold"] = feat["pct_change"] - 100  # 第四款 threshold ~100%
        else:
            feat["excess_over_threshold"] = feat["pct_change"]
    else:
        feat["excess_over_threshold"] = 0

    # Text length
    feat["text_len"] = len(text)

    return feat


def main():
    print("=" * 70)
    print("P19e: 改進預測模型 — 監管門檻距離")
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

    # Build attention events with reason text
    att = pd.DataFrame(att_raw).copy()
    att["stock_id"] = att["symbol"].astype(str).str.zfill(4)
    att["announce"] = pd.to_datetime(att["date"]).dt.normalize()

    # Find reason column
    reason_col = None
    for col in ["注意交易資訊", "註意原因", "原因", "reason"]:
        if col in att.columns:
            reason_col = col
            break

    if reason_col:
        att["reason_text"] = att[reason_col].astype(str)
    else:
        print("  WARNING: No reason column found!")
        att["reason_text"] = ""

    # Show sample reasons
    print(f"\n  Sample attention reasons:")
    for _, row in att.head(10).iterrows():
        print(f"    {row['reason_text'][:100]}")

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

    print(f"\n  Events: {len(att):,}, Upgraded: {att['leads_to_disposal'].sum():,} ({att['leads_to_disposal'].mean():.1%})")

    # === EXTRACT REGULATORY FEATURES ===
    print("\nExtracting regulatory features...")
    reg_features = att["reason_text"].apply(extract_regulatory_features).apply(pd.Series)
    print(f"  Features: {list(reg_features.columns)}")

    # === ANALYZE UPGRADE RATE BY FEATURE ===
    print("\n" + "=" * 70)
    print("升級率 by 監管門檻距離")
    print("=" * 70)

    att["kuai"] = reg_features["kuai"]
    att["is_supervisory"] = reg_features["is_supervisory"]
    att["pct_change"] = reg_features["pct_change"]
    att["volume_ratio_text"] = reg_features["volume_ratio_text"]

    print("\n  第X款 vs 升級率:")
    for n in sorted(att["kuai"].unique()):
        subset = att[att["kuai"] == n]
        if len(subset) < 10:
            continue
        rate = subset["leads_to_disposal"].mean()
        print(f"    第{n}款: upgrade={rate:.1%}, n={len(subset):,}")

    print("\n  漲幅程度 vs 升級率:")
    att["pct_bin"] = pd.cut(att["pct_change"], bins=[0, 35, 50, 100, 200, 999], labels=["<35%", "35-50%", "50-100%", "100-200%", ">200%"])
    for b, grp in att.groupby("pct_bin", observed=True):
        if len(grp) < 10:
            continue
        rate = grp["leads_to_disposal"].mean()
        print(f"    {b}: upgrade={rate:.1%}, n={len(grp):,}")

    print(f"\n  督導會報: upgrade={att[att['is_supervisory']==1]['leads_to_disposal'].mean():.1%}, n={(att['is_supervisory']==1).sum()}")

    # === BUILD COMBINED MODEL ===
    print("\n" + "=" * 70)
    print("Combined Model (regulatory + price features)")
    print("=" * 70)

    # Price features
    returns = close.pct_change()
    vol_20d = returns.rolling(20).std()
    ma20 = close.rolling(20).mean()
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Historical disposal
    dis_dates_by_stock = {sid: sorted(dates) for sid, dates in dis_by_stock.items()}
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = sorted(group["announce"].tolist())

    price_features = []
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
                feat["mom_5d"] = (c_now / c_5 - 1) if not np.isnan(c_5) and c_5 > 0 else 0
                feat["mom_10d"] = (c_now / c_10 - 1) if not np.isnan(c_10) and c_10 > 0 else 0
                feat["mom_20d"] = (c_now / c_20 - 1) if not np.isnan(c_20) and c_20 > 0 else 0
                feat["vol_20d"] = vol_20d.iloc[ai][sym] if sym in vol_20d.columns and not np.isnan(vol_20d.iloc[ai][sym]) else 0
                v5 = vol_ma5.iloc[ai][sym] if sym in vol_ma5.columns else np.nan
                v20 = vol_ma20.iloc[ai][sym] if sym in vol_ma20.columns else np.nan
                feat["vol_ratio"] = v5 / v20 if not np.isnan(v20) and v20 > 0 else 1
                feat["turnover"] = avg_turnover_5d.iloc[ai][sym] if sym in avg_turnover_5d.columns and not np.isnan(avg_turnover_5d.iloc[ai][sym]) else 0
                ma20_val = ma20.iloc[ai][sym] if sym in ma20.columns else np.nan
                feat["bias_20d"] = (c_now / ma20_val - 1) if not np.isnan(ma20_val) and ma20_val > 0 else 0
            else:
                for k in ["mom_5d","mom_10d","mom_20d","vol_20d","vol_ratio","turnover","bias_20d"]:
                    feat[k] = 0
        except (IndexError, KeyError):
            for k in ["mom_5d","mom_10d","mom_20d","vol_20d","vol_ratio","turnover","bias_20d"]:
                feat[k] = 0

        # Historical
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

        price_features.append(feat)

    price_df = pd.DataFrame(price_features, index=att.index)

    # Combine all features
    all_features = pd.concat([reg_features, price_df], axis=1)
    feature_cols = list(all_features.columns)
    print(f"  Total features: {len(feature_cols)}")

    # Time split
    att["year"] = att["announce"].dt.year
    train_mask = att["year"] <= 2023
    test_mask = att["year"] >= 2024

    X_train = all_features.loc[train_mask].values.astype(float)
    y_train = att.loc[train_mask, "leads_to_disposal"].astype(int).values
    X_test = all_features.loc[test_mask].values.astype(float)
    y_test = att.loc[test_mask, "leads_to_disposal"].astype(int).values

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Train: {X_train.shape[0]:,} (upgrade: {y_train.mean():.1%})")
    print(f"  Test: {X_test.shape[0]:,} (upgrade: {y_test.mean():.1%})")

    # Train model
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42, min_samples_leaf=20
    )
    gb.fit(X_train, y_train)
    y_prob = gb.predict_proba(X_test)[:, 1]

    print(f"\n  ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Feature importance
    print("\n  Top features:")
    for name, imp in sorted(zip(feature_cols, gb.feature_importances_), key=lambda x: x[1], reverse=True)[:15]:
        print(f"    {name}: {imp:.4f}")

    # Threshold analysis
    print("\n  Threshold analysis:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_prob >= t).astype(int)
        n_pred = y_pred.sum()
        if n_pred < 50:
            continue
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        print(f"    t={t:.1f}: precision={prec:.1%}, recall={rec:.1%}, selected={n_pred:,}")

    # === SIMPLE RULE-BASED APPROACH ===
    print("\n" + "=" * 70)
    print("Rule-Based: 直接用監管距離")
    print("=" * 70)

    # Rule: supervisory OR high severity (pct > 100% or volume > 10x)
    rule_high = (
        (att["is_supervisory"] == 1) |
        (att["pct_change"] >= 100) |
        (att["volume_ratio_text"] >= 10)
    )
    test_rule = rule_high & test_mask
    if test_rule.sum() > 50:
        prec_rule = att.loc[test_rule, "leads_to_disposal"].mean()
        print(f"\n  Rule (supervisory OR pct>=100% OR vol>=10x):")
        print(f"    Precision: {prec_rule:.1%}, n={test_rule.sum():,}")

    # Rule: supervisory OR pct >= 50%
    rule_mid = (
        (att["is_supervisory"] == 1) |
        (att["pct_change"] >= 50)
    )
    test_rule2 = rule_mid & test_mask
    if test_rule2.sum() > 50:
        prec_rule2 = att.loc[test_rule2, "leads_to_disposal"].mean()
        print(f"\n  Rule (supervisory OR pct>=50%):")
        print(f"    Precision: {prec_rule2:.1%}, n={test_rule2.sum():,}")

    # === COMBINED: Model + Rule ===
    print("\n" + "=" * 70)
    print("Combined: Model prob >= 0.5 OR Rule")
    print("=" * 70)

    model_high = pd.Series(y_prob >= 0.5, index=att.loc[test_mask].index)
    rule_high_test = rule_high.loc[test_mask]
    combined = model_high | rule_high_test
    combined_n = combined.sum()
    if combined_n > 50:
        combined_prec = att.loc[test_mask].loc[combined, "leads_to_disposal"].mean()
        print(f"  Combined precision: {combined_prec:.1%}, n={combined_n:,}")

    # Save
    save_cols = ["stock_id", "announce", "year", "leads_to_disposal", "kuai", "is_supervisory", "pct_change"]
    att_out = att[[c for c in save_cols if c in att.columns]].copy()
    att_out["prob_gb"] = np.nan
    att_out.loc[test_mask, "prob_gb"] = y_prob
    att_out.to_csv(OUT / "p19e_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
