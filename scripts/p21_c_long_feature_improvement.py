"""P21: C-Long 模型特徵改善 — 連續天數 + 門檻距離 + 板塊.

改善方向:
1. consecutive_days: 連續被注意的具體天數 (不是 binary)
2. days_to_threshold: 從文字解析距離處置門檻還差幾天/幾次
3. sector_attention_count: 同板塊近3日被注意的股票數
4. LightGBM 替代 sklearn GB

驗證: Walk-forward AUC + Portfolio CAGR
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.metrics import precision_score, roc_auc_score
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p21_feature_improve")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
MAX_POSITIONS = 5
MAX_HOLD_DAYS = 12
THRESHOLD = 0.55


def extract_improved_text_features(text):
    """Enhanced text parsing with threshold distance."""
    if pd.isna(text) or not isinstance(text, str):
        return {}
    feat = {}

    # 第X款
    kuai = re.search(r'[﹝（(]第([一二三四五六七八九十]+)款[﹞）)]', text)
    kuai_map = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'十一':11,'十二':12}
    feat["kuai"] = kuai_map.get(kuai.group(1), 0) if kuai else 0

    # 督導會報
    feat["is_supervisory"] = 1 if '督導會報' in text else 0

    # 漲幅/跌幅
    pct = re.search(r'[涨跌漲跌]幅達(\d+\.?\d*)%', text)
    feat["pct_change"] = float(pct.group(1)) if pct else 0

    # 振幅
    amp = re.search(r'振幅達(\d+\.?\d*)%', text)
    feat["amplitude"] = float(amp.group(1)) if amp else 0

    # 量能倍數
    vol_ratio = re.search(r'放大為\s*(\d+\.?\d*)倍', text)
    feat["volume_ratio_text"] = float(vol_ratio.group(1)) if vol_ratio else 0

    # 成交值比率
    val_ratio = re.search(r'成交值比率達(\d+\.?\d*)%', text)
    feat["value_ratio"] = float(val_ratio.group(1)) if val_ratio else 0

    # === NEW: 連續天數解析 ===
    # "連續第N日" or "最近N個營業日"
    consec_match = re.search(r'連續第?(\d+)日', text)
    if not consec_match:
        consec_match = re.search(r'最近(\d+)個?[營業]*日', text)
    feat["stated_days_window"] = int(consec_match.group(1)) if consec_match else 0

    # "10日內第N次"
    repeat_match = re.search(r'(\d+)日內第?(\d+)次', text)
    if repeat_match:
        feat["repeat_window"] = int(repeat_match.group(1))
        feat["repeat_count"] = int(repeat_match.group(2))
        feat["repeat_remaining"] = 6 - int(repeat_match.group(2))  # threshold is 6
    else:
        feat["repeat_window"] = 0
        feat["repeat_count"] = 0
        feat["repeat_remaining"] = 0

    # === NEW: 距離門檻的估計 ===
    # 第一款: 6日漲幅35% → 如果已達50%, 超過門檻了, 明天很可能繼續
    # 第二款: 5日漲幅50%
    # 連3日: 連續3日收盤價漲跌達X% → 如果已連續2日, 差1天
    # 10日內6次: 如果已5次, 差1次
    feat["excess_over_threshold"] = 0
    if feat["pct_change"] > 0:
        thresholds = {1: 35, 2: 50, 4: 100}
        thr = thresholds.get(feat["kuai"], 35)
        feat["excess_over_threshold"] = feat["pct_change"] - thr

    # 借券/當沖
    feat["is_lending"] = 1 if '借券' in text else 0
    feat["is_intraday"] = 1 if '當沖' in text else 0
    feat["is_price_trigger"] = 1 if ('漲幅' in text or '跌幅' in text or '振幅' in text) else 0
    feat["is_volume_trigger"] = 1 if ('成交量' in text or '成交值' in text) else 0
    feat["text_len"] = len(text)

    return feat


def main():
    print("=" * 70)
    print("P21: C-Long 模型特徵改善")
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
    reason_col = None
    for col in ["注意交易資訊", "註意原因", "原因", "reason"]:
        if col in att.columns:
            reason_col = col
            break
    att["reason_text"] = att[reason_col].astype(str) if reason_col else ""
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

    # === BUILD FEATURES ===
    print("\nBuilding features...")

    # Text features (improved)
    text_features = []
    for _, row in att.iterrows():
        tf = extract_improved_text_features(row["reason_text"])
        text_features.append(tf)
    text_df = pd.DataFrame(text_features, index=att.index)

    # Price features
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
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    price_features = []
    consec_days_list = []
    for _, row in att.iterrows():
        sym = row["stock_id"]
        ann_date = row["announce"]
        ai = int(row["ann_idx"])
        pf = {}
        try:
            if sym in close.columns and ai < len(close.index):
                c = close.iloc[max(0, ai-20):ai+1][sym]
                v = vol.iloc[max(0, ai-20):ai+1][sym]
                if len(c.dropna()) >= 5:
                    pf["mom_5d"] = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else 0
                    pf["mom_10d"] = float(c.iloc[-1] / c.iloc[-11] - 1) if len(c) >= 11 else 0
                    pf["mom_20d"] = float(c.iloc[-1] / c.iloc[0] - 1) if len(c) >= 21 else 0
                    pf["vol_20d"] = float(c.pct_change().dropna().std()) if len(c) >= 6 else 0
                    pf["vol_ratio"] = float(v.iloc[-5:].mean() / v.iloc[-20:].mean()) if len(v) >= 20 else 1
                    pf["turnover"] = float(c.iloc[-1] * v.iloc[-1])
                    ma20_val = float(c.mean())
                    pf["bias_20d"] = float(c.iloc[-1] / ma20_val - 1) if ma20_val > 0 else 0
                else:
                    for k in ["mom_5d","mom_10d","mom_20d","vol_20d","vol_ratio","turnover","bias_20d"]:
                        pf[k] = 0
            else:
                for k in ["mom_5d","mom_10d","mom_20d","vol_20d","vol_ratio","turnover","bias_20d"]:
                    pf[k] = 0
        except (IndexError, KeyError):
            for k in ["mom_5d","mom_10d","mom_20d","vol_20d","vol_ratio","turnover","bias_20d"]:
                pf[k] = 0

        # Historical
        hist_count = 0
        if sym in dis_dates_by_stock:
            cutoff = ann_date - pd.Timedelta(days=365)
            hist_count = sum(1 for d in dis_dates_by_stock[sym] if cutoff <= d < ann_date)
        pf["hist_disposal_count"] = hist_count

        days_since = 999
        if sym in att_dates_by_stock:
            prev = [d for d in att_dates_by_stock[sym] if d < ann_date]
            if prev:
                days_since = (ann_date - prev[-1]).days
        pf["days_since_last_att"] = min(days_since, 999)

        att_count = 0
        if sym in att_dates_by_stock:
            cutoff = ann_date - pd.Timedelta(days=30)
            att_count = sum(1 for d in att_dates_by_stock[sym] if cutoff <= d < ann_date)
        pf["att_count_30d"] = att_count

        price_features.append(pf)

        # === NEW: Consecutive days count ===
        consec = 0
        if sym in att_dates_by_stock:
            for d in range(1, 10):
                check_date = ann_date - pd.Timedelta(days=d)
                # Check trading days (approximate with calendar days 1-3 for weekends)
                found = False
                for offset in [0, 1, 2, 3]:
                    if (check_date - pd.Timedelta(days=offset)) in att_dates_by_stock[sym]:
                        found = True
                        break
                if found:
                    consec += 1
                else:
                    break
        consec_days_list.append(consec)

    price_df = pd.DataFrame(price_features, index=att.index)
    att["consecutive_days"] = consec_days_list

    # === NEW: Sector attention clustering ===
    # Use stock_id prefix as rough sector proxy (first digit)
    att["sector"] = att["stock_id"].str[0]
    sector_att_count = []
    for _, row in att.iterrows():
        sec = row["sector"]
        ann_date = row["announce"]
        cutoff = ann_date - pd.Timedelta(days=5)
        count = len(att[
            (att["sector"] == sec) &
            (att["announce"] >= cutoff) &
            (att["announce"] < ann_date) &
            (att["stock_id"] != row["stock_id"])
        ])
        sector_att_count.append(count)
    att["sector_attention_5d"] = sector_att_count

    # Combine all features
    all_features = pd.concat([text_df, price_df, att[["consecutive_days", "sector_attention_5d"]]], axis=1)
    feature_cols = list(all_features.columns)
    print(f"  Total features: {len(feature_cols)}")

    # === WALK-FORWARD ===
    print("\n" + "=" * 70)
    print("Walk-Forward 驗證")
    print("=" * 70)

    att["year"] = att["announce"].dt.year
    X_all = all_features.values.astype(float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = att["leads_to_disposal"].astype(int).values

    # Try both models
    from sklearn.ensemble import GradientBoostingClassifier
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False
        print("  LightGBM not available, using sklearn GB only")

    results = {}
    for model_name in ["sklearn_gb", "lightgbm"] if HAS_LGB else ["sklearn_gb"]:
        print(f"\n  --- {model_name} ---")
        all_oos_probs = np.full(len(att), np.nan)

        for test_year in range(2021, 2027):
            train_mask = (att["year"] <= test_year - 1).values
            test_mask = (att["year"] == test_year).values

            if train_mask.sum() < 1000 or test_mask.sum() < 50:
                continue

            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test = X_all[test_mask]

            if model_name == "sklearn_gb":
                model = GradientBoostingClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.1,
                    subsample=0.8, random_state=42, min_samples_leaf=20
                )
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=500, max_depth=8, learning_rate=0.05,
                    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
                )

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            all_oos_probs[test_mask] = probs

            auc = roc_auc_score(y_all[test_mask], probs)
            print(f"    {test_year}: AUC={auc:.4f}, n_test={test_mask.sum()}")

        att[f"prob_{model_name}"] = all_oos_probs
        valid = att[f"prob_{model_name}"].notna()
        overall_auc = roc_auc_score(y_all[valid.values], all_oos_probs[valid.values])
        print(f"    Overall AUC: {overall_auc:.4f}")
        results[model_name] = overall_auc

    # === PORTFOLIO COMPARISON ===
    print("\n" + "=" * 70)
    print("Portfolio 比較 (threshold=0.55, consec filter, 12d)")
    print("=" * 70)

    # Compute trade returns
    dis_by_stock_for_exit = dis_by_stock
    trade_rets = []
    exit_idxs = []
    hold_days_list = []
    for _, row in att.iterrows():
        sym = row["stock_id"]
        ai = int(row["ann_idx"])
        if sym not in valid_stocks:
            trade_rets.append(np.nan); exit_idxs.append(np.nan); hold_days_list.append(np.nan); continue
        entry_idx = ai + 1
        if entry_idx >= n_cal - 1:
            trade_rets.append(np.nan); exit_idxs.append(np.nan); hold_days_list.append(np.nan); continue
        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[ai][sym]
            avg_to = avg_turnover_5d.iloc[ai][sym]
        except (IndexError, KeyError):
            trade_rets.append(np.nan); exit_idxs.append(np.nan); hold_days_list.append(np.nan); continue
        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            trade_rets.append(np.nan); exit_idxs.append(np.nan); hold_days_list.append(np.nan); continue
        if not (np.isnan(avg_to) or avg_to >= 20_000_000):
            trade_rets.append(np.nan); exit_idxs.append(np.nan); hold_days_list.append(np.nan); continue
        gap = entry_price / prev_close - 1
        if not (-0.08 < gap < 0.04):
            trade_rets.append(np.nan); exit_idxs.append(np.nan); hold_days_list.append(np.nan); continue

        exit_idx = min(entry_idx + MAX_HOLD_DAYS - 1, n_cal - 1)
        ann_date = row["announce"]
        if sym in dis_by_stock_for_exit:
            for dis_date in dis_by_stock_for_exit[sym]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    dis_exit = cal.searchsorted(dis_date, side="left") - 1
                    if dis_exit > entry_idx:
                        exit_idx = min(exit_idx, dis_exit)
                    break
        try:
            exit_price = close.iloc[exit_idx][sym]
            if np.isnan(exit_price) or exit_price <= 0:
                trade_rets.append(np.nan); exit_idxs.append(np.nan); hold_days_list.append(np.nan); continue
        except (IndexError, KeyError):
            trade_rets.append(np.nan); exit_idxs.append(np.nan); hold_days_list.append(np.nan); continue
        ret = exit_price / entry_price - 1 - COST_RATE
        trade_rets.append(ret)
        exit_idxs.append(exit_idx)
        hold_days_list.append(exit_idx - entry_idx + 1)

    att["ret"] = trade_rets
    att["exit_idx"] = exit_idxs
    att["hold_days"] = hold_days_list

    # Consecutive filter
    att["consec_filter"] = att["consecutive_days"] >= 1

    for model_name in results:
        prob_col = f"prob_{model_name}"
        signals = att[
            (att[prob_col] >= THRESHOLD) &
            (att["consec_filter"]) &
            (att["ret"].notna())
        ].copy()
        signals["prob"] = signals[prob_col]
        signals = signals.sort_values(["announce", "prob"], ascending=[True, False])

        # Portfolio sim
        active = []
        executed = []
        for _, trade in signals.iterrows():
            ei = int(trade["ann_idx"]) + 1
            xi = min(int(trade["exit_idx"]), n_cal - 1)
            active = [(e, s) for e, s in active if e > ei]
            if len(active) >= MAX_POSITIONS:
                continue
            if any(s == trade["stock_id"] for _, s in active):
                continue
            active.append((xi, trade["stock_id"]))
            executed.append(trade)

        ex = pd.DataFrame(executed)
        if len(ex) < 30:
            continue

        # Metrics
        daily_returns = pd.Series(0.0, index=cal)
        cap = 1.0 / MAX_POSITIONS
        for _, t in ex.iterrows():
            ei = int(t["ann_idx"]) + 1
            xi = min(int(t["exit_idx"]), n_cal - 1)
            hold = max(xi - ei + 1, 1)
            dr = t["ret"] / hold
            for d in range(ei, min(xi + 1, n_cal)):
                daily_returns.iloc[d] += dr * cap

        dm = daily_returns.mean()
        ds = daily_returns.std()
        sharpe = dm / ds * np.sqrt(252) if ds > 0 else 0
        cum = (1 + daily_returns).cumprod()
        rm = cum.cummax()
        mdd = ((cum - rm) / rm).min()
        first_idx = int(ex["ann_idx"].min()) + 1
        total_days = n_cal - first_idx
        cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

        yearly_pass = 0
        yearly_total = 0
        for yr in sorted(ex["year"].unique()):
            yr_data = ex[ex["year"] == yr]
            if len(yr_data) < 5:
                continue
            yearly_total += 1
            if yr_data["ret"].mean() > 0:
                yearly_pass += 1

        print(f"\n  {model_name}:")
        print(f"    CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, MDD={mdd:.1%}")
        print(f"    AvgRet={ex['ret'].mean():.2%}, Win={(ex['ret']>0).mean():.1%}")
        print(f"    Years: {yearly_pass}/{yearly_total}, Trades: {len(ex)}")
        print(f"    AUC: {results[model_name]:.4f}")

    # === FEATURE IMPORTANCE (best model) ===
    print("\n" + "=" * 70)
    print("新模型特徵重要性 (trained on all data)")
    print("=" * 70)

    if HAS_LGB:
        best_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
        )
    else:
        best_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42, min_samples_leaf=20
        )
    best_model.fit(X_all, y_all)

    if HAS_LGB:
        importances = best_model.feature_importances_
    else:
        importances = best_model.feature_importances_

    print(f"\n  {'Rank':>4} {'Feature':<30} {'Importance':>10}")
    print(f"  {'-'*4} {'-'*30} {'-'*10}")
    for i, (name, imp) in enumerate(sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:20]):
        print(f"  {i+1:>4} {name:<30} {imp:>10.4f}")

    # Save
    att_out_cols = ["stock_id", "announce", "year", "leads_to_disposal", "ret", "hold_days",
                    "consecutive_days", "sector_attention_5d"]
    for mn in results:
        att_out_cols.append(f"prob_{mn}")
    att[att_out_cols].to_csv(OUT / "p21_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
