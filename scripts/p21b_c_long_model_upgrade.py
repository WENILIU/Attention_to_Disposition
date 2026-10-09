"""P21b: C-Long 模型升級測試 — LightGBM vs sklearn GB.

保持 p20b 的最佳配置 (threshold=0.55, consec=external filter, 12d)
只換模型算法，看 AUC 和 CAGR 是否提升。

另外測試：加入 improved text parsing (門檻距離) 但不加入 consecutive_days。
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p21_model_upgrade")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
MAX_POSITIONS = 5
MAX_HOLD_DAYS = 12
THRESHOLD = 0.55


def extract_text_features(text, enhanced=False):
    """Extract features from attention reason text."""
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

    if enhanced:
        # Parse "連續第N日" or "最近N個營業日"
        consec_m = re.search(r'連續第?(\d+)日', text)
        if not consec_m:
            consec_m = re.search(r'最近(\d+)個?[營業]*日', text)
        feat["stated_days_window"] = int(consec_m.group(1)) if consec_m else 0

        # "10日內第N次"
        repeat_m = re.search(r'(\d+)日內第?(\d+)次', text)
        if repeat_m:
            feat["repeat_count"] = int(repeat_m.group(2))
            feat["repeat_remaining"] = max(0, 6 - int(repeat_m.group(2)))
        else:
            feat["repeat_count"] = 0
            feat["repeat_remaining"] = 0

        # 借券比數值
        lend_m = re.search(r'借券比達(\d+\.?\d*)%', text)
        feat["lending_ratio"] = float(lend_m.group(1)) if lend_m else 0

        # 當沖占比
        dt_m = re.search(r'當沖.*?(\d+\.?\d*)%', text)
        feat["daytrade_ratio"] = float(dt_m.group(1)) if dt_m else 0

    return feat


def main():
    print("=" * 70)
    print("P21b: C-Long 模型升級測試")
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

    print(f"  Events: {len(att):,}, Upgraded: {att['leads_to_disposal'].sum():,}")

    # Build price features (same for all variants)
    print("Building price features...")
    returns = close.pct_change()
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    dis_dates_by_stock = {sid: sorted(dates) for sid, dates in dis_by_stock.items()}
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    price_features = []
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

    price_df = pd.DataFrame(price_features, index=att.index)

    # Consecutive (external filter only, NOT a model feature)
    att["consec_1d"] = False
    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann = row["announce"]
        if sid in att_dates_by_stock:
            for delta in [1, 2, 3]:
                if (ann - pd.Timedelta(days=delta)) in att_dates_by_stock[sid]:
                    att.loc[idx, "consec_1d"] = True
                    break

    # === TEST VARIANTS ===
    att["year"] = att["announce"].dt.year
    y_all = att["leads_to_disposal"].astype(int).values

    variants = {}

    # Variant A: Original features + sklearn GB (baseline = p20b)
    text_a = [extract_text_features(t, enhanced=False) for t in att["reason_text"]]
    feat_a = pd.concat([pd.DataFrame(text_a, index=att.index), price_df], axis=1)
    variants["A: original + sklearn GB"] = (feat_a, "sklearn")

    # Variant B: Original features + LightGBM
    variants["B: original + LightGBM"] = (feat_a, "lightgbm")

    # Variant C: Enhanced text + LightGBM
    text_c = [extract_text_features(t, enhanced=True) for t in att["reason_text"]]
    feat_c = pd.concat([pd.DataFrame(text_c, index=att.index), price_df], axis=1)
    variants["C: enhanced text + LightGBM"] = (feat_c, "lightgbm")

    # Run walk-forward for each variant
    print("\n" + "=" * 70)
    print("Walk-Forward 比較")
    print("=" * 70)

    for variant_name, (features, model_type) in variants.items():
        feature_cols = list(features.columns)
        X = features.values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        all_probs = np.full(len(att), np.nan)
        yearly_aucs = []

        for test_year in range(2021, 2027):
            train_mask = (att["year"] <= test_year - 1).values
            test_mask = (att["year"] == test_year).values
            if train_mask.sum() < 1000 or test_mask.sum() < 50:
                continue

            if model_type == "sklearn":
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

            model.fit(X[train_mask], y_all[train_mask])
            probs = model.predict_proba(X[test_mask])[:, 1]
            all_probs[test_mask] = probs
            auc = roc_auc_score(y_all[test_mask], probs)
            yearly_aucs.append(auc)

        valid_mask = ~np.isnan(all_probs)
        overall_auc = roc_auc_score(y_all[valid_mask], all_probs[valid_mask])

        # Portfolio simulation
        att["_prob"] = all_probs
        signals = att[
            (att["_prob"] >= THRESHOLD) &
            (att["consec_1d"])
        ].copy()
        signals["prob"] = signals["_prob"]

        # Compute returns
        trade_data = []
        for _, row in signals.iterrows():
            sym = row["stock_id"]
            ai = int(row["ann_idx"])
            if sym not in valid_stocks:
                continue
            entry_idx = ai + 1
            if entry_idx >= n_cal - 1:
                continue
            try:
                entry_price = open_p.iloc[entry_idx][sym]
                prev_close = close.iloc[ai][sym]
                avg_to = avg_turnover_5d.iloc[ai][sym]
            except (IndexError, KeyError):
                continue
            if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
                continue
            if not (np.isnan(avg_to) or avg_to >= 20_000_000):
                continue
            gap = entry_price / prev_close - 1
            if not (-0.08 < gap < 0.04):
                continue

            exit_idx = min(entry_idx + MAX_HOLD_DAYS - 1, n_cal - 1)
            ann_date = row["announce"]
            if sym in dis_by_stock:
                for dis_date in dis_by_stock[sym]:
                    delta = (dis_date - ann_date).days
                    if 0 < delta <= 30:
                        dis_exit = cal.searchsorted(dis_date, side="left") - 1
                        if dis_exit > entry_idx:
                            exit_idx = min(exit_idx, dis_exit)
                        break
            try:
                exit_price = close.iloc[exit_idx][sym]
                if np.isnan(exit_price) or exit_price <= 0:
                    continue
            except (IndexError, KeyError):
                continue
            ret = exit_price / entry_price - 1 - COST_RATE
            trade_data.append({
                "stock_id": sym, "announce": row["announce"], "year": row["year"],
                "entry_idx": entry_idx, "exit_idx": exit_idx, "ret": ret,
                "hold_days": exit_idx - entry_idx + 1, "prob": row["prob"],
            })

        if len(trade_data) < 50:
            print(f"\n  {variant_name}: insufficient trades")
            continue

        sig_df = pd.DataFrame(trade_data).sort_values(["announce", "prob"], ascending=[True, False])

        # Portfolio sim
        active = []
        executed = []
        for _, trade in sig_df.iterrows():
            ei = int(trade["entry_idx"])
            xi = min(int(trade["exit_idx"]), n_cal - 1)
            active = [(e, s) for e, s in active if e > ei]
            if len(active) >= MAX_POSITIONS:
                continue
            if any(s == trade["stock_id"] for _, s in active):
                continue
            active.append((xi, trade["stock_id"]))
            executed.append(trade)

        ex = pd.DataFrame(executed)
        daily_returns = pd.Series(0.0, index=cal)
        cap = 1.0 / MAX_POSITIONS
        for _, t in ex.iterrows():
            ei = int(t["entry_idx"])
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
        first_idx = int(ex["entry_idx"].min())
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

        print(f"\n  {variant_name}:")
        print(f"    AUC: {overall_auc:.4f} (yearly: {', '.join(f'{a:.3f}' for a in yearly_aucs)})")
        print(f"    CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, MDD={mdd:.1%}")
        print(f"    AvgRet={ex['ret'].mean():.2%}, Win={(ex['ret']>0).mean():.1%}")
        print(f"    Years: {yearly_pass}/{yearly_total}, Trades: {len(ex)}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
