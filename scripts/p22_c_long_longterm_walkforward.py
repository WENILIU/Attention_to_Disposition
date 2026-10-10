"""P22: C-Long 長週期 Walk-Forward 驗證 (2004-2026).

涵蓋完整市場週期:
  2008 金融海嘯, 2011 歐債危機, 2015 陸股崩盤,
  2018 貿易戰, 2020 COVID, 2022 升息循環

方法: 逐年 walk-forward, 用所有歷史訓練, 測下一年
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from finlab import data
import finlab

finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
import finlab.data as _fd
_fd._default_context._role = 'vip'

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p22_longterm")
OUT.mkdir(parents=True, exist_ok=True)

COST_RATE = 0.001425 + 0.003 + 0.003
MAX_POSITIONS = 5
MAX_HOLD_DAYS = 12
THRESHOLD = 0.55


def extract_text_features(text):
    if pd.isna(text) or not isinstance(text, str):
        return {}
    feat = {}
    kuai = re.search(r'[\uff5b\uff08(]\u7b2c([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+)\u6b3e[\uff5d\uff09)]', text)
    kuai_map = {'\u4e00':1,'\u4e8c':2,'\u4e09':3,'\u56db':4,'\u4e94':5,'\u516d':6,'\u4e03':7,'\u516b':8,'\u4e5d':9,'\u5341':10,'\u5341\u4e00':11,'\u5341\u4e8c':12}
    feat["kuai"] = kuai_map.get(kuai.group(1), 0) if kuai else 0
    feat["is_supervisory"] = 1 if '\u7763\u5c0e\u6703\u5831' in text else 0
    pct = re.search(r'[\u6da8\u8dcc\u6f32\u8dcc]\u5e45\u9054(\d+\.?\d*)%', text)
    feat["pct_change"] = float(pct.group(1)) if pct else 0
    amp = re.search(r'\u632f\u5e45\u9054(\d+\.?\d*)%', text)
    feat["amplitude"] = float(amp.group(1)) if amp else 0
    vol_ratio = re.search(r'\u653e\u5927\u70ba\s*(\d+\.?\d*)\u500d', text)
    feat["volume_ratio_text"] = float(vol_ratio.group(1)) if vol_ratio else 0
    feat["is_price_trigger"] = 1 if ('\u6f32\u5e45' in text or '\u8dcc\u5e45' in text or '\u632f\u5e45' in text) else 0
    feat["is_volume_trigger"] = 1 if ('\u6210\u4ea4\u91cf' in text or '\u6210\u4ea4\u503c' in text) else 0
    feat["is_lending"] = 1 if '\u501f\u5238' in text else 0
    feat["is_intraday"] = 1 if '\u7576\u6c92' in text else 0
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
    print("P22: C-Long \u9577\u9031\u671f Walk-Forward \u9a57\u8b49 (2004-2026)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    att_raw = data.get("trading_attention")
    dis_raw = data.get("disposal_information")
    close = data.get("price:\u6536\u76e4\u50f9")
    open_p = data.get("price:\u958b\u76e4\u50f9")
    vol = data.get("price:\u6210\u4ea4\u80a1\u6578")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)
    print(f"  Price data: {cal[0].date()} to {cal[-1].date()} ({n_cal} days)")

    # Build attention events (ALL years)
    att = pd.DataFrame(att_raw).copy()
    att["stock_id"] = att["symbol"].astype(str).str.zfill(4)
    att["announce"] = pd.to_datetime(att["date"]).dt.normalize()
    reason_col = None
    for col in ["\u6ce8\u610f\u4ea4\u6613\u8cc7\u8a0a", "\u8a3b\u610f\u539f\u56e0", "\u539f\u56e0", "reason"]:
        if col in att.columns:
            reason_col = col
            break
    att["reason_text"] = att[reason_col].astype(str) if reason_col else ""
    att = att[
        att["stock_id"].str.match(r"^\d{4}$") &
        att["stock_id"].isin(valid_stocks) &
        ~att["stock_id"].str.startswith(("00", "91"))
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

    att["year"] = att["announce"].dt.year
    print(f"  Events: {len(att):,}, Upgraded: {att['leads_to_disposal'].sum():,} ({att['leads_to_disposal'].mean():.1%})")

    # Build features
    print("\nBuilding features...")
    returns = close.pct_change()
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    dis_dates_by_stock = {sid: sorted(dates) for sid, dates in dis_by_stock.items()}
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    # Text features
    text_features = [extract_text_features(t) for t in att["reason_text"]]
    text_df = pd.DataFrame(text_features, index=att.index)

    # Price features
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
    all_features = pd.concat([text_df, price_df], axis=1)
    feature_cols = list(all_features.columns)
    print(f"  Features: {len(feature_cols)}")

    # Consecutive (external filter)
    att["consec_1d"] = False
    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann = row["announce"]
        if sid in att_dates_by_stock:
            for delta in [1, 2, 3]:
                if (ann - pd.Timedelta(days=delta)) in att_dates_by_stock[sid]:
                    att.loc[idx, "consec_1d"] = True
                    break

    # Compute trade returns
    print("Computing trade returns...")
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

    # === WALK-FORWARD ===
    print("\n" + "=" * 70)
    print("Walk-Forward: \u9010\u5e74\u8a13\u7df4\u6e2c\u8a66")
    print("=" * 70)

    X_all = all_features.values.astype(float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = att["leads_to_disposal"].astype(int).values

    all_oos_probs = np.full(len(att), np.nan)

    for test_year in range(2004, 2027):
        train_mask = (att["year"] <= test_year - 1).values
        test_mask = (att["year"] == test_year).values

        if train_mask.sum() < 2000 or test_mask.sum() < 50:
            continue

        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42, min_samples_leaf=20
        )
        model.fit(X_all[train_mask], y_all[train_mask])
        probs = model.predict_proba(X_all[test_mask])[:, 1]
        all_oos_probs[test_mask] = probs

        auc = roc_auc_score(y_all[test_mask], probs)
        print(f"  {test_year}: AUC={auc:.4f}, n_train={train_mask.sum():,}, n_test={test_mask.sum():,}")

    att["oos_prob"] = all_oos_probs

    # === PORTFOLIO SIMULATION ===
    print("\n" + "=" * 70)
    print("\u7d44\u5408\u6a2c\u64ec (5\u6a94, 12d, threshold=0.55, consec)")
    print("=" * 70)

    signals = att[
        (att["oos_prob"] >= THRESHOLD) &
        (att["consec_1d"]) &
        (att["ret"].notna())
    ].copy()
    signals["prob"] = signals["oos_prob"]
    signals = signals.sort_values(["announce", "prob"], ascending=[True, False])

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
    print(f"\n  Total executed: {len(ex):,} trades")
    print(f"  Date range: {ex['announce'].min().date()} to {ex['announce'].max().date()}")

    # Yearly breakdown
    print(f"\n  \u9010\u5e74\u8868\u73fe:")
    print(f"  {'Year':>6} {'n':>5} {'avg_ret':>8} {'win%':>6} {'mkt_ret':>8} {'\u5224\u5b9a':>6}")

    # Market benchmark (equal-weighted all stocks)
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_cum = (1 + mkt_ret).groupby(cal.year).apply(lambda x: (1+x).prod()-1)

    yearly_pass = 0
    yearly_total = 0
    yearly_results = []
    for yr in sorted(ex["year"].unique()):
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 5:
            continue
        avg = yr_data["ret"].mean()
        win = (yr_data["ret"] > 0).mean()
        mkt = mkt_cum.get(yr, 0)
        sign = "PASS" if avg > 0 else "FAIL"
        yearly_pass += (1 if avg > 0 else 0)
        yearly_total += 1
        yearly_results.append((yr, avg, win, len(yr_data), mkt))
        print(f"  {yr:>6} {len(yr_data):>5} {avg:>8.2%} {win:>6.1%} {mkt:>8.1%} {sign:>6}")

    print(f"\n  \u9010\u5e74\u901a\u904e: {yearly_pass}/{yearly_total}")

    # Bear market analysis
    print("\n" + "=" * 70)
    print("\u7a7a\u982d\u5e74\u4efd\u805a\u7126\u5206\u6790")
    print("=" * 70)

    bear_years = [2008, 2011, 2015, 2018, 2022]
    for yr in bear_years:
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 3:
            print(f"\n  {yr}: \u6a23\u672c\u4e0d\u8db3 (n={len(yr_data)})")
            continue
        avg = yr_data["ret"].mean()
        win = (yr_data["ret"] > 0).mean()
        mkt = mkt_cum.get(yr, 0)
        print(f"\n  {yr} (\u5927\u76e4{mkt:+.1%}):")
        print(f"    n={len(yr_data)}, avg={avg:.2%}, win={win:.1%}")
        # Monthly detail
        yr_data = yr_data.copy()
        yr_data["month"] = pd.to_datetime(yr_data["announce"]).dt.month
        for m in sorted(yr_data["month"].unique()):
            m_data = yr_data[yr_data["month"] == m]
            if len(m_data) < 3:
                continue
            print(f"      {m}\u6708: n={len(m_data)}, avg={m_data['ret'].mean():.2%}")

    # Portfolio metrics
    print("\n" + "=" * 70)
    print("\u7d44\u5408\u6307\u6a19")
    print("=" * 70)

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

    print(f"\n  CAGR: {cagr:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  MDD: {mdd:.1%}")
    print(f"  Avg return: {ex['ret'].mean():.2%}")
    print(f"  Win rate: {(ex['ret'] > 0).mean():.1%}")
    print(f"  Total trades: {len(ex):,}")
    print(f"  Period: {ex['announce'].min().date()} to {ex['announce'].max().date()}")

    # Save
    ex.to_csv(OUT / "p22_longterm_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_results, columns=["year", "avg_ret", "win", "n", "mkt_ret"]).to_csv(
        OUT / "p22_yearly_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
