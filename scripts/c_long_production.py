"""C-Long 實盤腳本 — 每日執行生成交易指令.

策略 v3.1 (優化後):
  訊號：注意公告 + 昨日也被注意(連續) + GB model prob >= 0.55
  進場：注意公告次日開盤
  出場：處置公告日-1 OR 最多12天（取較早者）
  濾網：流動性>20M, Gap -8%~+4%
  止損：無
  部位：5檔，每檔20%

使用方式:
  python c_long_production.py              # 生成今日訂單
  python c_long_production.py --backtest   # 完整回測驗證
  python c_long_production.py --train      # 重新訓練模型

輸出:
  - orders_today.csv: 今日買賣指令
  - portfolio_state.json: 目前持倉狀態
  - c_long_model.pkl: 訓練好的模型
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import finlab
from finlab import data as _data

# ==================== 0. 設定 ====================
TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\auth_token.txt")
OUTPUT_DIR = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\c_long_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "c_long_model.pkl"
STATE_PATH = OUTPUT_DIR / "portfolio_state.json"

# 策略參數
THRESHOLD = 0.55
MAX_HOLD_DAYS = 12
MAX_POSITIONS = 5
POSITION_SIZE = 0.20
MIN_TURNOVER = 20_000_000
GAP_LOW = -0.08
GAP_HIGH = 0.04
COST_RATE = 0.001425 + 0.003 + 0.003

# 新制生效日
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")


def auto_login():
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            finlab.login(token)
            import finlab.data as _fd
            _fd._default_context._role = 'vip'
            print("✅ FinLab VIP 登入成功")
    else:
        finlab.login('V0wvdB4G2yj5qcHkqX6yNCKIOW/GE2oJ9dIlFftXoeN3NqjQTvBYAqWuJox5k/ilk#vip_m')
        import finlab.data as _fd
        _fd._default_context._role = 'vip'
        print("✅ FinLab VIP 登入成功 (hardcoded token)")


# ==================== 1. 特徵工程 ====================
def extract_text_features(text):
    """從注意交易資訊文字提取監管特徵."""
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


def build_features(att_df, close, open_p, vol, dis_by_stock, att_dates_by_stock, cal):
    """Build full feature matrix for attention events."""
    returns = close.pct_change()
    vol_20d = returns.rolling(20).std()
    ma20 = close.rolling(20).mean()
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    dis_dates_by_stock = {sid: sorted(dates) for sid, dates in dis_by_stock.items()}

    text_features = []
    price_features = []

    for _, row in att_df.iterrows():
        text = row.get("reason_text", "")
        tf = extract_text_features(text)
        default_tf = {"kuai":0,"is_supervisory":0,"pct_change":0,"amplitude":0,
                      "volume_ratio_text":0,"is_price_trigger":0,"is_volume_trigger":0,
                      "is_lending":0,"is_intraday":0,"text_len":0,"excess_over_threshold":0}
        for k in default_tf:
            tf.setdefault(k, default_tf[k])
        text_features.append(tf)

        sym = row["stock_id"]
        ann_date = row["announce"]
        ai = cal.searchsorted(ann_date, side="left")
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

        # Historical features
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

    text_df = pd.DataFrame(text_features, index=att_df.index)
    price_df = pd.DataFrame(price_features, index=att_df.index)
    all_features = pd.concat([text_df, price_df], axis=1)
    return all_features


# ==================== 2. 模型訓練 ====================
def train_model(att_df, close, open_p, vol, dis_by_stock, cal):
    """Train GB model on all historical data (for production use)."""
    print("\n🧠 訓練模型...")

    att_dates_by_stock = {}
    for sid, group in att_df.groupby("stock_id"):
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    features = build_features(att_df, close, open_p, vol, dis_by_stock, att_dates_by_stock, cal)
    feature_cols = list(features.columns)

    X = features.values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = att_df["leads_to_disposal"].astype(int).values

    print(f"  Training samples: {len(X):,}, positive rate: {y.mean():.1%}")
    print(f"  Features: {len(feature_cols)}")

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42, min_samples_leaf=20
    )
    gb.fit(X, y)

    # Save model
    model_data = {"model": gb, "feature_cols": feature_cols}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  ✅ Model saved: {MODEL_PATH}")

    return gb, feature_cols


def load_model():
    """Load pre-trained model."""
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_cols"]


# ==================== 3. 資料載入 ====================
def load_all_data():
    """Load all required data from finlab."""
    print("⏳ 載入數據...")
    att_raw = _data.get("trading_attention")
    dis_raw = _data.get("disposal_information")
    close = _data.get("price:收盤價")
    open_p = _data.get("price:開盤價")
    vol = _data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
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
        ~att["stock_id"].str.startswith(("00", "91"))
    ].copy()

    # Build disposal lookup
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["dis_announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis_by_stock = {}
    for _, row in dis.iterrows():
        sid = row["stock_id"]
        if sid not in dis_by_stock:
            dis_by_stock[sid] = []
        dis_by_stock[sid].append(row["dis_announce"])

    # Label: leads_to_disposal
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

    print(f"  Attention events: {len(att):,}, Upgraded: {att['leads_to_disposal'].sum():,}")
    return att, close, open_p, vol, dis_by_stock, cal, valid_stocks


# ==================== 4. 訊號生成 ====================
def generate_signals(att, close, open_p, vol, dis_by_stock, cal, valid_stocks, model, feature_cols):
    """Generate today's trading signals."""
    print("\n🧠 生成訊號...")

    # Build features for ALL events (for model prediction)
    att_dates_by_stock = {}
    for sid, group in att.groupby("stock_id"):
        att_dates_by_stock[sid] = set(group["announce"].tolist())

    features = build_features(att, close, open_p, vol, dis_by_stock, att_dates_by_stock, cal)
    X = features[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    att["prob"] = model.predict_proba(X)[:, 1]

    # Consecutive detection
    att["consec_1d"] = False
    for idx, row in att.iterrows():
        sid = row["stock_id"]
        ann = row["announce"]
        if sid in att_dates_by_stock:
            for delta in [1, 2, 3]:
                if (ann - pd.Timedelta(days=delta)) in att_dates_by_stock[sid]:
                    att.loc[idx, "consec_1d"] = True
                    break

    # Filter: consecutive + prob >= threshold
    signals = att[
        (att["consec_1d"]) &
        (att["prob"] >= THRESHOLD)
    ].copy()

    # Apply liquidity + gap filters
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    valid_signals = []
    for _, row in signals.iterrows():
        sym = row["stock_id"]
        ann_date = row["announce"]
        ai = cal.searchsorted(ann_date, side="left")
        entry_idx = ai + 1
        if entry_idx >= len(cal):
            continue
        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[ai][sym]
            avg_to = avg_turnover_5d.iloc[ai][sym]
        except (IndexError, KeyError):
            continue
        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            continue
        if not (np.isnan(avg_to) or avg_to >= MIN_TURNOVER):
            continue
        gap = entry_price / prev_close - 1
        if not (GAP_LOW < gap < GAP_HIGH):
            continue

        # Determine exit
        exit_idx = min(entry_idx + MAX_HOLD_DAYS - 1, len(cal) - 1)
        if sym in dis_by_stock:
            for dis_date in dis_by_stock[sym]:
                delta = (dis_date - ann_date).days
                if 0 < delta <= 30:
                    dis_exit = cal.searchsorted(dis_date, side="left") - 1
                    if dis_exit > entry_idx:
                        exit_idx = min(exit_idx, dis_exit)
                    break

        valid_signals.append({
            "stock_id": sym,
            "announce_date": ann_date,
            "entry_date": cal[entry_idx],
            "exit_date": cal[exit_idx],
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "prob": row["prob"],
            "consec": True,
        })

    result = pd.DataFrame(valid_signals)
    if len(result) > 0:
        result = result.sort_values(["entry_date", "prob"], ascending=[True, False])
    print(f"  Valid signals: {len(result):,}")
    return result


# ==================== 5. 組合管理 ====================
def manage_portfolio(signals, cal):
    """Simulate portfolio with position limits, return executed trades."""
    print("\n📊 組合管理...")

    active_positions = []
    executed = []
    skipped_capacity = 0
    skipped_duplicate = 0

    for _, trade in signals.iterrows():
        entry_idx = int(trade["entry_idx"])
        exit_idx = int(trade["exit_idx"])

        active_positions = [(ei, s) for ei, s in active_positions if ei > entry_idx]
        if len(active_positions) >= MAX_POSITIONS:
            skipped_capacity += 1
            continue
        if any(s == trade["stock_id"] for _, s in active_positions):
            skipped_duplicate += 1
            continue
        active_positions.append((exit_idx, trade["stock_id"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)
    print(f"  Executed: {len(ex):,}")
    print(f"  Skipped (capacity): {skipped_capacity:,}")
    print(f"  Skipped (duplicate): {skipped_duplicate:,}")
    return ex


# ==================== 6. 回測模式 ====================
def run_backtest(att, close, open_p, vol, dis_by_stock, cal, valid_stocks, model, feature_cols):
    """Full backtest with yearly + regime breakdown."""
    print("\n" + "=" * 70)
    print("C-Long v3.1 完整回測")
    print("=" * 70)

    signals = generate_signals(att, close, open_p, vol, dis_by_stock, cal, valid_stocks, model, feature_cols)
    if len(signals) == 0:
        print("  No signals generated!")
        return

    # Compute returns
    returns_list = []
    for _, trade in signals.iterrows():
        sym = trade["stock_id"]
        ei = int(trade["entry_idx"])
        xi = int(trade["exit_idx"])
        try:
            entry_price = open_p.iloc[ei][sym]
            exit_price = close.iloc[xi][sym]
            if any(np.isnan(x) or x <= 0 for x in [entry_price, exit_price]):
                returns_list.append(np.nan)
            else:
                returns_list.append(exit_price / entry_price - 1 - COST_RATE)
        except (IndexError, KeyError):
            returns_list.append(np.nan)

    signals["ret"] = returns_list
    signals = signals[signals["ret"].notna()].copy()
    signals["year"] = pd.to_datetime(signals["entry_date"]).dt.year
    signals["hold_days"] = signals["exit_idx"] - signals["entry_idx"] + 1

    # Portfolio simulation
    ex = manage_portfolio(signals, cal)
    if len(ex) == 0:
        return

    # Metrics
    n_cal = len(cal)
    daily_returns = pd.Series(0.0, index=cal)
    cap = 1.0 / MAX_POSITIONS
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
        hold = max(xi - ei + 1, 1)
        daily_ret = t["ret"] / hold
        for d in range(ei, min(xi + 1, n_cal)):
            daily_returns.iloc[d] += daily_ret * cap

    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0

    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    mdd = ((cum - running_max) / running_max).min()

    first_idx = int(ex["entry_idx"].min())
    total_days = n_cal - first_idx
    cagr = cum.iloc[-1] ** (252 / total_days) - 1 if cum.iloc[-1] > 0 else -1

    print(f"\n{'='*60}")
    print(f"  C-Long v3.1 回測結果")
    print(f"{'='*60}")
    print(f"  CAGR: {cagr:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  MDD: {mdd:.1%}")
    print(f"  Avg return/trade: {ex['ret'].mean():.2%}")
    print(f"  Win rate: {(ex['ret'] > 0).mean():.1%}")
    print(f"  Total trades: {len(ex):,}")
    print(f"  Avg hold: {ex['hold_days'].mean():.1f} days")
    print(f"{'='*60}")

    # Yearly
    print(f"\n  逐年:")
    print(f"  {'Year':>6} {'n':>5} {'avg_ret':>8} {'win%':>6} {'判定':>6}")
    yearly_pass = 0
    yearly_total = 0
    for yr in sorted(ex["year"].unique()):
        yr_data = ex[ex["year"] == yr]
        if len(yr_data) < 5:
            continue
        avg = yr_data["ret"].mean()
        win = (yr_data["ret"] > 0).mean()
        sign = "PASS" if avg > 0 else "FAIL"
        yearly_pass += (1 if avg > 0 else 0)
        yearly_total += 1
        print(f"  {yr:>6} {len(yr_data):>5} {avg:>8.2%} {win:>6.1%} {sign:>6}")
    print(f"\n  逐年通過: {yearly_pass}/{yearly_total}")

    # New vs Old regime
    print(f"\n  新制 vs 舊制:")
    ex["is_new_regime"] = pd.to_datetime(ex["entry_date"]) >= NEW_REGIME_DATE
    old = ex[~ex["is_new_regime"]]
    new = ex[ex["is_new_regime"]]
    if len(old) > 10:
        print(f"    舊制: n={len(old):,}, avg={old['ret'].mean():.2%}, win={(old['ret']>0).mean():.1%}")
    if len(new) > 5:
        print(f"    新制: n={len(new):,}, avg={new['ret'].mean():.2%}, win={(new['ret']>0).mean():.1%}")
    elif len(new) > 0:
        print(f"    新制: n={len(new)}, avg={new['ret'].mean():.2%} (樣本不足,持續監測)")

    # Save
    ex.to_csv(OUTPUT_DIR / "c_long_backtest_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n  📁 Trades: {OUTPUT_DIR / 'c_long_backtest_trades.csv'}")


# ==================== 7. 實盤模式 ====================
def run_production(att, close, open_p, vol, dis_by_stock, cal, valid_stocks, model, feature_cols):
    """Generate today's orders."""
    print("\n" + "=" * 70)
    print("C-Long 實盤模式 — 今日訂單")
    print("=" * 70)

    today = cal[-1]  # Last available date
    print(f"\n  基準日期: {today.date()}")

    signals = generate_signals(att, close, open_p, vol, dis_by_stock, cal, valid_stocks, model, feature_cols)

    # Load portfolio state
    state = load_state()

    # Determine today's actions
    # 1. Check existing positions for exit
    exit_orders = []
    remaining_positions = []
    for pos in state["positions"]:
        exit_date = pd.Timestamp(pos["exit_date"])
        if exit_date <= today:
            exit_orders.append({
                "action": "SELL",
                "stock_id": pos["stock_id"],
                "reason": f"到期出場 (持有{pos.get('hold_days', '?')}天)" if pos.get("exit_reason") != "disposal" else "處置公告出場",
            })
        else:
            remaining_positions.append(pos)

    # 2. Check for new entries (signals with entry_date == today)
    today_signals = signals[signals["entry_date"] == today] if len(signals) > 0 else pd.DataFrame()

    # Apply position limit
    available_slots = MAX_POSITIONS - len(remaining_positions)
    new_orders = []
    if len(today_signals) > 0 and available_slots > 0:
        for _, sig in today_signals.head(available_slots).iterrows():
            if sig["stock_id"] in [p["stock_id"] for p in remaining_positions]:
                continue
            new_orders.append({
                "action": "BUY",
                "stock_id": sig["stock_id"],
                "reason": f"模型機率={sig['prob']:.2f}, 連續注意",
                "exit_date": str(sig["exit_date"].date()),
                "entry_idx": int(sig["entry_idx"]),
                "exit_idx": int(sig["exit_idx"]),
            })

    # 3. Output orders
    orders = exit_orders + new_orders
    if orders:
        orders_df = pd.DataFrame(orders)
        orders_df.to_csv(OUTPUT_DIR / "orders_today.csv", index=False, encoding="utf-8-sig")
        print(f"\n  📋 今日訂單 ({len(orders)} 筆):")
        for o in orders:
            print(f"    {o['action']:>4} {o['stock_id']} — {o['reason']}")
    else:
        print(f"\n  ✅ 今日無訂單")

    # 4. Update portfolio state
    state["positions"] = remaining_positions + [
        {"stock_id": o["stock_id"], "entry_date": str(today.date()),
         "exit_date": o["exit_date"], "prob": o.get("prob", 0)}
        for o in new_orders
    ]
    state["last_run"] = str(today.date())
    save_state(state)

    print(f"\n  📁 訂單: {OUTPUT_DIR / 'orders_today.csv'}")
    print(f"  📁 狀態: {STATE_PATH}")


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"positions": [], "last_run": None}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="C-Long 實盤腳本")
    parser.add_argument("--backtest", action="store_true", help="完整回測模式")
    parser.add_argument("--train", action="store_true", help="重新訓練模型")
    args = parser.parse_args()

    print("=" * 60)
    print("  C-Long v3.1 — 注意→處置 升級預測策略")
    print(f"  Config: threshold={THRESHOLD}, consec=Y, hold={MAX_HOLD_DAYS}d, pos={MAX_POSITIONS}")
    print("=" * 60)

    auto_login()

    # Load data
    att, close, open_p, vol, dis_by_stock, cal, valid_stocks = load_all_data()

    # Model
    if args.train or not MODEL_PATH.exists():
        model, feature_cols = train_model(att, close, open_p, vol, dis_by_stock, cal)
    else:
        model, feature_cols = load_model()
        print(f"  ✅ Model loaded: {MODEL_PATH}")

    # Run
    if args.backtest:
        run_backtest(att, close, open_p, vol, dis_by_stock, cal, valid_stocks, model, feature_cols)
    else:
        run_production(att, close, open_p, vol, dis_by_stock, cal, valid_stocks, model, feature_cols)

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
