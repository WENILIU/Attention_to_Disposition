"""組合策略每日訊號 — V20 + 廣度做空 + C-Long.

資金配置:
  V20 (處置期間做多): 70%
  廣度做空 (期貨/反向ETF): 20%
  C-Long (注意→處置): 10% (多頭時啟用)

使用方式:
  python portfolio_daily.py              # 生成今日訂單
  python portfolio_daily.py --backtest   # 完整回測
  python portfolio_daily.py --status     # 查看持倉
"""
from __future__ import annotations

import argparse
import json
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import finlab
from finlab import data as _data

# ==================== 0. 設定 ====================
TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\auth_token.txt")
OUTPUT_DIR = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\portfolio_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = OUTPUT_DIR / "portfolio_state.json"

# 資金配置
ALLOC_V20 = 0.70
ALLOC_SHORT = 0.20
ALLOC_CLONG = 0.10

# V20 參數
V20_MAX_POSITIONS = 5
V20_MIN_TURNOVER = 20_000_000
V20_GAP_LOW = -0.08
V20_GAP_HIGH = 0.04
V20_COST = 0.001425 + 0.003 + 0.003
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")

# V20 品質過濾 (S3 發現)
V20_BREADTH_MIN = 0.40
V20_EXCLUDE_CONDITIONS = ["連續5個營業日及沖銷標準"]

# 做空參數
BREADTH_DECLINE_THRESHOLD = -0.15  # 5天廣度下降>15%
SHORT_HOLD_DAYS = 3
SHORT_COST = 0.001  # 期貨交易成本

# S4 處置進場壓力（增強做空訊號）
S4_DISPOSAL_PRESSURE_MIN = 5  # 當日處置進場數 >= 5 視為高壓力
S4_SHORT_BONUS = 0.10  # S1+S4 同時觸發時，額外增加 10% 做空配置

# C-Long 參數
CLONG_THRESHOLD = 0.55
CLONG_MAX_POSITIONS = 5
CLONG_MAX_HOLD = 12


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
        print("✅ FinLab VIP 登入成功")


# ==================== 1. 廣度做空訊號 ====================
def compute_breadth_signal(close, cal):
    """Compute market breadth and generate short signal."""
    # % stocks above MA20
    ma20 = close.rolling(20).mean()
    above_ma20 = (close > ma20).sum(axis=1) / close.notna().sum(axis=1)

    # Breadth decline: 5-day change in breadth
    breadth_decline = above_ma20 - above_ma20.shift(5)

    # Signal: breadth dropped > 15% in 5 days
    short_signal = breadth_decline < BREADTH_DECLINE_THRESHOLD

    # Hold for N days after signal fires
    short_position = short_signal.rolling(SHORT_HOLD_DAYS, min_periods=1).max()

    return short_position, breadth_decline, above_ma20


def compute_disposal_pressure(dis, cal):
    """Count disposal entries per day — high count = market froth."""
    dis_start_counts = dis.groupby("start_idx").size()
    pressure = pd.Series(0, index=range(len(cal)))
    for idx, count in dis_start_counts.items():
        if 0 <= idx < len(cal):
            pressure[idx] = count
    return pressure


# ==================== 2. V20 訊號 ====================
def generate_v20_signals(dis, close, open_p, cal, avg_turnover_5d, danger_zone, today_idx, above_ma20, mkt_close, mkt_ma20):
    """Generate V20 BUY signals for disposals starting today."""
    signals = []
    today = cal[today_idx]

    starting_today = dis[
        (dis["start"] == today) &
        (dis["end"] > today) &
        (dis["duration"] >= 3)
    ].copy()

    if len(starting_today) == 0:
        return signals

    if danger_zone.iloc[today_idx]:
        return signals

    breadth = float(above_ma20.iloc[today_idx]) if today_idx < len(above_ma20) else 0.5
    mkt_below_ma20 = False
    try:
        mkt_below_ma20 = mkt_close.iloc[today_idx] < mkt_ma20.iloc[today_idx]
    except (IndexError, KeyError):
        pass

    if breadth < V20_BREADTH_MIN and mkt_below_ma20:
        print(f"  ⚠️ V20 過濾: 廣度{breadth:.1%}<{V20_BREADTH_MIN:.0%} + MA20下 → 跳過進場")
        return signals

    for _, row in starting_today.iterrows():
        sym = row["stock_id"]
        if row["is_new_regime"]:
            entry_offset = 0
        else:
            entry_offset = 3

        entry_idx = int(row["start_idx"]) + entry_offset
        exit_idx = int(row["end_idx"]) - 1

        if entry_idx != today_idx or exit_idx <= entry_idx:
            continue

        cond = str(row.get("condition", ""))
        if any(ex in cond for ex in V20_EXCLUDE_CONDITIONS):
            continue

        try:
            avg_to = avg_turnover_5d.iloc[today_idx - 1][sym]
            if not (np.isnan(avg_to) or avg_to >= V20_MIN_TURNOVER):
                continue
            prev_close = close.iloc[today_idx][sym]
            if np.isnan(prev_close) or prev_close <= 0:
                continue
        except (IndexError, KeyError):
            continue

        signals.append({
            "strategy": "V20",
            "action": "BUY",
            "stock_id": sym,
            "exit_date": str(cal[exit_idx].date()),
            "condition": row.get("condition", ""),
            "allocation": ALLOC_V20 / V20_MAX_POSITIONS,
        })

    return signals


# ==================== 3. 實盤模式 ====================
def run_production():
    print("\n" + "=" * 60)
    print("  組合策略 — 今日訂單")
    print(f"  V20: {ALLOC_V20:.0%} | 做空: {ALLOC_SHORT:.0%} | C-Long: {ALLOC_CLONG:.0%}")
    print("=" * 60)

    # Load data
    print("\n⏳ 載入數據...")
    dis_raw = _data.get("disposal_information")
    close = _data.get("price:收盤價")
    open_p = _data.get("price:開盤價")
    vol = _data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)
    today_idx = n_cal - 1
    today = cal[today_idx]

    # Build disposal
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis["condition"] = dis.get("處置條件", "").astype(str)
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$") &
        dis["stock_id"].isin(valid_stocks) &
        ~dis["stock_id"].str.startswith(("00", "91"))
    ].copy()
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")
    dis["end_idx"] = cal.searchsorted(dis["end"], side="right") - 1
    dis["duration"] = dis["end_idx"] - dis["start_idx"] + 1
    dis["is_new_regime"] = dis["announce"] >= NEW_REGIME_DATE

    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()
    daily_ret = close.pct_change()
    panic_count = (daily_ret < -0.06).sum(axis=1)
    is_panic = panic_count > 400
    danger_zone = is_panic.rolling(9, min_periods=1).max() > 0

    # Compute breadth signal
    short_position, breadth_decline, above_ma20 = compute_breadth_signal(close, cal)

    # S4: disposal pressure signal
    disposal_pressure = compute_disposal_pressure(dis, cal)
    mkt_ret_pf = close.pct_change().mean(axis=1)
    mkt_close_pf = (1 + mkt_ret_pf).cumprod()
    mkt_ma20_pf = mkt_close_pf.rolling(20).mean()
    mkt_weak = bool(mkt_close_pf.iloc[today_idx] < mkt_ma20_pf.iloc[today_idx]) if today_idx < len(mkt_ma20_pf) else False
    s4_active = bool(disposal_pressure.iloc[today_idx] >= S4_DISPOSAL_PRESSURE_MIN and mkt_weak)

    # Combined short: S1 primary, S4 confirms/enhances
    s1_active = bool(short_position.iloc[today_idx])
    short_allocation = ALLOC_SHORT
    if s1_active and s4_active:
        short_allocation = ALLOC_SHORT + S4_SHORT_BONUS
    elif s4_active and not s1_active:
        short_allocation = ALLOC_SHORT * 0.5

    print(f"\n  日期: {today.date()}")
    print(f"  市場廣度: {above_ma20.iloc[today_idx]:.1%} 股票在MA20上")
    print(f"  廣度5日變化: {breadth_decline.iloc[today_idx]:+.1%}")
    print(f"  處置進場壓力: {disposal_pressure.iloc[today_idx]} 檔 (門檻{S4_DISPOSAL_PRESSURE_MIN})")
    print(f"  S1 廣度做空: {'🔴 啟動' if s1_active else '⚪ 無'}")
    print(f"  S4 處置壓力: {'🔴 啟動' if s4_active else '⚪ 無'}")
    if s1_active or s4_active:
        print(f"  做空配置: {short_allocation:.0%}")

    # Market MA20 for V20 filter
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_close = (1 + mkt_ret).cumprod()
    mkt_ma20 = mkt_close.rolling(20).mean()

    # Generate V20 signals
    v20_signals = generate_v20_signals(
        dis, close, open_p, cal, avg_turnover_5d, danger_zone, today_idx,
        above_ma20, mkt_close, mkt_ma20
    )

    # Load state
    state = load_state()

    # Build orders
    orders = []

    # Short signal (S1 + S4 combined)
    short_active_now = s1_active or s4_active
    if short_active_now and not state.get("short_active", False):
        reasons = []
        if s1_active:
            reasons.append(f"廣度降{breadth_decline.iloc[today_idx]:.1%}")
        if s4_active:
            reasons.append(f"處置壓力{disposal_pressure.iloc[today_idx]}檔")
        orders.append({
            "strategy": "SHORT",
            "action": "SELL_SHORT",
            "instrument": "台指期貨 / 00649R",
            "reason": " + ".join(reasons),
            "hold_days": SHORT_HOLD_DAYS,
            "allocation": short_allocation,
        })
    elif not short_active_now and state.get("short_active", False):
        orders.append({
            "strategy": "SHORT",
            "action": "COVER",
            "instrument": "台指期貨 / 00649R",
            "reason": "做空到期平倉",
            "allocation": ALLOC_SHORT,
        })

    # V20 signals
    for sig in v20_signals[:V20_MAX_POSITIONS]:
        orders.append(sig)

    # Output
    if orders:
        print(f"\n  📋 今日訂單 ({len(orders)} 筆):")
        for o in orders:
            strat = o.get("strategy", "")
            action = o.get("action", "")
            if action == "SELL_SHORT":
                print(f"    📉 [{strat}] {action} {o['instrument']} — {o['reason']}")
            elif action == "COVER":
                print(f"    📈 [{strat}] {action} {o['instrument']} — {o['reason']}")
            elif action == "BUY":
                print(f"    📥 [{strat}] {action} {o['stock_id']} — {o.get('condition','')[:20]}")
    else:
        print(f"\n  ✅ 今日無訂單")

    # Update state
    state["short_active"] = bool(s1_active or s4_active)
    state["last_run"] = str(today.date())
    state["breadth"] = float(above_ma20.iloc[today_idx])
    save_state(state)

    if orders:
        pd.DataFrame(orders).to_csv(OUTPUT_DIR / "orders_today.csv", index=False, encoding="utf-8-sig")


# ==================== 4. 回測模式 ====================
def run_backtest():
    print("\n" + "=" * 60)
    print("  組合策略完整回測")
    print("=" * 60)

    # Load data
    print("\n⏳ 載入數據...")
    dis_raw = _data.get("disposal_information")
    close = _data.get("price:收盤價")
    open_p = _data.get("price:開盤價")
    vol = _data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)
    valid_stocks = set(close.columns)

    # Market returns
    mkt_ret = close.pct_change().mean(axis=1)
    mkt_close_bt = (1 + mkt_ret).cumprod()
    mkt_ma20_bt = mkt_close_bt.rolling(20).mean()

    # Build disposal DataFrame (needed for S4 and V20 filter)
    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$") &
        dis["stock_id"].isin(valid_stocks) &
        ~dis["stock_id"].str.startswith(("00", "91"))
    ].copy()
    dis["start_idx"] = cal.searchsorted(dis["start"], side="left")

    # Breadth signal (S1)
    short_position, breadth_decline, above_ma20 = compute_breadth_signal(close, cal)

    # S4: disposal pressure signal
    disposal_pressure = compute_disposal_pressure(dis, cal)
    mkt_weak_series = pd.Series(mkt_close_bt.values < mkt_ma20_bt.values, index=cal)

    s4_series = pd.Series(False, index=cal)
    for i in range(n_cal):
        if disposal_pressure.iloc[i] >= S4_DISPOSAL_PRESSURE_MIN and mkt_weak_series.iloc[i]:
            s4_series.iloc[i] = True

    # Combined short: S1 primary + S4 enhancement
    short_daily = pd.Series(0.0, index=cal)
    for i in range(n_cal):
        s1 = bool(short_position.iloc[i])
        s4 = bool(s4_series.iloc[i])
        if s1 and s4:
            alloc = ALLOC_SHORT + S4_SHORT_BONUS
        elif s1:
            alloc = ALLOC_SHORT
        elif s4:
            alloc = ALLOC_SHORT * 0.5
        else:
            continue
        short_daily.iloc[i] = -mkt_ret.iloc[i] * alloc - SHORT_COST * alloc

    # V20 returns (load from P23 results, apply S3 breadth filter)
    v20_path = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\p23_v20_longterm\p23_v20_longterm_trades.csv")
    v20_trades = pd.read_csv(v20_path, parse_dates=["entry_date", "exit_date"])

    # S3 filter: skip entries when breadth < 40% AND market below MA20
    v20_trades["entry_idx"] = cal.searchsorted(v20_trades["entry_date"], side="left")
    mask_bad = pd.Series(False, index=v20_trades.index)
    for idx, t in v20_trades.iterrows():
        ei = int(t["entry_idx"])
        if ei < len(above_ma20) and ei < len(mkt_close_bt):
            b = float(above_ma20.iloc[ei]) if not np.isnan(above_ma20.iloc[ei]) else 0.5
            below = bool(mkt_close_bt.iloc[ei] < mkt_ma20_bt.iloc[ei]) if not np.isnan(mkt_ma20_bt.iloc[ei]) else False
            if b < V20_BREADTH_MIN and below:
                mask_bad.iloc[idx] = True
    v20_trades = v20_trades[~mask_bad].copy()
    print(f"  V20 after breadth filter: {len(v20_trades)} trades (removed {mask_bad.sum()})")

    v20_daily = pd.Series(0.0, index=cal)
    for _, t in v20_trades.iterrows():
        ei = cal.searchsorted(t["entry_date"], side="left")
        xi = cal.searchsorted(t["exit_date"], side="left")
        hold = max(xi - ei + 1, 1)
        dr = t["ret"] / hold * (ALLOC_V20 / V20_MAX_POSITIONS)
        for d in range(ei, min(xi + 1, n_cal)):
            v20_daily.iloc[d] += dr

    # Combined
    combined = v20_daily + short_daily

    # Metrics
    for label, ret_series in [
        ("V20 單獨", v20_daily),
        ("做空單獨", short_daily),
        ("V20 + Short(S1+S4)", combined),
    ]:
        cum = (1 + ret_series).cumprod()
        sharpe = np.mean(ret_series) / np.std(ret_series) * np.sqrt(252) if np.std(ret_series) > 0 else 0
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()

        yearly = {}
        for yr in range(2010, 2027):
            yr_mask = cal.year == yr
            if yr_mask.sum() > 50:
                yearly[yr] = np.prod(1 + ret_series.values[yr_mask]) - 1
        n_pos = sum(1 for v in yearly.values() if v > 0)

        print(f"\n  {label}:")
        print(f"    Sharpe: {sharpe:.2f}, MDD: {mdd:.1%}")
        print(f"    Years positive: {n_pos}/{len(yearly)}")
        for yr in [2011, 2015, 2018, 2022]:
            if yr in yearly:
                print(f"      {yr}: {yearly[yr]:+.1%}")

    # Full yearly table
    print(f"\n  組合逐年:")
    print(f"  {'Year':>6} {'V20':>8} {'Short':>8} {'Combined':>8} {'判定':>6}")
    for yr in range(2010, 2027):
        yr_mask = cal.year == yr
        if yr_mask.sum() < 50:
            continue
        v20_r = np.prod(1 + v20_daily.values[yr_mask]) - 1
        short_r = np.prod(1 + short_daily.values[yr_mask]) - 1
        comb_r = np.prod(1 + combined.values[yr_mask]) - 1
        sign = "PASS" if comb_r > 0 else "FAIL"
        print(f"  {yr:>6} {v20_r:>8.1%} {short_r:>8.1%} {comb_r:>8.1%} {sign:>6}")


# ==================== 5. 狀態 ====================
def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"short_active": False, "positions": [], "last_run": None}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def show_status():
    state = load_state()
    print(f"\n  最後執行: {state.get('last_run', '未執行')}")
    print(f"  做空狀態: {'啟動' if state.get('short_active') else '關閉'}")
    print(f"  市場廣度: {state.get('breadth', 'N/A')}")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="組合策略每日訊號")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  組合策略 — V20 + 廣度做空 + C-Long")
    print(f"  配置: V20 {ALLOC_V20:.0%} | Short {ALLOC_SHORT:.0%} | C-Long {ALLOC_CLONG:.0%}")
    print(f"  做空訊號: 廣度5日降>{abs(BREADTH_DECLINE_THRESHOLD):.0%} → 做空{SHORT_HOLD_DAYS}天")
    print("=" * 60)

    if args.status:
        show_status()
        return

    auto_login()

    if args.backtest:
        run_backtest()
    else:
        run_production()

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
