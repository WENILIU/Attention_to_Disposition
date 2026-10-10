"""V20 每日訊號腳本 — 處置期間做多策略.

策略邏輯（純規則，無 ML 模型）:
  新制 (處置5天): Day 1 進場, Day 4 出場
  舊制 (處置10天): Day 4 進場, Day 9 出場
  濾網: 流動性>20M, Gap -8%~+4%
  熔斷: >400家跌>6% → 清倉9天
  部位: 5檔, 每檔20%

使用方式:
  python v20_daily_signal.py              # 生成今日訂單
  python v20_daily_signal.py --backtest   # 完整回測
  python v20_daily_signal.py --status     # 查看目前持倉

輸出:
  - v20_output/orders_today.csv: 今日買賣指令
  - v20_output/portfolio_state.json: 目前持倉狀態
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
OUTPUT_DIR = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\v20_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = OUTPUT_DIR / "portfolio_state.json"

# 策略參數
MIN_TURNOVER = 20_000_000
GAP_LOW = -0.08
GAP_HIGH = 0.04
MAX_POSITIONS = 5
POSITION_SIZE = 0.20
PANIC_THRESHOLD = 400
PANIC_COOLDOWN = 9
COST_RATE = 0.001425 + 0.003 + 0.003
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
        print("✅ FinLab VIP 登入成功")


# ==================== 1. 資料載入 ====================
def load_data():
    print("⏳ 載入數據...")
    dis_raw = _data.get("disposal_information")
    close = _data.get("price:收盤價")
    open_p = _data.get("price:開盤價")
    vol = _data.get("price:成交股數")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    valid_stocks = set(close.columns)

    # Build disposal events
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

    # Turnover for liquidity filter
    turnover = close * vol
    avg_turnover_5d = turnover.rolling(5).mean()

    # Circuit breaker
    daily_ret = close.pct_change()
    panic_count = (daily_ret < -0.06).sum(axis=1)
    is_panic = panic_count > PANIC_THRESHOLD
    danger_zone = is_panic.rolling(PANIC_COOLDOWN, min_periods=1).max() > 0

    print(f"  處置事件: {len(dis):,}, 價格數據: {cal[0].date()} ~ {cal[-1].date()}")
    return dis, close, open_p, vol, cal, valid_stocks, avg_turnover_5d, danger_zone


# ==================== 2. 訊號生成 ====================
def generate_entry_signals(dis, close, open_p, cal, avg_turnover_5d, danger_zone, today_idx):
    """Generate BUY signals for disposals starting on today."""
    signals = []

    # Find disposals that START today
    today = cal[today_idx]
    starting_today = dis[
        (dis["start"] == today) &
        (dis["end"] > today) &
        (dis["duration"] >= 3)
    ].copy()

    if len(starting_today) == 0:
        return signals

    # Check circuit breaker
    if danger_zone.iloc[today_idx]:
        print(f"  🚨 熔斷中，跳過所有進場")
        return signals

    for _, row in starting_today.iterrows():
        sym = row["stock_id"]
        dur = int(row["duration"])

        # Determine entry day
        if row["is_new_regime"]:
            entry_offset = 0  # Day 1 = start day
        else:
            entry_offset = 3  # Day 4 = start + 3

        entry_idx = int(row["start_idx"]) + entry_offset
        exit_idx = int(row["end_idx"]) - 1

        if entry_idx != today_idx:
            continue  # Not entering today
        if exit_idx <= entry_idx:
            continue  # Invalid: exit before entry

        # Liquidity filter
        try:
            avg_to = avg_turnover_5d.iloc[today_idx - 1][sym]
            if not (np.isnan(avg_to) or avg_to >= MIN_TURNOVER):
                continue
        except (IndexError, KeyError):
            continue

        # Gap filter (check tomorrow's open vs today's close)
        # In production, we can't check gap until tomorrow opens
        # So we flag it as "pending" and verify at market open
        try:
            prev_close = close.iloc[today_idx][sym]
            if np.isnan(prev_close) or prev_close <= 0:
                continue
        except (IndexError, KeyError):
            continue

        signals.append({
            "stock_id": sym,
            "action": "BUY",
            "entry_date": str(today.date()),
            "exit_date": str(cal[exit_idx].date()),
            "exit_idx": exit_idx,
            "duration": dur,
            "condition": row["condition"],
            "is_new_regime": bool(row["is_new_regime"]),
            "note": "開盤時確認 Gap 在 -8%~+4% 範圍內再買入",
        })

    return signals


def generate_exit_signals(state, cal, today_idx):
    """Generate SELL signals for positions reaching exit date."""
    exits = []
    today = cal[today_idx]

    for pos in state["positions"]:
        exit_date = pd.Timestamp(pos["exit_date"])
        if exit_date <= today:
            exits.append({
                "stock_id": pos["stock_id"],
                "action": "SELL",
                "reason": f"持有到期 (處置期間結束)",
                "entry_date": pos["entry_date"],
                "exit_date": pos["exit_date"],
            })

    return exits


# ==================== 3. 實盤模式 ====================
def run_production(dis, close, open_p, cal, avg_turnover_5d, danger_zone):
    """Generate today's orders."""
    print("\n" + "=" * 60)
    print("  V20 實盤模式 — 今日訂單")
    print("=" * 60)

    today_idx = len(cal) - 1
    today = cal[today_idx]
    print(f"\n  基準日期: {today.date()}")

    # Load portfolio state
    state = load_state()

    # Generate exit signals
    exit_signals = generate_exit_signals(state, cal, today_idx)

    # Generate entry signals
    entry_signals = generate_entry_signals(
        dis, close, open_p, cal, avg_turnover_5d, danger_zone, today_idx
    )

    # Apply position limit
    remaining_positions = [p for p in state["positions"]
                          if pd.Timestamp(p["exit_date"]) > today]
    available_slots = MAX_POSITIONS - len(remaining_positions)

    # Sort entries by condition priority (連5日 > 其他)
    priority_conditions = ["連5日", "借券比"]
    entry_signals.sort(
        key=lambda s: (0 if any(c in s.get("condition", "") for c in priority_conditions) else 1)
    )
    entry_signals = entry_signals[:available_slots]

    # Output
    orders = exit_signals + entry_signals
    if orders:
        print(f"\n  📋 今日訂單 ({len(orders)} 筆):")
        for o in orders:
            if o["action"] == "SELL":
                print(f"    📤 SELL {o['stock_id']} — {o['reason']}")
            else:
                cond = o.get("condition", "")[:20]
                print(f"    📥 BUY  {o['stock_id']} — 處置條件: {cond}, 出場: {o['exit_date']}")
    else:
        print(f"\n  ✅ 今日無訂單")

    # Update state
    new_positions = [
        {"stock_id": s["stock_id"], "entry_date": s["entry_date"],
         "exit_date": s["exit_date"], "condition": s.get("condition", "")}
        for s in entry_signals
    ]
    state["positions"] = remaining_positions + new_positions
    state["last_run"] = str(today.date())
    save_state(state)

    # Save orders CSV
    if orders:
        pd.DataFrame(orders).to_csv(OUTPUT_DIR / "orders_today.csv", index=False, encoding="utf-8-sig")

    print(f"\n  📁 訂單: {OUTPUT_DIR / 'orders_today.csv'}")
    print(f"  📁 狀態: {STATE_PATH}")
    print(f"  📊 目前持倉: {len(state['positions'])} 檔")


# ==================== 4. 回測模式 ====================
def run_backtest(dis, close, open_p, cal, avg_turnover_5d, danger_zone):
    """Full backtest."""
    print("\n" + "=" * 60)
    print("  V20 完整回測")
    print("=" * 60)

    n_cal = len(cal)
    valid = dis[
        (dis["start_idx"] >= 0) &
        (dis["end_idx"] >= 0) &
        (dis["end_idx"] < n_cal) &
        (dis["duration"] >= 3)
    ].copy()

    trades = []
    for _, row in valid.iterrows():
        sym = row["stock_id"]
        si = int(row["start_idx"])
        ei = int(row["end_idx"])

        if row["is_new_regime"]:
            entry_idx = si
        else:
            entry_idx = si + 3
        exit_idx = ei - 1

        if entry_idx < 0 or exit_idx >= n_cal or entry_idx >= exit_idx:
            continue
        if danger_zone.iloc[entry_idx]:
            continue

        try:
            entry_price = open_p.iloc[entry_idx][sym]
            prev_close = close.iloc[entry_idx - 1][sym]
            avg_to = avg_turnover_5d.iloc[entry_idx - 1][sym]
        except (IndexError, KeyError):
            continue
        if any(np.isnan(x) or x <= 0 for x in [entry_price, prev_close]):
            continue
        if not (np.isnan(avg_to) or avg_to >= MIN_TURNOVER):
            continue
        gap = entry_price / prev_close - 1
        if not (GAP_LOW < gap < GAP_HIGH):
            continue
        try:
            exit_price = close.iloc[exit_idx][sym]
            if np.isnan(exit_price) or exit_price <= 0:
                continue
        except (IndexError, KeyError):
            continue

        ret = exit_price / entry_price - 1 - COST_RATE
        trades.append({
            "stock_id": sym, "entry_date": cal[entry_idx], "exit_date": cal[exit_idx],
            "entry_idx": entry_idx, "exit_idx": exit_idx,
            "year": cal[entry_idx].year, "ret": ret,
            "hold_days": exit_idx - entry_idx + 1,
        })

    all_trades = pd.DataFrame(trades)
    all_trades = all_trades.sort_values(["entry_date", "ret"], ascending=[True, False])

    # Portfolio sim
    active = []
    executed = []
    for _, trade in all_trades.iterrows():
        ei = int(trade["entry_idx"])
        xi = int(trade["exit_idx"])
        active = [(e, s) for e, s in active if e > ei]
        if len(active) >= MAX_POSITIONS:
            continue
        if any(s == trade["stock_id"] for _, s in active):
            continue
        active.append((xi, trade["stock_id"]))
        executed.append(trade)

    ex = pd.DataFrame(executed)

    # Metrics
    daily_returns = pd.Series(0.0, index=cal)
    cap = 1.0 / MAX_POSITIONS
    for _, t in ex.iterrows():
        ei = int(t["entry_idx"])
        xi = int(t["exit_idx"])
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

    print(f"\n  CAGR: {cagr:.1%}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  MDD: {mdd:.1%}")
    print(f"  Avg return: {ex['ret'].mean():.2%}")
    print(f"  Win rate: {(ex['ret'] > 0).mean():.1%}")
    print(f"  Total trades: {len(ex):,}")

    # Yearly
    print(f"\n  逐年:")
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
        print(f"    {yr}: n={len(yr_data)}, avg={avg:.2%}, win={win:.1%} {sign}")
    print(f"\n  逐年通過: {yearly_pass}/{yearly_total}")

    ex.to_csv(OUTPUT_DIR / "v20_backtest_trades.csv", index=False, encoding="utf-8-sig")


# ==================== 5. 狀態管理 ====================
def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"positions": [], "last_run": None}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def show_status():
    state = load_state()
    print(f"\n  最後執行: {state.get('last_run', '未執行')}")
    print(f"  目前持倉: {len(state['positions'])} 檔")
    for pos in state["positions"]:
        print(f"    {pos['stock_id']}: 進場 {pos['entry_date']}, 出場 {pos['exit_date']}")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="V20 每日訊號")
    parser.add_argument("--backtest", action="store_true", help="完整回測")
    parser.add_argument("--status", action="store_true", help="查看持倉")
    args = parser.parse_args()

    print("=" * 60)
    print("  V20 — 處置期間做多 (每日訊號)")
    print(f"  部位: {MAX_POSITIONS}檔, 每檔{POSITION_SIZE:.0%}")
    print(f"  濾網: 流動性>{MIN_TURNOVER//10000}萬, Gap {GAP_LOW:.0%}~{GAP_HIGH:.0%}")
    print(f"  熔斷: >{PANIC_THRESHOLD}家跌>6% → 暫停{PANIC_COOLDOWN}天")
    print("=" * 60)

    if args.status:
        show_status()
        return

    auto_login()
    dis, close, open_p, vol, cal, valid_stocks, avg_turnover_5d, danger_zone = load_data()

    if args.backtest:
        run_backtest(dis, close, open_p, cal, avg_turnover_5d, danger_zone)
    else:
        run_production(dis, close, open_p, cal, avg_turnover_5d, danger_zone)

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
