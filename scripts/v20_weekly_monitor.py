"""V20 Weekly Monitor - 每週自動執行，追蹤樣本累積與策略健康度.

Usage: python v20_weekly_monitor.py
Output: prints status report, saves to monitoring log.

Alerts:
  - n >= 200: 完整複檢提醒
  - n >= 500: 加入條件/市值加權提醒
  - 30d win rate < 40%: 策略暫停警報
  - 30d avg return < 0%: 策略暫停警報
  - Max single loss > -30%: 尾部風險警報
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import finlab
from finlab import data
from finlab.backtest import sim

# ==================== Config ====================
TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\auth_token.txt")
MONITOR_DIR = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs\v20_monitor")
MONITOR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = MONITOR_DIR / "monitor_log.csv"
STATE_FILE = MONITOR_DIR / "monitor_state.json"

# V20 Parameters
NEW_REGIME_DATE = "2026-08-10"
ENTRY_OFFSET = 0
EXIT_OFFSET = -1
MIN_TURNOVER = 20_000_000
GAP_LOW, GAP_HIGH = -0.08, 0.04
MAX_BIAS = 999  # Bias filter removed (P13: no effect)
STOP_LOSS = 0.99  # Effectively no stop loss (P13 validated)
POSITION_LIMIT = 0.20
FEE_RATIO = 1.425 / 1000 / 3

# Alert thresholds
ALERT_N_200 = 200
ALERT_N_500 = 500
ALERT_WIN_RATE_LOW = 0.40
ALERT_AVG_RETURN_NEG = 0.0
ALERT_MAX_LOSS = -0.30


def login():
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            finlab.login(f.read().strip())


def run_backtest():
    """Run V20 backtest and return trades + stats."""
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    dis_raw = data.get("disposal_information")

    dis = pd.DataFrame(dis_raw).copy()
    dis["stock_id"] = dis["symbol"].astype(str).str.zfill(4)
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()

    valid_stocks = set(close.columns)
    dis = dis[
        dis["stock_id"].str.match(r"^\d{4}$") &
        dis["stock_id"].isin(valid_stocks) &
        ~dis["stock_id"].str.startswith(("00", "91")) &
        (dis["announce"] >= NEW_REGIME_DATE)
    ].copy()

    trading_days = close.index
    n_days = len(trading_days)

    all_stocks = list(set(dis["stock_id"]) & valid_stocks)
    filtered_close = close[all_stocks].astype(np.float32)
    filtered_vol = vol[all_stocks].astype(np.float32)
    turnover = filtered_close * filtered_vol
    avg_turnover_5d = turnover.rolling(5).mean()
    ma20 = filtered_close.rolling(20).mean()

    position = pd.DataFrame(False, index=close.index, columns=close.columns)

    for row in dis.itertuples():
        sym = row.stock_id
        if sym not in all_stocks:
            continue
        start_idx = trading_days.searchsorted(row.start)
        end_idx = trading_days.searchsorted(row.end, side="right") - 1
        if start_idx >= n_days or end_idx >= n_days or end_idx < start_idx:
            continue

        signal_idx = start_idx + ENTRY_OFFSET
        exec_idx = signal_idx + 1
        exit_idx = end_idx + EXIT_OFFSET
        if exec_idx >= n_days or exit_idx < exec_idx:
            continue

        signal_day = trading_days[signal_idx]
        exec_day = trading_days[exec_idx]

        try:
            prev_c = filtered_close.loc[signal_day, sym]
            e_open = open_p.loc[exec_day, sym]
            current_ma20 = ma20.loc[signal_day, sym]
            avg_to = avg_turnover_5d.loc[signal_day, sym]
        except KeyError:
            continue

        if any(pd.isna(x) or x <= 0 for x in [prev_c, e_open]):
            continue

        gap_pct = (e_open - prev_c) / prev_c
        if not (GAP_LOW < gap_pct < GAP_HIGH):
            continue
        if pd.isna(avg_to) or avg_to < MIN_TURNOVER:
            continue
        # Bias filter removed (P13: no effect)

        position.loc[exec_day:trading_days[exit_idx], sym] = True

    # Circuit breaker
    panic_count = (close.pct_change() < -0.06).sum(axis=1)
    is_panic = panic_count > 400
    danger_zone = is_panic.rolling(9, min_periods=1).max() > 0
    position.loc[danger_zone, :] = False

    # Run sim
    report = sim(position, trade_at_price="open", fee_ratio=FEE_RATIO,
                 position_limit=POSITION_LIMIT, stop_loss=STOP_LOSS,
                 upload=False, name="V20_monitor")

    stats = report.get_stats()
    trades = report.get_trades()
    return stats, trades, len(dis)


def compute_health(trades):
    """Compute strategy health metrics."""
    if len(trades) == 0:
        return {}

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])

    # Overall
    overall = {
        "n_trades": len(trades),
        "win_rate": (trades["return"] > 0).mean(),
        "avg_return": trades["return"].mean(),
        "median_return": trades["return"].median(),
        "max_loss": trades["return"].min(),
        "max_gain": trades["return"].max(),
        "total_return": trades["return"].sum(),
    }

    # 30-day rolling
    cutoff_30d = trades["entry_date"].max() - timedelta(days=30)
    recent = trades[trades["entry_date"] >= cutoff_30d]
    if len(recent) >= 5:
        overall["n_30d"] = len(recent)
        overall["win_rate_30d"] = (recent["return"] > 0).mean()
        overall["avg_return_30d"] = recent["return"].mean()
    else:
        overall["n_30d"] = len(recent)
        overall["win_rate_30d"] = np.nan
        overall["avg_return_30d"] = np.nan

    # Weekly returns
    trades["week"] = trades["entry_date"].dt.isocalendar().week.astype(str)
    weekly = trades.groupby("week")["return"].agg(["mean", "count",
                                                    lambda x: (x > 0).mean()])
    weekly.columns = ["mean_ret", "n", "win"]
    overall["recent_weeks"] = weekly.tail(4).to_dict("index")

    return overall


def check_alerts(health, stats):
    """Check alert conditions and return list of alerts."""
    alerts = []

    n = health.get("n_trades", 0)

    if n >= ALERT_N_500:
        alerts.append(f"[GOAL] n={n} >= 500: 加入條件加權+市值偏好，完整重跑 P7")
    elif n >= ALERT_N_200:
        alerts.append(f"[NEXT] n={n} >= 200: 完整複檢 V20 規則，驗證濾網有效性")
    else:
        alerts.append(f"[INFO] n={n}/{ALERT_N_200}: 樣本累積中，繼續監測")

    wr_30d = health.get("win_rate_30d", np.nan)
    if not np.isnan(wr_30d) and wr_30d < ALERT_WIN_RATE_LOW:
        alerts.append(f"[ALERT] 30日勝率 {wr_30d:.1%} < {ALERT_WIN_RATE_LOW:.0%}: 考慮暫停策略")

    ar_30d = health.get("avg_return_30d", np.nan)
    if not np.isnan(ar_30d) and ar_30d < ALERT_AVG_RETURN_NEG:
        alerts.append(f"[ALERT] 30日平均報酬 {ar_30d:.2%} < 0: 考慮暫停策略")

    max_loss = health.get("max_loss", 0)
    if max_loss < ALERT_MAX_LOSS:
        alerts.append(f"[WARN] 最大單筆虧損 {max_loss:.1%} > {ALERT_MAX_LOSS:.0%}: 檢查止損是否生效")

    mdd = stats.get("max_drawdown", 0)
    if isinstance(mdd, (int, float)) and mdd < -0.20:
        alerts.append(f"[ALERT] 組合回撤 {mdd:.1%} > 20%: 策略可能失效")

    return alerts


def save_state(health, stats, alerts):
    """Save monitoring state for next week comparison."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "n_trades": health.get("n_trades", 0),
        "win_rate": health.get("win_rate", 0),
        "avg_return": health.get("avg_return", 0),
        "max_loss": health.get("max_loss", 0),
        "cagr": stats.get("cagr", 0),
        "mdd": stats.get("max_drawdown", 0),
        "alerts": alerts,
    }

    # Append to log
    log_entry = pd.DataFrame([state])
    if LOG_FILE.exists():
        log_entry.to_csv(LOG_FILE, mode="a", header=False,
                         index=False, encoding="utf-8-sig")
    else:
        log_entry.to_csv(LOG_FILE, header=True, index=False,
                         encoding="utf-8-sig")

    # Save latest state
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def print_report(health, stats, alerts, n_events):
    """Print formatted status report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("\n" + "=" * 60)
    print(f"  監獄兔 V20 每週監測報告")
    print(f"  {now}")
    print("=" * 60)

    print(f"\n--- 樣本累積 ---")
    print(f"  新制處置事件總數: {n_events}")
    print(f"  通過濾網交易數: {health.get('n_trades', 0)}")
    print(f"  目標: {ALERT_N_200} 筆後完整複檢")
    progress = min(health.get('n_trades', 0) / ALERT_N_200 * 100, 100)
    bar = "#" * int(progress // 5) + "-" * (20 - int(progress // 5))
    print(f"  進度: [{bar}] {progress:.0f}%")

    print(f"\n--- 策略健康度 ---")
    print(f"  勝率: {health.get('win_rate', 0):.1%}")
    print(f"  平均報酬: {health.get('avg_return', 0):.2%}")
    print(f"  中位數報酬: {health.get('median_return', 0):.2%}")
    print(f"  最大虧損: {health.get('max_loss', 0):.1%}")
    print(f"  最大獲利: {health.get('max_gain', 0):.1%}")

    if not np.isnan(health.get("win_rate_30d", np.nan)):
        print(f"\n--- 30日滾動 ---")
        print(f"  30日勝率: {health['win_rate_30d']:.1%}")
        print(f"  30日平均報酬: {health['avg_return_30d']:.2%}")
        print(f"  30日筆數: {health.get('n_30d', 0)}")

    print(f"\n--- 回測統計 ---")
    if isinstance(stats.get("cagr"), (int, float)):
        print(f"  CAGR: {stats['cagr'] * 100:.1f}%")
    if isinstance(stats.get("max_drawdown"), (int, float)):
        print(f"  最大回撤: {stats['max_drawdown'] * 100:.1f}%")
    if isinstance(stats.get("win_ratio"), (int, float)):
        print(f"  勝率(sim): {stats['win_ratio'] * 100:.1f}%")

    print(f"\n--- 警報 ---")
    for alert in alerts:
        print(f"  {alert}")

    print("\n" + "=" * 60)


def main():
    print("V20 Weekly Monitor starting...")
    login()

    # Run backtest
    stats, trades, n_events = run_backtest()

    # Compute health
    health = compute_health(trades)

    # Check alerts
    alerts = check_alerts(health, stats)

    # Print report
    print_report(health, stats, alerts, n_events)

    # Save state
    save_state(health, stats, alerts)

    # Save trades
    if len(trades) > 0:
        trades.to_csv(MONITOR_DIR / "latest_trades.csv",
                      index=False, encoding="utf-8-sig")

    print(f"\nState saved to: {MONITOR_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
