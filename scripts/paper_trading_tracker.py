"""Paper Trading 追蹤器 — 記錄模擬交易、計算損益、生成報告.

使用方式:
  python paper_trading_tracker.py --log      # 記錄今日訂單（每日執行後呼叫）
  python paper_trading_tracker.py --report   # 生成目前狀態報告
  python paper_trading_tracker.py --weekly   # 生成週報
"""
from __future__ import annotations

import argparse
import json
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd

TRACKER_DIR = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\paper_trading")
TRACKER_DIR.mkdir(parents=True, exist_ok=True)

V20_OUTPUT = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\v20_output")
CLONG_OUTPUT = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\c_long_output")

LOG_PATH = TRACKER_DIR / "paper_trades_log.csv"
EXEC_LOG_PATH = TRACKER_DIR / "execution_assumptions.csv"
STATE_PATH = TRACKER_DIR / "tracker_state.json"


def load_tracker_state():
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "v20": {"positions": [], "closed_trades": [], "total_pnl": 0, "start_date": None},
        "c_long": {"positions": [], "closed_trades": [], "total_pnl": 0, "start_date": None},
    }


def save_tracker_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def log_orders():
    """Read today's orders from both strategies and log them."""
    print("📝 記錄今日訂單...")

    state = load_tracker_state()
    today = datetime.now().strftime("%Y-%m-%d")
    logged = 0

    for strategy, output_dir in [("v20", V20_OUTPUT), ("c_long", CLONG_OUTPUT)]:
        orders_file = output_dir / "orders_today.csv"
        if not orders_file.exists():
            continue

        try:
            orders = pd.read_csv(orders_file, encoding="utf-8-sig")
        except Exception:
            continue

        if len(orders) == 0:
            continue

        for _, order in orders.iterrows():
            action = order.get("action", "")
            stock = str(order.get("stock_id", ""))

            if action == "BUY":
                entry = {
                    "date": today,
                    "strategy": strategy,
                    "action": "BUY",
                    "stock_id": stock,
                    "exit_date": order.get("exit_date", ""),
                    "reason": order.get("reason", order.get("condition", "")),
                }
                state[strategy]["positions"].append(entry)
                if state[strategy]["start_date"] is None:
                    state[strategy]["start_date"] = today
                logged += 1

            elif action == "SELL":
                # Find matching position
                remaining = []
                for pos in state[strategy]["positions"]:
                    if pos["stock_id"] == stock:
                        # Close this position
                        closed = {
                            "entry_date": pos["date"],
                            "exit_date": today,
                            "stock_id": stock,
                            "strategy": strategy,
                        }
                        state[strategy]["closed_trades"].append(closed)
                        logged += 1
                    else:
                        remaining.append(pos)
                state[strategy]["positions"] = remaining

    save_tracker_state(state)

    # Append to CSV log
    log_rows = []
    for strategy in ["v20", "c_long"]:
        for pos in state[strategy]["positions"]:
            log_rows.append({**pos, "status": "OPEN"})
        for trade in state[strategy]["closed_trades"]:
            if trade.get("exit_date") == today:
                log_rows.append({**trade, "action": "SELL", "status": "CLOSED"})

    if log_rows:
        df = pd.DataFrame(log_rows)
        if LOG_PATH.exists():
            df.to_csv(LOG_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    print(f"  記錄了 {logged} 筆操作")
    print(f"  V20 持倉: {len(state['v20']['positions'])} 檔")
    print(f"  C-Long 持倉: {len(state['c_long']['positions'])} 檔")


def show_report():
    """Show current paper trading status."""
    state = load_tracker_state()

    print("\n" + "=" * 60)
    print("  Paper Trading 狀態報告")
    print(f"  生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    for strategy, name in [("v20", "V20 (處置期間做多)"), ("c_long", "C-Long (注意→處置)")]:
        s = state[strategy]
        print(f"\n  {'='*40}")
        print(f"  {name}")
        print(f"  {'='*40}")
        print(f"  開始日期: {s.get('start_date', '未開始')}")
        print(f"  目前持倉: {len(s['positions'])} 檔")
        print(f"  已平倉: {len(s['closed_trades'])} 筆")

        if s["positions"]:
            print(f"\n  持倉明細:")
            for pos in s["positions"]:
                print(f"    {pos['stock_id']}: 進場 {pos['date']}, 出場 {pos.get('exit_date', '?')}")

        if s["closed_trades"]:
            print(f"\n  最近平倉:")
            for trade in s["closed_trades"][-5:]:
                print(f"    {trade['stock_id']}: {trade['entry_date']} → {trade['exit_date']}")

    # Summary
    total_open = len(state["v20"]["positions"]) + len(state["c_long"]["positions"])
    total_closed = len(state["v20"]["closed_trades"]) + len(state["c_long"]["closed_trades"])
    print(f"\n  {'='*40}")
    print(f"  總計: 持倉 {total_open} 檔, 已平倉 {total_closed} 筆")
    print(f"  {'='*40}")


def weekly_report():
    """Generate weekly summary."""
    state = load_tracker_state()
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("  Paper Trading 週報")
    print(f"  期間: {week_ago} ~ {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)

    for strategy, name in [("v20", "V20"), ("c_long", "C-Long")]:
        s = state[strategy]
        week_trades = [t for t in s["closed_trades"] if t.get("exit_date", "") >= week_ago]
        week_opens = [p for p in s["positions"] if p.get("date", "") >= week_ago]

        print(f"\n  {name}:")
        print(f"    本週開倉: {len(week_opens)} 筆")
        print(f"    本週平倉: {len(week_trades)} 筆")
        print(f"    目前持倉: {len(s['positions'])} 檔")

    # Save weekly report
    report_path = TRACKER_DIR / f"weekly_{datetime.now().strftime('%Y%m%d')}.txt"
    print(f"\n  📁 報告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading 追蹤器")
    parser.add_argument("--log", action="store_true", help="記錄今日訂單")
    parser.add_argument("--report", action="store_true", help="查看狀態")
    parser.add_argument("--weekly", action="store_true", help="生成週報")
    args = parser.parse_args()

    if args.log:
        log_orders()
        log_execution_assumptions()
    elif args.report:
        show_report()
    elif args.weekly:
        weekly_report()
    else:
        show_report()


if __name__ == "__main__":
    main()
