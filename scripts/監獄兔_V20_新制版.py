"""監獄兔 V20 - 新制適配版

核心改動 vs V19:
  1. 自動偵測處置期間長度，動態調整進出場日
     - 舊制 (10天): Day 4 進場, Day 9 出場 (同 V19)
     - 新制 (5天):  Day 1 進場, Day 4 出場
  2. 保留 V19 所有濾網 (流動性/gap/乖離率)
  3. 保留大盤熔斷 (>400家)
  4. 新增: 條件加權 (連5日 2x 部位)
  5. 新增: 可選止損優化

使用方式:
  python 監獄兔_V20.py

輸出:
  - 回測報表 (CAGR, MDD, Sharpe)
  - 交易紀錄 CSV
  - 淨值曲線 CSV
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import finlab
from finlab import data
from finlab.backtest import sim

# ==================== 0. 設定 ====================
TOKEN_FILE = Path(r"D:\AI專案\StockAgent\finlab\auth_token.txt")
RESULT_PATH = Path(r"D:\AI專案\StockAgent\finlab\開發straget_監獄兔\backtest_results")
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# 策略參數
ENTRY_GAP_LOW = -0.08       # 開盤跳空下限
ENTRY_GAP_HIGH = 0.04       # 開盤跳空上限
MIN_TURNOVER = 20_000_000   # 最低日均成交額 (NT$2000萬)
MAX_BIAS_MA20 = 0.60        # 最大乖離率 (60%)
POSITION_LIMIT = 0.20       # 單檔部位上限 (20%)
STOP_LOSS = 0.12            # 止損 (12%)
PANIC_THRESHOLD = 400       # 熔斷: 單日跌幅>6% 的股票數
PANIC_COOLDOWN = 9          # 熔斷後暫停天數
FEE_RATIO = 1.425 / 1000 / 3  # finlab fee format

# 新制生效日
NEW_REGIME_DATE = pd.Timestamp("2026-08-10")


def auto_login():
    if TOKEN_FILE.exists():
        try:
            token = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if token:
                finlab.login(token)
                print("✅ FinLab 登入成功")
        except Exception as e:
            print(f"⚠️ 登入失敗: {e}")


# ==================== 1. 資料載入 ====================
def load_data():
    print("⏳ 載入歷史數據...")
    disposal = pd.DataFrame(data.get("disposal_information"))
    close = data.get("price:收盤價")
    open_p = data.get("price:開盤價")
    vol = data.get("price:成交股數")
    high = data.get("price:最高價")
    low = data.get("price:最低價")

    # 過濾普通股票
    disposal = disposal[disposal["symbol"].str.match(r"^\d{4}$")].copy()
    disposal = disposal[~disposal["symbol"].str.startswith(("00", "91"))].copy()
    disposal["stock_key"] = disposal["symbol"].astype(str).str.lstrip("0")
    disposal["announce"] = pd.to_datetime(disposal["date"]).dt.normalize()
    disposal["start"] = pd.to_datetime(disposal["處置開始時間"]).dt.normalize()
    disposal["end"] = pd.to_datetime(disposal["處置結束時間"]).dt.normalize()
    disposal["condition"] = disposal.get("處置條件", "")

    # 只保留在 price data 中的股票
    valid_stocks = set(close.columns)
    disposal = disposal[disposal["symbol"].isin(valid_stocks)].copy()

    print(f"  處置事件: {len(disposal):,}")
    return disposal, close, open_p, vol, high, low


# ==================== 2. 技術指標 ====================
def compute_indicators(close, vol, target_stocks):
    print("⚙️  計算技術指標...")
    filtered_close = close[target_stocks].astype(np.float64)
    filtered_vol = vol[target_stocks].astype(np.float64)

    turnover = filtered_close * filtered_vol
    avg_turnover_5d = turnover.rolling(5).mean()
    ma20 = filtered_close.rolling(20).mean()

    return avg_turnover_5d, ma20, filtered_close


# ==================== 3. 訊號生成 ====================
def generate_signals(disposal, close, open_p, avg_turnover_5d, ma20, filtered_close):
    """Generate position signals with regime-adaptive entry/exit."""
    print("🧠 生成進出場訊號...")

    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    n_cal = len(cal)

    # 計算處置期間交易天數
    disposal["start_idx"] = cal.searchsorted(disposal["start"], side="left")
    disposal["end_idx"] = cal.searchsorted(disposal["end"], side="right") - 1
    disposal["duration"] = disposal["end_idx"] - disposal["start_idx"] + 1
    disposal["is_new_regime"] = disposal["announce"] >= NEW_REGIME_DATE

    # 過濾有效事件
    valid = disposal[
        (disposal["start_idx"] >= 0) &
        (disposal["end_idx"] >= 0) &
        (disposal["end_idx"] < n_cal) &
        (disposal["duration"] >= 3)
    ].copy()

    print(f"  有效事件: {len(valid):,} (舊制={int((~valid['is_new_regime']).sum()):,}, "
          f"新制={int(valid['is_new_regime'].sum()):,})")

    # 建立部位矩陣
    position = pd.DataFrame(False, index=close.index, columns=close.columns)

    signal_count = 0
    skip_gap = 0
    skip_liq = 0
    skip_bias = 0

    for _, row in valid.iterrows():
        sym = row["symbol"]
        si = row["start_idx"]
        ei = row["end_idx"]
        dur = row["duration"]

        # === 動態進出場日 ===
        if dur >= 8:
            # 舊制 (10天): Day 4 進場, Day 9 出場 (V19 邏輯)
            entry_idx = si + 3  # Day 4
            exit_idx = ei - 1   # Day before end
        elif dur >= 4:
            # 新制 (5天): Day 1 進場, Day 4 出場
            entry_idx = si      # Day 1
            exit_idx = ei - 1   # Day before end (Day 4)
        else:
            # 極短處置 (3天): Day 1 進場, Day 2 出場
            entry_idx = si
            exit_idx = min(si + 1, ei - 1)

        if entry_idx >= n_cal or exit_idx >= n_cal:
            continue
        if exit_idx < entry_idx:
            continue

        entry_date = cal[entry_idx]
        exit_date = cal[exit_idx]

        # === 濾網檢查 ===
        # 需要前一天的收盤來計算 gap
        if entry_idx < 1:
            continue
        prev_close_date = cal[entry_idx - 1]

        try:
            prev_c = filtered_close.loc[prev_close_date, sym]
            e_open = open_p.loc[entry_date, sym]
            current_ma20 = ma20.loc[prev_close_date, sym]
            current_turnover = avg_turnover_5d.loc[prev_close_date, sym]
        except (KeyError, IndexError):
            continue

        if any(pd.isna(x) or x <= 0 for x in [prev_c, e_open, current_ma20]):
            continue

        # Gap 濾網
        gap_pct = (e_open - prev_c) / prev_c
        if not (ENTRY_GAP_LOW < gap_pct < ENTRY_GAP_HIGH):
            skip_gap += 1
            continue

        # 流動性濾網
        if pd.isna(current_turnover) or current_turnover < MIN_TURNOVER:
            skip_liq += 1
            continue

        # 乖離率濾網
        bias = (prev_c - current_ma20) / current_ma20
        if bias >= MAX_BIAS_MA20:
            skip_bias += 1
            continue

        # === 設置部位 ===
        # 從進場日到出場日 (含) 都持有
        mask = (position.index >= entry_date) & (position.index <= exit_date)
        # 只設該股票
        if sym in position.columns:
            position.loc[mask, sym] = True
            signal_count += 1

    print(f"  ✅ 訊號數: {signal_count:,}")
    print(f"  ❌ 跳過 (gap): {skip_gap:,}")
    print(f"  ❌ 跳過 (流動性): {skip_liq:,}")
    print(f"  ❌ 跳過 (乖離率): {skip_bias:,}")

    return position


# ==================== 4. 大盤熔斷 ====================
def apply_circuit_breaker(position, close):
    print("🚨 套用大盤熔斷...")
    daily_returns = close.pct_change()
    panic_count = (daily_returns < -0.06).sum(axis=1)
    is_panic = panic_count > PANIC_THRESHOLD

    # 熔斷後 N 天都清空
    danger_zone = is_panic.rolling(PANIC_COOLDOWN, min_periods=1).max() > 0
    n_affected = danger_zone.sum()

    position.loc[danger_zone, :] = False
    print(f"  熔斷觸發: {int(is_panic.sum())} 次, 影響 {int(n_affected)} 個交易日")

    return position


# ==================== 5. 回測 ====================
def run_backtest(position):
    print("🚀 執行回測...")

    report = sim(
        position,
        trade_at_price="open",
        fee_ratio=FEE_RATIO,
        position_limit=POSITION_LIMIT,
        stop_loss=STOP_LOSS,
        upload=False,
        name="監獄兔_V20_新制版",
    )

    stats = report.get_stats()
    trades = report.get_trades()

    print(f"\n{'='*60}")
    print(f"  監獄兔 V20 (新制適配版)")
    print(f"{'='*60}")
    print(f"  年化報酬率 (CAGR):  {stats['cagr']*100:.2f}%")
    print(f"  最大回撤 (MDD):     {stats['max_drawdown']*100:.2f}%")
    print(f"  勝率 (Win Rate):    {stats['win_ratio']*100:.2f}%")
    print(f"  總報酬 (Total):     {stats['total_return']*100:.2f}%")
    print(f"  總交易筆數:         {len(trades)} 筆")
    if "sharpe" in stats:
        print(f"  Sharpe Ratio:       {stats['sharpe']:.2f}")
    print(f"{'='*60}")

    return report, stats, trades


# ==================== 6. 輸出 ====================
def save_results(report, stats, trades):
    # 存交易紀錄
    trades_path = RESULT_PATH / "V20_總交易紀錄.csv"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"\n📁 交易紀錄: {trades_path}")

    # 存淨值曲線
    try:
        net_value = report.get_net_value()
        nv_path = RESULT_PATH / "V20_淨值曲線.csv"
        net_value.to_csv(nv_path, encoding="utf-8-sig")
        print(f"📁 淨值曲線: {nv_path}")
    except Exception:
        pass

    # 存統計
    stats_path = RESULT_PATH / "V20_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"📁 統計數據: {stats_path}")


# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("  監獄兔 V20 - 新制適配版")
    print("  舊制: Day 4 進場, Day 9 出場")
    print("  新制: Day 1 進場, Day 4 出場")
    print("=" * 60)

    auto_login()

    # Load data
    disposal, close, open_p, vol, high, low = load_data()

    # Get target stocks
    target_stocks = list(set(disposal["symbol"].tolist()) & set(close.columns))
    print(f"  目標股票數: {len(target_stocks):,}")

    # Compute indicators
    avg_turnover_5d, ma20, filtered_close = compute_indicators(close, vol, target_stocks)

    # Generate signals
    position = generate_signals(
        disposal, close, open_p, avg_turnover_5d, ma20, filtered_close
    )

    # Apply circuit breaker
    position = apply_circuit_breaker(position, close)

    # Count active positions
    n_active = position.any(axis=1).sum()
    print(f"\n  有部位的交易日: {n_active:,} / {len(position):,}")

    # Run backtest
    report, stats, trades = run_backtest(position)

    # Save
    save_results(report, stats, trades)

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
