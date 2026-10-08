"""Audit: Can disposal stocks actually be shorted?

Checks:
  1. 融券限額 > 0 on the disposal start date
  2. 暫停融券賣出 is not True
  3. Coverage rate by year and disposal type
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\p3_short_feasibility"
)
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Load disposal events
    dis = pd.DataFrame(data.get("disposal_information"))
    dis["stock_key"] = dis["symbol"].astype(str).str.lstrip("0")
    dis["announce"] = pd.to_datetime(dis["date"]).dt.normalize()
    dis["start"] = pd.to_datetime(dis["處置開始時間"]).dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"]).dt.normalize()

    # Filter: ordinary stocks, 2018+
    dis = dis[dis["symbol"].str.match(r"^\d{4}$")].copy()
    dis = dis[dis["announce"] >= "2018-01-01"].copy()
    print(f"Disposal events (ordinary, 2018+): {len(dis):,}")
    print(f"Unique stocks: {dis['symbol'].nunique():,}")

    # Load short sell data
    print("Loading short sell data...")
    quota = data.get("margin_transactions:融券限額")
    suspended = data.get("margin_short_sell_mark:暫停融券賣出")
    print(f"Quota: {quota.shape}, Suspended: {suspended.shape}")

    # Use quota's own date index for lookup
    quota_dates = pd.DatetimeIndex(quota.index).normalize().unique().sort_values()

    # For each disposal event, check short availability on the START date
    # (that's when you'd want to short - the day disposal begins)
    results = []
    stocks_in_quota = set(quota.columns)
    stocks_in_susp = set(suspended.columns)

    for _, row in dis.iterrows():
        sym = row["symbol"]
        start_date = row["start"]
        yr = row["announce"].year

        # Check if stock is in the data at all
        in_quota_data = sym in stocks_in_quota
        in_susp_data = sym in stocks_in_susp

        # Find the nearest date in quota index for start_date
        idx = quota_dates.searchsorted(start_date, side="left")
        if idx >= len(quota_dates):
            continue

        # Check quota
        has_quota = False
        quota_val = np.nan
        if in_quota_data:
            try:
                quota_val = quota.iloc[idx][sym]
                has_quota = (not np.isnan(quota_val)) and (quota_val > 0)
            except (IndexError, KeyError):
                pass

        # Check suspension
        is_suspended = False
        if in_susp_data:
            try:
                susp_val = suspended.iloc[idx][sym]
                is_suspended = (str(susp_val).lower() == "true")
            except (IndexError, KeyError):
                pass

        can_short = has_quota and not is_suspended

        results.append({
            "symbol": sym,
            "announce": row["announce"],
            "start": start_date,
            "end": row["end"],
            "year": yr,
            "condition": row.get("處置條件", ""),
            "in_quota_data": in_quota_data,
            "quota_value": quota_val,
            "has_quota": has_quota,
            "is_suspended": is_suspended,
            "can_short": can_short,
        })

    res = pd.DataFrame(results)
    res.to_csv(OUT / "p3_short_feasibility_detail.csv",
               index=False, encoding="utf-8-sig")

    # Summary
    print("\n" + "=" * 60)
    print("SHORT SELLING FEASIBILITY SUMMARY")
    print("=" * 60)

    total = len(res)
    in_data = res["in_quota_data"].sum()
    has_q = res["has_quota"].sum()
    susp = res["is_suspended"].sum()
    can = res["can_short"].sum()

    print(f"\nTotal disposal events: {total:,}")
    print(f"In 融券限額 data at all: {int(in_data):,} ({in_data/total:.1%})")
    print(f"Has quota > 0: {int(has_q):,} ({has_q/total:.1%})")
    print(f"Suspended (暫停融券賣出): {int(susp):,} ({susp/total:.1%})")
    print(f"CAN SHORT (quota > 0 AND not suspended): {int(can):,} ({can/total:.1%})")

    # By year
    print("\n--- By Year ---")
    yearly = res.groupby("year").agg(
        n=("can_short", "size"),
        can_short=("can_short", "sum"),
        has_quota=("has_quota", "sum"),
        suspended=("is_suspended", "sum"),
    ).reset_index()
    yearly["rate"] = yearly["can_short"] / yearly["n"]
    print(yearly.to_string(index=False))
    yearly.to_csv(OUT / "p3_short_feasibility_by_year.csv",
                  index=False, encoding="utf-8-sig")

    # By disposal condition
    print("\n--- By Disposal Condition (top 10) ---")
    cond = res.groupby("condition").agg(
        n=("can_short", "size"),
        can_short=("can_short", "sum"),
    ).reset_index()
    cond["rate"] = cond["can_short"] / cond["n"]
    cond = cond.sort_values("n", ascending=False).head(10)
    print(cond.to_string(index=False))
    cond.to_csv(OUT / "p3_short_feasibility_by_condition.csv",
                index=False, encoding="utf-8-sig")

    # For the P3 test period specifically (2023+)
    test_period = res[res["year"] >= 2023]
    tp_can = test_period["can_short"].sum()
    tp_total = len(test_period)
    print(f"\n--- TEST PERIOD (2023+) ---")
    print(f"Disposal events: {tp_total:,}")
    print(f"Can short: {int(tp_can):,} ({tp_can/tp_total:.1%})")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
