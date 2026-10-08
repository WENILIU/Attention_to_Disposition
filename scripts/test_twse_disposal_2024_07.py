import json
import re
from pathlib import Path

import pandas as pd
from finlab import data

OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\twse_disposal_date_test_2024_07"
)
RAW = OUT / "raw_date_range.txt"
MONTH_START = pd.Timestamp("2024-07-01")
MONTH_END = pd.Timestamp("2024-07-31")


def parse_date(value):
    if value is None or pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    digits = re.sub(r"\D", "", text)
    try:
        if re.fullmatch(r"\d{7}", digits):
            return pd.Timestamp(
                int(digits[:3]) + 1911,
                int(digits[3:5]),
                int(digits[5:7]),
            )
        if re.fullmatch(r"\d{8}", digits):
            return pd.Timestamp(
                int(digits[:4]),
                int(digits[4:6]),
                int(digits[6:8]),
            )
        parts = re.split(r"[/\-]", text)
        if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) < 1911:
            parts[0] = str(int(parts[0]) + 1911)
            return pd.to_datetime("/".join(parts[:3]), errors="coerce")
        return pd.to_datetime(text, errors="coerce")
    except (TypeError, ValueError):
        return pd.NaT


def parse_period(value):
    parts = re.split(r"[～~至]", str(value), maxsplit=1)
    if len(parts) != 2:
        return pd.NaT, pd.NaT
    return parse_date(parts[0]), parse_date(parts[1])


def sid(value):
    if value is None or pd.isna(value):
        return None
    return str(value).strip().upper().removesuffix(".0")


def main():
    if not RAW.is_file():
        raise FileNotFoundError(RAW)

    payload = json.loads(RAW.read_text(encoding="utf-8-sig"))
    fields = payload.get("fields")
    rows = payload.get("data")
    title = str(payload.get("title", ""))

    if not isinstance(fields, list) or not isinstance(rows, list):
        raise ValueError("官方回應缺 fields 或 data")

    # 驗證查詢期間，而不是要求公布日期落在 7 月。
    dates_in_title = re.findall(r"\d{3}/\d{2}/\d{2}", title)
    if len(dates_in_title) != 2:
        raise ValueError(f"無法確認官方查詢期間：{title}")
    if (parse_date(dates_in_title[0]) != MONTH_START
            or parse_date(dates_in_title[1]) != MONTH_END):
        raise ValueError(f"官方回應不是 2024-07 查詢：{title}")

    official = pd.DataFrame(rows, columns=fields)
    required = {"公布日期", "證券代號", "處置起迄時間"}
    if not required.issubset(official.columns):
        raise ValueError(f"缺少官方欄位：{required - set(official.columns)}")

    official["official_stock_id"] = official["證券代號"].map(sid)
    official["official_announce_date"] = official["公布日期"].map(parse_date)
    starts_ends = official["處置起迄時間"].map(parse_period)
    official["official_start_date"] = starts_ends.map(lambda pair: pair[0])
    official["official_end_date"] = starts_ends.map(lambda pair: pair[1])
    official["official_row_id"] = official.index.astype(str)

    invalid = official.loc[
        official["official_announce_date"].isna()
        | official["official_start_date"].isna()
        | official["official_end_date"].isna()
    ]
    if not invalid.empty:
        invalid.to_csv(
            OUT / "unparsed_official_rows.csv",
            index=False, encoding="utf-8-sig"
        )
        raise ValueError("官方日期有無法解析的列；已輸出 unparsed_official_rows.csv")

    # 必須與「處置期間」相交；不要求公告日在 7 月。
    overlap = (
        official["official_start_date"].le(MONTH_END)
        & official["official_end_date"].ge(MONTH_START)
    )
    if not overlap.all():
        official.loc[~overlap].to_csv(
            OUT / "official_outside_requested_period.csv",
            index=False, encoding="utf-8-sig"
        )
        raise ValueError("有官方紀錄的處置期與查詢期間不相交，停止對帳")

    official.to_csv(
        OUT / "official_2024_07_normalized.csv",
        index=False, encoding="utf-8-sig"
    )

    fin = pd.DataFrame(data.get("disposal_information")).copy()
    fin["finlab_stock_id"] = fin["stock_id"].map(sid)
    fin["finlab_record_date"] = fin["date"].map(parse_date)
    fin["finlab_start_date"] = fin["處置開始時間"].map(parse_date)
    fin["finlab_end_date"] = fin["處置結束時間"].map(parse_date)
    fin["finlab_row_id"] = fin.index.astype(str)

    # 同樣只選「處置期間與 2024-07 相交」的 FinLab 事件。
    fin = fin.loc[
        fin["finlab_start_date"].le(MONTH_END)
        & fin["finlab_end_date"].ge(MONTH_START)
    ].copy()

    merged = fin.merge(
        official[
            ["official_row_id", "official_stock_id",
             "official_announce_date", "official_start_date",
             "official_end_date", "公布日期", "處置起迄時間"]
        ],
        left_on=["finlab_stock_id", "finlab_start_date"],
        right_on=["official_stock_id", "official_start_date"],
        how="left",
        validate="many_to_many"
    )

    merged["candidate_count"] = merged.groupby(
        "finlab_row_id"
    )["official_row_id"].transform(lambda s: s.notna().sum())
    merged["status"] = "official_not_matched"
    merged.loc[
        merged["candidate_count"].gt(1), "status"
    ] = "multiple_candidates"
    one = merged["candidate_count"].eq(1)
    merged.loc[
        one & merged["finlab_record_date"].eq(
            merged["official_announce_date"]
        ), "status"
    ] = "date_match"
    merged.loc[
        one & merged["finlab_record_date"].ne(
            merged["official_announce_date"]
        ), "status"
    ] = "date_mismatch"
    merged["date_delta_days"] = (
        merged["finlab_record_date"] - merged["official_announce_date"]
    ).dt.days
    merged["end_date_matches"] = merged["finlab_end_date"].eq(
        merged["official_end_date"]
    )

    cols = [
        "finlab_row_id", "finlab_stock_id",
        "finlab_record_date", "finlab_start_date", "finlab_end_date",
        "official_row_id", "official_stock_id", "official_announce_date",
        "official_start_date", "official_end_date",
        "candidate_count", "date_delta_days", "end_date_matches", "status"
    ]
    merged[cols].to_csv(
        OUT / "row_audit_2024_07.csv",
        index=False, encoding="utf-8-sig"
    )
    summary = (
        merged.drop_duplicates("finlab_row_id")
        .groupby("status", dropna=False)
        .size()
        .reset_index(name="n_finlab_events")
    )
    summary.to_csv(
        OUT / "summary_2024_07.csv",
        index=False, encoding="utf-8-sig"
    )
    print("官方查詢標題：", title)
    print("官方處置期與 7 月相交筆數：", len(official))
    print("FinLab 處置期與 7 月相交筆數：", len(fin))
    print(summary.to_string(index=False))
    print("逐筆表：", OUT / "row_audit_2024_07.csv")


if __name__ == "__main__":
    main()