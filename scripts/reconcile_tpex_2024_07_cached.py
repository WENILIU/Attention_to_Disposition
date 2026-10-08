from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs")
TWSE_DIR = ROOT / "twse_disposal_date_test_2024_07"
TPEX_DIR = ROOT / "tpex_disposal_probe_2024_07"
OUT = ROOT / "disposal_reconciliation_2024_07"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2024-07-31")

def dt(value):
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    digits = re.sub(r"\D", "", s)
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
        return pd.to_datetime(s, errors="coerce")
    except (ValueError, TypeError):
        return pd.NaT

def parse_period(value):
    parts = re.split(r"[～~至]", str(value), maxsplit=1)
    if len(parts) != 2:
        return pd.NaT, pd.NaT
    return dt(parts[0]), dt(parts[1])

def sid(value):
    if pd.isna(value):
        return None
    return str(value).strip().upper().removesuffix(".0")

def name_without_link(value):
    return re.sub(r"\([^()]*\.(?:html|php)\?[^()]*\)", "",
                  str(value)).strip()

def main():
    twse_path = TWSE_DIR / "row_audit_2024_07.csv"
    tpex_path = TPEX_DIR / "api_ad_slash.bin"
    if not twse_path.exists() or not tpex_path.exists():
        raise FileNotFoundError(
            f"需要既有檔案：{twse_path}；{tpex_path}"
        )

    twse = pd.read_csv(twse_path, dtype=str, encoding="utf-8-sig")
    passed = twse.loc[
        twse["status"].eq("date_match")
        & twse["candidate_count"].eq("1")
        & twse["end_date_matches"].str.lower().eq("true")
    ].copy()
    passed.to_csv(
        OUT / "twse_date_verified_2024_07.csv",
        index=False, encoding="utf-8-sig"
    )

    pending = twse.loc[
        twse["status"].eq("official_not_matched")
    ].copy()
    if pending["finlab_row_id"].duplicated().any():
        raise ValueError("TWSE 未配對表含重複 finlab_row_id")

    payload = json.loads(tpex_path.read_bytes().decode("utf-8-sig"))
    tables = payload.get("tables", [])
    if len(tables) != 1:
        raise ValueError("TPEx 回應表數非 1，須人工檢查")

    table = tables[0]
    title = str(table.get("title2", ""))
    found = re.findall(r"\d{3}/\d{2}/\d{2}", title)
    if len(found) != 2 or dt(found[0]) != START or dt(found[1]) != END:
        raise ValueError(f"不是 2024-07 的櫃買歷史回應：{title}")

    official = pd.DataFrame(table["data"], columns=table["fields"])
    official["tpex_raw_row_id"] = official.index.astype(str)

    # 官方結果有「本日無處置資料」佔位列，不是證券事件。
    placeholder = (
        official["證券代號"].fillna("").astype(str).str.strip().eq("")
        | official["處置內容"].fillna("").astype(str)
          .str.contains("本日無處置資料", regex=False)
    )
    placeholders = official.loc[placeholder].copy()
    placeholders.to_csv(
        OUT / "tpex_placeholders_2024_07.csv",
        index=False, encoding="utf-8-sig"
    )
    official = official.loc[~placeholder].copy()

    official["tpex_stock_id"] = official["證券代號"].map(sid)
    official["tpex_security_name"] = official["證券名稱"].map(
        name_without_link
    )
    official["tpex_announce_date"] = official["公布日期"].map(dt)
    parsed = official["處置起訖時間"].map(parse_period)
    official["tpex_start_date"] = parsed.map(lambda x: x[0])
    official["tpex_end_date"] = parsed.map(lambda x: x[1])

    bad = official.loc[
        official[["tpex_announce_date", "tpex_start_date",
                  "tpex_end_date"]].isna().any(axis=1)
    ]
    if not bad.empty:
        bad.to_csv(OUT / "tpex_unparsed_rows_2024_07.csv",
                   index=False, encoding="utf-8-sig")
        raise ValueError("櫃買非佔位列有日期無法解析；停止對帳")

    official.to_csv(
        OUT / "tpex_official_normalized_2024_07.csv",
        index=False, encoding="utf-8-sig"
    )

    pending["finlab_stock_id"] = pending["finlab_stock_id"].map(sid)
    pending["finlab_record_date"] = pending[
        "finlab_record_date"
    ].map(dt)
    pending["finlab_start_date"] = pending[
        "finlab_start_date"
    ].map(dt)
    pending["finlab_end_date"] = pending[
        "finlab_end_date"
    ].map(dt)

    merged = pending.merge(
        official[[
            "tpex_raw_row_id", "tpex_stock_id",
            "tpex_security_name", "tpex_announce_date",
            "tpex_start_date", "tpex_end_date",
            "處置原因", "處置措施",
        ]],
        left_on=["finlab_stock_id", "finlab_start_date"],
        right_on=["tpex_stock_id", "tpex_start_date"],
        how="left",
        validate="one_to_many",
    )
    merged["tpex_candidate_count"] = merged.groupby(
        "finlab_row_id"
    )["tpex_raw_row_id"].transform(lambda x: x.notna().sum())

    merged["second_pass_status"] = "still_unmatched"
    merged.loc[
        merged["tpex_candidate_count"].gt(1),
        "second_pass_status"
    ] = "tpex_multiple_candidates"

    unique = merged["tpex_candidate_count"].eq(1)
    merged.loc[
        unique & merged["finlab_record_date"].eq(
            merged["tpex_announce_date"]
        ),
        "second_pass_status"
    ] = "tpex_date_match"
    merged.loc[
        unique & merged["finlab_record_date"].ne(
            merged["tpex_announce_date"]
        ),
        "second_pass_status"
    ] = "tpex_date_mismatch"
    merged["tpex_end_date_matches"] = merged[
        "finlab_end_date"
    ].eq(merged["tpex_end_date"])
    merged["date_delta_days"] = (
        merged["finlab_record_date"] -
        merged["tpex_announce_date"]
    ).dt.days

    merged.to_csv(
        OUT / "tpex_second_pass_row_audit_2024_07.csv",
        index=False, encoding="utf-8-sig"
    )
    summary = (
        merged.drop_duplicates("finlab_row_id")
        .groupby("second_pass_status", dropna=False)
        .size()
        .reset_index(name="n_finlab_events")
    )
    summary.to_csv(
        OUT / "tpex_second_pass_summary_2024_07.csv",
        index=False, encoding="utf-8-sig"
    )

    # 名稱及代號只能產生「待核候選」，不可據此確認普通股。
    product = merged.drop_duplicates("finlab_row_id")[[
        "finlab_row_id", "finlab_stock_id",
        "tpex_security_name", "second_pass_status",
    ]].copy()
    product["candidate_from_id_only"] = product[
        "finlab_stock_id"
    ].astype(str).str.fullmatch(r"\d{4}").map({
        True: "ordinary_share_candidate",
        False: "other_or_unknown_candidate"
    })
    product["verified_security_type"] = "unknown"
    product["include_in_attention_to_disposal_main"] = False
    product["classification_basis"] = (
        "No point-in-time official security-type master joined yet"
    )
    product.to_csv(
        OUT / "security_type_audit_2024_07.csv",
        index=False, encoding="utf-8-sig"
    )

    print("TWSE date+end verified:", len(passed))
    print("TWSE pending for TPEx:", len(pending))
    print("TPEx raw rows:", len(table["data"]))
    print("TPEx placeholders:", len(placeholders))
    print("TPEx security rows:", len(official))
    print(summary.to_string(index=False))
    print("輸出位置：", OUT)

if __name__ == "__main__":
    main()