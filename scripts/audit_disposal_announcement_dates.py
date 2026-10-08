from pathlib import Path
import re
import pandas as pd
from finlab import data

ROOT = Path(r"D:\AI專案\StockAgent\finlab\database")
OFFICIAL_DIR = ROOT / "disposal_official_exports"
OUT = ROOT / "disposal_outputs" / "disposal_date_reconciliation"
OUT.mkdir(parents=True, exist_ok=True)

def col(df, names, label):
    found = next((x for x in names if x in df.columns), None)
    if found is None:
        raise ValueError(f"{label} 找不到 {names}；實際欄位：{list(df.columns)}")
    return found

def sid(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    return x[:-2] if x.endswith(".0") and x[:-2].isdigit() else x

def tw_date(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip().replace(".", "/").replace("-", "/")
    if not s:
        return pd.NaT
    # 民國 115/09/11、1150911，以及西元日期。
    digits = re.sub(r"\D", "", s)
    if re.fullmatch(r"\d{7}", digits):
        y, m, d = int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:7])
        return pd.Timestamp(y, m, d)
    parts = s.split("/")
    if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) < 1911:
        parts[0] = str(int(parts[0]) + 1911)
        s = "/".join(parts)
    return pd.to_datetime(s, errors="coerce")

def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"無法解碼：{path}")

# 1. FinLab 全事件表；此步不先限縮 F02 主樣本。
fin = pd.DataFrame(data.get("disposal_information")).copy()
f_stock = col(fin, ["stock_id", "symbol"], "FinLab 股票")
f_date = col(fin, ["date"], "FinLab date")
f_start = col(fin, ["處置開始時間"], "FinLab 起日")
f_end = col(fin, ["處置結束時間"], "FinLab 迄日")
fin["_stock"] = fin[f_stock].map(sid)
fin["_record_date"] = fin[f_date].map(tw_date)
fin["_start"] = fin[f_start].map(tw_date)
fin["_end"] = fin[f_end].map(tw_date)
fin["_fin_row_id"] = range(len(fin))

# 2. 讀取官方匯出；檔名以 TWSE_ 或 TPEX_ 開頭，避免市場別靠猜。
files = sorted(OFFICIAL_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError(
        f"{OFFICIAL_DIR} 沒有官方 CSV。請放入 TWSE_*.csv / TPEX_*.csv"
    )

official_frames = []
file_audit = []
for path in files:
    market = "TWSE" if path.name.upper().startswith("TWSE_") else (
        "TPEX" if path.name.upper().startswith("TPEX_") else None
    )
    if market is None:
        file_audit.append((path.name, "skipped_unknown_market", 0))
        continue
    df = read_csv(path)
    c_stock = col(df, ["證券代號", "股票代號", "代號"], path.name)
    c_announce = col(df, ["公布日期", "公告日期"], path.name)
    c_period = col(df, ["處置起迄時間", "處置期間"], path.name)

    norm = pd.DataFrame({
        "official_file": path.name,
        "market": market,
        "official_stock_id": df[c_stock].map(sid),
        "official_announce_date": df[c_announce].map(tw_date),
        "official_period_raw": df[c_period],
    })
    # 取區間開頭。遇到無法解析的格式，保留原文供人工檢查。
    norm["official_start_date"] = norm["official_period_raw"].astype(str).map(
        lambda x: tw_date(re.split(r"[～~至]", x, maxsplit=1)[0])
    )
    norm["official_row_id"] = [
        f"{path.name}:{i + 2}" for i in range(len(norm))
    ]
    official_frames.append(norm)
    file_audit.append((path.name, "loaded", len(norm)))

if not official_frames:
    raise ValueError("沒有成功載入任何 TWSE_/TPEX_ 官方檔")

official = pd.concat(official_frames, ignore_index=True)
official.to_csv(OUT / "official_normalized.csv",
                index=False, encoding="utf-8-sig")
pd.DataFrame(file_audit, columns=["file", "status", "rows"]).to_csv(
    OUT / "official_file_audit.csv", index=False, encoding="utf-8-sig"
)

# 同一筆官方紀錄可能在多次年度匯出中出現；只移除完全相同的重複列。
official = official.drop_duplicates(
    ["market", "official_stock_id", "official_announce_date",
     "official_start_date", "official_period_raw"]
)

# 3. 股票 + 起日配對；不任意替多候選選第一筆。
joined = fin.merge(
    official,
    left_on=["_stock", "_start"],
    right_on=["official_stock_id", "official_start_date"],
    how="left",
    sort=False
)
n_candidates = joined.groupby("_fin_row_id")["official_row_id"].transform(
    lambda x: x.notna().sum()
)
joined["candidate_count"] = n_candidates
joined["status"] = "official_not_covered"
joined.loc[joined["candidate_count"] > 1, "status"] = "multiple_candidates"
one = joined["candidate_count"].eq(1)
joined.loc[one & joined["_record_date"].eq(
    joined["official_announce_date"]), "status"] = "date_match"
joined.loc[one & joined["_record_date"].ne(
    joined["official_announce_date"]), "status"] = "date_mismatch"
joined["date_delta_days"] = (
    joined["_record_date"] - joined["official_announce_date"]
).dt.days

# 沒有候選者不可直接當作日期不一致；可能是官方匯出缺年份或市場。
columns = [
    "_fin_row_id", "_stock", "_record_date", "_start", "_end",
    "market", "official_file", "official_row_id",
    "official_announce_date", "official_start_date",
    "official_period_raw", "candidate_count", "date_delta_days", "status"
]
joined[columns].to_csv(OUT / "disposal_date_row_audit.csv",
                       index=False, encoding="utf-8-sig")
summary = joined.drop_duplicates("_fin_row_id").groupby(
    "status", dropna=False
).size().reset_index(name="n_finlab_events")
summary.to_csv(OUT / "disposal_date_summary.csv",
               index=False, encoding="utf-8-sig")
print("FinLab rows:", len(fin))
print("Official loaded rows:", len(official))
print(summary.to_string(index=False))
print("Output:", OUT)