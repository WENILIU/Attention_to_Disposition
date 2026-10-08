from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
           r"\a01_clause_parser")
SRC = OUT / "a01_attention_with_clauses.csv"
OPEN = r"[﹝\(（\[［]"
CLOSE = r"[﹞\)）\]］]"
CLAUSE = re.compile(OPEN + r"\s*第([一二三四五六七八九十\s]+?)款\s*" + CLOSE)
PCT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")


def clause1_segment(text):
    text = "" if pd.isna(text) else str(text).replace("<br>", " ")
    pos = 0
    for m in CLAUSE.finditer(text):
        if re.sub(r"\s+", "", m.group(1)) == "一":
            return text[pos:m.start()]
        pos = m.end()
    return None


def main():
    df = pd.read_csv(
        SRC, usecols=["stock_key", "attention_date", "t_idx", "year",
                      "注意交易資訊"],
        dtype={"stock_key": str}, encoding="utf-8-sig", low_memory=False)
    df = df.loc[(df["year"] >= 2018) & (df["t_idx"] >= 8)].copy()
    df["attention_date"] = pd.to_datetime(df["attention_date"])
    seg = df["注意交易資訊"].map(clause1_segment)
    df = df.loc[seg.notna()].copy()
    seg = seg.loc[df.index].map(lambda s: re.sub(r"\s+", "", s))
    pct = []
    for s in seg:
        mm = PCT.search(s)
        pct.append(float(mm.group(1)) / 100 if mm else np.nan)
    df["pct"] = pct
    df["direction"] = np.where(seg.str.contains("跌幅"), -1, 1)
    df["wording"] = np.where(
        seg.str.contains("最後成交價"), "最後成交價",
        np.where(seg.str.contains("收盤價"), "收盤價", "other"))

    close = pd.DataFrame(data.get("price:收盤價")).sort_index()
    close.index = pd.DatetimeIndex(close.index).normalize()
    t_all = df["t_idx"].astype(int).to_numpy()
    if not (close.index[t_all].to_numpy()
            == df["attention_date"].to_numpy()).all():
        raise ValueError("t_idx 與價格表日期不一致")
    col_pos = {str(c): j for j, c in enumerate(close.columns)}
    df["j"] = df["stock_key"].map(col_pos)
    sub = df.loc[df["j"].notna() & df["pct"].notna()].copy()
    t = sub["t_idx"].astype(int).to_numpy()
    jj = sub["j"].astype(int).to_numpy()
    signed = (sub["pct"] * sub["direction"]).to_numpy()
    C = close.to_numpy(float)

    rows = []
    for k in range(3, 9):
        r = C[t, jj] / C[t - k, jj] - 1
        hit = np.abs(r - signed) <= 0.0006
        row = {"base_offset_k": k, "n": len(sub),
               "match_rate_all": float(hit.mean())}
        for w in sorted(sub["wording"].unique()):
            m = (sub["wording"] == w).to_numpy()
            row[f"match_{w}"] = float(hit[m].mean()) if m.any() else np.nan
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "a04c_base_match.csv", index=False,
               encoding="utf-8-sig")

    thr = (sub.groupby(["wording", "year", "direction"])["pct"]
           .agg(n="size", min_pct="min",
                p01=lambda x: x.quantile(0.01)).reset_index())
    thr.to_csv(OUT / "a04c_threshold_by_year.csv", index=False,
               encoding="utf-8-sig")
    print(res.to_string(index=False))
    print("輸出：", OUT)


if __name__ == "__main__":
    main()