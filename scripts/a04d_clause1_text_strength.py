from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
           r"\a01_clause_parser")
SRC = OUT / "a04_features.csv"

OPEN = r"[﹝\(（\[［]"
CLOSE = r"[﹞\)）\]］]"
CLAUSE = re.compile(
    OPEN + r"\s*第([一二三四五六七八九十\s]+?)款\s*" + CLOSE
)

# 只接受緊接在「累積的…價格漲/跌幅達」附近的百分比。
PATTERN = re.compile(
    r"(?:六個營業日.*?累積.*?"
    r"(?:收盤價|最後成交價).*?"
    r"(漲幅|跌幅)\s*(?:達|為)?\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*%)"
)


def clause1_segment(text):
    text = "" if pd.isna(text) else str(text).replace("<br>", " ")
    pos = 0
    for match in CLAUSE.finditer(text):
        number = re.sub(r"\s+", "", match.group(1))
        if number == "一":
            return text[pos:match.start()]
        pos = match.end()
    return None


def extract(seg):
    if seg is None:
        return None, None, None, "no_clause1"
    clean = re.sub(r"\s+", "", seg)
    matches = list(PATTERN.finditer(clean))
    if not matches:
        return None, None, None, "no_target_pct"
    # 最後一個通常最接近第一款標記；保留全部候選做稽核。
    m = matches[-1]
    direction = 1 if m.group(1) == "漲幅" else -1
    wording = "最後成交價" if "最後成交價" in m.group(0) else "收盤價"
    return float(m.group(2)) / 100, direction, wording, "ok"


def main():
    # A4 特徵表含武裝日、隔日結果與年別，但不保留注意文字。
    df = pd.read_csv(
        SRC,
        dtype={"stock_key": str},
        encoding="utf-8-sig",
        low_memory=False,
    )
    for c in ("a3", "disposal_tomorrow", "listed_c1_tomorrow"):
        df[c] = df[c].astype(str).str.lower().eq("true")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.loc[df["a3"]].copy()

    # 從 A1 完整注意事件表回接原始注意文字。
    clause_src = OUT / "a01_attention_with_clauses.csv"
    attention = pd.read_csv(
        clause_src,
        usecols=["stock_key", "attention_date", "注意交易資訊"],
        dtype={"stock_key": str},
        encoding="utf-8-sig",
        low_memory=False,
    )
    attention["attention_date"] = pd.to_datetime(
        attention["attention_date"], errors="coerce"
    ).dt.normalize()

    # 同一股、同一注意日最多應有一列；若重複就停止，不靜默選一列。
    duplicate = attention.duplicated(
        ["stock_key", "attention_date"], keep=False
    )
    if duplicate.any():
        attention.loc[duplicate].to_csv(
            OUT / "a04d_duplicate_attention_text_keys.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            "A1 注意文字表有同股同日重複鍵；"
            "已輸出 a04d_duplicate_attention_text_keys.csv"
        )

    df = df.merge(
        attention,
        left_on=["stock_key", "date"],
        right_on=["stock_key", "attention_date"],
        how="left",
        validate="many_to_one",
    )

    missing_text = df["注意交易資訊"].isna()
    if missing_text.any():
        df.loc[missing_text, ["stock_key", "date", "year"]].to_csv(
            OUT / "a04d_missing_attention_text.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("找不到注意文字的 a3 武裝日：", int(missing_text.sum()))

    seg = df["注意交易資訊"].map(clause1_segment)
    parsed = seg.map(extract)
    df[["reported_pct", "direction", "wording", "parse_status"]] = (
        pd.DataFrame(parsed.tolist(), index=df.index)
    )
    ok = df["parse_status"].eq("ok")
    df["direction_text"] = df["direction"].map({1: "up", -1: "down"})

    # 只用訓練期（<=2022）估計工作門檻；避免測試期偷看。
    train = df.loc[ok & (df["year"] <= 2022)]
    threshold = (train.groupby(["wording", "direction_text"])["reported_pct"]
                 .quantile(0.01).rename("train_p01_threshold")
                 .reset_index())
    df = df.merge(threshold, on=["wording", "direction_text"], how="left")
    df["margin_to_threshold"] = (
        df["reported_pct"] - df["train_p01_threshold"]
    )

    df.to_csv(OUT / "a04d_clause1_text_features.csv",
              index=False, encoding="utf-8-sig")

    quality = (df.groupby(["parse_status", "wording"], dropna=False)
               .size().reset_index(name="n"))
    quality.to_csv(OUT / "a04d_parse_quality.csv",
                   index=False, encoding="utf-8-sig")

    samples = df.loc[:, ["date", "stock_key", "注意交易資訊",
                         "parse_status", "reported_pct",
                         "direction_text", "wording"]].copy()
    samples["注意交易資訊"] = samples["注意交易資訊"].astype(str).str.slice(0, 250)
    samples.groupby("parse_status", dropna=False).head(30).to_csv(
        OUT / "a04d_parse_samples.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    for subset, mask in {
        "train": ok & (df["year"] <= 2022),
        "test": ok & (df["year"] >= 2023),
    }.items():
        for (w, dr), g in df.loc[mask].groupby(
            ["wording", "direction_text"]
        ):
            try:
                bins = pd.qcut(g["margin_to_threshold"], 5,
                               duplicates="drop")
            except ValueError:
                continue
            for label, h in g.groupby(bins, observed=True):
                rows.append({
                    "subset": subset, "wording": w, "direction": dr,
                    "bin": str(label), "n": len(h),
                    "mean_margin": h["margin_to_threshold"].mean(),
                    "p_disposal_tomorrow": h["disposal_tomorrow"].mean(),
                    "p_listed_c1_tomorrow": h["listed_c1_tomorrow"].mean(),
                })
    pd.DataFrame(rows).to_csv(OUT / "a04d_strength_bins.csv",
                              index=False, encoding="utf-8-sig")

    print("a3 武裝日：", len(df))
    print("成功解析第一款強度：", int(ok.sum()))
    print(quality.to_string(index=False))
    print("輸出：", OUT)


if __name__ == "__main__":
    main()