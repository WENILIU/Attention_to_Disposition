from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs")
SRC = ROOT / "a00_attention_master_v2" / "a00_v2_attention_master.csv"
OUT = ROOT / "a01_clause_parser"
OUT.mkdir(parents=True, exist_ok=True)

OPEN = r"[﹝\(（\[［]"
CLOSE = r"[﹞\)）\]］]"
CLAUSE = re.compile(OPEN + r"\s*第([一二三四五六七八九十\s]+?)款\s*" + CLOSE)
DIGITS = {c: i for i, c in enumerate("一二三四五六七八九", start=1)}


def zh_int(s):
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + DIGITS.get(s[1:], 0)
    if s.endswith("十"):
        return DIGITS.get(s[0], 0) * 10
    if "十" in s:
        a, b = s.split("十", 1)
        return DIGITS.get(a, 0) * 10 + DIGITS.get(b, 0)
    return DIGITS.get(s)


def parse(text):
    text = "" if pd.isna(text) else str(text).replace("<br>", " ")
    out, pos = [], 0
    for m in CLAUSE.finditer(text):
        out.append((zh_int(m.group(1)), text[pos:m.start()]))
        pos = m.end()
    return out


def template(seg):
    s = re.sub(r"[0-9０-９][0-9０-９\.,]*", "#", seg)
    s = re.sub(r"[\s。，、；:：%％]+", "", s).lstrip("且")
    return s[:26]


def guess(seg):
    s = re.sub(r"\s+", "", seg)
    for kw, name in (
        ("存託憑證", "TDR溢折價"), ("督導會報", "督導會報"),
        ("當日沖銷", "當沖比"), ("借券賣出", "借券比"),
        ("價差達", "價差"), ("累積週轉率", "累積週轉率"),
        ("券資比", "券資比"), ("本益比", "本益比/淨值比"),
        ("股價淨值比", "本益比/淨值比"), ("週轉率", "週轉率"),
        ("成交量", "成交量"), ("漲幅", "漲跌幅"), ("跌幅", "漲跌幅"),
    ):
        if kw in s:
            return name
    return "other"


def to_bool(s):
    return s.astype(str).str.lower().eq("true")


def main():
    df = pd.read_csv(
        SRC, dtype={"stock_key": str, "raw_id": str},
        encoding="utf-8-sig", low_memory=False
    )
    df["attention_date"] = pd.to_datetime(df["attention_date"])
    df["year"] = df["attention_date"].dt.year
    for c in ("same_day_disposal", "in_disposal_period",
              "ordinary_stock", "in_sample"):
        df[c] = to_bool(df[c])
    df["t_idx"] = df["t_idx"].astype(int)

    parsed = df["注意交易資訊"].map(parse)
    df["clauses"] = parsed.map(lambda xs: sorted({c for c, _ in xs if c}))
    df["clause_set"] = df["clauses"].map(lambda xs: "|".join(map(str, xs)))
    df["no_clause_parsed"] = df["clauses"].map(len).eq(0)
    df["has_c1"] = df["clauses"].map(lambda xs: 1 in xs)
    df["has_c1to8"] = df["clauses"].map(
        lambda xs: any(1 <= c <= 8 for c in xs)
    )

    years = df["year"].to_numpy()
    rows = []
    for i, xs in enumerate(parsed.tolist()):
        for c, seg in xs:
            rows.append((years[i], c, template(seg), guess(seg)))
    seg_df = pd.DataFrame(
        rows, columns=["year", "clause", "template", "keyword_guess"]
    )
    seg_df.groupby(["clause", "year"]).size().unstack(
        fill_value=0
    ).to_csv(OUT / "a01_clause_by_year.csv", encoding="utf-8-sig")
    seg_df["era"] = pd.cut(
        seg_df["year"], [0, 2019, 2022, 2100],
        labels=["<=2019", "2020-2022", ">=2023"]
    )
    (
        seg_df.groupby(
            ["clause", "era", "keyword_guess", "template"], observed=True
        ).size().reset_index(name="n")
        .sort_values(["clause", "era", "n"],
                     ascending=[True, True, False])
        .groupby(["clause", "era"], observed=True).head(6)
        .to_csv(OUT / "a01_clause_templates.csv",
                index=False, encoding="utf-8-sig")
    )
    miss = df.loc[df["no_clause_parsed"],
                  ["attention_date", "stock_key", "注意交易資訊"]]
    miss.sample(min(200, len(miss)), random_state=1).to_csv(
        OUT / "a01_unparsed_samples.csv",
        index=False, encoding="utf-8-sig"
    )

    n = len(df)
    R3, R5, R10, R30 = (np.zeros(n, bool) for _ in range(4))
    valid = df["t_idx"] >= 0
    for _, g in df.loc[valid].groupby("stock_key", sort=False):
        pos = g.index.to_numpy()
        t = g["t_idx"].to_numpy()
        f1 = g["has_c1"].to_numpy()
        f18 = g["has_c1to8"].to_numpy()
        order = np.argsort(t)
        pos, t, f1, f18 = pos[order], t[order], f1[order], f18[order]
        s1 = set(t[f1].tolist())
        t18 = t[f18]
        s18 = set(t18.tolist())
        for i, ti in enumerate(t.tolist()):
            if ti in s1 and (ti - 1) in s1 and (ti - 2) in s1:
                R3[pos[i]] = True
            if ti in s18:
                R5[pos[i]] = all((ti - j) in s18 for j in range(5))
                n10 = (np.searchsorted(t18, ti, side="right")
                       - np.searchsorted(t18, ti - 9, side="left"))
                n30 = (np.searchsorted(t18, ti, side="right")
                       - np.searchsorted(t18, ti - 29, side="left"))
                R10[pos[i]] = n10 >= 6
                R30[pos[i]] = n30 >= 12
    df["r3"], df["r5"], df["r10"], df["r30"] = R3, R5, R10, R30
    df["pred"] = df[["r3", "r5", "r10", "r30"]].any(axis=1)
    df["actual"] = df["same_day_disposal"]

    base = df["in_sample"] & df["ordinary_stock"]
    scopes = {
        "not_in_disposal": base & ~df["in_disposal_period"],
        "all": base,
    }
    conf_rows = []
    for scope, mask in scopes.items():
        sub = df.loc[mask]
        groups = list(sub.groupby("year")) + [("ALL", sub)]
        for y, g in groups:
            tp = int((g.pred & g.actual).sum())
            fp = int((g.pred & ~g.actual).sum())
            fn = int((~g.pred & g.actual).sum())
            tn = int((~g.pred & ~g.actual).sum())
            conf_rows.append({
                "scope": scope, "year": y, "TP": tp, "FP": fp,
                "FN": fn, "TN": tn,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
            })
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(OUT / "a03_rule_reproduction_confusion.csv",
                index=False, encoding="utf-8-sig")

    sub = df.loc[scopes["not_in_disposal"]]
    act = sub.loc[sub.actual]
    pd.DataFrame({
        "rule": ["r3", "r5", "r10", "r30", "none"],
        "n_actual_disposals": [
            int(act.r3.sum()), int(act.r5.sum()),
            int(act.r10.sum()), int(act.r30.sum()),
            int((~act.pred).sum()),
        ],
    }).to_csv(OUT / "a03_rule_hits_among_actual.csv",
              index=False, encoding="utf-8-sig")

    cols = ["attention_date", "stock_key", "clause_set", "pred", "actual",
            "r3", "r5", "r10", "r30", "注意交易資訊"]
    for name, m in (("fn", ~sub.pred & sub.actual),
                    ("fp", sub.pred & ~sub.actual)):
        x = sub.loc[m, cols].copy()
        x["注意交易資訊"] = x["注意交易資訊"].astype(str).str.slice(0, 160)
        x.head(300).to_csv(OUT / f"a03_mismatch_{name}.csv",
                           index=False, encoding="utf-8-sig")

    df.drop(columns=["clauses"]).to_csv(
        OUT / "a01_attention_with_clauses.csv",
        index=False, encoding="utf-8-sig"
    )
    print(conf.query("year == 'ALL'").to_string(index=False))
    print("無款號列數：", int(df["no_clause_parsed"].sum()))
    print("輸出：", OUT)


if __name__ == "__main__":
    main()