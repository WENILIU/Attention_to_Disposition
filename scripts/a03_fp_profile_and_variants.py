from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

ROOT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs")
SRC = ROOT / "a01_clause_parser" / "a01_attention_with_clauses.csv"
OUT = ROOT / "a01_clause_parser"


def to_bool(s):
    return s.astype(str).str.lower().eq("true")


def raw_id(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    return s[:-2] if s.endswith(".0") else s


def key_of(raw):
    if raw is None:
        return None
    return (raw.lstrip("0") or "0") if raw.isdigit() else raw


def reproduce(df, c1_col, c18_col):
    pred = np.zeros(len(df), bool)
    for _, g in df.loc[df["t_idx"] >= 0].groupby("stock_key", sort=False):
        pos = g.index.to_numpy()
        t = g["t_idx"].to_numpy()
        f1, f18 = g[c1_col].to_numpy(), g[c18_col].to_numpy()
        o = np.argsort(t)
        pos, t, f1, f18 = pos[o], t[o], f1[o], f18[o]
        s1 = set(t[f1].tolist())
        t18 = t[f18]
        s18 = set(t18.tolist())
        for i, ti in enumerate(t.tolist()):
            hit = ti in s1 and (ti - 1) in s1 and (ti - 2) in s1
            if not hit and ti in s18:
                hit = all((ti - j) in s18 for j in range(5))
                if not hit:
                    hi = np.searchsorted(t18, ti, side="right")
                    n10 = hi - np.searchsorted(t18, ti - 9, side="left")
                    n30 = hi - np.searchsorted(t18, ti - 29, side="left")
                    hit = n10 >= 6 or n30 >= 12
            pred[pos[i]] = hit
    return pred


def main():
    df = pd.read_csv(
        SRC, dtype={"stock_key": str, "clause_set": str},
        encoding="utf-8-sig", low_memory=False
    )
    for c in ("same_day_disposal", "in_disposal_period", "ordinary_stock",
              "in_sample", "has_c1", "has_c1to8", "no_clause_parsed",
              "r3", "r5", "r10", "r30", "pred", "actual"):
        df[c] = to_bool(df[c])
    df["attention_date"] = pd.to_datetime(df["attention_date"])
    df["year"] = df["attention_date"].dt.year
    df["t_idx"] = df["t_idx"].astype(int)
    lists = df["clause_set"].fillna("").map(
        lambda s: [int(x) for x in s.split("|") if x]
    )
    df["c1_to_7"] = lists.map(lambda xs: any(1 <= c <= 7 for c in xs))

    scope = df["in_sample"] & df["ordinary_stock"] & ~df["in_disposal_period"]

    variants = {
        "A_current_1to8": df["has_c1to8"],
        "B_1to7_until_2019": pd.Series(
            np.where(df["year"] <= 2019, df["c1_to_7"], df["has_c1to8"]),
            index=df.index
        ),
        "C_1to7_all_years": df["c1_to_7"],
    }
    rows = []
    for name, col in variants.items():
        d = df[["stock_key", "t_idx"]].copy()
        d["f1"], d["f18"] = df["has_c1"], col
        pred = reproduce(d, "f1", "f18")
        groups = list(df.loc[scope].groupby("year").groups.items())
        groups.append(("ALL", df.index[scope]))
        for y, idx in groups:
            ix = np.asarray(idx)
            a = df["actual"].to_numpy()[ix]
            p = pred[ix]
            tp, fp, fn = int((p & a).sum()), int((p & ~a).sum()), int((~p & a).sum())
            rows.append({
                "variant": name, "year": y, "TP": tp, "FP": fp, "FN": fn,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
            })
    pd.DataFrame(rows).to_csv(
        OUT / "a03_variant_comparison.csv", index=False, encoding="utf-8-sig"
    )

    dis = pd.DataFrame(data.get("disposal_information"))
    sym = "symbol" if "symbol" in dis.columns else "stock_id"
    dis["stock_key"] = dis[sym].map(raw_id).map(key_of)
    dis["end"] = pd.to_datetime(dis["處置結束時間"], errors="coerce").dt.normalize()
    dis = dis.dropna(subset=["stock_key", "end"])
    ends = {k: np.sort(g["end"].to_numpy()) for k, g in dis.groupby("stock_key")}

    sub = df.loc[scope & df["pred"]].copy()
    gaps = []
    for key, d in zip(sub["stock_key"], sub["attention_date"].to_numpy()):
        arr = ends.get(key)
        if arr is None:
            gaps.append(np.nan)
            continue
        p = np.searchsorted(arr, d, side="left") - 1
        gaps.append(np.nan if p < 0 else (d - arr[p]) / np.timedelta64(1, "D"))
    sub["days_since_last_end"] = gaps
    labels = ["0-7", "8-14", "15-30", "31-60", "61-180", ">180"]
    sub["gap_bin"] = pd.cut(
        sub["days_since_last_end"],
        bins=[-1, 7, 14, 30, 60, 180, 100000], labels=labels
    ).astype(object)
    sub.loc[sub["days_since_last_end"].isna(), "gap_bin"] = "no_prior"
    sub["outcome"] = np.where(sub["actual"], "TP", "FP")

    sub.groupby(["gap_bin", "outcome"]).size().unstack(fill_value=0).to_csv(
        OUT / "a03_gap_since_last_disposal_end.csv", encoding="utf-8-sig"
    )
    sub.groupby("outcome")[["r3", "r5", "r10", "r30"]].sum().assign(
        n=sub.groupby("outcome").size()
    ).to_csv(OUT / "a03_rule_flags_tp_vs_fp.csv", encoding="utf-8-sig")
    df.loc[df["no_clause_parsed"]].groupby("year").size().rename(
        "n_unparsed"
    ).to_csv(OUT / "a01_unparsed_by_year.csv", encoding="utf-8-sig")
    print("輸出：", OUT)


if __name__ == "__main__":
    main()