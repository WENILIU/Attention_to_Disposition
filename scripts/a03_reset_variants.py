from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

ROOT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs")
OUT = ROOT / "a01_clause_parser"
SRC = OUT / "a01_attention_with_clauses.csv"
VARIANTS = (
    "A_current",
    "D_reset_at_prev_announce",
    "E_reset_at_prev_end",
    "F_ignore_days_in_disposal",
)
NEG = -10**9


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


def trig(ti, c, s1, s18, t18):
    if ti - 2 > c and ti in s1 and (ti - 1) in s1 and (ti - 2) in s1:
        return True
    if ti not in s18:
        return False
    if ti - 4 > c and all((ti - j) in s18 for j in range(5)):
        return True
    hi = np.searchsorted(t18, ti, side="right")
    n10 = hi - np.searchsorted(t18, max(c + 1, ti - 9), side="left")
    if n10 >= 6:
        return True
    n30 = hi - np.searchsorted(t18, max(c + 1, ti - 29), side="left")
    return n30 >= 12


def main():
    use = ["stock_key", "attention_date", "t_idx", "year",
           "same_day_disposal", "in_disposal_period", "ordinary_stock",
           "in_sample", "has_c1", "has_c1to8"]
    df = pd.read_csv(SRC, usecols=use, dtype={"stock_key": str},
                     encoding="utf-8-sig", low_memory=False)
    for c in ("same_day_disposal", "in_disposal_period", "ordinary_stock",
              "in_sample", "has_c1", "has_c1to8"):
        df[c] = to_bool(df[c])
    df["t_idx"] = df["t_idx"].astype(int)

    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    dis = pd.DataFrame(data.get("disposal_information")).copy()
    sym = "symbol" if "symbol" in dis.columns else "stock_id"
    dis["stock_key"] = dis[sym].map(raw_id).map(key_of)
    dis["announce"] = pd.to_datetime(dis["date"], errors="coerce").dt.normalize()
    dis["end"] = pd.to_datetime(dis["處置結束時間"], errors="coerce").dt.normalize()
    dis = dis.dropna(subset=["stock_key", "announce", "end"])
    dis["a_idx"] = cal.searchsorted(dis["announce"].values, side="left")
    dis["e_idx"] = cal.searchsorted(dis["end"].values, side="left")
    disp = {}
    for k, g in dis.groupby("stock_key"):
        g = g.sort_values("a_idx")
        disp[k] = (g["a_idx"].to_numpy(), g["e_idx"].to_numpy())

    n = len(df)
    pred = {v: np.zeros(n, bool) for v in VARIANTS}
    gap = np.full(n, np.nan)
    for key, g in df.loc[df["t_idx"] >= 0].groupby("stock_key", sort=False):
        order = np.argsort(g["t_idx"].to_numpy())
        pos = g.index.to_numpy()[order]
        t = g["t_idx"].to_numpy()[order]
        f1 = g["has_c1"].to_numpy()[order]
        f18 = g["has_c1to8"].to_numpy()[order]
        ind = g["in_disposal_period"].to_numpy()[order]
        t18 = t[f18]
        s1, s18 = set(t[f1].tolist()), set(t18.tolist())
        t18f = t[f18 & ~ind]
        s1f = set(t[f1 & ~ind].tolist())
        s18f = set(t18f.tolist())
        d = disp.get(key)
        for i, ti in enumerate(t.tolist()):
            cD = cE = NEG
            if d is not None:
                j = int(np.searchsorted(d[0], ti, side="left")) - 1
                if j >= 0:
                    cD, cE = int(d[0][j]), int(d[1][j])
                    gap[pos[i]] = ti - cE
            p = pos[i]
            pred["A_current"][p] = trig(ti, NEG, s1, s18, t18)
            pred["D_reset_at_prev_announce"][p] = trig(ti, cD, s1, s18, t18)
            pred["E_reset_at_prev_end"][p] = trig(ti, cE, s1, s18, t18)
            pred["F_ignore_days_in_disposal"][p] = trig(
                ti, NEG, s1f, s18f, t18f)

    scope = df["in_sample"] & df["ordinary_stock"] & ~df["in_disposal_period"]
    actual = df["same_day_disposal"]
    years = sorted(df.loc[scope, "year"].unique().tolist()) + ["ALL"]
    gbin = pd.cut(
        pd.Series(gap), [-10**9, 7, 14, 30, 60, 180, 10**9],
        labels=["0-7", "8-14", "15-30", "31-60", "61-180", ">180"]
    ).astype(object).fillna("no_prior")

    def counts(m, p):
        tp = int((m & p & actual).sum())
        fp = int((m & p & ~actual).sum())
        fn = int((m & ~p & actual).sum())
        return tp, fp, fn

    by_year, by_gap = [], []
    for v in VARIANTS:
        p = pd.Series(pred[v], index=df.index)
        for y in years:
            m = scope if y == "ALL" else scope & (df["year"] == y)
            tp, fp, fn = counts(m, p)
            by_year.append({
                "variant": v, "year": y, "TP": tp, "FP": fp, "FN": fn,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
            })
        for b in ["0-7", "8-14", "15-30", "31-60", "61-180", ">180", "no_prior"]:
            tp, fp, fn = counts(scope & (gbin == b), p)
            by_gap.append({"variant": v, "gap_bin": b,
                           "TP": tp, "FP": fp, "FN": fn})

    res = pd.DataFrame(by_year)
    res.to_csv(OUT / "a03_reset_variants_by_year.csv",
               index=False, encoding="utf-8-sig")
    pd.DataFrame(by_gap).to_csv(OUT / "a03_reset_variants_by_gap.csv",
                                index=False, encoding="utf-8-sig")
    print(res.query("year == 'ALL'").to_string(index=False))
    print("距離單位為交易日；輸出：", OUT)


if __name__ == "__main__":
    main()