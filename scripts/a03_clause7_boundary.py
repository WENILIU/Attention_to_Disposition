from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

ROOT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs")
OUT = ROOT / "a01_clause_parser"
SRC = OUT / "a01_attention_with_clauses.csv"
NEG = -10**9
YEARS = tuple(range(2018, 2025))
ALL8 = frozenset(range(1, 9))
VARIANTS = {"base_1to8": ALL8, "drop_clause_7": ALL8 - {7}}


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


def hit(ti, c, s1, s18, t18):
    if ti - 2 > c and ti in s1 and (ti - 1) in s1 and (ti - 2) in s1:
        return True
    if ti not in s18:
        return False
    if ti - 4 > c and all((ti - j) in s18 for j in range(5)):
        return True
    hi = np.searchsorted(t18, ti, side="right")
    if hi - np.searchsorted(t18, max(c + 1, ti - 9), side="left") >= 6:
        return True
    return hi - np.searchsorted(t18, max(c + 1, ti - 29), side="left") >= 12


def main():
    use = ["stock_key", "attention_date", "t_idx", "year",
           "same_day_disposal", "in_disposal_period", "ordinary_stock",
           "in_sample", "clause_set"]
    df = pd.read_csv(SRC, usecols=use,
                     dtype={"stock_key": str, "clause_set": str},
                     encoding="utf-8-sig", low_memory=False)
    for c in ("same_day_disposal", "in_disposal_period",
              "ordinary_stock", "in_sample"):
        df[c] = to_bool(df[c])
    df["t_idx"] = df["t_idx"].astype(int)
    df["attention_date"] = pd.to_datetime(df["attention_date"])
    df["cs"] = df["clause_set"].fillna("").map(
        lambda x: frozenset(int(v) for v in x.split("|") if v)
    )

    sc = pd.DataFrame(data.get("security_categories"))
    market = dict(zip(sc["stock_id"].map(raw_id).map(key_of), sc["market"]))
    df["market_now"] = df["stock_key"].map(market).fillna("unknown")

    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    dis = pd.DataFrame(data.get("disposal_information")).copy()
    sym = "symbol" if "symbol" in dis.columns else "stock_id"
    dis["stock_key"] = dis[sym].map(raw_id).map(key_of)
    dis["announce"] = pd.to_datetime(dis["date"], errors="coerce").dt.normalize()
    dis = dis.dropna(subset=["stock_key", "announce"])
    dis["a_idx"] = cal.searchsorted(dis["announce"].values, side="left")
    disp = {k: np.sort(g["a_idx"].to_numpy())
            for k, g in dis.groupby("stock_key")}

    n = len(df)
    evalmask = (df["in_sample"] & df["ordinary_stock"]
                & ~df["in_disposal_period"]
                & df["year"].isin(YEARS)).to_numpy()
    actual = df["same_day_disposal"].to_numpy()
    pred = {v: np.zeros(n, bool) for v in VARIANTS}

    for key, g in df.loc[df["t_idx"] >= 0].groupby("stock_key", sort=False):
        order = np.argsort(g["t_idx"].to_numpy())
        pos = g.index.to_numpy()[order]
        if not evalmask[pos].any():
            continue
        t = g["t_idx"].to_numpy()[order]
        cs = g["cs"].to_numpy()[order]
        s1 = {int(x) for x, c in zip(t, cs) if 1 in c}
        t18, s18 = {}, {}
        for name, S in VARIANTS.items():
            m = np.array([bool(S & c) for c in cs])
            t18[name], s18[name] = t[m], set(t[m].tolist())
        d = disp.get(key)
        for i, ti in enumerate(t.tolist()):
            p = pos[i]
            if not evalmask[p]:
                continue
            c = NEG
            if d is not None:
                j = int(np.searchsorted(d, ti, side="left")) - 1
                if j >= 0:
                    c = int(d[j])
            for name in VARIANTS:
                pred[name][p] = hit(ti, c, s1, s18[name], t18[name])

    year = df["year"].to_numpy()
    mk = df["market_now"].to_numpy()
    rows = []
    for name in VARIANTS:
        for y in YEARS:
            for market_name in ("sii", "otc"):
                m = evalmask & (year == y) & (mk == market_name)
                tp = int((m & pred[name] & actual).sum())
                fp = int((m & pred[name] & ~actual).sum())
                fn = int((m & ~pred[name] & actual).sum())
                rows.append({
                    "variant": name, "year": y, "market_now": market_name,
                    "TP": tp, "FP": fp, "FN": fn,
                    "precision": tp / (tp + fp) if tp + fp else np.nan,
                    "recall": tp / (tp + fn) if tp + fn else np.nan,
                })
    pd.DataFrame(rows).to_csv(OUT / "a03_c7_by_year_market.csv",
                              index=False, encoding="utf-8-sig")

    dep = evalmask & pred["base_1to8"] & ~pred["drop_clause_7"]
    sub = df.loc[dep].copy()
    sub["outcome"] = np.where(actual[dep], "TP", "FP")
    sub["ym"] = sub["attention_date"].dt.to_period("M").astype(str)
    sub.groupby(["ym", "market_now", "outcome"]).size().reset_index(
        name="n"
    ).to_csv(OUT / "a03_c7_dependent_by_month.csv",
             index=False, encoding="utf-8-sig")
    print("只靠第七款才觸發的預測共", int(dep.sum()), "筆")
    print(sub.groupby(["year", "outcome"]).size().to_string())
    print("輸出：", OUT)


if __name__ == "__main__":
    main()