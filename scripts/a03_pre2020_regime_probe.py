from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

ROOT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs")
OUT = ROOT / "a01_clause_parser"
SRC = OUT / "a01_attention_with_clauses.csv"
NEG = -10**9
YEARS = (2018, 2019, 2020, 2021)
ALL8 = set(range(1, 9))

VARIANTS = {"base_1to8": {"S": ALL8}}
for rule in ("r5", "r10", "r30"):
    VARIANTS[f"no_{rule}"] = {"S": ALL8, rule: False}
VARIANTS["no_r10_r30"] = {"S": ALL8, "r10": False, "r30": False}
for k in range(2, 9):
    VARIANTS[f"drop_clause_{k}"] = {"S": ALL8 - {k}}


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


def flags(ti, c, s1, s18, t18):
    r3 = ti - 2 > c and ti in s1 and (ti - 1) in s1 and (ti - 2) in s1
    if ti not in s18:
        return r3, False, False, False
    r5 = ti - 4 > c and all((ti - j) in s18 for j in range(5))
    hi = np.searchsorted(t18, ti, side="right")
    r10 = hi - np.searchsorted(t18, max(c + 1, ti - 9), side="left") >= 6
    r30 = hi - np.searchsorted(t18, max(c + 1, ti - 29), side="left") >= 12
    return r3, r5, bool(r10), bool(r30)


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

    market = {}
    try:
        sc = pd.DataFrame(data.get("security_categories"))
        print("security_categories 欄位：", list(sc.columns))
        id_col = next((c for c in ("stock_id", "symbol")
                       if c in sc.columns), None)
        mk_col = next((c for c in sc.columns
                       if "market" in str(c).lower() or "市場" in str(c)),
                      None)
        if id_col and mk_col:
            market = dict(zip(sc[id_col].map(raw_id).map(key_of),
                              sc[mk_col]))
    except Exception as exc:
        print("市場別資料不可用：", repr(exc))
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
    scope = (df["in_sample"] & df["ordinary_stock"]
             & ~df["in_disposal_period"]).to_numpy()
    evalmask = scope & df["year"].isin(YEARS).to_numpy()
    actual = df["same_day_disposal"].to_numpy()
    pred = {v: np.zeros(n, bool) for v in VARIANTS}
    sig = np.full(n, "", dtype=object)

    for key, g in df.loc[df["t_idx"] >= 0].groupby("stock_key", sort=False):
        order = np.argsort(g["t_idx"].to_numpy())
        pos = g.index.to_numpy()[order]
        if not evalmask[pos].any():
            continue
        t = g["t_idx"].to_numpy()[order]
        cs = g["cs"].to_numpy()[order]
        s1 = {int(x) for x, c in zip(t, cs) if 1 in c}
        t18, s18 = {}, {}
        for name, spec in VARIANTS.items():
            m = np.array([bool(spec["S"] & c) for c in cs])
            t18[name] = t[m]
            s18[name] = set(t[m].tolist())
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
            for name, spec in VARIANTS.items():
                r3, r5, r10, r30 = flags(ti, c, s1, s18[name], t18[name])
                pred[name][p] = (
                    (r3 and spec.get("r3", True))
                    or (r5 and spec.get("r5", True))
                    or (r10 and spec.get("r10", True))
                    or (r30 and spec.get("r30", True))
                )
                if name == "base_1to8":
                    sig[p] = "+".join(
                        k for k, v in (("r3", r3), ("r5", r5),
                                       ("r10", r10), ("r30", r30)) if v
                    )

    year = df["year"].to_numpy()
    rows = []
    for name in VARIANTS:
        for y in YEARS:
            m = evalmask & (year == y)
            tp = int((m & pred[name] & actual).sum())
            fp = int((m & pred[name] & ~actual).sum())
            fn = int((m & ~pred[name] & actual).sum())
            rows.append({
                "variant": name, "year": y, "TP": tp, "FP": fp, "FN": fn,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
            })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "a03_pre2020_variants.csv",
               index=False, encoding="utf-8-sig")

    df["sig"], df["pred"], df["actual"] = sig, pred["base_1to8"], actual
    sub = df.loc[evalmask & df["pred"]].copy()
    sub["outcome"] = np.where(sub["actual"], "TP", "FP")
    sub.groupby(["year", "outcome", "sig"]).size().reset_index(
        name="n"
    ).to_csv(OUT / "a03_regime_signature.csv",
             index=False, encoding="utf-8-sig")
    sub.groupby(["year", "outcome", "market_now"]).size().reset_index(
        name="n"
    ).to_csv(OUT / "a03_regime_market.csv",
             index=False, encoding="utf-8-sig")
    sub.loc[sub["outcome"] == "TP"].groupby("sig")["attention_date"].agg(
        ["min", "max", "size"]
    ).reset_index().to_csv(OUT / "a03_regime_first_dates.csv",
                           index=False, encoding="utf-8-sig")
    print(res.query("year in [2018, 2019]")
             .sort_values(["year", "FP"]).to_string(index=False))
    print("輸出：", OUT)


if __name__ == "__main__":
    main()