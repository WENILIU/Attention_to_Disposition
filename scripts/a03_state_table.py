from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

ROOT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs")
OUT = ROOT / "a01_clause_parser"
SRC = OUT / "a01_attention_with_clauses.csv"
NEG = -10**9
C7_CUTOFF = pd.Timestamp("2019-05-01")
SAMPLE_START = pd.Timestamp("2018-01-01")
ALL8 = frozenset(range(1, 9))


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


def main():
    use = ["stock_key", "attention_date", "t_idx", "ordinary_stock",
           "clause_set"]
    df = pd.read_csv(SRC, usecols=use,
                     dtype={"stock_key": str, "clause_set": str},
                     encoding="utf-8-sig", low_memory=False)
    df["ordinary_stock"] = to_bool(df["ordinary_stock"])
    df["t_idx"] = df["t_idx"].astype(int)
    df["attention_date"] = pd.to_datetime(df["attention_date"])
    df["cs"] = df["clause_set"].fillna("").map(
        lambda x: frozenset(int(v) for v in x.split("|") if v)
    )
    old = (df["attention_date"] < C7_CUTOFF).to_numpy()
    df["has1"] = df["cs"].map(lambda c: 1 in c)
    df["has18"] = [
        bool(c & ((ALL8 - {7}) if o else ALL8))
        for c, o in zip(df["cs"], old)
    ]

    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()
    dis = pd.DataFrame(data.get("disposal_information")).copy()
    sym = "symbol" if "symbol" in dis.columns else "stock_id"
    dis["stock_key"] = dis[sym].map(raw_id).map(key_of)
    for col, name in (("date", "announce"), ("處置開始時間", "start"),
                      ("處置結束時間", "end")):
        dis[name] = pd.to_datetime(dis[col], errors="coerce").dt.normalize()
    dis = dis.dropna(subset=["stock_key", "announce", "start", "end"])
    dis["a_idx"] = cal.searchsorted(dis["announce"].values, side="left")
    dis["s_idx"] = cal.searchsorted(dis["start"].values, side="left")
    dis["e_idx"] = cal.searchsorted(dis["end"].values, side="left")
    disp = {}
    for k, g in dis.groupby("stock_key"):
        g = g.sort_values("a_idx")
        disp[k] = (g["a_idx"].to_numpy(), g["s_idx"].to_numpy(),
                   g["e_idx"].to_numpy())

    last_date = min(df["attention_date"].max(), dis["announce"].max())
    last_idx = int(cal.searchsorted(np.datetime64(last_date),
                                    side="right")) - 1
    sidx = int(cal.searchsorted(np.datetime64(SAMPLE_START), side="left"))
    empty = np.array([], dtype=int)

    frames = []
    for key, g in df.loc[df["t_idx"] >= 0].groupby("stock_key", sort=False):
        if not g["ordinary_stock"].any():
            continue
        tl = np.sort(g["t_idx"].to_numpy())
        t1 = np.sort(g.loc[g["has1"], "t_idx"].to_numpy())
        t18 = np.sort(g.loc[g["has18"], "t_idx"].to_numpy())
        days = np.unique((tl[:, None] + np.arange(0, 30)[None, :]).ravel())
        days = days[(days >= sidx) & (days + 1 <= last_idx)]
        if len(days) == 0:
            continue
        ann, st, en = disp.get(key, (empty, empty, empty))
        if len(ann):
            j = np.searchsorted(ann, days, side="right") - 1
            c = np.where(j >= 0, ann[np.maximum(j, 0)], NEG)
            announced_today = np.isin(days, ann)
            in_disp = ((days[:, None] >= st[None, :])
                       & (days[:, None] <= en[None, :])).any(axis=1)
        else:
            c = np.full(len(days), NEG)
            announced_today = np.zeros(len(days), bool)
            in_disp = np.zeros(len(days), bool)
        keep = ~announced_today & ~in_disp
        days, c = days[keep], c[keep]
        if len(days) == 0:
            continue

        a3 = np.isin(days, t1) & np.isin(days - 1, t1) & (days - 1 > c)
        a5 = (days - 3 > c)
        for q in range(4):
            a5 &= np.isin(days - q, t18)
        hi = np.searchsorted(t18, days, side="right")
        n10 = hi - np.searchsorted(t18, np.maximum(c + 1, days - 8),
                                   side="left")
        n30 = hi - np.searchsorted(t18, np.maximum(c + 1, days - 28),
                                   side="left")
        nxt = days + 1
        frames.append(pd.DataFrame({
            "stock_key": key, "day_idx": days, "date": cal[days],
            "listed_today": np.isin(days, tl),
            "n10": n10, "n30": n30,
            "a3": a3, "a5": a5, "a10": n10 >= 5, "a30": n30 >= 11,
            "listed_c1_tomorrow": np.isin(nxt, t1),
            "listed_c18_tomorrow": np.isin(nxt, t18),
            "disposal_tomorrow": np.isin(nxt, ann),
        }))

    st = pd.concat(frames, ignore_index=True)
    st["year"] = st["date"].dt.year
    st["armed"] = st[["a3", "a5", "a10", "a30"]].any(axis=1)
    st["armed_type"] = st[["a3", "a5", "a10", "a30"]].apply(
        lambda r: "+".join(k for k in ("a3", "a5", "a10", "a30") if r[k]),
        axis=1
    )
    st["fires_tomorrow"] = (
        (st["a3"] & st["listed_c1_tomorrow"])
        | ((st["a5"] | st["a10"] | st["a30"]) & st["listed_c18_tomorrow"])
    )
    st.to_csv(OUT / "a03_state_table.csv", index=False, encoding="utf-8-sig")

    def summarize(g):
        return pd.Series({
            "n_days": len(g),
            "p_disposal_tomorrow": g["disposal_tomorrow"].mean(),
            "p_listed_c18_tomorrow": g["listed_c18_tomorrow"].mean(),
            "p_fires_tomorrow": g["fires_tomorrow"].mean(),
            "n_disposal_tomorrow": int(g["disposal_tomorrow"].sum()),
        })

    overall = st.groupby("armed").apply(summarize).reset_index()
    by_type = st.loc[st["armed"]].groupby("armed_type").apply(
        summarize).reset_index()
    by_year = st.loc[st["armed"]].groupby("year").apply(
        summarize).reset_index()
    overall.to_csv(OUT / "a03_state_overall.csv", index=False,
                   encoding="utf-8-sig")
    by_type.to_csv(OUT / "a03_state_by_type.csv", index=False,
                   encoding="utf-8-sig")
    by_year.to_csv(OUT / "a03_state_by_year.csv", index=False,
                   encoding="utf-8-sig")
    agree = (st["fires_tomorrow"] == st["disposal_tomorrow"]).mean()
    print(overall.to_string(index=False))
    print("規則預測與實際隔日處置一致比例：", round(float(agree), 4))
    print("輸出：", OUT)


if __name__ == "__main__":
    main()