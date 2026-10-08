from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
           r"\a01_clause_parser")
SRC = OUT / "a04_features.csv"
LIMIT = 0.10
LABELS = ["unreachable", "needs_favorable_move",
          "tolerates_small_adverse", "certain_even_limit_move"]


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
    st = pd.read_csv(SRC, dtype={"stock_key": str},
                     encoding="utf-8-sig", low_memory=False)
    for c in ("a3", "disposal_tomorrow", "listed_c1_tomorrow"):
        st[c] = st[c].astype(str).str.lower().eq("true")
    st = (st.loc[st["a3"]]
          .replace([np.inf, -np.inf], np.nan)
          .dropna(subset=["ret6", "dropoff"]).copy())

    sc = pd.DataFrame(data.get("security_categories"))
    market = dict(zip(sc["stock_id"].map(raw_id).map(key_of), sc["market"]))
    st["market_now"] = st["stock_key"].map(market).fillna("unknown")
    st["direction"] = np.where(st["ret6"] >= 0, "up", "down")
    st["era"] = np.where(st["year"] <= 2019, "2018-19", "2020+")

    rows, rows_m = [], []
    for label, th_map in (("sii32_otc30", {"sii": 0.32, "otc": 0.30}),
                          ("all32", {"sii": 0.32, "otc": 0.32})):
        theta = st["market_now"].map(th_map).fillna(0.32).to_numpy()
        d = st["dropoff"].to_numpy()
        tol = np.where(st["direction"].to_numpy() == "up",
                       1 - (1 + theta) * d, (1 - theta) * d - 1)
        cls = pd.cut(tol, [-np.inf, -LIMIT, 0, LIMIT, np.inf],
                     labels=LABELS, right=False)
        tmp = st.assign(tol=tol, tol_class=cls)
        for (era, direction, k), g in tmp.groupby(
                ["era", "direction", "tol_class"], observed=True):
            rows.append({
                "theta_set": label, "era": era, "direction": direction,
                "tol_class": str(k), "n": len(g),
                "p_disposal_tomorrow": g["disposal_tomorrow"].mean(),
                "p_listed_c1_tomorrow": g["listed_c1_tomorrow"].mean(),
            })
        for (mk, direction, k), g in tmp.groupby(
                ["market_now", "direction", "tol_class"], observed=True):
            rows_m.append({
                "theta_set": label, "market_now": mk,
                "direction": direction, "tol_class": str(k), "n": len(g),
                "p_disposal_tomorrow": g["disposal_tomorrow"].mean(),
            })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "a04b_tolerance_by_era.csv",
               index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_m).to_csv(OUT / "a04b_tolerance_by_market.csv",
                                index=False, encoding="utf-8-sig")
    print(res.query("theta_set == 'sii32_otc30'").to_string(index=False))
    print("輸出：", OUT)


if __name__ == "__main__":
    main()