from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(
    r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
    r"\a00_attention_master_v2"
)
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_START = pd.Timestamp("2018-01-01")
WINDOWS = (1, 5, 10, 20)


def find_col(df, names, label):
    found = next((c for c in names if c in df.columns), None)
    if found is None:
        raise ValueError(f"{label} 找不到 {names}；實際={list(df.columns)}")
    return found


def raw_id(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    return s[:-2] if s.endswith(".0") else s


def key_of(raw):
    if raw is None:
        return None
    return (raw.lstrip("0") or "0") if raw.isdigit() else raw


def is_ordinary(raw):
    return bool(raw) and raw.isdigit() and len(raw) == 4 and 1000 <= int(raw) <= 9999


def main():
    att = pd.DataFrame(data.get("trading_attention")).copy()
    dis = pd.DataFrame(data.get("disposal_information")).copy()
    close = data.get("price:收盤價")
    cal = pd.DatetimeIndex(close.index).normalize().unique().sort_values()

    a_sym = find_col(att, ["symbol", "stock_id"], "注意股代號")
    d_sym = find_col(dis, ["symbol", "stock_id"], "處置股代號")
    a_date = find_col(att, ["date"], "注意日")
    d_date = find_col(dis, ["date"], "處置公布日")
    d_start = find_col(dis, ["處置開始時間"], "處置起日")
    d_end = find_col(dis, ["處置結束時間"], "處置迄日")

    mismatch = {
        "attention_symbol_ne_stock_id": int(
            (att["symbol"].astype(str) != att["stock_id"].astype(str)).sum()
        ) if {"symbol", "stock_id"} <= set(att.columns) else None,
        "disposal_symbol_ne_stock_id": int(
            (dis["symbol"].astype(str) != dis["stock_id"].astype(str)).sum()
        ) if {"symbol", "stock_id"} <= set(dis.columns) else None,
    }

    att["raw_id"] = att[a_sym].map(raw_id)
    att["stock_key"] = att["raw_id"].map(key_of)
    att["attention_date"] = pd.to_datetime(
        att[a_date], errors="coerce"
    ).dt.normalize()
    att = att.dropna(subset=["stock_key", "attention_date"])

    ev = (
        att.sort_values(["stock_key", "attention_date"])
        .drop_duplicates(["stock_key", "attention_date"])
        .reset_index(drop=True)
    )
    ev["t_idx"] = cal.get_indexer(ev["attention_date"])

    dis["raw_id"] = dis[d_sym].map(raw_id)
    dis["stock_key"] = dis["raw_id"].map(key_of)
    dis["announce"] = pd.to_datetime(dis[d_date], errors="coerce").dt.normalize()
    dis["start"] = pd.to_datetime(dis[d_start], errors="coerce").dt.normalize()
    dis["end"] = pd.to_datetime(dis[d_end], errors="coerce").dt.normalize()
    dis = dis.dropna(subset=["stock_key", "announce", "start", "end"]).copy()
    dis["announce_idx"] = cal.searchsorted(dis["announce"].values, side="left")

    n = len(ev)
    cols = {k: np.full(n, np.nan) for k in
            ("streak", "c10", "c30", "days_to_next")}
    flags = {k: np.zeros(n, bool) for k in ("same_day", "in_disposal")}
    disp_groups = {
        k: g.sort_values("announce") for k, g in dis.groupby("stock_key")
    }

    for key, g in ev.groupby("stock_key", sort=False):
        pos = g.index.to_numpy()
        t = g["t_idx"].to_numpy()
        dates = g["attention_date"].to_numpy()
        ok = t >= 0
        tv, pv = t[ok], pos[ok]
        if len(tv):
            s = np.ones(len(tv))
            for i in range(1, len(tv)):
                if tv[i] == tv[i - 1] + 1:
                    s[i] = s[i - 1] + 1
            ar = np.arange(len(tv))
            cols["streak"][pv] = s
            cols["c10"][pv] = ar - np.searchsorted(tv, tv - 9, side="left") + 1
            cols["c30"][pv] = ar - np.searchsorted(tv, tv - 29, side="left") + 1

        dg = disp_groups.get(key)
        if dg is None:
            continue
        ann = dg["announce"].to_numpy()
        st = dg["start"].to_numpy()
        en = dg["end"].to_numpy()
        ai = dg["announce_idx"].to_numpy()
        flags["same_day"][pos] = (dates[:, None] == ann[None, :]).any(axis=1)
        flags["in_disposal"][pos] = (
            (dates[:, None] >= st[None, :]) & (dates[:, None] <= en[None, :])
        ).any(axis=1)
        nxt = np.searchsorted(ann, dates, side="right")
        use = (nxt < len(ann)) & ok
        d2 = np.full(len(pos), np.nan)
        d2[use] = ai[nxt[use]] - t[use]
        cols["days_to_next"][pos] = d2

    for k, v in cols.items():
        ev[k] = v
    ev["same_day_disposal"] = flags["same_day"]
    ev["in_disposal_period"] = flags["in_disposal"]
    ev["ordinary_stock"] = ev["raw_id"].map(is_ordinary)
    ev["in_sample"] = (ev["attention_date"] >= SAMPLE_START) & (ev["t_idx"] >= 0)

    att_max, dis_max = ev["attention_date"].max(), dis["announce"].max()
    last_idx = cal.searchsorted(
        np.datetime64(min(att_max, dis_max)), side="right"
    ) - 1
    for k in WINDOWS:
        observed = (ev["t_idx"] >= 0) & (ev["t_idx"] + k <= last_idx)
        hit = ev["days_to_next"].between(1, k)
        ev[f"observed_{k}d"] = observed
        ev[f"disposed_within_{k}d"] = np.where(observed, hit.astype(float), np.nan)

    m1 = ev["in_sample"]
    m2 = m1 & ev["ordinary_stock"]
    m3 = m2 & ~ev["same_day_disposal"]
    m4 = m3 & ~ev["in_disposal_period"]
    ev["eligible_risk_set"] = m4

    funnel = pd.DataFrame([
        ("all attention stock-dates", len(ev)),
        ("2018+ and trading day", int(m1.sum())),
        ("+ ordinary stock 1000-9999", int(m2.sum())),
        ("- same-day disposal announcement", int(m3.sum())),
        ("- already in disposal period (eligible)", int(m4.sum())),
    ], columns=["step", "n_events"])
    funnel.to_csv(OUT / "a00_v2_funnel.csv", index=False, encoding="utf-8-sig")

    def rates(mask, col, clip):
        sub = ev.loc[mask].copy()
        sub["bucket"] = sub[col].clip(upper=clip)
        rows = []
        for b, g in sub.groupby("bucket"):
            row = {"feature": col, "bucket": b, "n_events": len(g)}
            for k in WINDOWS:
                o = g[f"observed_{k}d"]
                row[f"n_{k}d"] = int(o.sum())
                row[f"rate_{k}d"] = (
                    g.loc[o, f"disposed_within_{k}d"].mean() if o.any() else np.nan
                )
            rows.append(row)
        return pd.DataFrame(rows)

    pd.concat([
        rates(m4, "streak", 6),
        rates(m4, "c10", 7),
        rates(m4, "c30", 13),
    ]).to_csv(OUT / "a00_v2_rates_by_state.csv", index=False,
              encoding="utf-8-sig")

    keys = set(zip(ev["stock_key"], ev["attention_date"]))
    dis["has_same_day_attention"] = [
        (k, d) in keys for k, d in zip(dis["stock_key"], dis["announce"])
    ]
    missing = dis.loc[~dis["has_same_day_attention"]]
    cond_cols = [c for c in ("處置條件", "處置措施") if c in dis.columns]
    missing[["stock_key", "announce"] + cond_cols].to_csv(
        OUT / "a00_v2_disposal_without_same_day_attention.csv",
        index=False, encoding="utf-8-sig"
    )

    pd.Series(mismatch).to_csv(OUT / "a00_v2_id_mismatch.csv", header=["n"])
    ev.to_csv(OUT / "a00_v2_attention_master.csv", index=False,
              encoding="utf-8-sig")

    print(funnel.to_string(index=False))
    print("處置列數：", len(dis), " 無同日注意：", len(missing))
    if cond_cols:
        print(missing[cond_cols[0]].value_counts().head(8).to_string())
    print("symbol/stock_id 不一致：", mismatch)
    print("輸出：", OUT)


if __name__ == "__main__":
    main()