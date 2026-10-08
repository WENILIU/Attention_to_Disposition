from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from finlab import data

OUT = Path(r"D:\AI專案\StockAgent\finlab\database\disposal_outputs"
           r"\a01_clause_parser")
SRC = OUT / "a03_state_table.csv"
FEATURES = ["ret6", "ret1", "dropoff", "vol_ratio", "close", "n10", "n30"]
OUTCOMES = ("disposal_tomorrow", "listed_c18_tomorrow")


def uni(df, feature, label):
    x = df[feature].replace([np.inf, -np.inf], np.nan)
    ok = x.notna()
    if ok.sum() < 50:
        return []
    q = pd.qcut(x[ok], 5, duplicates="drop")
    rows = []
    for b, g in df.loc[ok].groupby(q, observed=True):
        rows.append({
            "subset": label, "feature": feature, "bin": str(b), "n": len(g),
            **{f"p_{o}": float(g[o].mean()) for o in OUTCOMES},
        })
    return rows


def main():
    st = pd.read_csv(SRC, dtype={"stock_key": str},
                     encoding="utf-8-sig", low_memory=False)
    for c in ("a3", "a5", "a10", "a30", "armed", "listed_today",
              "listed_c1_tomorrow", "listed_c18_tomorrow",
              "disposal_tomorrow", "fires_tomorrow"):
        st[c] = st[c].astype(str).str.lower().eq("true")
    st = st.loc[st["armed"]].copy()
    st["date"] = pd.to_datetime(st["date"])

    close = pd.DataFrame(data.get("price:收盤價")).sort_index()
    volume = pd.DataFrame(data.get("price:成交股數")).sort_index()
    close.index = pd.DatetimeIndex(close.index).normalize()
    volume.index = pd.DatetimeIndex(volume.index).normalize()
    if not close.index.is_unique:
        raise ValueError("價格表日期有重複")
    volume = volume.reindex(index=close.index, columns=close.columns)

    d = st["day_idx"].astype(int).to_numpy()
    if not (close.index[d].to_numpy() == st["date"].to_numpy()).all():
        raise ValueError("day_idx 與價格表日期不一致，請回報")

    col_pos = {str(c): j for j, c in enumerate(close.columns)}
    st["j"] = st["stock_key"].map(col_pos)
    print("價格表找不到的武裝日：", int(st["j"].isna().sum()))
    st = st.dropna(subset=["j"]).copy()
    d = st["day_idx"].astype(int).to_numpy()
    j = st["j"].astype(int).to_numpy()

    C = close.to_numpy(float)
    V = volume.to_numpy(float)
    vma = volume.rolling(60, min_periods=40).mean().shift(1).to_numpy(float)
    ct = C[d, j]
    st["close"] = ct
    st["ret1"] = ct / C[d - 1, j] - 1
    st["ret6"] = ct / C[d - 6, j] - 1
    st["dropoff"] = C[d - 5, j] / ct
    st["vol_ratio"] = V[d, j] / vma[d, j]
    st.to_csv(OUT / "a04_features.csv", index=False, encoding="utf-8-sig")

    subsets = {
        "all_armed": pd.Series(True, index=st.index),
        "a3_any": st["a3"],
        "count_only": ~st["a3"],
    }
    rows = []
    for label, m in subsets.items():
        for f in FEATURES:
            rows += uni(st.loc[m], f, label)
    pd.DataFrame(rows).to_csv(OUT / "a04_univariate.csv",
                              index=False, encoding="utf-8-sig")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("沒有 scikit-learn，略過 AUC")
        return
    st["log_dropoff"] = np.log(st["dropoff"])
    st["log_vol"] = np.log(st["vol_ratio"])
    st["log_close"] = np.log(st["close"])
    sets = {
        "armed_type_only": ["a3", "a5", "a10", "a30"],
        "with_price_volume": ["a3", "a5", "a10", "a30", "ret6", "ret1",
                              "log_dropoff", "log_vol", "log_close",
                              "n10", "n30", "listed_today"],
    }
    allcols = sorted({c for v in sets.values() for c in v})
    ds = st.replace([np.inf, -np.inf], np.nan).dropna(subset=allcols)
    tr, te = ds[ds["year"] <= 2022], ds[ds["year"] >= 2023]
    lines = [f"train n={len(tr)}, test n={len(te)}"]
    for name, cols in sets.items():
        Xtr, Xte = tr[cols].astype(float), te[cols].astype(float)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        model = LogisticRegression(max_iter=2000).fit(
            (Xtr - mu) / sd, tr["disposal_tomorrow"])
        auc = roc_auc_score(te["disposal_tomorrow"],
                            model.predict_proba((Xte - mu) / sd)[:, 1])
        lines.append(f"{name}: test AUC = {auc:.4f}")
    (OUT / "a04_auc.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("輸出：", OUT)


if __name__ == "__main__":
    main()