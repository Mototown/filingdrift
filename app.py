#!/usr/bin/env python3
"""Streamlit demo. Historical simulation only."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from filingdrift.config import (
    CURVES_PATH,
    DATASET_PATH,
    DISCLAIMER,
    FEATURE_COLS,
    IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
)
from filingdrift.model import load_model, permutation_importance_df, top_reasons
from filingdrift.split import walk_forward_split

st.set_page_config(page_title="FilingDrift", layout="wide")
st.markdown(
    f"<div style='background:#111;color:#f3f3f3;padding:10px 14px;border-radius:6px'>{DISCLAIMER}</div>",
    unsafe_allow_html=True,
)
st.title("FilingDrift")
st.caption("GIBC V2 Track 02 — fundamentals + prices → 21-day 10% drawdown flag")

if not DATASET_PATH.exists() or not MODEL_PATH.exists():
    st.error("Run `python scripts/generate_sample.py` then `python scripts/train_model.py` first.")
    st.stop()

df = pd.read_csv(DATASET_PATH)
train, test = walk_forward_split(df)
model = load_model()
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

left, right = st.columns((1.2, 1))
with left:
    st.subheader("Inspect a held-out quarter")
    tickers = sorted(test["ticker"].unique())
    ticker = st.selectbox("Ticker", tickers)
    subset = test[test["ticker"] == ticker].sort_values("asof_date")
    asof = st.selectbox("As-of date (report date)", list(subset["asof_date"]))
    row = subset[subset["asof_date"] == asof].iloc[0]
    X = row[FEATURE_COLS].to_frame().T
    proba = float(model.predict_proba(X)[0, 1])
    st.metric("P(next 21 sessions drop ≥ 10%)", f"{proba:.1%}")
    st.metric("Actual next-21d return (history)", f"{float(row['fwd_21d_return']):+.1%}")
    st.metric("Actual drawdown label", "YES" if int(row["y_drawdown_21d"]) == 1 else "no")
    try:
        imp = permutation_importance_df(model, test, n_repeats=4)
        reasons = top_reasons(row, imp)
    except Exception:
        reasons = ["see feature table"]
    st.write("Why this score")
    for r in reasons:
        st.write(f"- {r}")
    st.dataframe(X.T.rename(columns={X.index[0]: "value"}), use_container_width=True)

with right:
    st.subheader("Walk-forward test")
    gbdt = metrics.get("hist_gbdt", {})
    base = metrics.get("baseline_logreg", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Model ROC AUC", "—" if gbdt.get("roc_auc") is None else f"{gbdt['roc_auc']:.3f}")
    c2.metric("Baseline ROC AUC", "—" if base.get("roc_auc") is None else f"{base['roc_auc']:.3f}")
    c3.metric("Test rows", str(metrics.get("n_test", len(test))))
    st.json(metrics.get("paper_sim", {}))
    if CURVES_PATH.exists():
        st.image(str(CURVES_PATH), caption="Held-out ROC")
    if IMPORTANCE_PATH.exists():
        st.image(str(IMPORTANCE_PATH), caption="Permutation importance")

st.divider()
st.subheader("Panel (test window)")
show = test.copy()
show["proba"] = model.predict_proba(show[FEATURE_COLS])[:, 1]
st.dataframe(
    show[["ticker", "asof_date", "fwd_21d_return", "y_drawdown_21d", "proba"] + FEATURE_COLS].reset_index(drop=True),
    use_container_width=True,
    height=320,
)
st.caption("Research prototype. Not financial advice. No live money.")
