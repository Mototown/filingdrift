"""Build a leakage-safe panel from fundamentals + prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from filingdrift.config import DATE_COL, FEATURE_COLS, TARGET_COL


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = prices.groupby("ticker", sort=False)
    prices["ret_1d"] = g["close"].pct_change()
    prices["mom_21d"] = g["close"].pct_change(21)
    prices["mom_63d"] = g["close"].pct_change(63)
    prices["vol_21d"] = g["ret_1d"].transform(lambda s: s.rolling(21).std())
    prices["fwd_21d_return"] = g["close"].shift(-21) / prices["close"] - 1.0
    prices[TARGET_COL] = (prices["fwd_21d_return"] <= -0.10).astype("float")
    return prices


def add_fundamental_features(fun: pd.DataFrame) -> pd.DataFrame:
    fun = fun.copy()
    fun["report_date"] = pd.to_datetime(fun["report_date"])
    fun = fun.sort_values(["ticker", "report_date"]).reset_index(drop=True)
    g = fun.groupby("ticker", sort=False)
    fun["revenue_yoy"] = g["revenue"].pct_change(4)
    fun["net_income_yoy"] = g["net_income"].pct_change(4)
    fun["asset_growth"] = g["assets"].pct_change(4)
    fun["leverage"] = fun["liabilities"] / fun["assets"].replace(0, np.nan)
    fun["current_ratio"] = fun["current_assets"] / fun["current_liabilities"].replace(0, np.nan)
    fun["margin"] = fun["net_income"] / fun["revenue"].replace(0, np.nan)
    if "mda_excerpt" in fun.columns:
        text = fun["mda_excerpt"].fillna("")
    else:
        text = pd.Series([""] * len(fun), index=fun.index)
    risk_words = r"risk|uncertain|litigation|impairment|going concern|downturn|liquidity"
    hedge_words = r"may|might|could|approximately|expect"
    n_words = text.str.split().str.len().clip(lower=1)
    fun["risk_keyword_density"] = text.str.lower().str.count(risk_words) / n_words
    fun["uncertainty_score"] = text.str.lower().str.count(hedge_words) / n_words
    return fun


def asof_join(fun: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Each quarter uses only prices known on report_date (no peeking)."""
    fun = fun.copy()
    prices = prices.copy()
    fun["report_date"] = pd.to_datetime(fun["report_date"])
    prices["date"] = pd.to_datetime(prices["date"])
    rows = []
    for ticker, fpart in fun.groupby("ticker"):
        ppart = (
            prices[prices["ticker"] == ticker]
            .sort_values("date")
            .drop(columns=["ticker"], errors="ignore")
        )
        if ppart.empty:
            continue
        fpart = fpart.copy()
        if "ticker" not in fpart.columns:
            fpart.insert(0, "ticker", ticker)
        merged = pd.merge_asof(
            fpart.sort_values("report_date"),
            ppart,
            left_on="report_date",
            right_on="date",
            direction="backward",
        )
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out[DATE_COL] = out["report_date"]
    keep = [
        "ticker",
        DATE_COL,
        "report_date",
        "revenue",
        "net_income",
        "assets",
        TARGET_COL,
        "fwd_21d_return",
        *FEATURE_COLS,
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].dropna(subset=FEATURE_COLS + [TARGET_COL])
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    return out.reset_index(drop=True)


def build_dataset(fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    fun = add_fundamental_features(fundamentals)
    px = add_price_features(prices)
    return asof_join(fun, px)
