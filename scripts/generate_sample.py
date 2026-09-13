#!/usr/bin/env python3
"""Create bundled sample CSVs so the demo runs with no API keys."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filingdrift.config import DATA_DIR, DATASET_PATH, FUNDAMENTALS_PATH, PRICES_PATH
from filingdrift.dataset import build_dataset

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "GS", "WFC",
    "XOM", "CVX", "JNJ", "PFE", "UNH", "WMT", "TGT", "HD", "PG", "KO",
    "PEP", "DIS", "NFLX", "AMD", "INTC", "IBM", "ORCL", "CRM", "CAT", "BA",
    "GE", "F", "GM", "T", "VZ", "MRK", "ABBV", "LLY", "COST", "NKE",
]

MDA = [
    "Management discusses competitive pressure, supply chain risk, and uncertain demand.",
    "Results were in line with expectations. Liquidity remains adequate for planned operations.",
    "We may face litigation, impairment risk, and a downturn in discretionary spending.",
    "Guidance could change. Customers might delay orders. Approximately stable margins.",
    "Going concern is not in doubt. Cash generation supports dividends and buybacks.",
    "Material weakness remediation is ongoing. Cybersecurity risk remains elevated.",
]


def main() -> None:
    rng = np.random.default_rng(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    quarters = pd.date_range("2019-03-31", "2025-12-31", freq="QE")
    fun_rows = []
    for i, ticker in enumerate(TICKERS):
        rev = 8_000 + i * 400 + rng.normal(0, 200)
        ni = rev * (0.08 + 0.02 * np.sin(i))
        assets = rev * 3.2
        for q, dt in enumerate(quarters):
            shock = rng.normal(0.015, 0.04)
            if rng.random() < 0.06:
                shock -= 0.18
            rev = max(400, rev * (1 + shock))
            ni = ni * (1 + shock * 1.4) + rng.normal(0, 40)
            assets = max(rev, assets * (1 + shock * 0.5) + rng.normal(0, 80))
            liab = assets * (0.35 + 0.15 * abs(np.sin(i + q / 3)))
            ca = assets * (0.28 + 0.05 * rng.random())
            cl = liab * (0.45 + 0.1 * rng.random())
            fun_rows.append(
                {
                    "ticker": ticker,
                    "report_date": dt.date().isoformat(),
                    "revenue": round(rev, 2),
                    "net_income": round(ni, 2),
                    "assets": round(assets, 2),
                    "liabilities": round(liab, 2),
                    "current_assets": round(ca, 2),
                    "current_liabilities": round(cl, 2),
                    "mda_excerpt": MDA[(i + q) % len(MDA)],
                }
            )
    fundamentals = pd.DataFrame(fun_rows)

    price_rows = []
    dates = pd.bdate_range("2018-10-01", "2026-03-31")
    for i, ticker in enumerate(TICKERS):
        price = 40 + i * 3
        drift = 0.00025 + 0.00005 * np.sin(i)
        for d in dates:
            vol = 0.012 + 0.004 * abs(np.sin((d.timetuple().tm_yday + i) / 18))
            r = rng.normal(drift, vol)
            if rng.random() < 0.012:
                r -= abs(rng.normal(0.07, 0.03))
            price = max(2.0, price * (1 + r))
            price_rows.append({"ticker": ticker, "date": d.date().isoformat(), "close": round(price, 4)})
    prices = pd.DataFrame(price_rows)
    prices["date"] = pd.to_datetime(prices["date"])

    # Planted demo signal: high leverage often precedes a weak 21-session window.
    # Live EDGAR will not look like this. Documented in README.
    fun_tmp = fundamentals.copy()
    fun_tmp["report_date"] = pd.to_datetime(fun_tmp["report_date"])
    fun_tmp["lev"] = fun_tmp["liabilities"] / fun_tmp["assets"]
    lev_cut = fun_tmp["lev"].quantile(0.80)
    stressed = fun_tmp[fun_tmp["lev"] >= lev_cut]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    for rec in stressed.itertuples(index=False):
        mask = (
            (prices["ticker"] == rec.ticker)
            & (prices["date"] > rec.report_date)
            & (prices["date"] <= rec.report_date + pd.Timedelta(days=32))
        )
        idx = prices.index[mask]
        if len(idx) == 0:
            continue
        closes = prices.loc[idx, "close"].to_numpy(dtype=float)
        shock = np.linspace(0.0, -0.13, num=len(closes))
        prices.loc[idx, "close"] = np.maximum(2.0, closes * (1.0 + shock))
    prices["date"] = prices["date"].dt.date.astype(str)

    dataset = build_dataset(fundamentals, prices)
    fundamentals.to_csv(FUNDAMENTALS_PATH, index=False)
    prices.to_csv(PRICES_PATH, index=False)
    dataset.to_csv(DATASET_PATH, index=False)
    print(f"fundamentals {fundamentals.shape} -> {FUNDAMENTALS_PATH}")
    print(f"prices        {prices.shape} -> {PRICES_PATH}")
    print(f"dataset       {dataset.shape} -> {DATASET_PATH}")
    print(dataset["y_drawdown_21d"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
