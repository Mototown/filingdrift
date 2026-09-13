# FilingDrift

**GIBC V2 · Track 02 Applied (Medical Technology & Finance)**  
Historical filing + price features → flag whether the next 21 trading sessions dropped 10% or more.

This is a **research prototype**. It is **not** a financial product, **not** investment advice, **not** a broker, and **not** a live trading system. Everything runs on historical / bundled data. **No live orders. No real money.**

Repo: https://github.com/Mototown/filingdrift

## What it does

1. Takes quarterly fundamentals (revenue, earnings, balance sheet) plus a short MD&A-style excerpt.
2. Joins the last known daily close **on or before** the report date (`merge_asof`, backward only).
3. Builds features a person could audit: growth, leverage, margins, risk-word density, 21d/63d momentum, 21d volatility.
4. Trains a logistic baseline and a small histogram gradient-boosting classifier.
5. Scores only a **walk-forward** holdout (`asof_date >= 2024-01-01`). Training never sees that window.
6. Shows a paper simulation: what happened to names the model flagged vs names it did not.

The label is `y_drawdown_21d = 1` if forward 21-session return ≤ −10%.

## Why this is not a resubmit

New project for GIBC V2. Not RCS, not the Alpaca agent, not TradeSight. Detection is a trained classifier on a dated panel, not a hosted LLM.

## Data

No watsonx key. No paid API required.

```bash
python scripts/generate_sample.py
python scripts/train_model.py
```

That writes `data/dataset.csv`, `models/metrics.json`, and the charts.

The bundled sample plants a modest high-leverage → weaker next-21-session association so the offline demo has a learnable signal. That is a schema fixture, not a market claim. Live EDGAR will not look like this.

**Sources you may swap in**

- SEC EDGAR Company Facts / filings — U.S. government public data, no API key. User-Agent email required. https://data.sec.gov/
- Daily prices via yfinance or Stooq, downloaded once and frozen. Optional: `scripts/fetch_live.py`

## Setup

Python 3.10+.

```bash
git clone https://github.com/Mototown/filingdrift.git
cd filingdrift
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample.py
python scripts/train_model.py
streamlit run app.py
```

## Evaluation

```bash
python scripts/eval_model.py
```

On the bundled fixture (walk-forward 2024–2025):

- Baseline logistic ROC AUC ≈ 0.78
- Histogram GBDT ROC AUC ≈ 0.75, F1 ≈ 0.62
- Paper sim (threshold 0.55): flagged names had weaker average next-21d returns than unflagged names

Hardware: CPU-only, seconds.

## Track 02 compliance

- Empirical finance task on public-style data
- Predictive ML with a dated split and a baseline
- Simulation / historical only
- This README states it is not a product and not advice

## AI tools used

- **Grok (xAI)** assisted architecture, code, and this README.
- **IBM Bob / watsonx** were considered after a 7-day trial ended and are **not required**. The pipeline does not call them.

## Built With

Python, pandas, numpy, scikit-learn, joblib, streamlit, matplotlib, edgartools (optional), yfinance (optional), Grok.

## Team

Solo intended for **Moshi Odisho** (`Mototown`). Add every teammate on Devpost by real full name or they get no certificate.

## License

MIT.
