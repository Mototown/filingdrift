# Devpost draft — FilingDrift

**Track:** 02 Applied (Medical Technology & Finance)

## Tagline

Public-style filings and prices in, a dated drawdown flag out — simulation only.

## What we built

FilingDrift turns quarterly fundamentals and daily closes into a walk-forward classifier that estimates whether the next 21 sessions fell 10% or more. A Streamlit app inspects one ticker-quarter at a time.

## Disclaimer (paste on the Devpost page)

Research prototype only. Not a medical device, not a diagnostic, not a financial product, not investment advice. No live patients, no live orders, no real money.

## Built With

Python, pandas, scikit-learn, Streamlit, matplotlib, SEC EDGAR public data schema, Grok (xAI).

## Screenshots

1. Streamlit ticker inspector
2. assets/roc_curve.png (after training)
3. assets/feature_importance.png (after training)

## Demo video outline

1. Disclaimer on screen
2. generate_sample + train_model + streamlit run
3. Pick a ticker/date, show score vs realized return
4. metrics.json + ROC
5. dataset.py asof join (no shuffle leak)
