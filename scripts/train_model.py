#!/usr/bin/env python3
"""Train baseline + boosting on the walk-forward split and write artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from filingdrift.config import (
    CURVES_PATH,
    DATASET_PATH,
    DISCLAIMER,
    FEATURE_COLS,
    IMPORTANCE_PATH,
    TARGET_COL,
    TEST_START,
    TRAIN_END,
)
from filingdrift.metrics_lib import classification_report_dict, plot_importance, plot_roc, save_metrics
from filingdrift.model import fit_predict, make_baseline, make_boosting, permutation_importance_df, save_model
from filingdrift.simulate import paper_backtest
from filingdrift.split import walk_forward_split


def main() -> None:
    print(DISCLAIMER)
    df = pd.read_csv(DATASET_PATH)
    train, test = walk_forward_split(df)
    if train.empty or test.empty:
        raise SystemExit(f"empty split train={len(train)} test={len(test)}")

    baseline, b_proba, b_pred = fit_predict(make_baseline(), train, test)
    model, proba, pred = fit_predict(make_boosting(), train, test)

    payload = {
        "disclaimer": DISCLAIMER,
        "split": {"train_end": TRAIN_END, "test_start": TEST_START},
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": FEATURE_COLS,
        "baseline_logreg": classification_report_dict(test[TARGET_COL], b_proba, b_pred),
        "hist_gbdt": classification_report_dict(test[TARGET_COL], proba, pred),
        "paper_sim": paper_backtest(test, proba),
        "hardware": "CPU-only laptop / sandbox. Seconds, not GPU-hours.",
        "data": "Bundled sample from scripts/generate_sample.py. Optional live pull: scripts/fetch_live.py.",
    }
    save_metrics(payload)
    save_model(model)
    plot_roc(test[TARGET_COL], proba, "Held-out ROC — next-21d 10% drawdown", CURVES_PATH)
    imp = permutation_importance_df(model, test)
    plot_importance(imp, IMPORTANCE_PATH)
    print(json.dumps(payload, indent=2))
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
