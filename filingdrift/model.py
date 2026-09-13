from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from filingdrift.config import FEATURE_COLS, FEATURE_LIST_PATH, MODEL_PATH, TARGET_COL


def make_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.08,
        max_iter=160,
        min_samples_leaf=15,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=42,
    )


def make_baseline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]
    )


def fit_predict(model, train: pd.DataFrame, test: pd.DataFrame):
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[FEATURE_COLS]
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return model, proba, pred


def save_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    FEATURE_LIST_PATH.write_text(json.dumps(FEATURE_COLS, indent=2))


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def permutation_importance_df(model, test: pd.DataFrame, n_repeats: int = 8) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        test[FEATURE_COLS],
        test[TARGET_COL],
        n_repeats=n_repeats,
        random_state=42,
        scoring="roc_auc",
    )
    return (
        pd.DataFrame(
            {
                "feature": FEATURE_COLS,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def top_reasons(row: pd.Series, importances: pd.DataFrame, n: int = 3) -> list[str]:
    reasons = []
    for feat in importances["feature"].head(6):
        val = float(row.get(feat, np.nan))
        if not np.isfinite(val):
            continue
        if feat in {"leverage", "vol_21d", "risk_keyword_density", "uncertainty_score"} and val > 0:
            reasons.append(f"{feat}={val:.3f} (elevated)")
        elif feat in {"revenue_yoy", "net_income_yoy", "margin", "mom_21d", "mom_63d"} and val < 0:
            reasons.append(f"{feat}={val:.3f} (weak)")
        elif feat == "current_ratio" and val < 1:
            reasons.append(f"{feat}={val:.3f} (thin coverage)")
        if len(reasons) >= n:
            break
    if not reasons:
        reasons.append("score driven by combined fundamentals + momentum, no single spike")
    return reasons[:n]
