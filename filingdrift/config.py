from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

DATASET_PATH = DATA_DIR / "dataset.csv"
FUNDAMENTALS_PATH = DATA_DIR / "fundamentals.csv"
PRICES_PATH = DATA_DIR / "prices.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "drawdown_model.joblib"
FEATURE_LIST_PATH = MODELS_DIR / "feature_names.json"
CURVES_PATH = ASSETS_DIR / "roc_curve.png"
IMPORTANCE_PATH = ASSETS_DIR / "feature_importance.png"

TARGET_COL = "y_drawdown_21d"
DATE_COL = "asof_date"
ID_COL = "ticker"

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

FEATURE_COLS = [
    "revenue_yoy",
    "net_income_yoy",
    "asset_growth",
    "leverage",
    "current_ratio",
    "margin",
    "risk_keyword_density",
    "uncertainty_score",
    "mom_21d",
    "vol_21d",
    "mom_63d",
]

DISCLAIMER = (
    "RESEARCH PROTOTYPE ONLY. Not a financial product, not investment advice, "
    "and not a trading system. Simulation on historical / bundled data only. "
    "No live orders. No real money."
)
