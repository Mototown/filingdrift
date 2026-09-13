"""Time-based split. Never shuffle a panel with a forward-looking label."""

from __future__ import annotations

import pandas as pd

from filingdrift.config import DATE_COL, TEST_START, TRAIN_END


def walk_forward_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    train = df[df[DATE_COL] <= pd.Timestamp(TRAIN_END)].copy()
    test = df[df[DATE_COL] >= pd.Timestamp(TEST_START)].copy()
    return train, test
