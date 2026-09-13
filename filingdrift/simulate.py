"""Paper portfolio on the held-out window. No live orders."""

from __future__ import annotations

import pandas as pd

from filingdrift.config import TARGET_COL


def paper_backtest(test: pd.DataFrame, proba, threshold: float = 0.55) -> dict:
    frame = test.copy()
    frame["proba"] = proba
    frame["flag"] = (frame["proba"] >= threshold).astype(int)
    if "fwd_21d_return" not in frame.columns:
        return {"error": "missing fwd_21d_return"}

    hold = frame.loc[frame["flag"] == 0, "fwd_21d_return"]
    avoid = frame.loc[frame["flag"] == 1, "fwd_21d_return"]
    always = frame["fwd_21d_return"]
    flagged_hits = frame.loc[frame["flag"] == 1, TARGET_COL]

    def _mean(s):
        return float(s.mean()) if len(s) else None

    return {
        "threshold": threshold,
        "n_test": int(len(frame)),
        "n_flagged": int(frame["flag"].sum()),
        "hit_rate_among_flags": float(flagged_hits.mean()) if len(flagged_hits) else None,
        "mean_fwd_21d_all": _mean(always),
        "mean_fwd_21d_held": _mean(hold),
        "mean_fwd_21d_avoided": _mean(avoid),
        "note": "Historical simulation only. Costs, borrow, and slippage omitted.",
    }
