#!/usr/bin/env python3
"""Print saved metrics so judges can verify without retraining."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filingdrift.config import DISCLAIMER, METRICS_PATH


def main() -> None:
    print(DISCLAIMER)
    print(METRICS_PATH.read_text())


if __name__ == "__main__":
    main()
