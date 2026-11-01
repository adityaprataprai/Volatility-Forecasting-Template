"""Format check on a tiny synthetic panel (no real data needed; runs in CI).

    python tests/check_submission.py
Exit code 0 = your submission follows the contract.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from challenge import metrics as M                         # noqa: E402
from challenge.evaluate import SubmissionError, run_one_origin   # noqa: E402
from challenge.synthetic import make_panel                 # noqa: E402


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    panel = M.clean_panel(make_panel(n_tickers=40, start="2025-06-02", end="2026-02-27", seed=7))
    origin = M.trading_calendar(panel)[-10]
    hist = panel[panel["Date"] <= origin]
    sub = ROOT / "submission"
    if not (sub / "predict.py").exists():
        fail("submission/predict.py not found")
    try:
        p1, secs = run_one_origin(hist, origin, sub, timeout=300)
        p2, _ = run_one_origin(hist, origin, sub, timeout=300)
    except SubmissionError as e:
        fail(f"predict() crashed:\n{e}")
    expected = set(hist.loc[hist["Date"] == origin, "Ticker"])
    got = set(p1.index)
    if len(p1.index) != len(got):
        fail("duplicate tickers in the returned Series")
    if len(expected & got) < 0.99 * len(expected):
        fail(f"only {len(expected & got)}/{len(expected)} tickers present on the origin date were predicted")
    if not np.isfinite(p1.reindex(sorted(expected & got))).all():
        fail("NaN/inf predictions for tickers that were returned")
    if not np.allclose(p1.sort_index(), p2.sort_index(), equal_nan=True, atol=1e-8):
        fail("predictions are not deterministic (set seeds / avoid nondeterministic ops)")
    lo, hi = M.PRED_CLIP
    outside = ((p1 < lo) | (p1 > hi)).mean()
    print(f"OK: {len(p1)} predictions, {secs:.1f}s, deterministic. "
          f"Range [{p1.min():.2f}, {p1.max():.2f}] (clipped to [{lo}, {hi}] when scored; {outside:.1%} outside).")
    if outside > 0.2:
        print("WARNING: many predictions fall outside the clip range - did you predict log-variance (not volatility)?")


if __name__ == "__main__":
    main()
