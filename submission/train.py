"""Fit the baseline and write submission/artifacts/har_coefs.json.

    python -m submission.train                       # train on all public data
    python -m submission.train --end 2026-01-30      # train only up to a date (honest validation)
    python -m submission.train --persistence         # reference 'persistence' baseline (no fitting)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from challenge import metrics as M
from submission.features import NAMES, feature_matrices

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "data" / "train_panel.parquet"))
    ap.add_argument("--end", default=None, help="use only data up to this date (inclusive)")
    ap.add_argument("--persistence", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "submission" / "artifacts" / "har_coefs.json"))
    a = ap.parse_args()

    if a.persistence:
        coefs = {"intercept": 0.0, "coef": {"f1": 0.0, "f5": 1.0, "f22": 0.0}, "note": "persistence"}
    else:
        panel = M.load_panel(a.panel)
        if a.end:
            panel = panel[panel["Date"] <= pd.Timestamp(a.end)]
        cal = M.trading_calendar(panel)
        V = M.daily_variance_matrix(panel, cal)
        Y = M.target_matrix(V)
        F = feature_matrices(V)
        cols = [F[n].stack() for n in NAMES] + [Y.stack()]
        D = pd.concat(cols, axis=1, keys=NAMES + ["y"]).dropna()
        X = np.column_stack([np.ones(len(D))] + [D[n].to_numpy() for n in NAMES])
        beta, *_ = np.linalg.lstsq(X, D["y"].to_numpy(), rcond=None)
        coefs = {"intercept": float(beta[0]), "coef": dict(zip(NAMES, map(float, beta[1:]))),
                 "trained_on_rows": int(len(D)), "trained_until": str(panel["Date"].max().date())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(coefs, indent=2))
    print("wrote", a.out, coefs)


if __name__ == "__main__":
    main()
