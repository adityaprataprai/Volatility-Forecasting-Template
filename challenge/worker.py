"""Runs a submission's predict() for ONE origin in its own process (called by evaluate.py).

The process receives only a parquet file with history up to and including the origin date,
so nothing from the future is ever on disk for the submission to read.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path


def load_predict(submission_dir):
    sub = Path(submission_dir).resolve()
    team_root, trusted_root = str(sub.parent), str(Path(__file__).resolve().parent.parent)
    for p in (trusted_root, team_root):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, trusted_root)
    sys.path.insert(0, team_root)          # the team's own package wins on name clashes
    mod = importlib.import_module(f"{sub.name}.predict")
    if not hasattr(mod, "predict"):
        raise AttributeError(f"{sub.name}/predict.py must define predict(history, origin)")
    return mod.predict


def normalize(out):
    import pandas as pd
    if isinstance(out, pd.DataFrame):
        out = out.iloc[:, 0]
    s = out.astype("float64") if isinstance(out, pd.Series) else pd.Series(out, dtype="float64")
    s.index = s.index.astype(str)
    s.index.name, s.name = "Ticker", "pred"
    return s


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--origin", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    predict = load_predict(a.submission_dir)
    history = pd.read_parquet(a.history)
    t0 = time.time()
    s = normalize(predict(history, pd.Timestamp(a.origin)))
    s.to_csv(a.out)
    print(f"worker: {len(s)} predictions in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
