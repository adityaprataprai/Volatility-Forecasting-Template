#!/usr/bin/env python
"""Local validation (participants) and hidden-test scoring (organisers) - same code path.

    python challenge/evaluate.py --mode validate
    python challenge/evaluate.py --mode test --hidden ../release/hidden --submission-dir teams/alpha/submission
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from challenge import metrics as M          # noqa: E402
from challenge import worker as W           # noqa: E402


class SubmissionError(RuntimeError):
    pass


def run_one_origin(history, origin, submission_dir, python_exe=None, timeout=600, inproc=False):
    """Returns (Series of predictions, seconds). Raises SubmissionError on any failure."""
    t0 = time.time()
    try:
        if inproc:
            return W.normalize(W.load_predict(submission_dir)(history.copy(), pd.Timestamp(origin))), time.time() - t0
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            history.to_parquet(tmp / "history.parquet", index=False)
            cmd = [python_exe or sys.executable, str(HERE / "worker.py"),
                   "--submission-dir", str(submission_dir), "--history", str(tmp / "history.parquet"),
                   "--origin", pd.Timestamp(origin).strftime("%Y-%m-%d"), "--out", str(tmp / "pred.csv")]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                  cwd=str(Path(submission_dir).resolve().parent))
            if proc.returncode != 0:
                raise SubmissionError(proc.stderr.strip()[-1500:] or f"exit code {proc.returncode}")
            return pd.read_csv(tmp / "pred.csv", dtype={"Ticker": str}).set_index("Ticker")["pred"], time.time() - t0
    except subprocess.TimeoutExpired:
        raise SubmissionError(f"timed out after {timeout}s")
    except SubmissionError:
        raise
    except Exception as e:                       # in-process failures
        raise SubmissionError(f"{type(e).__name__}: {e}")


def build_context(mode, public_path, hidden_dir=None, val_origins=20):
    pub = M.load_panel(public_path)
    if mode == "validate":
        panel = pub
        cal = M.trading_calendar(panel)
        positions = M.validation_origin_positions(len(cal), val_origins)
    else:
        hid = M.load_panel(Path(hidden_dir) / "test_panel.parquet")
        panel = M.clean_panel(pd.concat([pub, hid], ignore_index=True))
        cal = M.trading_calendar(panel)
        start = int(cal.searchsorted(pub["Date"].max(), side="right")) - 1
        positions = M.forward_origin_positions(len(cal), start)
    V = M.daily_variance_matrix(panel, cal)
    return panel, cal, positions, M.forward_targets(V, positions)


def evaluate(mode, public_path, hidden_dir=None, submission_dir="submission", python_exe=None,
             timeout=600, val_origins=20, inproc=False, fail_fast=False, verbose=True):
    panel, cal, positions, truth = build_context(mode, public_path, hidden_dir, val_origins)
    preds, errors, secs = {}, {}, []
    for p in positions:
        origin = cal[p]
        if len(truth[origin]) == 0:
            continue
        hist = panel[panel["Date"] <= origin]
        try:
            preds[origin], dt = run_one_origin(hist, origin, submission_dir, python_exe, timeout, inproc)
            secs.append(dt)
            if verbose:
                print(f"  origin {origin.date()}  ok  {dt:5.1f}s  {len(preds[origin])} preds", flush=True)
        except SubmissionError as e:
            errors[str(origin.date())] = str(e)
            if verbose:
                print(f"  origin {origin.date()}  FAILED: {str(e).splitlines()[-1] if str(e) else e}", flush=True)
            if fail_fast:
                break
    res = M.score_predictions(preds, truth) if preds else dict(valid=False, qlike=float("nan"), rmse=float("nan"))
    res.update(mode=mode, first_origin=str(cal[positions[0]].date()), last_origin=str(cal[positions[-1]].date()),
               errors=errors, max_seconds_per_origin=max(secs) if secs else None,
               total_seconds=sum(secs) if secs else None)
    if not errors and not res.get("valid", False):
        res["errors"]["coverage"] = "fewer than 99% of scored (stock, origin) pairs received a finite prediction"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["validate", "test"], default="validate")
    ap.add_argument("--public", default=str(ROOT / "data" / "train_panel.parquet"))
    ap.add_argument("--hidden", default=None, help="organisers only: folder with test_panel.parquet")
    ap.add_argument("--submission-dir", default=str(ROOT / "submission"))
    ap.add_argument("--python", default=None, help="interpreter to run the submission with")
    ap.add_argument("--timeout", type=int, default=600, help="seconds allowed per origin")
    ap.add_argument("--val-origins", type=int, default=20)
    ap.add_argument("--inproc", action="store_true", help="debug: run in this process (no isolation)")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    a = ap.parse_args()
    if a.mode == "test" and not a.hidden:
        ap.error("--mode test needs --hidden")
    print(f"[{a.mode}] submission: {a.submission_dir}")
    res = evaluate(a.mode, a.public, a.hidden, a.submission_dir, a.python, a.timeout, a.val_origins,
                   a.inproc, fail_fast=(a.mode == "validate"))
    print("-" * 70)
    if res.get("n_pairs"):
        print(f"origins {res['first_origin']} -> {res['last_origin']}  ({res['n_origins']} origins, {res['n_pairs']:,} pairs)")
        print(f"QLIKE (primary, lower=better): {res['qlike']:.5f}")
        print(f"RMSE of log-variance (tie-break): {res['rmse']:.5f}   |   mean cross-sectional Spearman: {res['spearman_mean']:.3f}")
        print(f"coverage {res['coverage']:.2%}   valid: {'YES' if res['valid'] else 'NO'}   slowest origin: {res['max_seconds_per_origin']:.1f}s")
    for k, v in res["errors"].items():
        print(f"ERROR [{k}]: {v}")
    if a.mode == "validate":
        print(f"\nHonest validation tip: train with `--end {res['first_origin']}` so the model never sees the validation period.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    sys.exit(0 if res.get("valid") else 1)


if __name__ == "__main__":
    main()
