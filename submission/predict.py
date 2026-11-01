"""BASELINE submission: HAR-style regression on log Garman-Klass variance.

CONTRACT (do not change the name/signature):
    predict(history: pd.DataFrame, origin: pd.Timestamp) -> pd.Series

    history : long-format rows with columns Date, Ticker, Open, High, Low, Close, Volume,
              containing ONLY dates <= origin (you can never see the future).
    origin  : the forecast origin (last date in `history`).
    returns : Series indexed by Ticker, one value per ticker that traded on `origin`:
              your forecast of  ln( mean Garman-Klass variance over the next 5 trading days ).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from submission.features import NAMES, features_at_origin

_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "har_coefs.json"
_COEF = json.loads(_ARTIFACT.read_text())      # loaded once per process


def predict(history: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    X = features_at_origin(history, origin)
    yhat = _COEF["intercept"] + sum(_COEF["coef"][n] * X[n] for n in NAMES)
    return yhat.fillna(yhat.median()).rename("pred")     # fallback for tickers with too little history
