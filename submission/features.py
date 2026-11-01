"""HAR-style features (Corsi 2009) used by the baseline. Replace or extend freely."""
import numpy as np
import pandas as pd

from challenge.metrics import daily_variance_matrix

EPS = 1e-6
NAMES = ["f1", "f5", "f22"]


def feature_matrices(V: pd.DataFrame) -> dict:
    """V: dates x tickers Garman-Klass daily variance -> dict of log-features (same shape)."""
    r1 = V
    r5 = V.rolling(5, min_periods=3).mean()
    r22 = V.rolling(22, min_periods=10).mean()
    return {"f1": np.log(r1 + EPS), "f5": np.log(r5 + EPS), "f22": np.log(r22 + EPS)}


def features_at_origin(history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """One row per ticker that has a bar on `origin`; columns f1, f5, f22 (NaN if not computable)."""
    present = history.loc[history["Date"] == origin, "Ticker"].unique()
    recent_dates = np.sort(history["Date"].unique())[-40:]
    recent = history[history["Date"].isin(recent_dates)]
    V = daily_variance_matrix(recent)
    F = feature_matrices(V)
    out = pd.DataFrame({k: F[k].loc[origin] for k in NAMES})
    return out.reindex(present)
