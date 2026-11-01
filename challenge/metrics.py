"""Target construction and scoring for the Nifty Volatility Forecasting Challenge.

Everything that defines "the right answer" lives here, so the organisers' scorer
and the participants' local validation use identical code.

TARGET (per stock i, forecast origin t, i.e. after the close of day t):
    gk_d  = 0.5*ln(H/L)^2 - (2 ln2 - 1)*ln(C/O)^2          (Garman-Klass daily variance)
    y_it  = ln( mean_{k=1..5} gk_{i,t+k} )                  (log 5-day forward variance)

SCORE (lower is better):
    QLIKE = exp(y - yhat) - (y - yhat) - 1,  averaged over all scored (stock, origin) pairs
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 5            # trading days in the forecast window
STEP = 5               # trading days between evaluation origins (non-overlapping windows)
VAR_FLOOR = 1e-8       # windows with mean variance below this are untraded -> not scored
PRED_CLIP = (-12.0, -3.0)   # predictions are clipped to this range before scoring
MIN_COVERAGE = 0.99    # share of scored pairs that must receive a finite prediction
REQUIRED_COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
_K = 2.0 * np.log(2.0) - 1.0


# ----------------------------------------------------------------------------- data
def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")
    out = df[REQUIRED_COLS].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out["Ticker"] = out["Ticker"].astype(str)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.drop_duplicates(["Date", "Ticker"], keep="last")
    return out.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def load_panel(path) -> pd.DataFrame:
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return clean_panel(df)


def trading_calendar(panel: pd.DataFrame, min_frac: float = 0.5) -> pd.DatetimeIndex:
    """Dates on which at least half of the usual universe traded (ignores stray/partial dates)."""
    counts = panel.groupby("Date")["Ticker"].nunique()
    return pd.DatetimeIndex(counts[counts >= min_frac * counts.max()].index).sort_values()


# ----------------------------------------------------------------------------- target
def gk_variance(o, h, l, c) -> np.ndarray:
    """Garman-Klass daily variance. NaN where prices are invalid. Always >= 0."""
    o, h, l, c = (np.asarray(x, dtype="float64") for x in (o, h, l, c))
    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    with np.errstate(all="ignore"):
        hh = np.maximum.reduce([h, o, c])      # repair tiny High/Low inconsistencies
        ll = np.minimum.reduce([l, o, c])
        v = 0.5 * np.log(hh / ll) ** 2 - _K * np.log(c / o) ** 2
    return np.where(ok & np.isfinite(v), np.maximum(v, 0.0), np.nan)


def daily_variance_matrix(panel: pd.DataFrame, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """dates x tickers matrix of Garman-Klass daily variance."""
    v = gk_variance(panel["Open"], panel["High"], panel["Low"], panel["Close"])
    idx = pd.MultiIndex.from_arrays([panel["Date"], panel["Ticker"]], names=["Date", "Ticker"])
    V = pd.Series(v, index=idx).unstack("Ticker").sort_index()
    return V.reindex(calendar) if calendar is not None else V


def target_matrix(V: pd.DataFrame) -> pd.DataFrame:
    """y for every (origin date, ticker): log of the mean variance over the next HORIZON days.
    NaN when the origin bar or any of the next HORIZON bars is invalid/missing, or the window is untraded."""
    fwd = V.rolling(HORIZON, min_periods=HORIZON).mean().shift(-HORIZON)
    fwd = fwd.where(fwd >= VAR_FLOOR).where(V.notna())
    return np.log(fwd)


def forward_targets(V: pd.DataFrame, positions) -> dict:
    Y = target_matrix(V)
    return {V.index[p]: Y.iloc[p].dropna() for p in positions}


def validation_origin_positions(n_dates: int, n_origins: int, step: int = STEP, h: int = HORIZON) -> list:
    last = n_dates - 1 - h
    return [last - step * j for j in reversed(range(n_origins))]


def forward_origin_positions(n_dates: int, start_pos: int, step: int = STEP, h: int = HORIZON) -> list:
    return list(range(start_pos, n_dates - h, step))


# ----------------------------------------------------------------------------- score
def qlike(y, yhat) -> np.ndarray:
    d = np.asarray(y, dtype="float64") - np.asarray(yhat, dtype="float64")
    return np.exp(d) - d - 1.0


def score_predictions(preds: dict, truth: dict, min_coverage: float = MIN_COVERAGE) -> dict:
    """preds: {origin -> Series(ticker -> yhat)};  truth: {origin -> Series(ticker -> y)}."""
    rows, q_all, se_all = [], [], []
    n_total = n_missing = 0
    for origin, y in truth.items():
        if len(y) == 0:
            continue
        p = preds.get(origin)
        if p is None:
            p = pd.Series(np.nan, index=y.index)
        else:
            p = pd.to_numeric(pd.Series(p), errors="coerce").replace([np.inf, -np.inf], np.nan)
            p = p[~p.index.duplicated(keep="last")].reindex(y.index)
        miss = p.isna()
        n_total += len(y)
        n_missing += int(miss.sum())
        fill = p.median()
        p = p.fillna(PRED_CLIP[0] if np.isnan(fill) else fill).clip(*PRED_CLIP)
        d = y.to_numpy() - p.to_numpy()
        q = np.exp(d) - d - 1.0
        rho = float(y.rank().corr(p.rank())) if p.nunique() > 1 else float("nan")
        rows.append(dict(origin=str(pd.Timestamp(origin).date()), n=int(len(y)),
                         qlike=float(q.mean()), rmse=float(np.sqrt((d ** 2).mean())), spearman=rho))
        q_all.append(q)
        se_all.append(d ** 2)
    if not rows:
        raise ValueError("no scorable (stock, origin) pairs")
    q_all, se_all = np.concatenate(q_all), np.concatenate(se_all)
    coverage = 1.0 - n_missing / n_total
    return dict(qlike=float(q_all.mean()), rmse=float(np.sqrt(se_all.mean())),
                spearman_mean=(float(np.nanmean(sp)) if (sp := [r["spearman"] for r in rows if r["spearman"] == r["spearman"]]) else float("nan")),
                n_pairs=int(n_total), n_origins=len(rows), coverage=float(coverage),
                valid=bool(coverage >= min_coverage), per_origin=rows)
