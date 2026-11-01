"""Synthetic OHLCV generator with persistent, market+stock-specific volatility.
Used ONLY for CI/format checks and for dry-running the pipeline. Not real data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_panel(n_tickers=40, start="2021-06-09", end="2026-10-30", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T, N, S = len(dates), n_tickers, 16
    base = rng.normal(np.log(0.017), 0.3, N)
    mkt, stk = np.zeros(T), np.zeros((T, N))
    for t in range(1, T):
        mkt[t] = 0.97 * mkt[t - 1] + 0.10 * rng.standard_normal()
        stk[t] = 0.95 * stk[t - 1] + 0.12 * rng.standard_normal(N)
    sigma = np.clip(np.exp(base + 0.6 * mkt[:, None] + stk), 0.003, 0.15)
    steps = rng.standard_normal((T, N, S)) * (sigma[:, :, None] / np.sqrt(S))
    gap = 0.3 * sigma * rng.standard_normal((T, N))
    close_prev, rows = np.full(N, 100.0), []
    tickers = np.array([f"SYN{i:03d}" for i in range(N)])
    listed_from = np.where(rng.random(N) < 0.1, rng.integers(50, 600, N), 0)   # some late listings
    for t in range(T):
        o = close_prev * np.exp(gap[t])
        path = o[:, None] * np.exp(np.cumsum(steps[t], axis=1))
        h, l, c = np.maximum(o, path.max(1)), np.minimum(o, path.min(1)), path[:, -1]
        vol = np.exp(rng.normal(12, 0.8, N))
        keep = (t >= listed_from) & (rng.random(N) > 0.002)                       # a few missing bars
        rows.append(pd.DataFrame({"Date": dates[t], "Ticker": tickers[keep], "Open": o[keep], "High": h[keep],
                                  "Low": l[keep], "Close": c[keep], "Volume": vol[keep]}))
        close_prev = c
    return pd.concat(rows, ignore_index=True)
