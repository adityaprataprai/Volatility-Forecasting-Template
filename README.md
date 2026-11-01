# Nifty Volatility Forecasting Challenge — starter repo

Predict each Nifty 500 stock's **log 5-day forward Garman–Klass variance**. Scored by **QLIKE** (lower is better)
on a hidden, later time window. Read the Problem Statement and the Submission Guide first.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put train_panel.parquet (from the data link in the Problem Statement) into data/
python -m submission.train                               # fit the HAR baseline on all public data
python challenge/evaluate.py --mode validate             # local score, same code the organisers use
python tests/check_submission.py                         # format check (also runs on every push via GitHub Actions)
```

Edit `submission/predict.py` (keep the `predict(history, origin)` signature). Do **not** edit `challenge/`.
Fill in `TEAM.md`. Never commit `data/`.
