# Snapshot Analysis

Snapshot Analysis is an independently developed model that estimates where a stock is likely to move next. Common approaches like Transformers or ARIMA read the price history as one long sequence. This model takes a snapshot of every past trading day instead, built only from data available on that day, which rules out lookahead bias. It then compares the latest day with all earlier snapshots, groups days that looked alike and measures how the price moved afterwards. A signal is only given when a pattern has appeared often enough to hold up in the backtest.

The full explanation and ten example stocks are on my website:
https://nicolascybergit.github.io/snapshot/

## Run it

```bash
pip install -r requirements.txt
python build_site.py            # every CSV in data/prices -> docs/data/*.json
python -m http.server -d docs   # then open http://localhost:8000
```

Add a stock by dropping its daily CSV (Nasdaq export, or any file with `Date` and `Close` columns, `Volume` optional) into `data/prices/` and building again.
