"""
Build the static showcase: run the engine once per stock and write the results
the page reads (``docs/data/<SYMBOL>.json`` plus ``docs/data/index.json``), then
copy the engine itself into ``docs/engine/`` so the page can run it on an upload.

    python build_site.py              # every CSV in data/prices
    python build_site.py MU NVDA      # just these
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from snapshot_engine.analyse import ROOT, analyse, load_config

SITE = ROOT / "docs"

NAMES = {
    "AAPL": "Apple", "AMD": "AMD", "AMZN": "Amazon", "ASML": "ASML", "AVGO": "Broadcom",
    "GOOGL": "Alphabet", "LLY": "Eli Lilly", "META": "Meta", "MSFT": "Microsoft",
    "MU": "Micron", "NFLX": "Netflix", "NVDA": "Nvidia", "TSM": "TSMC",
}


def _round(value, digits: int = 4):
    """Shorter JSON: floats rounded, structure unchanged."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


def copy_engine() -> None:
    """The package and its parameters, as the page loads them into Pyodide."""
    target = SITE / "engine"
    shutil.rmtree(target, ignore_errors=True)
    files = []
    for path in sorted((ROOT / "snapshot_engine").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / rel)
        files.append(rel)
    (target / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "config" / "parameters.yaml", target / "config" / "parameters.yaml")
    files.append("config/parameters.yaml")
    (target / "manifest.json").write_text(json.dumps(files, indent=1), encoding="utf-8")


def main(argv: list[str]) -> int:
    config = load_config()
    symbols = [s.upper() for s in argv] or sorted(config.get("csv_map"))
    out_dir = SITE / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    stocks = {s["symbol"]: s for s in index.get("stocks", [])}
    for symbol in symbols:
        started = time.time()
        result = _round(analyse(config, symbol, name=NAMES.get(symbol, symbol)))
        (out_dir / f"{symbol}.json").write_text(json.dumps(result, separators=(",", ":")),
                                                encoding="utf-8")
        today = result.get("today") or {}
        stocks[symbol] = {"symbol": symbol, "name": result["name"], "date": today.get("date"),
                          "score": today.get("score")}
        print(f"{symbol:6s} {time.time() - started:5.1f}s  score {today.get('score')}")
    index = {"generated": time.strftime("%Y-%m-%d"),
             "radius_lo": float(config.get("street.default_radius_lo")),
             "stocks": sorted(stocks.values(), key=lambda s: s["symbol"])}
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    copy_engine()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
