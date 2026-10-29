"""
ANALYSE — one call, one symbol, everything the page draws.

Used twice with the same code: ``build_site.py`` runs it once per showcase stock
and writes the result as static JSON; the page runs it in the browser (Pyodide)
for a CSV someone uploads. Statements are only written for the showcase stocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .config.loader import Config
from .meta import cache as meta_cache
from .meta import robustness as robustness_module
from .meta import snapshot as snapshot_module
from . import score, statements, view

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Optional[str | Path] = None, csv_dir: Optional[str | Path] = None) -> Config:
    """The exported parameters; every CSV in ``csv_dir`` is mapped to its file stem."""
    config = Config.load(path or ROOT / "config" / "parameters.yaml")
    folder = Path(csv_dir or ROOT / config.get("data.csv_dir"))
    config.data["data"]["csv_dir"] = str(folder)
    config.data["data"]["primitive_dir"] = str(ROOT / config.get("data.primitive_dir"))
    config.data["csv_map"] = {p.stem.upper(): p.name for p in folder.glob("*.csv")}
    return config


def analyse(config: Config, symbol: str, *, name: Optional[str] = None,
            with_statements: bool = True) -> dict[str, Any]:
    """Prices, every layer, today's snapshot score and (optionally) the statements."""
    out: dict[str, Any] = {"symbol": symbol, "name": name or symbol}
    out.update(view.price_view(config, symbol))
    if not out["prices"]:
        out["error"] = "no usable price rows"
        return out
    store = meta_cache.MetaCache(config)
    try:
        out["ath"] = view.ath_view(config, symbol)
        out["snapshots"] = view.snapshot_view(config, symbol, store)
        out["ath_meta"] = view.ath_meta_view(config, symbol, store)
        out["trend_meta"] = view.trend_meta_view(config, symbol, store)
        out["today"] = score.today(config, symbol, store)
        out["statements"] = []
        if with_statements:
            params = robustness_module.params_from_config(config)
            snaps = snapshot_module.build_snapshots(symbol, params=params.snapshot, cache=store)
            groups = robustness_module.all_groups(symbol, params=params, cache=store)
            dates = [p["date"] for p in out["prices"]]
            closes = [p["close"] for p in out["prices"]]
            out["statements"] = (statements.snapshot_statements(out["name"], groups, snaps, config)
                                 + statements.recovery_statements(out["name"], dates, closes, config))
    finally:
        store.close()
    return out
