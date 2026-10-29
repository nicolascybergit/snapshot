"""
META — the lazy primitive cache. The ONE door between the meta level and the
primitives below it.

    value = cache.get(symbol, primitive, params)

  * a stored result for exactly this ``(symbol, primitive, params)`` -> returned
  * nothing stored -> the real primitive runs ONCE, the result is stored, returned

The caller cannot tell which of the two happened, and that is the point: the meta
level asks for a primitive's answer and gets it, without knowing whether it cost
a computation. Everything is stored as JSON, and a freshly computed result is
handed back THROUGH the same JSON round-trip as a cached one — otherwise the
first call could return pandas/dataclass objects and the second plain dicts, and
a snapshot built at 9:00 would differ from the identical one built at 9:01.

KEY = symbol + primitive + a stable hash of the params that affect the result AND
of the input SERIES itself. Change a tunable and the hash changes; replace the CSV
and it changes too, so the next call recomputes instead of serving a stale answer.

The series belongs in the key because "historical data does not change" is an
assumption about the world, not about the file: a CSV gets re-exported with a
longer history, a vendor restates a close, a symbol is repointed at a different
file. When that happened here (five years of history swapped for ten) every
params-only key still matched, and the meta level went on reading five-year
primitives against a ten-year close series without anything looking wrong — the
ATH view drew 184 new highs while the ATH-meta, one door further in, still counted
99. Nothing invalidates by TIME; identity of the input is what invalidates.

The wrappers at the bottom are the vocabulary the meta level actually uses
(``plateaus(...)``, ``trailing_volume(...)``, ``ath_distance(...)`` …). Each one
names a primitive entry point and the params it depends on; the meta layer picks
from this list and never reaches past it.

Storage: ONE SQLite file PER SYMBOL in ``data.primitive_dir``
(``data/primitive/<TICKER>.sqlite``) — the "primitive analysis" level of the data
layout: every primitive result of one stock in one file. A single shared file
(``data.meta_cache_path`` or an explicit ``db_path``) is still accepted, which is
what the tests use. A corrupt row, an unreadable file or a missing table is treated
as a MISS — the primitive simply runs again. A cache that cannot answer must never
break the analysis above it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from ..applied_data import ath as ath_module
from ..applied_data import breakout as breakout_module
from ..applied_data import plateau as plateau_module
from ..applied_data import trend as trend_module
from ..applied_data import volume as volume_module
from ..raw_data import long_term_csv

SOURCE = "meta.cache"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS primitive_cache (
    symbol      TEXT NOT NULL,
    primitive   TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (symbol, primitive, params_hash)
)
"""


def params_hash(params: dict[str, Any]) -> str:
    """
    A stable fingerprint of the params that affect a result.

    Stable across processes and orderings: the dict is dumped sorted, so
    ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` hash the same, and floats keep
    their exact repr so 0.06 and 0.060000001 do not collide.
    """
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class MetaCache:
    """Get-or-compute-and-store, keyed by symbol + primitive + params + series."""

    def __init__(self, config, db_path: Optional[str] = None) -> None:
        self.config = config
        # One shared file when asked for explicitly, else one file per symbol.
        self.db_path = db_path or config.get("data.meta_cache_path")
        self.primitive_dir = config.get("data.primitive_dir", "data/primitive")
        self.computes: int = 0          # how many real primitive runs this instance caused
        self.hits: int = 0
        self._conns: dict[str, sqlite3.Connection] = {}
        self._fingerprints: dict[str, str] = {}
        self._touched: set[tuple[str, str, str]] = set()   # (symbol, primitive, key)

    # ---------------------------------------------------------------- storage

    def path_for(self, symbol: str) -> str:
        """The file a symbol's primitives live in."""
        return self.db_path or str(Path(self.primitive_dir) / f"{symbol.upper()}.sqlite")

    def _connection(self, symbol: str) -> Optional[sqlite3.Connection]:
        path = self.path_for(symbol)
        if path in self._conns:
            return self._conns[path]
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path)
            conn.execute(_SCHEMA)
            conn.commit()
        except (sqlite3.Error, OSError):
            return None                 # unusable store -> every call is a miss
        self._conns[path] = conn
        return conn

    def _read(self, symbol: str, primitive: str, key: str) -> Optional[Any]:
        conn = self._connection(symbol)
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT payload FROM primitive_cache "
                "WHERE symbol=? AND primitive=? AND params_hash=?",
                (symbol, primitive, key),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None                 # corrupt row -> recompute over it

    def _write(self, symbol: str, primitive: str, key: str, payload: Any) -> None:
        conn = self._connection(symbol)
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT OR REPLACE INTO primitive_cache VALUES (?,?,?,?)",
                (symbol, primitive, key, json.dumps(payload)),
            )
            conn.commit()
        except (sqlite3.Error, TypeError, ValueError):
            pass                        # a store that cannot store is still not fatal

    def prune_untouched(self, symbol: str) -> int:
        """
        Delete every stored row of ``symbol`` this instance did not ask for.

        Meant right after a FULL reading of the symbol (the dossier asks for every
        primitive): what was not touched then belongs to an older series and can
        never be hit again. Returns the number of rows removed. Anything removed by
        mistake is only a recompute, never a wrong answer.
        """
        conn = self._connection(symbol)
        if conn is None:
            return 0
        keep = {(p, k) for s, p, k in self._touched if s == symbol}
        if not keep:
            return 0                    # nothing read -> nothing proven stale
        try:
            rows = conn.execute("SELECT primitive, params_hash FROM primitive_cache "
                                "WHERE symbol=?", (symbol,)).fetchall()
            stale = [r for r in rows if (r[0], r[1]) not in keep]
            conn.executemany("DELETE FROM primitive_cache WHERE symbol=? AND primitive=? "
                             "AND params_hash=?", [(symbol, p, k) for p, k in stale])
            conn.commit()
        except sqlite3.Error:
            return 0
        return len(stale)

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns = {}

    # ---------------------------------------------------------------- the door

    def series_fingerprint(self, symbol: str) -> str:
        """
        A short, stable fingerprint of the close series a symbol currently loads.

        Cheap to take (the series is read anyway) and specific enough that any
        real change moves it: the number of days, the first and last date, and a
        digest of every close. An appended day, a restated close, a longer export
        or a repointed ``csv_map`` entry all produce a different fingerprint and
        therefore a different cache key. Memoised per instance so a run that asks
        for twenty primitives fingerprints the series once.
        """
        known = self._fingerprints.get(symbol)
        if known is not None:
            return known
        series = self.closes(symbol)
        if series is None or series.empty:
            mark = "empty"
        else:
            blob = ",".join(repr(float(v)) for v in series.to_numpy())
            mark = "|".join((
                str(len(series)),
                str(series.index[0].date()),
                str(series.index[-1].date()),
                hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
            ))
        self._fingerprints[symbol] = mark
        return mark

    def get(self, symbol: str, primitive: str, params: dict[str, Any]) -> Any:
        """
        The primitive's answer for this symbol under these params — from the
        store if it is there, computed once and stored if it is not.

        The key carries the SERIES fingerprint beside the params, so a refreshed
        CSV misses instead of matching. Old rows are simply never hit again; the
        store grows rather than corrupting, and a stale row can be deleted at
        leisure or left alone.
        """
        key = params_hash({"params": params,
                           "series": self.series_fingerprint(symbol)})
        self._touched.add((symbol, primitive, key))
        cached = self._read(symbol, primitive, key)
        if cached is not None:
            self.hits += 1
            return cached

        compute = _COMPUTERS.get(primitive)
        if compute is None:
            raise KeyError(f"unknown primitive: {primitive}")
        payload = json.loads(json.dumps(compute(self, symbol, params)))
        self.computes += 1
        self._write(symbol, primitive, key, payload)
        return payload

    # ------------------------------------------------------------ source data

    def closes(self, symbol: str) -> pd.Series:
        """The close series behind a symbol (not cached: it IS the input)."""
        series = long_term_csv.load(symbol, self.config.get("data.csv_dir"),
                                    self.config.get("csv_map", {}))
        return series if series is not None else pd.Series(dtype=float)

    def volumes(self, symbol: str) -> pd.Series:
        series = long_term_csv.load_volume(symbol, self.config.get("data.csv_dir"),
                                           self.config.get("csv_map", {}))
        return series if series is not None else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# The computers. Each one runs exactly one primitive entry point and returns a
# JSON-able payload. They are the only place in the meta layer where a primitive
# is called at all.
# ---------------------------------------------------------------------------


def _compute_plateaus(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    tiers = [plateau_module.Tier(t["name"], t["band_tol"], t["min_days"],
                                 t["max_slope_pct_per_day"], t["max_drift_pct"])
             for t in params["tiers"]]
    found = plateau_module.detect_tiers(symbol, cache.closes(symbol), tiers)
    return [p.to_dict() for p in found]


def _compute_breakouts(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    tiers = [breakout_module.Tier(t["name"], breakout_module.Params(**t["params"]))
             for t in params["tiers"]]
    found = breakout_module.detect_tiers(symbol, cache.closes(symbol), tiers=tiers)
    return [b.to_dict() for b in found]


def _compute_plateau_causal(cache: MetaCache, symbol: str,
                            params: dict[str, Any]) -> Any:
    """Past-only plateau scan per size class: x is the right edge, every day."""
    series = cache.closes(symbol)
    return {t["name"]: plateau_module.causal_scan(
        symbol, series, band_tol=t["band_tol"], min_days=t["min_days"],
        max_slope_pct_per_day=t["max_slope_pct_per_day"],
        max_drift_pct=t["max_drift_pct"], tier=t["name"])
        for t in params["tiers"]}


def _compute_breakout_causal(cache: MetaCache, symbol: str,
                             params: dict[str, Any]) -> Any:
    """Past-only breakout detection per size class, dated by when each was knowable."""
    series = cache.closes(symbol)
    out: list[dict[str, Any]] = []
    for t in params["tiers"]:
        out.extend(breakout_module.causal(
            symbol, series, params=breakout_module.Params(**t["params"]), tier=t["name"]))
    out.sort(key=lambda b: (b["known_date"] or "", b["start_date"]))
    return out


def _compute_breakout_trailing(cache: MetaCache, symbol: str,
                               params: dict[str, Any]) -> Any:
    return breakout_module.in_progress(
        cache.closes(symbol), params=breakout_module.Params(**params["params"]))


def _compute_trend_segments(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    found = trend_module.trend_segments(cache.closes(symbol), **params["settings"])
    return [s.to_dict() for s in found]


def _compute_trend_tiles(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    states = trend_module.tile_states(cache.closes(symbol), **params["settings"])
    return [s.to_dict() for s in states]


def _compute_rolling_ath(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    return ath_module.rolling_ath(cache.closes(symbol))


def _compute_ath_events(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    return [e.to_dict() for e in ath_module.ath_events(cache.closes(symbol),
                                                       symbol=symbol)]


def _compute_ath_distance(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    """The primitive's own per-day distance answer, asked once for every day."""
    series = cache.closes(symbol)
    return [ath_module.distance_to_ath(series, i) for i in range(len(series))]


def _compute_trend_at(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    """The primitive's trailing fit at EVERY day, with its four-way label."""
    series = cache.closes(symbol)
    window = int(params["window"])
    min_slope = float(params["min_slope"])
    min_r2 = float(params["min_r2"])
    out = []
    for i in range(len(series)):
        fit = trend_module.trend_at(series, i, window=window)
        out.append({
            "date": fit.end_date,
            "slope_per_day": fit.slope_per_day,
            "r2": fit.r2,
            "direction": fit.direction,
            "label": trend_module.classify(fit, min_slope=min_slope, min_r2=min_r2),
        })
    return out


def _compute_trailing_volume(cache: MetaCache, symbol: str, params: dict[str, Any]) -> Any:
    """Volume's own two measurements over the window ENDING on each day."""
    volumes = cache.volumes(symbol)
    window = int(params["window"])
    flat_band = float(params["flat_band"])
    out = []
    for i in range(len(volumes)):
        start = max(0, i - window + 1)
        out.append({
            "ratio": volume_module.ratio(volumes, start, i, window=window),
            "trend": volume_module.trend(volumes, start, i, flat_band=flat_band),
        })
    return out


_COMPUTERS: dict[str, Callable[[MetaCache, str, dict[str, Any]], Any]] = {
    "plateau.detect_tiers": _compute_plateaus,
    "plateau.causal_scan": _compute_plateau_causal,
    "breakout.detect_tiers": _compute_breakouts,
    "breakout.causal": _compute_breakout_causal,
    "breakout.in_progress": _compute_breakout_trailing,
    "trend.trend_segments": _compute_trend_segments,
    "trend.at_series": _compute_trend_at,
    "trend.tile_states": _compute_trend_tiles,
    "ath.rolling_ath": _compute_rolling_ath,
    "ath.ath_events": _compute_ath_events,
    "ath.distance": _compute_ath_distance,
    "volume.trailing": _compute_trailing_volume,
}


# ---------------------------------------------------------------------------
# The vocabulary the meta level uses. Each wrapper knows one primitive's name and
# the params it depends on — so a tunable change lands in the hash automatically.
# ---------------------------------------------------------------------------


def plateau_tier_params(config) -> dict[str, Any]:
    """
    The plateau size classes, as the plain dict that goes into the hash.

    Every tolerance a detection depends on belongs here: the dict IS the cache key,
    so a limit that is used but not listed would serve yesterday's plateaus under
    today's rules.
    """
    return {"tiers": [{"name": t.name, "band_tol": t.band_tol, "min_days": t.min_days,
                       "max_slope_pct_per_day": t.max_slope_pct_per_day,
                       "max_drift_pct": t.max_drift_pct}
                      for t in plateau_module.tiers_from_config(config)]}


def breakout_tier_params(config) -> dict[str, Any]:
    return {"tiers": [{"name": t.name, "params": vars(t.params)}
                      for t in breakout_module.breakout_tiers_from_config(config)]}


def trend_settings_params(config) -> dict[str, Any]:
    settings = trend_module.settings_from_config(config)
    return {"settings": {
        "tile_window": settings.tile_window, "tile_step": settings.tile_step,
        "slope_tol": settings.slope_tol,
        "break_up_factor": settings.break_up_factor,
        "break_down_factor": settings.break_down_factor,
        "min_slope": settings.min_slope, "min_r2": settings.min_r2,
    }}


def plateaus(cache: MetaCache, symbol: str) -> list[dict[str, Any]]:
    """Every detected plateau, both size classes. Tier is kept, never averaged."""
    return cache.get(symbol, "plateau.detect_tiers", plateau_tier_params(cache.config))


def breakouts(cache: MetaCache, symbol: str) -> list[dict[str, Any]]:
    """Every detected breakout, both size classes, with tier and direction."""
    return cache.get(symbol, "breakout.detect_tiers", breakout_tier_params(cache.config))


def causal_plateaus(cache: MetaCache, symbol: str) -> dict[str, Any]:
    """
    The PAST-ONLY plateau scan, per size class: for every day, the plateau that
    reaches it (recomputed with that day as the right edge) and the plateaus whose
    end was already established, each with the day it became knowable.

    Cached under its own key, so past-only results can never be confused with the
    forward-informed ``plateau.detect_tiers`` payload.
    """
    return cache.get(symbol, "plateau.causal_scan", plateau_tier_params(cache.config))


def causal_breakouts(cache: MetaCache, symbol: str) -> list[dict[str, Any]]:
    """
    The PAST-ONLY breakout detection, both size classes, each carrying the
    ``known_date`` on which it first became decidable — never earlier.
    """
    return cache.get(symbol, "breakout.causal", breakout_tier_params(cache.config))


def breakout_in_progress(cache: MetaCache, symbol: str) -> list[Optional[str]]:
    """Per day: the direction of a move breaking out as of that day, or None."""
    small = breakout_module.breakout_tiers_from_config(cache.config)[0]
    return cache.get(symbol, "breakout.in_progress", {"params": vars(small.params)})


def trend_classification_enabled(config) -> bool:
    """
    Is the trend PRIMITIVE's classification still in use? Default NO.

    DEACTIVATED — replaced by trend-meta v0.2 (:mod:`snapshot_engine.meta.trend_regime`).
    The primitive's tile states (``consistent`` / ``break_up`` / ``break_down``) and
    the segments merged out of them proved too unreliable to separate days by: as a
    grouping key they split groups on noise, which costs members twice over — the
    key fragments ``n`` AND the split follows something that was never a fact about
    the day. The module and its maths stay exactly where they are (nothing was
    deleted); only these two readers are switched off, and with them every path that
    fed the classification into a snapshot.

    NOT covered by the switch: :func:`trend_at_series`, the plain rolling regression.
    A slope with its R^2 is a measurement, not a classification, and the modules that
    read it (the old event-based TREND-META, the dashboard's trend view) never used
    it to group anything.
    """
    return bool(config.get("trend.primitive_enabled", False))                # tunable


def trend_tiles(cache: MetaCache, symbol: str) -> list[dict[str, Any]]:
    """
    Per tile: the consistency state, dated by the day the tile ends on.

    Empty while the classification is switched off (see
    :func:`trend_classification_enabled`) — the ONE place that decision is enforced,
    so no caller can reach the states around it.
    """
    if not trend_classification_enabled(cache.config):
        return []
    return cache.get(symbol, "trend.tile_states", trend_settings_params(cache.config))


def trend_segments(cache: MetaCache, symbol: str) -> list[dict[str, Any]]:
    """
    The merged consistency spans. Empty while the classification is switched off.

    They used to supply Block B's event list and, through it, most snapshots'
    forward horizon. Both are now built from plateaus and breakouts alone.
    """
    if not trend_classification_enabled(cache.config):
        return []
    return cache.get(symbol, "trend.trend_segments", trend_settings_params(cache.config))


def ath_distance(cache: MetaCache, symbol: str) -> list[dict[str, Any]]:
    """Per day: the rolling high, the gap to it and the days since it."""
    return cache.get(symbol, "ath.distance", {})


def rolling_ath(cache: MetaCache, symbol: str) -> list[float]:
    """Per day: the highest close reached up to and including it."""
    return cache.get(symbol, "ath.rolling_ath", {})


def ath_events(cache: MetaCache, symbol: str) -> list[dict[str, Any]]:
    """The days a new all-time high was set."""
    return cache.get(symbol, "ath.ath_events", {})


def trend_at_series(cache: MetaCache, symbol: str, *, window: int) -> list[dict[str, Any]]:
    """
    Per day: the trailing trend fit over ``window`` days ending there, labelled.

    The standard, forward-free-of-nothing form — this is the primitive's ordinary
    answer, which is what the full-history metas want. (The snapshot's strictly
    past-only variants live under their own keys and are not used here.)
    """
    min_slope = float(cache.config.get("trend.min_slope"))    # tunable
    min_r2 = float(cache.config.get("trend.min_r2"))          # tunable
    return cache.get(symbol, "trend.at_series",
                     {"window": window, "min_slope": min_slope, "min_r2": min_r2})


def trailing_volume(cache: MetaCache, symbol: str, *, window: int) -> list[dict[str, Any]]:
    """Per day: volume's ratio and direction over the window ending there."""
    flat_band = float(cache.config.get("volume.flat_band"))           # tunable
    return cache.get(symbol, "volume.trailing",
                     {"window": window, "flat_band": flat_band})
