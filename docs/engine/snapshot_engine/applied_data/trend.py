"""
Applied Data — trend (a FILE, not a level). A PRIMITIVE.

Deterministic, pure maths, no LLM. Trend does not ask "which way is price going
on average" — an average slope over a long window says almost nothing about a
move that changed its character halfway through. It asks the sharper question:

    *** HOW CONSISTENT is the slope, and WHERE does the consistency BREAK? ***

The series is walked in short overlapping TILES (a ``tile_window``-day regression
every ``tile_step`` days). Each tile's normalised slope is compared with the
PREVIOUS tile's, and the difference decides the tile's STATE:

  * inside the tolerance band (``slope_tol``)      -> ``consistent``
        The move keeps its character. Mildly steeper or mildly flatter is still
        the same move, which is exactly what the band is for.
  * clearly ABOVE the band  (``break_up_factor``)  -> ``break_up``
        Acceleration — the move got decisively faster. Often what a breakout
        looks like from the inside.
  * clearly BELOW the band  (``break_down_factor``)-> ``break_down``
        Deceleration or a turn: the order is coming apart.
  * directionless chop (``min_slope`` AND ``min_r2`` both missed) -> ``break_down``
        Chaos IS a loss of order. A move that dissolves into scatter has broken
        just as surely as one that reversed, and it usually happens first.

The two break gates are deliberately SEPARATE factors: an upward break and a
downward break are not the same event and need not be equally hard to trigger.
Consistency breaking down often PRECEDES a fall — that is the whole value of the
detector. It is a per-tile STATE, never a prediction: it says what has happened
to the slope, not what price will do next.

Consecutive tiles of equal state merge into one SEGMENT, and each segment is
refitted over its own span so it can be drawn as one straight line.

The building blocks stay available on their own:

  * ``slope_r2`` — one straight fit: ``slope_per_day`` (raw slope divided by the
    span's mean close, so 8 USD and 800 USD stocks compare directly), ``r2``
    (how well the line describes the span) and the plain sign as ``direction``.
  * ``trend_at`` — that fit over the window ENDING at one day.
  * ``classify`` — the older four-way reading of a single fit
    (``up`` · ``down`` · ``flat`` · ``noisy``), still used by ``rolling_trend``.

PAST-ONLY throughout. A tile is fitted over ``[end - tile_window + 1 .. end]``
and is dated by the day it ENDS on, so nothing ever reads a day later than the
one it describes; a segment refit covers only the days its own tiles ended on.

In:  a date-indexed close series for ONE symbol (raw_data/long_term_csv)
Out: one fit, a rolling series of fits, or the consistency segments with their
     lines as two endpoints in PRICE space (``y_start`` / ``y_end``), so a view
     draws from two given points and computes nothing.

All thresholds are central parameters (``trend.*`` in
``config/parameters.yaml``); ``settings_from_config`` is the ONE place they are
read and every entry point takes them as arguments — this module hard-codes no
threshold, does no I/O and prints nothing (``_report`` is the CLI end). It takes
NO event data: no plateau, no breakout, no volume, no ATH.

The two readings that need trend TOGETHER with events — a breakout WITH or
AGAINST the background trend, and a trend as the bracket that groups several
same-direction breakouts into one staged move — are META-level questions. They
are deliberately NOT here; a primitive that looked at events would stop being one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd

SOURCE = "trend"

UP, DOWN, FLAT, NOISY = "up", "down", "flat", "noisy"

# The three consistency STATES a tile (and therefore a segment) can be in.
CONSISTENT, BREAK_UP, BREAK_DOWN = "consistent", "break_up", "break_down"


@dataclass(frozen=True)
class Settings:
    """The tunables this primitive needs: the tile, the band, the two break gates."""

    tile_window: int
    tile_step: int
    slope_tol: float
    break_up_factor: float
    break_down_factor: float
    min_slope: float
    min_r2: float


def settings_from_config(config) -> Settings:
    """The ONE place these tunables are read (dashboard, CLI, later the core)."""
    return Settings(
        tile_window=int(config.get("trend.tile_window")),                  # tunable
        tile_step=int(config.get("trend.tile_step")),                      # tunable
        slope_tol=float(config.get("trend.slope_tol")),                    # tunable
        break_up_factor=float(config.get("trend.break_up_factor")),        # tunable
        break_down_factor=float(config.get("trend.break_down_factor")),    # tunable
        min_slope=float(config.get("trend.min_slope")),                    # tunable
        min_r2=float(config.get("trend.min_r2")),                          # tunable
    )


@dataclass(frozen=True)
class TrendFit:
    """One straight line fitted through one span of closes."""

    start_date: str        # ISO date of the first day of the span
    end_date: str          # ISO date of the last day of the span
    slope_per_day: float   # raw slope / mean close — a fraction of price per day
    r2: float              # how much of the movement the line explains (0 .. 1)
    direction: str         # the SIGN of the slope: "up" | "down" | "flat"
    intercept: float       # the line's price at series index 0 (raw, price space)
    n_days: int            # days the fit was made from

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "slope_per_day": self.slope_per_day,
            "r2": self.r2,
            "direction": self.direction,
            "intercept": self.intercept,
            "n_days": self.n_days,
        }


@dataclass(frozen=True)
class TrendPoint:
    """The trailing trend of ONE day: the window ending on it, already labelled."""

    date: str
    slope_per_day: float
    r2: float
    label: str             # "up" | "down" | "flat" | "noisy"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "date": self.date,
            "slope_per_day": self.slope_per_day,
            "r2": self.r2,
            "label": self.label,
        }


@dataclass(frozen=True)
class TrendSegment:
    """
    One run of neighbouring tiles that shared a consistency STATE, plus the line
    refitted over that run.

    A segment is NOT "an up stretch" or "a down stretch" any more — it is a
    stretch over which the slope behaved the same way: it kept its character, it
    broke upward, or it broke down. The line's angle still shows which way price
    went; ``state`` says what happened to the consistency of that movement.

    ``y_start`` / ``y_end`` are the refit line's own endpoints in PRICE space,
    so a view draws the segment from two points it was handed.
    """

    start_date: str
    end_date: str
    slope_per_day: float
    r2: float
    state: str             # "consistent" | "break_up" | "break_down"
    y_start: float
    y_end: float

    def to_dict(self) -> dict[str, Any]:
        """The full record, JSON-safe (dates already ISO strings)."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "slope_per_day": self.slope_per_day,
            "r2": self.r2,
            "state": self.state,
            "y_start": self.y_start,
            "y_end": self.y_end,
        }

    def to_geometry(self) -> dict[str, Any]:
        """
        What a read-only VIEW is handed: the two endpoints of the line and the
        state that picks its line style. ``slope_per_day`` and ``r2`` stay
        behind — no analysis figure is ever rendered as text (DESIGN_CODEX /
        BUILD_NOTES).
        """
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "y_start": self.y_start,
            "y_end": self.y_end,
            "state": self.state,
        }


class _Line(NamedTuple):
    """The raw regression, before anything is normalised or named."""

    slope: float       # price change per day (NOT normalised)
    intercept: float   # price at series index 0
    r2: float
    mean: float        # mean close over the span — what the slope is divided by
    n_days: int

    def at(self, index: int) -> float:
        """The fitted price at one absolute series index."""
        return self.intercept + self.slope * index


def _iso(label: Any) -> str:
    """ISO date string for one index label; plain labels fall back to str()."""
    if isinstance(label, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(label).date().isoformat()
    return str(label)


def _closes(series: Optional[pd.Series]) -> np.ndarray:
    if series is None or len(series) == 0:
        return np.empty(0, dtype=float)
    return np.asarray(series.to_numpy(), dtype=float)


def _span(values: np.ndarray, start_idx: int, end_idx: int) -> tuple[int, int]:
    """Clamp a requested span onto the series; an empty span comes back (0, -1)."""
    if len(values) == 0:
        return 0, -1
    start = max(0, int(start_idx))
    end = min(len(values) - 1, int(end_idx))
    return start, end


def _regress(values: np.ndarray, start: int, end: int) -> _Line:
    """
    Least squares over ``[start .. end]`` on ABSOLUTE series indices, so
    ``line.at(i)`` is the fitted price on day ``i`` of the series and the two
    endpoints of a span fall out of the same equation.

    Degenerate spans get a safe, finite answer instead of a NaN: an empty or
    single-day span is a flat line at that day's price (r2 0.0), and a span with
    no variance at all is a flat line that describes it exactly (r2 1.0).
    """
    if end < start or len(values) == 0:
        return _Line(0.0, 0.0, 0.0, 0.0, 0)
    span = values[start:end + 1]
    mean = float(np.mean(span))
    if len(span) < 2:
        return _Line(0.0, float(span[0]), 0.0, mean, len(span))

    x = np.arange(start, end + 1, dtype=float)
    slope, intercept = np.polyfit(x, span, 1)
    total = float(np.sum((span - mean) ** 2))
    if total <= 0.0:                       # perfectly flat: fitted exactly
        return _Line(0.0, float(mean), 1.0, mean, len(span))
    predicted = slope * x + intercept
    residual = float(np.sum((span - predicted) ** 2))
    return _Line(float(slope), float(intercept), float(1.0 - residual / total),
                 mean, len(span))


def _per_day(line: _Line) -> float:
    """
    The slope as a FRACTION of the span's own price level, per day. This is the
    normalisation that makes an 8 USD stock and an 800 USD stock comparable: the
    same 1%-per-day climb reads as 0.01 in both. A non-positive mean cannot be
    divided by and yields 0.0 rather than an infinity.
    """
    if line.mean <= 0:
        return 0.0
    return line.slope / line.mean


def _sign(slope_per_day: float) -> str:
    """The plain direction of a fit, with no threshold applied yet."""
    if slope_per_day > 0:
        return UP
    if slope_per_day < 0:
        return DOWN
    return FLAT


def _fit_of(series: pd.Series, values: np.ndarray, start: int, end: int) -> TrendFit:
    """Build the public fit object for one already-clamped span."""
    line = _regress(values, start, end)
    per_day = _per_day(line)
    labels = list(series.index)
    if end < start:
        return TrendFit("", "", 0.0, 0.0, FLAT, 0.0, 0)
    return TrendFit(
        start_date=_iso(labels[start]),
        end_date=_iso(labels[end]),
        slope_per_day=per_day,
        r2=line.r2,
        direction=_sign(per_day),
        intercept=line.intercept,
        n_days=line.n_days,
    )


def slope_r2(series: pd.Series, start_idx: int, end_idx: int) -> TrendFit:
    """
    Fit one straight line through ``[start_idx .. end_idx]`` and report it.

    ``slope_per_day`` is normalised by the span's mean close, ``r2`` says how
    well the line describes the span, and ``direction`` is only the SIGN of the
    slope — the thresholds that turn a fit into a verdict live in
    :func:`classify`, which is where the tunables belong.
    """
    values = _closes(series)
    start, end = _span(values, start_idx, end_idx)
    return _fit_of(series, values, start, end)


def trend_at(series: pd.Series, idx: int, *, window: int) -> TrendFit:
    """
    The trailing trend AT day ``idx``: the fit over the ``window`` days ENDING on
    it (``[idx - window + 1 .. idx]``).

    Past-only by construction — the window never reaches beyond ``idx``, so the
    answer is the one that day itself could have had. Early days simply have a
    shorter window (``n_days`` says how short); a negative index counts from the
    end (``-1`` = last day).
    """
    values = _closes(series)
    if len(values) == 0:
        return TrendFit("", "", 0.0, 0.0, FLAT, 0.0, 0)
    i = int(idx)
    if i < 0:
        i += len(values)
    i = max(0, min(len(values) - 1, i))
    start = max(0, i - max(1, int(window)) + 1)
    return _fit_of(series, values, start, i)


def classify(fit: TrendFit, *, min_slope: float, min_r2: float) -> str:
    """
    Turn one fit into a verdict: ``up`` · ``down`` · ``flat`` · ``noisy``.

    Order matters. FLAT is decided first, because a span that went nowhere has no
    direction to be clean or noisy about — its R^2 describes a horizontal line
    and says nothing worth reading. Only a span that actually moved is then asked
    whether the straight line describes it well enough to be trusted; if not, the
    move was real but the path was not a line, and that is ``noisy``.
    """
    if abs(fit.slope_per_day) < min_slope:
        return FLAT
    if fit.r2 < min_r2:
        return NOISY
    return UP if fit.slope_per_day > 0 else DOWN


def rolling_trend(series: pd.Series, *, window: int, step: int = 1,
                  min_slope: float, min_r2: float) -> list[TrendPoint]:
    """
    The trailing trend across the series: one :class:`TrendPoint` every ``step``
    days, each holding the fit of the ``window`` days ending on it plus its label.

    Starts at the first day that has a FULL window behind it — a short window is
    a different measurement wearing the same name, so it is not offered. Past-only
    throughout.

    ``min_slope`` / ``min_r2`` are passed in (they are what :func:`classify`
    needs); this module declares no default for either.
    """
    values = _closes(series)
    if len(values) == 0 or window < 2:
        return []
    stride = max(1, int(step))
    labels = list(series.index)

    points: list[TrendPoint] = []
    for i in range(int(window) - 1, len(values), stride):
        fit = _fit_of(series, values, i - int(window) + 1, i)
        points.append(
            TrendPoint(
                date=_iso(labels[i]),
                slope_per_day=fit.slope_per_day,
                r2=fit.r2,
                label=classify(fit, min_slope=min_slope, min_r2=min_r2),
            )
        )
    return points


class _Tile(NamedTuple):
    """One short regression, dated by the day it ENDS on."""

    start: int
    end: int
    slope_per_day: float
    r2: float


def _tiles(series: pd.Series, values: np.ndarray, *, tile_window: int,
           tile_step: int) -> list[_Tile]:
    """
    Walk a short regression across the series: one tile every ``tile_step`` days,
    each fitted over the ``tile_window`` days ENDING on it. The first tile is the
    first day with a full window behind it — a short window is a different
    measurement wearing the same name, so it is not offered.
    """
    if len(values) == 0 or tile_window < 2:
        return []
    stride = max(1, int(tile_step))
    out: list[_Tile] = []
    for end in range(int(tile_window) - 1, len(values), stride):
        begin = end - int(tile_window) + 1
        line = _regress(values, begin, end)
        out.append(_Tile(begin, end, _per_day(line), line.r2))
    return out


def _state_of(tile: _Tile, previous: Optional[_Tile], *, slope_tol: float,
              break_up_factor: float, break_down_factor: float,
              min_slope: float, min_r2: float) -> str:
    """
    What happened to the slope between the previous tile and this one.

    The very first tile has nothing to be compared against and is ``consistent``
    by definition. After that the difference of the two normalised slopes decides:
    inside the band the move kept its character; clearly above or below it, the
    consistency broke, upward or downward, with the two gates set separately
    because the two events are not mirror images of each other.

    The last rule is the one that is easy to miss and matters most: a tile that
    is going nowhere AND explains nothing (``min_slope`` and ``min_r2`` both
    missed) counts as ``break_down`` even though its slope barely moved. Order
    dissolving into chop is a break — usually the earliest one there is.
    """
    if previous is None:
        return CONSISTENT
    diff = tile.slope_per_day - previous.slope_per_day
    if abs(diff) <= slope_tol:
        return CONSISTENT
    if diff > break_up_factor * slope_tol:
        return BREAK_UP
    if diff < -break_down_factor * slope_tol:
        return BREAK_DOWN
    if abs(tile.slope_per_day) < min_slope and tile.r2 < min_r2:
        return BREAK_DOWN                      # directionless chop = order lost
    return CONSISTENT                          # mild drift, still the same move


@dataclass(frozen=True)
class TileState:
    """One tile's consistency verdict, dated by the day the tile ENDS on."""

    date: str
    state: str             # "consistent" | "break_up" | "break_down"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {"date": self.date, "state": self.state}


def tile_states(series: pd.Series, *, tile_window: int, tile_step: int,
                slope_tol: float, break_up_factor: float, break_down_factor: float,
                min_slope: float, min_r2: float) -> list[TileState]:
    """
    The consistency state of every tile, before any merging — the per-DAY form of
    the detector.

    :func:`trend_segments` merges these into spans; a caller that needs to know
    what ONE day knew must read them here instead, because a segment can extend
    past that day and its span is therefore not something the day could see. Each
    tile is dated by the day it ends on and is fitted over the days before it, so
    this list is past-only in the strict sense.
    """
    values = _closes(series)
    if len(values) == 0:
        return []
    tiles = _tiles(series, values, tile_window=tile_window, tile_step=tile_step)
    labels = list(series.index)
    out: list[TileState] = []
    for i, tile in enumerate(tiles):
        out.append(TileState(
            date=_iso(labels[tile.end]),
            state=_state_of(
                tile, tiles[i - 1] if i else None,
                slope_tol=slope_tol, break_up_factor=break_up_factor,
                break_down_factor=break_down_factor,
                min_slope=min_slope, min_r2=min_r2,
            ),
        ))
    return out


def trend_segments(series: pd.Series, *, tile_window: int, tile_step: int,
                   slope_tol: float, break_up_factor: float,
                   break_down_factor: float, min_slope: float,
                   min_r2: float) -> list[TrendSegment]:
    """
    THE segmentation: tile the series, state each tile against its predecessor,
    merge neighbouring tiles of equal state and refit each run over its own span.

    Two different fits are involved and the distinction is the point. The TILES
    decide where the spans break — each one asks "did the slope keep its
    character since the last tile?" — while the REFIT over the merged run is what
    gets drawn as one straight line. A segment therefore reads as: over these
    days the movement behaved this way (``state``), and this is the line it made.

    Segments are dated by the days their tiles ENDED on, so they are contiguous
    and never overlap even though the tiles themselves do. Past-only: no fit here
    ever reaches beyond the day it describes.
    """
    values = _closes(series)
    if len(values) == 0:
        return []
    tiles = _tiles(series, values, tile_window=tile_window, tile_step=tile_step)
    if not tiles:
        return []
    states = [t.state for t in tile_states(
        series, tile_window=tile_window, tile_step=tile_step, slope_tol=slope_tol,
        break_up_factor=break_up_factor, break_down_factor=break_down_factor,
        min_slope=min_slope, min_r2=min_r2)]

    labels = list(series.index)
    found: list[TrendSegment] = []
    run_start = 0
    for k in range(len(tiles) + 1):
        if k < len(tiles) and states[k] == states[run_start]:
            continue
        start = tiles[run_start].end          # a tile stands on the day it ends
        end = tiles[k - 1].end
        line = _regress(values, start, end)
        found.append(
            TrendSegment(
                start_date=_iso(labels[start]),
                end_date=_iso(labels[end]),
                slope_per_day=_per_day(line),
                r2=line.r2,
                state=states[run_start],
                y_start=line.at(start),
                y_end=line.at(end),
            )
        )
        run_start = k
    return found
