"""
Applied Data — volume (a FILE, not a level). A PRIMITIVE, not an event finder.

Volume is the number of shares traded per day (the ``Volume`` column already in
every CSV). It says HOW MANY took part in a move, never whether price moved.

This module detects nothing and returns NO list of findings. It only MEASURES:
"how much trading happened over this span, compared with what is normal for THIS
stock?" Plateau and breakout call it to attach a measured value to findings they
already hold; nothing here ever looks at a plateau or a breakout. The dependency
runs one way only:

    plateau  ──calls──►  volume
    breakout ──calls──►  volume
    volume   ──────────►  (nothing)

Every answer is RELATIVE — a ratio against a baseline of the same stock's own
recent days. Absolute share counts are meaningless across stocks (a mega-cap
trades a thousand times what a micro-cap does), so they never leave this module
as a judgement.

The API works on INDEX ranges, because plateau and breakout already hold the
index span of each finding; ``span_of`` converts their ISO dates when needed.
``window`` and ``flat_band`` are passed IN by the caller, resolved from
``config/parameters.yaml`` via ``settings_from_config``. Pure maths, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

SOURCE = "volume"

# Returned when there is nothing to compare against (no baseline days, or a
# baseline of zero). Neutral by construction: "no busier and no quieter than
# normal", so a missing measurement can never masquerade as a signal.
NEUTRAL_RATIO = 1.0

RISING, FALLING, FLAT = "rising", "falling", "flat"


@dataclass(frozen=True)
class Settings:
    """The two tunables this primitive needs."""

    baseline_window: int
    flat_band: float


def settings_from_config(config) -> Settings:
    """The ONE place these tunables are read (plateau, breakout, dashboard, CLI)."""
    return Settings(
        baseline_window=int(config.get("volume.baseline_window")),   # tunable
        flat_band=float(config.get("volume.flat_band")),             # tunable
    )


def _values(volume_series: pd.Series) -> np.ndarray:
    if volume_series is None or len(volume_series) == 0:
        return np.empty(0, dtype=float)
    return np.asarray(volume_series.to_numpy(), dtype=float)


def _mean(values: np.ndarray) -> float:
    """
    Mean that IGNORES missing days rather than counting them as zero — some
    exports leave the Volume cell empty (QNC has 191 such rows), and a zero
    would silently drag the average down. All-missing gives 0.0.
    """
    if len(values) == 0:
        return 0.0
    usable = values[np.isfinite(values)]
    if len(usable) == 0:
        return 0.0
    return float(np.mean(usable))


def baseline(volume_series: pd.Series, *, window: int, before: Optional[int] = None) -> float:
    """
    Mean daily volume over ``window`` days.

    ``before`` is the index the span starts at: the baseline is then taken from
    the ``window`` days immediately PRECEDING it — the stock's own normal right
    up to the moment of interest. Without ``before`` the last ``window`` days of
    the series are used. Returns 0.0 when there is nothing to average.
    """
    values = _values(volume_series)
    if len(values) == 0 or window <= 0:
        return 0.0
    if before is None:
        window_values = values[-window:]
    else:
        start = max(0, before - window)
        window_values = values[start:before]
    return _mean(window_values)


def ratio(volume_series: pd.Series, start_idx: int, end_idx: int, *, window: int) -> float:
    """
    Mean volume over ``[start_idx .. end_idx]`` divided by the baseline of the
    ``window`` days immediately BEFORE it.

        > 1  busier than normal      < 1  quieter than normal

    Returns :data:`NEUTRAL_RATIO` when no baseline exists (start of the series)
    or the baseline is zero — never a division by zero, never a fake signal.
    """
    values = _values(volume_series)
    if len(values) == 0:
        return NEUTRAL_RATIO
    start = max(0, int(start_idx))
    end = min(len(values) - 1, int(end_idx))
    if end < start:
        return NEUTRAL_RATIO

    span_mean = _mean(values[start:end + 1])
    if span_mean <= 0.0:
        return NEUTRAL_RATIO
    base = baseline(volume_series, window=window, before=start)
    if base <= 0.0:
        return NEUTRAL_RATIO
    return span_mean / base


def trend(volume_series: pd.Series, start_idx: int, end_idx: int, *, flat_band: float) -> str:
    """
    Which way volume is going across ``[start_idx .. end_idx]``.

    A straight line is fitted over the span and its slope is normalised by the
    span's own mean volume and its length, so the answer is comparable across
    stocks. Anything inside ``flat_band`` counts as ``"flat"`` — which is what a
    genuinely quiet plateau usually looks like, or ``"falling"`` when interest
    is drying up.
    """
    values = _values(volume_series)
    if len(values) == 0:
        return FLAT
    start = max(0, int(start_idx))
    end = min(len(values) - 1, int(end_idx))
    span = values[start:end + 1]
    keep = np.isfinite(span)
    if int(keep.sum()) < 2:
        return FLAT

    mean = _mean(span)
    if mean <= 0.0:
        return FLAT
    x = np.arange(len(span), dtype=float)[keep]   # gaps keep their position on the axis
    slope = float(np.polyfit(x, span[keep], 1)[0])
    # total drift across the span, as a fraction of its own mean level
    normalised = slope * (len(span) - 1) / mean
    if abs(normalised) <= flat_band:
        return FLAT
    return RISING if normalised > 0 else FALLING


def span_of(volume_series: pd.Series, start_date: str, end_date: str) -> Optional[tuple[int, int]]:
    """Index span for two ISO dates, for callers that hold dates rather than positions."""
    if volume_series is None or len(volume_series) == 0:
        return None
    position = {pd.Timestamp(label).date().isoformat(): i
                for i, label in enumerate(volume_series.index)}
    start, end = position.get(start_date), position.get(end_date)
    if start is None or end is None or end < start:
        return None
    return start, end


def measure(volume_series: pd.Series, start_idx: int, end_idx: int,
            settings: Settings) -> dict[str, Any]:
    """Both measurements for one span — what plateau and breakout attach."""
    return {
        "volume_ratio": ratio(volume_series, start_idx, end_idx,
                              window=settings.baseline_window),
        "volume_trend": trend(volume_series, start_idx, end_idx,
                              flat_band=settings.flat_band),
    }
