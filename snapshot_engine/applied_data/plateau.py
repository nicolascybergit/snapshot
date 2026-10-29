"""
Applied Data — plateau detection (a FILE, not a level).

Deterministic, pure maths, no LLM: find the time spans where the close price
clusters inside a narrow y-band over a sustained run of trading days. A plateau
is the "quiet ground" a move later departs from — the reference level the
compound-dip analysis in ``baseline.py`` already reasons about, made explicit as
its own object so later layers can anchor on it.

In:  a date-indexed close series for ONE symbol (raw_data/long_term_csv)
Out: a chronological, NON-overlapping list of Plateau objects

Algorithm (greedy expansion, left to right, deterministic):
  Start a candidate at day i with running_min = running_max = price[i]. Extend to
  the next day j while ALL THREE of the span's own measurements hold:

      (max(span) - min(span)) / mean(span)  <=  band_tol              the BAND
      |slope(span) / mean(span)|            <=  max_slope_pct_per_day  the SLOPE
      |last - first| / mean(span)           <=  max_drift_pct          the DRIFT

  The first day that breaks any of them ends the candidate. A candidate of at
  least ``min_days`` days is recorded and scanning resumes at the day AFTER the
  plateau; a shorter one is discarded and scanning resumes at i + 1. Plateaus are
  never merged — a gap ends a plateau.

*** A BAND ALONE DOES NOT MEAN FLAT. *** The band only says how WIDE a span is,
not whether it went anywhere, so a steady climb that stays inside a wide band
passed as a plateau: a large-class span could rise a fifth of its own price from
first day to last and still be called quiet ground, which put a "plateau" exactly
on top of a breakout. The slope of the span therefore has to be flat too, and the
two flatness gates say that in the two ways it can fail:

  SLOPE  the least-squares slope per day, RELATIVE to the span's own mean — the
         average rise over the COURSE of the span. This is what stops a steady
         ramp: ten days at half a percent a day is a move, however narrow the band
         it happens to fit into.
  DRIFT  the ACTUAL change from the first close of the span to the last, again as
         a share of the mean. A different question from the slope, and it has to be
         asked separately: a span that falls, recovers and falls again fits a flat
         line through its middle while ending a fifth below where it began. A
         plateau has to END where it STARTED.

Both are fractions of the span's own mean, never absolute money, so "flat" means
the same at 20 EUR and at 700 EUR — a 15-EUR rise is nothing to a 700-EUR stock
and an entire regime to a 20-EUR one, and only the relative form can tell them
apart. A sideways span that genuinely goes nowhere is allowed to run for a year;
length is not what disqualifies a plateau, direction is.

The band is the span's ACTUAL min/max close, not a derived envelope. Every
tolerance is RELATIVE to the span's own mean, so one setting means the same thing
at 2 USD and at 900 USD.

A PLATEAU AND A BREAKOUT MAY HOLD ON THE SAME DAY, and nothing here prevents it.
This module knows nothing about breakouts and asks nothing about them: whether a
span is flat follows from the span alone, exactly as whether a move is a breakout
follows from that move alone. Where both are true — the usual case is an expiring
plateau that a breakout starts inside — BOTH are reported, because the transition
is a real thing that happened and suppressing either one would hide it. In practice
they collide rarely (a flat band is seldom also an escape), and that rarity is a
RESULT of the two definitions, never something imposed on them.

SIZE CLASSES: the same scan runs once per tolerance, so one symbol yields
several kinds of plateau at once — a SMALL, tight one and a LARGE, slack one
that may swallow a whole run of small ones. Inside a class plateaus never
overlap; ACROSS classes they may, and that overlap is the point. Each Plateau
carries the ``tier`` it came from. ``detect_tiers`` runs the set,
``tiers_from_config`` resolves it from parameters.yaml.

``band_tol`` and ``min_days`` are passed IN by the caller, which resolves them
from ``config/parameters.yaml`` (``plateau.band_tol`` / ``plateau.min_days``).
``detect`` hard-codes no threshold, does no I/O and prints nothing.

``cache()`` is the storage side and belongs to the engine: it runs the
detection over the configured universe and writes each symbol's plateaus into
Trade Memory (`patterns`, kind ``symbol_plateaus``), the same way
``baseline.cache`` stores its pattern payload. The dashboard never writes here —
it only calls ``detect`` for its own display.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from . import volume as volume_module

SOURCE = "plateau"


@dataclass
class Plateau:
    """One detected flat span of a symbol's close series."""

    symbol: str
    start_date: str      # ISO date of the first day in the span
    end_date: str        # ISO date of the last day in the span
    band_low: float      # actual min close over the span
    band_high: float     # actual max close over the span
    mean: float          # mean close over the span
    duration_days: int   # number of days in the span
    # The two flatness measurements, both RELATIVE to `mean` and both signed, so
    # a plateau can always be asked how flat it actually was rather than only
    # whether it passed. `slope_pct_per_day` is the fitted line over the course,
    # `drift_pct` the ACTUAL change from the first close to the last.
    slope_pct_per_day: float = 0.0
    drift_pct: float = 0.0
    tier: str = "small"  # which size class produced it
    # Measured by the volume primitive over this same span; None when no volume
    # series was supplied. A genuinely quiet plateau reads below 1.
    volume_ratio: Optional[float] = None
    volume_trend: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "band_low": self.band_low,
            "band_high": self.band_high,
            "mean": self.mean,
            "duration_days": self.duration_days,
            "slope_pct_per_day": self.slope_pct_per_day,
            "drift_pct": self.drift_pct,
            "tier": self.tier,
            "volume_ratio": self.volume_ratio,
            "volume_trend": self.volume_trend,
        }


def _iso(label: Any) -> str:
    """ISO date string for one index label; plain labels fall back to str()."""
    if isinstance(label, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(label).date().isoformat()
    return str(label)


def flatness(values: list[float]) -> tuple[float, float]:
    """
    How flat a span is: ``(slope_pct_per_day, drift_pct)``, both relative and signed.

    ``slope_pct_per_day`` is the least-squares slope of the closes against day
    number divided by the span's own mean — the average rise over the COURSE of the
    span. ``drift_pct`` is ``(last - first) / mean`` — where the span actually ENDED
    against where it began.

    THE TWO ARE DIFFERENT QUESTIONS and a plateau has to answer both with roughly
    zero. A fitted line runs flat through a span that fell, recovered and fell again
    while ending a fifth below its first day; and a span can end where it began after
    climbing steadily and giving it all back in one session. Each gate catches the
    one the other misses.

    Relative on purpose, and that is the whole point of dividing by the mean: 15 EUR
    of rise is noise at 700 EUR and a different stock at 20 EUR. Flat has to mean the
    same thing at both prices, so it is measured as a share of the level, never as an
    amount of money and never against the size of the chart around it.

    A span shorter than two days, or one with a non-positive mean, has nothing to
    measure: ``(0.0, 0.0)``.
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    mean_y = sum(values) / n
    if mean_y <= 0:
        return 0.0, 0.0
    sum_x = n * (n - 1) / 2.0
    sum_xx = (n - 1) * n * (2 * n - 1) / 6.0
    sum_y = sum(values)
    sum_xy = sum(i * v for i, v in enumerate(values))
    denominator = n * sum_xx - sum_x * sum_x
    if denominator <= 0:
        return 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope / mean_y, (values[-1] - values[0]) / mean_y


def _running_flatness(n: int, sum_y: float, sum_xy: float, first: float,
                      last: float) -> tuple[float, float]:
    """
    :func:`flatness` from the sums a left-to-right scan already carries.

    Identical arithmetic, accumulated instead of recomputed: the x-sums of
    ``0 .. n-1`` are closed forms, so extending a span by one day costs one
    multiplication instead of a fresh pass over it. A test holds the two forms
    against each other, because two implementations of one formula are exactly the
    kind of thing that drifts apart silently.
    """
    if n < 2:
        return 0.0, 0.0
    mean_y = sum_y / n
    if mean_y <= 0:
        return 0.0, 0.0
    sum_x = n * (n - 1) / 2.0
    sum_xx = (n - 1) * n * (2 * n - 1) / 6.0
    denominator = n * sum_xx - sum_x * sum_x
    if denominator <= 0:
        return 0.0, 0.0
    slope_pct = ((n * sum_xy - sum_x * sum_y) / denominator) / mean_y
    return slope_pct, (last - first) / mean_y


def _is_flat(n: int, sum_y: float, sum_xy: float, first: float, last: float, *,
             min_days: int, max_slope_pct_per_day: float,
             max_drift_pct: float) -> bool:
    """
    Do the two flatness gates hold for this span?

    Only asked once a span is long enough to BE a plateau. Below ``min_days`` a
    slope is one or two steps of noise rather than a direction, and refusing on it
    would end candidates before they could show what they are; the band alone
    governs there, and the gates decide the moment the span could qualify.
    """
    if n < max(2, min_days):
        return True
    slope_pct, drift_pct = _running_flatness(n, sum_y, sum_xy, first, last)
    return (abs(slope_pct) <= max_slope_pct_per_day
            and abs(drift_pct) <= max_drift_pct)


def _volume_of(volumes: Optional[pd.Series], settings, start: int, end: int) -> dict[str, Any]:
    """Attach the volume measurement for one span, or nothing when unavailable."""
    if volumes is None or settings is None or len(volumes) == 0:
        return {}
    return volume_module.measure(volumes, start, end, settings)


def detect(
    symbol: str,
    series: pd.Series,
    *,
    band_tol: float,
    min_days: int,
    max_slope_pct_per_day: float,
    max_drift_pct: float,
    tier: str = "small",
    volumes: Optional[pd.Series] = None,
    volume_settings: Optional[volume_module.Settings] = None,
) -> list[Plateau]:
    """
    Detect the flat spans of one symbol's close series.

    ``series`` is the close series WITH its dates (a date-indexed pd.Series, as
    returned by ``raw_data.long_term_csv.load``). Returns the plateaus in
    chronological order; they never overlap. Empty list when nothing qualifies.

    Three tolerances, all of them relative and all of them passed in: the BAND
    (how wide the span may be), the SLOPE (how steep its fitted line may be per
    day) and the DRIFT (how far that line may travel from the first day to the
    last). A span must satisfy all three; the band says the span is narrow, the
    other two say it is going nowhere, and only the three together mean "flat".

    ``volumes`` + ``volume_settings`` are optional: supply the same series' daily
    volume and each plateau is measured against the stock's own normal (see
    ``applied_data/volume.py``). Detection itself is untouched by volume.
    """
    if series is None or len(series) == 0 or min_days <= 0:
        return []

    prices = [float(v) for v in series.to_numpy()]
    labels = list(series.index)
    total_days = len(prices)

    found: list[Plateau] = []
    i = 0
    while i < total_days:
        running_min = running_max = prices[i]
        running_sum = prices[i]
        running_xy = 0.0          # sum of (day - i) * price, for the fitted line
        last = i
        j = i + 1
        while j < total_days:
            new_min = min(running_min, prices[j])
            new_max = max(running_max, prices[j])
            length = j - i + 1
            new_sum = running_sum + prices[j]
            new_xy = running_xy + (j - i) * prices[j]
            mean_so_far = new_sum / length
            if mean_so_far <= 0 or (new_max - new_min) / mean_so_far > band_tol:
                break
            if not _is_flat(length, new_sum, new_xy, prices[i], prices[j],
                            min_days=min_days,
                            max_slope_pct_per_day=max_slope_pct_per_day,
                            max_drift_pct=max_drift_pct):
                break             # narrow, but going somewhere: not a plateau
            running_min, running_max = new_min, new_max
            running_sum, running_xy = new_sum, new_xy
            last = j
            j += 1

        duration_days = last - i + 1
        if duration_days >= min_days:
            slope_pct, drift_pct = _running_flatness(duration_days, running_sum,
                                                     running_xy, prices[i],
                                                     prices[last])
            found.append(
                Plateau(
                    symbol=symbol,
                    start_date=_iso(labels[i]),
                    end_date=_iso(labels[last]),
                    band_low=running_min,
                    band_high=running_max,
                    mean=running_sum / duration_days,
                    duration_days=duration_days,
                    slope_pct_per_day=slope_pct,
                    drift_pct=drift_pct,
                    tier=tier,
                    **_volume_of(volumes, volume_settings, i, last),
                )
            )
            i = last + 1          # continue AFTER the plateau: no overlap
        else:
            i += 1                # discard the candidate, shift by one day
    return found


def causal_scan(symbol: str, series: pd.Series, *, band_tol: float, min_days: int,
                max_slope_pct_per_day: float, max_drift_pct: float,
                tier: str = "small") -> dict[str, Any]:
    """
    The SAME greedy scan as :func:`detect`, reported the way a day could know it.

    Two answers come out of one left-to-right pass, and both are strictly
    past-only — day x is described using ``prices[0 .. x]`` and nothing else:

    ``reaching[x]``
        the plateau that REACHES day x, closed at x, with its band, mean and
        duration recomputed over ``[i .. x]`` alone — or ``None`` when the
        current candidate is still shorter than ``min_days``. This is what a
        detection run on the truncated series ``[0 .. x]`` returns, because the
        truncation is exactly what closes the candidate.
    ``completed``
        the plateaus whose end was already ESTABLISHED, each with the
        ``known_date`` on which it became knowable — the day the band broke,
        which is the first day one can say the plateau is over. A plateau still
        running at the end of the series is not completed and carries no date.

    Why this exists at all: a finished plateau from :func:`detect` is shaped by
    every day it contains, including days after x. Its mean, its band and even
    whether it qualified are post-x facts, so clipping such an object at x does
    NOT make it past-only — the numbers are already contaminated. The honest
    answer for day x has to be recomputed with x as the right edge, which is what
    this scan does for every day at once.
    """
    empty: dict[str, Any] = {"completed": [], "reaching": []}
    if series is None or len(series) == 0 or min_days <= 0:
        return empty

    prices = [float(v) for v in series.to_numpy()]
    labels = list(series.index)
    total = len(prices)

    completed: list[dict[str, Any]] = []
    reaching: list[Optional[dict[str, Any]]] = [None] * total

    def _record_reach(start: int, end: int, low: float, high: float, mean: float,
                      slope_pct: float, drift_pct: float) -> None:
        if end - start + 1 < min_days:
            return
        reaching[end] = Plateau(
            symbol=symbol,
            start_date=_iso(labels[start]),
            end_date=_iso(labels[end]),
            band_low=low,
            band_high=high,
            mean=mean,
            duration_days=end - start + 1,
            slope_pct_per_day=slope_pct,
            drift_pct=drift_pct,
            tier=tier,
        ).to_dict()

    i = 0
    while i < total:
        running_min = running_max = prices[i]
        running_sum = prices[i]
        running_xy = 0.0
        last = i
        _record_reach(i, i, running_min, running_max, running_sum, 0.0, 0.0)
        j = i + 1
        while j < total:
            new_min = min(running_min, prices[j])
            new_max = max(running_max, prices[j])
            length = j - i + 1
            new_sum = running_sum + prices[j]
            new_xy = running_xy + (j - i) * prices[j]
            mean_so_far = new_sum / length
            if mean_so_far <= 0 or (new_max - new_min) / mean_so_far > band_tol:
                break
            if not _is_flat(length, new_sum, new_xy, prices[i], prices[j],
                            min_days=min_days,
                            max_slope_pct_per_day=max_slope_pct_per_day,
                            max_drift_pct=max_drift_pct):
                break
            running_min, running_max = new_min, new_max
            running_sum, running_xy = new_sum, new_xy
            last = j
            # the view day x = j would close the candidate here
            slope_pct, drift_pct = _running_flatness(length, running_sum, running_xy,
                                                     prices[i], prices[j])
            _record_reach(i, j, running_min, running_max, running_sum / length,
                          slope_pct, drift_pct)
            j += 1

        duration = last - i + 1
        if duration >= min_days:
            slope_pct, drift_pct = _running_flatness(duration, running_sum, running_xy,
                                                    prices[i], prices[last])
            plateau = Plateau(
                symbol=symbol,
                start_date=_iso(labels[i]),
                end_date=_iso(labels[last]),
                band_low=running_min,
                band_high=running_max,
                mean=running_sum / duration,
                duration_days=duration,
                slope_pct_per_day=slope_pct,
                drift_pct=drift_pct,
                tier=tier,
            ).to_dict()
            # knowable only once the band has visibly broken: the day after it
            plateau["known_date"] = _iso(labels[last + 1]) if last + 1 < total else None
            completed.append(plateau)
            i = last + 1          # continue AFTER the plateau: no overlap
        else:
            i += 1                # discard the candidate, shift by one day
    return {"completed": completed, "reaching": reaching}


# ---------------------------------------------------------------------------
# Size classes. Every caller (dashboard, cache, CLI) resolves the SAME set here,
# so there is exactly one place that reads the tunables.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    """One size class: a name, its relative band width, minimum length and flatness."""

    name: str
    band_tol: float
    min_days: int
    max_slope_pct_per_day: float
    max_drift_pct: float


def tiers_from_config(config) -> list[Tier]:
    """
    Resolve the configured size classes. ``large`` is expressed as a FACTOR on
    the small band, so widening the small class widens the large one with it.
    """
    band_tol = float(config.get("plateau.band_tol"))            # tunable
    min_days = int(config.get("plateau.min_days"))              # tunable
    large_factor = float(config.get("plateau.large_factor"))    # tunable
    large_min_days = int(config.get("plateau.large_min_days"))  # tunable
    max_slope = float(config.get("plateau.max_slope_pct_per_day"))     # tunable
    max_drift = float(config.get("plateau.max_drift_pct"))             # tunable
    flat_factor = float(config.get("plateau.large_flat_factor"))       # tunable
    return [
        Tier("small", band_tol, min_days, max_slope, max_drift),
        Tier("large", band_tol * large_factor, large_min_days,
             max_slope * flat_factor, max_drift * flat_factor),
    ]


def detect_tiers(symbol: str, series: pd.Series, tiers: list[Tier],
                 volumes: Optional[pd.Series] = None,
                 volume_settings: Optional[volume_module.Settings] = None) -> list[Plateau]:
    """
    Run the scan once per size class and return everything together, sorted by
    start date. Plateaus of the SAME class never overlap; plateaus of different
    classes may — a large one typically spans several small ones.
    """
    found: list[Plateau] = []
    for tier in tiers:
        found.extend(
            detect(symbol, series, band_tol=tier.band_tol, min_days=tier.min_days,
                   max_slope_pct_per_day=tier.max_slope_pct_per_day,
                   max_drift_pct=tier.max_drift_pct,
                   tier=tier.name, volumes=volumes, volume_settings=volume_settings)
        )
    found.sort(key=lambda p: (p.start_date, p.end_date, p.tier))
    return found


# ---------------------------------------------------------------------------
# Core-side storage. The analysis lives in the engine: it is
# computed once and persisted; readers read, they do not recompute.
# ---------------------------------------------------------------------------

PATTERN_KIND = "symbol_plateaus"


def analyze(symbol: str, closes: pd.Series, tiers: list[Tier],
            volumes: Optional[pd.Series] = None,
            volume_settings: Optional[volume_module.Settings] = None) -> dict[str, Any]:
    """The cache payload for one symbol: every plateau plus the classes used."""
    found = detect_tiers(symbol, closes, tiers, volumes=volumes,
                         volume_settings=volume_settings)
    return {
        "tiers": [{"name": t.name, "band_tol": t.band_tol, "min_days": t.min_days,
                   "max_slope_pct_per_day": t.max_slope_pct_per_day,
                   "max_drift_pct": t.max_drift_pct} for t in tiers],
        "count": len(found),
        "counts_by_tier": {t.name: sum(1 for p in found if p.tier == t.name) for t in tiers},
        "plateaus": [p.to_dict() for p in found],
    }
