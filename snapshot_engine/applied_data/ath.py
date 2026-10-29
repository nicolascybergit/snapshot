"""
Applied Data — all-time high history (a FILE, not a level). A PRIMITIVE.

Deterministic, pure maths, no LLM. An all-time high is the highest close the
series has reached SO FAR. This module measures the HISTORY and STRUCTURE of
those highs: how often a new one was set, when, how long the market waited
between them, how big each step up was, and how far below the last one price
stands right now.

*** ROLLING ATH ONLY — NEVER THE GLOBAL MAX OF THE SERIES. ***

At day x the ATH is ``max(close[0 .. x])`` — the past and nothing else. The
maximum of the whole series is a fact of the day it is finally reached, never a
value an earlier day is allowed to know. Every function here walks left to right
and reads only what has already happened, so a caller may ask about any index
and gets the answer that day itself would have had. Taking the series maximum
anywhere would leak the future and is forbidden.

A NEW ATH is set on day x when ``close[x] > max(close[0 .. x-1])`` — strictly
greater, so merely touching the old high again is not a new one. The FIRST day
of the series is the first ATH by definition (there is no earlier high to beat);
it is the baseline every later step is measured from and therefore carries no
previous price, no gap and no step.

In:  a date-indexed close series for ONE symbol (``raw_data.long_term_csv.load``
     / ``close_series`` already return the dates together with the prices)
Out: the running ATH per day, the chronological list of new-ATH events, one
     aggregate summary, and the distance to the last ATH at any day.

This is the LOW level: price and dates only. It knows nothing about plateaus,
breakouts or volume, imports none of them and takes no events as input — it runs
on any CSV that has none of that. Reading an ATH together with those events is a
BEHAVIOUR analysis one level above and does not live here.

There is no threshold to set: "a new high" is a definition, not a judgement, so
this primitive declares NO tunable (reasoning in ``BUILD_NOTES.md``). Callers
that want to ignore tiny new highs filter on the ``step_pct`` every event
already carries. Pure maths, no I/O, nothing printed (``_report`` is the CLI
end).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from statistics import median
from typing import Any, Optional

import pandas as pd

SOURCE = "ath"


@dataclass
class AthEvent:
    """One day on which the close set a new all-time high."""

    symbol: str
    date: str                              # ISO date the new high was set
    price: float                           # the new ATH close
    prev_ath_price: Optional[float]        # the high it surpassed (None on the first)
    days_since_prev_ath: Optional[int]     # trading days since the previous ATH
    step_pct: Optional[float]              # (price - prev_ath_price) / prev_ath_price

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "symbol": self.symbol,
            "date": self.date,
            "price": self.price,
            "prev_ath_price": self.prev_ath_price,
            "days_since_prev_ath": self.days_since_prev_ath,
            "step_pct": self.step_pct,
        }


@dataclass
class AthStats:
    """Aggregate summary over every ATH event of one series."""

    symbol: str
    ath_count: int
    first_ath_date: str
    last_ath_date: str
    gap_days_min: Optional[int]            # None while only the baseline ATH exists
    gap_days_median: Optional[float]
    gap_days_max: Optional[int]
    step_pct_median: Optional[float]
    step_pct_max: Optional[float]
    current_gap_pct: float                 # last close vs. last rolling ATH (<= 0)
    current_days_since_ath: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "symbol": self.symbol,
            "ath_count": self.ath_count,
            "first_ath_date": self.first_ath_date,
            "last_ath_date": self.last_ath_date,
            "gap_days_min": self.gap_days_min,
            "gap_days_median": self.gap_days_median,
            "gap_days_max": self.gap_days_max,
            "step_pct_median": self.step_pct_median,
            "step_pct_max": self.step_pct_max,
            "current_gap_pct": self.current_gap_pct,
            "current_days_since_ath": self.current_days_since_ath,
        }


def _iso(label: Any) -> str:
    """ISO date string for one index label; plain labels fall back to str()."""
    if isinstance(label, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(label).date().isoformat()
    return str(label)


def _closes(series: Optional[pd.Series]) -> list[float]:
    if series is None or len(series) == 0:
        return []
    return [float(v) for v in series.to_numpy()]


def _step_pct(price: float, prev_ath_price: float) -> Optional[float]:
    """Relative step up from one ATH to the next; None for a degenerate base."""
    if prev_ath_price <= 0:
        return None
    return (price - prev_ath_price) / prev_ath_price


def rolling_ath(series: pd.Series) -> list[float]:
    """
    The running all-time high, one value per day: ``max(close[0 .. i])``.

    Same length as the series, monotonically non-decreasing, and past-only by
    construction — the loop has never seen day ``i + 1`` when it writes day
    ``i``. This is THE guard against a future leak: everything else in this
    module is derived from this walk, never from ``series.max()``.
    """
    running: list[float] = []
    high: Optional[float] = None
    for price in _closes(series):
        high = price if high is None or price > high else high
        running.append(high)
    return running


def ath_events(series: pd.Series, *, symbol: str = "") -> list[AthEvent]:
    """
    Every day that set a NEW all-time high, in chronological order.

    The first day is the first ATH (the baseline: no previous high, hence no gap
    and no step). Every later day qualifies only on a STRICTLY higher close than
    the running high before it, so a flat re-touch of the old high is not an
    event. Empty list for an empty series.
    """
    prices = _closes(series)
    if not prices:
        return []
    labels = list(series.index)

    events = [
        AthEvent(
            symbol=symbol,
            date=_iso(labels[0]),
            price=prices[0],
            prev_ath_price=None,
            days_since_prev_ath=None,
            step_pct=None,
        )
    ]
    high, high_idx = prices[0], 0
    for i in range(1, len(prices)):
        if prices[i] > high:               # strictly higher than the PAST high
            events.append(
                AthEvent(
                    symbol=symbol,
                    date=_iso(labels[i]),
                    price=prices[i],
                    prev_ath_price=high,
                    days_since_prev_ath=i - high_idx,
                    step_pct=_step_pct(prices[i], high),
                )
            )
            high, high_idx = prices[i], i
    return events


def distance_to_ath(series: pd.Series, idx: int) -> dict[str, Any]:
    """
    How far below the last all-time high day ``idx`` stands, and how long since.

    The high is taken from ``close[0 .. idx]`` ONLY — a higher close later in the
    series is invisible here, which is the whole point: this answers what that
    day itself could have known. ``gap_pct`` is negative below the high and 0.0
    on the day the high is set. Negative indices count from the end
    (``-1`` = last day); an empty series measures nothing and returns ``{}``.
    """
    prices = _closes(series)
    if not prices:
        return {}
    labels = list(series.index)

    total = len(prices)
    i = int(idx)
    if i < 0:
        i += total
    i = max(0, min(total - 1, i))

    high_idx = 0
    for k in range(1, i + 1):              # walk the past, never beyond idx
        if prices[k] > prices[high_idx]:
            high_idx = k
    high = prices[high_idx]

    return {
        "last_ath_value": high,
        "last_ath_date": _iso(labels[high_idx]),
        "days_since_ath": i - high_idx,
        "gap_pct": (prices[i] - high) / high if high > 0 else 0.0,
    }


def ath_stats(series: pd.Series, *, symbol: str = "") -> Optional[AthStats]:
    """
    Aggregate summary over all ATH events: how often, when, the time gaps and the
    price steps between consecutive highs, plus where the LAST day of the series
    stands relative to the high it knows. None for an empty series.
    """
    events = ath_events(series, symbol=symbol)
    if not events:
        return None

    gaps = [e.days_since_prev_ath for e in events if e.days_since_prev_ath is not None]
    steps = [e.step_pct for e in events if e.step_pct is not None]
    current = distance_to_ath(series, len(series) - 1)

    return AthStats(
        symbol=symbol,
        ath_count=len(events),
        first_ath_date=events[0].date,
        last_ath_date=events[-1].date,
        gap_days_min=min(gaps) if gaps else None,
        gap_days_median=float(median(gaps)) if gaps else None,
        gap_days_max=max(gaps) if gaps else None,
        step_pct_median=float(median(steps)) if steps else None,
        step_pct_max=max(steps) if steps else None,
        current_gap_pct=float(current["gap_pct"]),
        current_days_since_ath=int(current["days_since_ath"]),
    )
