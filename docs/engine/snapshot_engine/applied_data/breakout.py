"""
Applied Data — breakout detection (a FILE, not a level). Sibling of plateau.py.

Deterministic, pure maths, no LLM. A breakout is a CHANGE OF CHARACTER in the
price path: a strong, fast, directional move that leaves a contained "before"
regime and establishes a genuinely different "after" regime.

What that means concretely — and what it deliberately does NOT mean:

  * A plateau before the move is an ENRICHER, never a requirement — and never an
    obstacle either: a plateau that is still running does not forbid a breakout,
    and a breakout does not delete a plateau it starts inside. The two are decided
    independently and are allowed to overlap; where both hold, both are reported.
    Price that was far too jumpy to qualify as a plateau can still break out; the
    plateau only supplies ``band_ref`` / ``had_plateau_before`` when one happens to sit
    right in front of the move.
  * "Zig-zag before" is not the qualifier. The qualifier is a directional move
    PLUS a real regime change, measured by comparing the three windows
    BEFORE / MOVE / AFTER against each other.
  * A big zig-zag that reverts is NOT a breakout: if the after-window sits at
    the same level as the before-window, no new regime was established
    (``level_shift``). The level also has to clear the before-window's OWN
    noise (``min_regime_gap_std``) — otherwise "a different level" is just a
    swing the old regime made anyway, which is exactly the similar-level +
    similar-volatility case.
  * A move that merely continues what the before-window was already doing at a
    similar rate is NOT a breakout: the move has to be clearly steeper than the
    pre-slope (``pre_slope_factor``) — acceleration, not continuation. That
    question is asked ONLY of a move going the same way as the run-up; a move
    that turns AGAINST it is a reversal and is judged by ``reversal_slope_factor``
    instead (it must at least match the pace of what it turns).
  * A V — a sharp move immediately reversed by a sharp move the other way — is
    TWO breakouts, and both are kept. The before-window of the second leg is the
    first leg, so its "noise" is really that leg's amplitude; asking the rebound
    to clear it would ask it to out-travel the crash it is rebounding from, so
    the regime gap is scaled by ``v_reversal_relax`` for a reversal. The
    de-duplication is per DIRECTION for the same reason: the two legs share their
    turning day and must not crowd each other out.
  * plateau -> zig-zag -> plateau: entering the zig-zag fails on directionality
    (``move_r2``) and on level shift, so it drops out on its own.

Algorithm (three-window comparison, deterministic):
  For every candidate move window [t1, t2] of length L (L scanned from
  MIN_MOVE_LEN up to ``max_days``), with ``context_days`` of context on each
  side, compute the move (delta, speed, direction, R^2 of a linear fit), the
  before window (slope per day relative to its mean, mean, std) and the after
  window (mean, std). A candidate qualifies when ALL six thresholds hold. The
  candidate windows overlap heavily, so qualifying candidates are clustered and
  only the strongest survives per cluster (largest |delta_pct|, then speed) —
  the result never overlaps WITHIN a tier.

SIZE CLASSES mirror plateau.py exactly: ``small`` carries the base thresholds,
``large`` raises the required move by ``large_factor`` and allows a longer
``max_days_large``. Each class scans independently: no overlap inside a class,
overlap ACROSS classes is allowed and expected. ``breakout_tiers_from_config``
is the ONE place these tunables are read.

All thresholds are passed IN by the caller; this module hard-codes nothing,
does no I/O and prints nothing (``cache``/``_report`` are the core-side ends).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import volume as volume_module
from .plateau import Plateau

SOURCE = "breakout"

# Structural floor, not a threshold: R^2 over a straight line needs at least
# three points to say anything, so the shortest move window spans two steps.
MIN_MOVE_LEN = 2


@dataclass
class Breakout:
    """One detected change-of-character move."""

    symbol: str
    start_date: str
    end_date: str
    price_from: float
    price_to: float
    delta_abs: float
    delta_pct: float
    speed_per_day: float
    direction: str                  # "up" | "down"
    move_r2: float
    band_ref: Optional[float]       # plateau edge in the move's direction, else None
    had_plateau_before: bool
    tier: str = "small"             # which size class produced it
    # Measured by the volume primitive over the move span; None when no volume
    # series was supplied. A conviction break reads above 1, a thin one below.
    volume_ratio: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "price_from": self.price_from,
            "price_to": self.price_to,
            "delta_abs": self.delta_abs,
            "delta_pct": self.delta_pct,
            "speed_per_day": self.speed_per_day,
            "direction": self.direction,
            "move_r2": self.move_r2,
            "band_ref": self.band_ref,
            "had_plateau_before": self.had_plateau_before,
            "tier": self.tier,
            "volume_ratio": self.volume_ratio,
        }


@dataclass(frozen=True)
class Params:
    """The six qualifying thresholds plus the context window length."""

    min_move_pct: float
    min_speed_pct_per_day: float
    max_days: int
    min_move_r2: float
    pre_slope_factor: float
    min_level_shift: float
    min_regime_gap_std: float
    context_days: int
    reversal_slope_factor: float
    v_reversal_relax: float


@dataclass(frozen=True)
class Tier:
    """One size class: a name and the thresholds it scans with."""

    name: str
    params: Params


@dataclass(frozen=True)
class Candidate:
    """A qualifying window before de-duplication: the move plus its index span."""

    strength: float
    speed: float
    start: int
    end: int
    breakout: Breakout


def _iso(label: Any) -> str:
    """ISO date string for one index label; plain labels fall back to str()."""
    if isinstance(label, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(label).date().isoformat()
    return str(label)


def _slope_and_r2(values: np.ndarray) -> tuple[float, float]:
    """Least-squares slope (per step) and R^2 of a straight fit through values."""
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    predicted = slope * x + intercept
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= 0.0:                       # a perfectly flat window fits exactly
        return float(slope), 1.0
    residual = float(np.sum((values - predicted) ** 2))
    return float(slope), float(1.0 - residual / total)


def _is_continuation(move_slope: float, pre_slope: float, params: Params) -> bool:
    """
    Is this move merely MORE OF THE SAME? Only then does the steepness gate apply.

    The gate exists to reject continuation: price was already running this way, it
    just kept running. That question only makes sense when the move goes the SAME
    way as the run-up — a move that turns AGAINST it is a reversal, and a reversal
    is never a continuation of what it reverses. Comparing the two speeds without
    looking at their signs is what made AMD's June crash unqualifiable: it was
    asked to fall three times faster than the rally before it had risen.
    """
    if move_slope * pre_slope <= 0:
        # A REVERSAL. It cannot be a continuation of what it turns, so the
        # continuation factor does not apply — but it still has to be worth the
        # name: a counter-move slower than the run it reverses is a pullback
        # inside that run, not a change of character. Hence its own, much lower
        # bar: at least the pace of what it is turning.
        return abs(move_slope) < params.reversal_slope_factor * abs(pre_slope)
    return abs(move_slope) < params.pre_slope_factor * abs(pre_slope)


def _regime_gap_needed(pre_std: float, pre_slope: float, move_slope: float,
                       params: Params) -> float:
    """
    How far the new level must sit from the old one to count as a new regime.

    Normally that is ``min_regime_gap_std`` times the before-window's own noise:
    a level inside the old regime's swing range is not a new level. But when the
    before-window is itself a fast move in the OPPOSITE direction — the other leg
    of a V — its "noise" is not noise at all, it is the amplitude of that move.
    Requiring the reversal to clear it means requiring it to out-travel the very
    move it is reversing, which no rebound off a crash can do while it is still a
    rebound. In that case the requirement is scaled by ``v_reversal_relax``.
    """
    needed = params.min_regime_gap_std * pre_std
    if move_slope * pre_slope < 0:                  # the other leg of a V
        return needed * params.v_reversal_relax
    return needed


def _plateau_index(plateaus: list[Plateau],
                   labels: list[Any]) -> dict[int, list[tuple[int, Plateau]]]:
    """Map each plateau to the index its span ENDS on, plus where it started."""
    position = {_iso(label): i for i, label in enumerate(labels)}
    ends: dict[int, list[tuple[int, Plateau]]] = {}
    for p in plateaus:
        end = position.get(p.end_date)
        start = position.get(p.start_date)
        if end is not None and start is not None:
            ends.setdefault(end, []).append((start, p))
    return ends


def _plateau_before(ends: dict[int, list[tuple[int, Plateau]]], t1: int, t2: int,
                    reach: int, direction: str) -> tuple[bool, Optional[float]]:
    """
    Enrichment only: did a plateau sit right in front of the move? It must have
    STARTED before the move and may end anywhere from ``reach`` days earlier up
    to the end of the move — the first steps out of a band often still fall
    inside it, so a plateau routinely bleeds a day or two into the break. The
    nearest such plateau wins. Never a qualifying condition: a breakout is
    perfectly valid with no plateau at all.
    """
    for i in range(t2, max(-1, t1 - reach - 1), -1):
        for start, p in ends.get(i, []):
            if start < t1:
                return True, (p.band_high if direction == "up" else p.band_low)
    return False, None


def detect(
    symbol: str,
    series: pd.Series,
    *,
    params: Params,
    plateaus: Optional[list[Plateau]] = None,
    tier: str = "small",
    volumes: Optional[pd.Series] = None,
    volume_settings: Optional[volume_module.Settings] = None,
) -> list[Breakout]:
    """
    Detect the change-of-character moves in one symbol's close series.

    ``series`` is the date-indexed close series (``raw_data.long_term_csv.load``).
    ``plateaus`` is optional enrichment: pass the plateaus of the same series to
    populate ``band_ref`` / ``had_plateau_before``. ``volumes`` +
    ``volume_settings`` are optional too: supply them and each move is measured
    against the stock's own normal volume. Neither changes what qualifies.
    Returns the surviving breakouts in chronological order; they never overlap.
    """
    if series is None or len(series) == 0 or params.max_days < MIN_MOVE_LEN:
        return []

    prices = np.asarray(series.to_numpy(), dtype=float)
    labels = list(series.index)
    total = len(prices)
    context = int(params.context_days)
    ends = _plateau_index(plateaus or [], labels)

    candidates: list[Candidate] = []
    for t1 in range(context, total):
        pre = prices[t1 - context:t1]
        if len(pre) < 2 or pre.mean() <= 0:
            continue
        pre_slope_raw, _ = _slope_and_r2(pre)
        pre_mean = float(pre.mean())
        pre_slope = pre_slope_raw / pre_mean                 # relative, per day

        for length in range(MIN_MOVE_LEN, int(params.max_days) + 1):
            t2 = t1 + length
            if t2 >= total or t2 + context >= total:
                break                                        # no after window left

            price_from, price_to = float(prices[t1]), float(prices[t2])
            if price_from <= 0:
                continue
            delta_abs = price_to - price_from
            delta_pct = delta_abs / price_from
            if abs(delta_pct) < params.min_move_pct:                    # (a)
                continue
            speed = abs(delta_pct) / length
            if speed < params.min_speed_pct_per_day:                    # (b)
                continue
            # (c) holds by construction: length never exceeds max_days
            _, move_r2 = _slope_and_r2(prices[t1:t2 + 1])
            if move_r2 < params.min_move_r2:                            # (d)
                continue
            move_slope = delta_pct / length
            if _is_continuation(move_slope, pre_slope, params):              # (e)
                continue
            post = prices[t2 + 1:t2 + 1 + context]
            if len(post) < 2:
                continue
            post_mean = float(post.mean())
            level_shift = abs(post_mean - pre_mean) / pre_mean
            if level_shift < params.min_level_shift:                    # (f)
                continue
            pre_std = float(pre.std(ddof=0))
            if abs(post_mean - pre_mean) < _regime_gap_needed(                # (g)
                    pre_std, pre_slope, move_slope, params):
                continue

            direction = "up" if delta_abs > 0 else "down"
            had_plateau, band_ref = _plateau_before(ends, t1, t2, context, direction)
            candidates.append(Candidate(
                strength=abs(delta_pct),
                speed=speed,
                start=t1,
                end=t2,
                breakout=Breakout(
                    symbol=symbol,
                    start_date=_iso(labels[t1]),
                    end_date=_iso(labels[t2]),
                    price_from=price_from,
                    price_to=price_to,
                    delta_abs=delta_abs,
                    delta_pct=delta_pct,
                    speed_per_day=speed,
                    direction=direction,
                    move_r2=move_r2,
                    band_ref=band_ref,
                    had_plateau_before=had_plateau,
                    tier=tier,
                    volume_ratio=(
                        volume_module.ratio(volumes, t1, t2,
                                            window=volume_settings.baseline_window)
                        if volumes is not None and volume_settings is not None
                        and len(volumes) > 0 else None
                    ),
                ),
            ))

    return _strongest_per_cluster(candidates)


def in_progress(series: pd.Series, *, params: Params) -> list[Optional[str]]:
    """
    Per day: is a move ENDING on that day forming into a breakout? Returns its
    direction (``"up"`` / ``"down"``) or ``None``.

    This is the PAST-ONLY half of :func:`detect`. A finished breakout is judged
    on three windows — before, move, after — but the after window lies in the
    future of the day being asked about, and so does the level shift it measures.
    A day can only know the first four gates: the move is big enough
    (``min_move_pct``), fast enough (``min_speed_pct_per_day``), straight enough
    (``min_move_r2``) and clearly steeper than the run-up before it
    (``pre_slope_factor``). A move that passes those is breaking out AS FAR AS
    THAT DAY CAN TELL — and stays so in the record even if the level later
    reverts and the finished detection rejects it. That is the honest answer to
    "is something breaking right now", and it is deliberately not the same
    question as "did a breakout happen here".

    The longest qualifying move wins, so a day inside a long clean run reports
    the run rather than its last two days.
    """
    if series is None or len(series) == 0 or params.max_days < MIN_MOVE_LEN:
        return [None] * (0 if series is None else len(series))

    prices = np.asarray(series.to_numpy(), dtype=float)
    total = len(prices)
    context = int(params.context_days)
    out: list[Optional[str]] = [None] * total

    for t2 in range(total):
        best: Optional[str] = None
        best_strength = 0.0
        for length in range(MIN_MOVE_LEN, int(params.max_days) + 1):
            t1 = t2 - length
            if t1 - context < 0:
                break                                   # no before window left
            pre = prices[t1 - context:t1]
            if len(pre) < 2 or pre.mean() <= 0:
                continue
            pre_slope_raw, _ = _slope_and_r2(pre)
            pre_slope = pre_slope_raw / float(pre.mean())

            price_from, price_to = float(prices[t1]), float(prices[t2])
            if price_from <= 0:
                continue
            delta_pct = (price_to - price_from) / price_from
            if abs(delta_pct) < params.min_move_pct:                     # (a)
                continue
            if abs(delta_pct) / length < params.min_speed_pct_per_day:   # (b)
                continue
            _, move_r2 = _slope_and_r2(prices[t1:t2 + 1])
            if move_r2 < params.min_move_r2:                             # (d)
                continue
            if abs(delta_pct / length) < params.pre_slope_factor * abs(pre_slope):   # (e)
                continue
            if abs(delta_pct) > best_strength:
                best_strength = abs(delta_pct)
                best = "up" if delta_pct > 0 else "down"
        out[t2] = best
    return out


def causal(symbol: str, series: pd.Series, *, params: Params,
           tier: str = "small") -> list[dict[str, Any]]:
    """
    The same detection as :func:`detect`, ordered by when each breakout could
    first be KNOWN — and never using a day beyond that moment.

    A finished breakout is judged on three windows, and the last of them lies
    AFTER the move: a move ending on ``t2`` is only decidable once
    ``context_days`` further days exist, i.e. on day ``t2 + context_days``. That
    day is the breakout's ``known_date``, and every gate is evaluated from days
    up to it — never past it. So this function answers "which breakouts had
    happened, as far as day x could tell" by taking those with
    ``known_date <= x``, without a single post-x value entering the numbers.

    De-duplication is causal too, and that is the one place where this differs
    from :func:`detect`: the global scan keeps the strongest of a cluster, which
    may be a candidate that only became decidable later; here a candidate is
    dropped when it overlaps one that was already known. Within a single
    knowledge day the strongest still wins. Judging a day by a stronger move that
    could not yet be seen would be the very leak this function exists to close.
    """
    if series is None or len(series) == 0 or params.max_days < MIN_MOVE_LEN:
        return []

    prices = np.asarray(series.to_numpy(), dtype=float)
    labels = list(series.index)
    total = len(prices)
    context = int(params.context_days)

    admitted: list[dict[str, Any]] = []
    spans: list[tuple[int, int, str]] = []
    for known in range(total):
        t2 = known - context
        if t2 <= 0:
            continue
        found: list[Candidate] = []
        for length in range(MIN_MOVE_LEN, int(params.max_days) + 1):
            t1 = t2 - length
            if t1 - context < 0:
                break                                    # no before window left
            pre = prices[t1 - context:t1]
            if len(pre) < 2 or pre.mean() <= 0:
                continue
            pre_slope_raw, _ = _slope_and_r2(pre)
            pre_mean = float(pre.mean())
            pre_slope = pre_slope_raw / pre_mean

            price_from, price_to = float(prices[t1]), float(prices[t2])
            if price_from <= 0:
                continue
            delta_abs = price_to - price_from
            delta_pct = delta_abs / price_from
            if abs(delta_pct) < params.min_move_pct:                     # (a)
                continue
            speed = abs(delta_pct) / length
            if speed < params.min_speed_pct_per_day:                     # (b)
                continue
            _, move_r2 = _slope_and_r2(prices[t1:t2 + 1])
            if move_r2 < params.min_move_r2:                             # (d)
                continue
            if _is_continuation(delta_pct / length, pre_slope, params):       # (e)
                continue
            post = prices[t2 + 1:known + 1]              # ends ON the knowledge day
            if len(post) < 2:
                continue
            post_mean = float(post.mean())
            level_shift = abs(post_mean - pre_mean) / pre_mean
            if level_shift < params.min_level_shift:                     # (f)
                continue
            pre_std = float(pre.std(ddof=0))
            if abs(post_mean - pre_mean) < _regime_gap_needed(                 # (g)
                    pre_std, pre_slope, delta_pct / length, params):
                continue

            found.append(Candidate(
                strength=abs(delta_pct), speed=speed, start=t1, end=t2,
                breakout=Breakout(
                    symbol=symbol,
                    start_date=_iso(labels[t1]),
                    end_date=_iso(labels[t2]),
                    price_from=price_from,
                    price_to=price_to,
                    delta_abs=delta_abs,
                    delta_pct=delta_pct,
                    speed_per_day=speed,
                    direction="up" if delta_abs > 0 else "down",
                    move_r2=move_r2,
                    band_ref=None,          # plateau enrichment is not a past-only fact
                    had_plateau_before=False,
                    tier=tier,
                ),
            ))

        found.sort(key=lambda c: (-c.strength, -c.speed, c.start))
        for candidate in found:
            # only a move of the SAME direction can crowd this one out: the two
            # legs of a V overlap on the turning day and are both real
            if any(candidate.start <= end and start <= candidate.end
                   for start, end, direction in spans
                   if direction == candidate.breakout.direction):
                continue                                  # an older, already-known move
            payload = candidate.breakout.to_dict()
            payload["known_date"] = _iso(labels[known])
            admitted.append(payload)
            spans.append((candidate.start, candidate.end, candidate.breakout.direction))
    return admitted


def _strongest_per_cluster(candidates: list["Candidate"]) -> list[Breakout]:
    """
    Candidate windows overlap heavily. Take the strongest one (largest move,
    then fastest, then earliest for a stable order), drop everything of the SAME
    DIRECTION that overlaps it, repeat. Survivors of one direction never overlap;
    an up and a down leg may, and must — a V turns on a single day that belongs
    to both of its legs.
    """
    ranked = sorted(candidates, key=lambda c: (-c.strength, -c.speed, c.start))
    kept: list[Breakout] = []
    taken: list[tuple[int, int, str]] = []
    for candidate in ranked:
        # a move only crowds out an overlapping one of the SAME direction — the
        # two legs of a V share their turning day and are two different events
        if any(candidate.start <= end and start <= candidate.end
               for start, end, direction in taken
               if direction == candidate.breakout.direction):
            continue
        taken.append((candidate.start, candidate.end, candidate.breakout.direction))
        kept.append(candidate.breakout)
    kept.sort(key=lambda b: (b.start_date, b.end_date))
    return kept


# ---------------------------------------------------------------------------
# Size classes. Every caller (dashboard, cache, CLI) resolves the SAME set here,
# so there is exactly one place that reads the tunables.
# ---------------------------------------------------------------------------


def breakout_tiers_from_config(config) -> list[Tier]:
    """
    Resolve the configured size classes. ``large`` raises the required move by a
    FACTOR on the small one, so retuning the small class carries the large one
    with it, and allows a longer window of its own.
    """
    min_move_pct = float(config.get("breakout.min_move_pct"))                    # tunable
    min_speed = float(config.get("breakout.min_speed_pct_per_day"))              # tunable
    max_days = int(config.get("breakout.max_days"))                              # tunable
    min_move_r2 = float(config.get("breakout.min_move_r2"))                      # tunable
    pre_slope_factor = float(config.get("breakout.pre_slope_factor"))            # tunable
    min_level_shift = float(config.get("breakout.min_level_shift"))              # tunable
    min_regime_gap_std = float(config.get("breakout.min_regime_gap_std"))        # tunable
    reversal_slope_factor = float(config.get("breakout.reversal_slope_factor"))  # tunable
    v_reversal_relax = float(config.get("breakout.v_reversal_relax"))            # tunable
    context_days = int(config.get("breakout.context_days"))                      # tunable
    large_factor = float(config.get("breakout.large_factor"))                    # tunable
    max_days_large = int(config.get("breakout.max_days_large"))                  # tunable

    base = dict(
        min_speed_pct_per_day=min_speed,
        min_move_r2=min_move_r2,
        pre_slope_factor=pre_slope_factor,
        min_level_shift=min_level_shift,
        min_regime_gap_std=min_regime_gap_std,
        context_days=context_days,
        reversal_slope_factor=reversal_slope_factor,
        v_reversal_relax=v_reversal_relax,
    )
    return [
        Tier("small", Params(min_move_pct=min_move_pct, max_days=max_days, **base)),
        Tier("large", Params(min_move_pct=min_move_pct * large_factor,
                             max_days=max_days_large, **base)),
    ]


def detect_tiers(
    symbol: str,
    series: pd.Series,
    *,
    tiers: list[Tier],
    plateaus: Optional[list[Plateau]] = None,
    volumes: Optional[pd.Series] = None,
    volume_settings: Optional[volume_module.Settings] = None,
) -> list[Breakout]:
    """
    Run the scan once per size class and return everything together, sorted by
    start date. Breakouts of the SAME class never overlap; breakouts of
    different classes may — a large move can contain smaller ones.
    """
    found: list[Breakout] = []
    for tier in tiers:
        found.extend(
            detect(symbol, series, params=tier.params, plateaus=plateaus, tier=tier.name,
                   volumes=volumes, volume_settings=volume_settings)
        )
    found.sort(key=lambda b: (b.start_date, b.end_date, b.tier))
    return found


# ---------------------------------------------------------------------------
# Core-side storage. The analysis lives in the engine: it is
# computed once and persisted; readers read, they do not recompute.
# ---------------------------------------------------------------------------

PATTERN_KIND = "symbol_breakouts"


def analyze(symbol: str, closes: pd.Series, tiers: list[Tier],
            plateaus: Optional[list[Plateau]] = None,
            volumes: Optional[pd.Series] = None,
            volume_settings: Optional[volume_module.Settings] = None) -> dict[str, Any]:
    """The cache payload for one symbol: every breakout plus the classes used."""
    found = detect_tiers(symbol, closes, tiers=tiers, plateaus=plateaus,
                         volumes=volumes, volume_settings=volume_settings)
    return {
        "tiers": [
            {"name": t.name, "min_move_pct": t.params.min_move_pct, "max_days": t.params.max_days}
            for t in tiers
        ],
        "count": len(found),
        "counts_by_tier": {t.name: sum(1 for b in found if b.tier == t.name) for t in tiers},
        "counts_by_direction": {
            "up": sum(1 for b in found if b.direction == "up"),
            "down": sum(1 for b in found if b.direction == "down"),
        },
        "breakouts": [b.to_dict() for b in found],
    }
