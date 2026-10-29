"""
META — TREND-META v0.2: the trend as a REGIME plus a SHAPE, in two truth regimes.

The trend primitive is switched off (see
:func:`snapshot_engine.meta.cache.trend_classification_enabled`). It labelled the slope of a
short window and called that a trend, which flipped with noise and, used as a
grouping key, split days on something that was never a fact about them. This module
replaces the PRODUCT, not the maths: it answers the two questions a trend is
actually asked, and it keeps them strictly apart because they have different truth
regimes.

    *** OUTPUT 1 — THE MOMENT (strictly past-only, one row per day) ***

What day x could KNOW about its own regime, computed from ``series[0..x]`` and
nothing else. Four signals vote, each of them past-only:

  SLOPE      the rolling regression over the last ``slope_window`` days, expressed
             RELATIVE to the price level, with its R^2. A slope whose fit is poor
             votes sideways rather than pretending to a direction.
  DRAWDOWN   how far below the ROLLING peak (the highest close up to x) the day
             sits. Deep enough is a bear vote whatever the slope says; close enough
             to the peak is a bull vote.
  STRUCTURE  higher highs / higher lows against lower highs / lower lows, read off
             the breakouts that were KNOWABLE by x (the causal detection, never the
             forward-informed one).
  ATH RATE   how many new all-time highs fell inside the window before x. A market
             printing highs is not in the same state as one that has not printed
             one in months.

The regime is the majority of those votes, ties go to sideways, and the vote is
carried in the row so a label can always be taken apart. Beside it stands a
CONFIDENCE, which is what the right edge demands: a regime one day old is not the
same statement as one confirmed for three months, even when both read "bull".

    *** THE RIGHT EDGE IS RECOMPUTED, NEVER CLIPPED ***

Output 1 is never derived by cutting output 2 short. Output 2 knows where each
segment ENDED — that is precisely what output 1 cannot know at x, and clipping it
would smuggle the ending back in. The two therefore DISAGREE near a turn, on
purpose: output 1 still calls the old regime with falling confidence while output 2
has already placed the break. That disagreement is the honest measurement of what a
day could and could not see.

    *** OUTPUT 2 — THE STRUCTURE (full history, forward information allowed) ***

A description of what the whole series did, for a later model to learn from. The
same four signals, but read with the full history in hand (a CENTERED slope window,
the peak of the entire series, every breakout including the ones only decidable
afterwards). Days with the same label become stretches; stretches too short to be a
regime are absorbed; same-direction stretches separated only by a PAUSE are merged
into one move.

TIERS DECIDE WHAT A PAUSE IS, and they are never mixed. A SMALL plateau between two
up-legs is a breather and the two legs are one move; a LARGE plateau is the move
being over. Averaging the two classes would erase the only thing that distinguishes
a pause from an ending.

Each merged segment carries a FEATURE VECTOR — direction, span, amplitude, slope,
uniformity, shape, volume signature, position against the all-time high — and the
transitions between segments are counted with the event that sat at the boundary.

THIS MODULE PREDICTS NOTHING. It states what happened and how often one thing
followed another; a forecast built on that is a later, separate build.

STANDALONE. Nothing here feeds Block A, Block B or the similarity comparison, and
no snapshot imports it: output 2 is forward-informed and must never touch a grouping
key, and output 1 stays unattached until its usefulness has been measured rather
than assumed. Primitives are only READ, through :mod:`snapshot_engine.meta.cache` (plateau,
breakout, volume, all-time high) — never recomputed, and the deactivated trend
primitive is not among them. Deterministic and pure: same series, same params, same
regimes, every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from . import cache as meta_cache

SOURCE = "meta.trend_regime"

#: The three regimes. "sideways" is a real answer, not a failure to decide.
BULL, BEAR, SIDEWAYS = "bull", "bear", "sideways"
#: A segment's direction, which is the regime seen as a movement.
UP, DOWN, FLAT = "up", "down", "flat"
#: What the recent structure of highs and lows looks like.
HH_HL, LH_LL, MIXED, NONE = "hh_hl", "lh_ll", "mixed", "none"
#: Where a segment sits against the all-time high.
BLUE_SKY, AT_CEILING, UNDER_CEILING = "blue_sky", "at_ceiling", "under_ceiling"
#: The shape of a move over its own span.
LINEAR, ACCELERATING, DECELERATING, STEPPED = ("linear", "accelerating",
                                               "decelerating", "stepped")


@dataclass(frozen=True)
class Params:
    """The tunables of both outputs. Resolved once, never read from config here."""

    slope_window: int
    slope_min_r2: float
    flat_slope_pct: float
    bear_drawdown_pct: float
    near_peak_pct: float
    structure_events: int
    ath_window_days: int
    ath_stale_days: int
    confirm_days: int
    min_segment_days: int
    max_pause_days: int
    push_window_days: int


def params_from_config(config) -> Params:
    """The ONE place this meta's tunables are read."""
    return Params(
        slope_window=int(config.get("trend_regime.slope_window")),              # tunable
        slope_min_r2=float(config.get("trend_regime.slope_min_r2")),            # tunable
        flat_slope_pct=float(config.get("trend_regime.flat_slope_pct")),        # tunable
        bear_drawdown_pct=float(config.get("trend_regime.bear_drawdown_pct")),  # tunable
        near_peak_pct=float(config.get("trend_regime.near_peak_pct")),          # tunable
        structure_events=int(config.get("trend_regime.structure_events")),      # tunable
        ath_window_days=int(config.get("trend_regime.ath_window_days")),        # tunable
        ath_stale_days=int(config.get("trend_regime.ath_stale_days")),          # tunable
        confirm_days=int(config.get("trend_regime.confirm_days")),              # tunable
        min_segment_days=int(config.get("trend_regime.min_segment_days")),      # tunable
        max_pause_days=int(config.get("trend_regime.max_pause_days")),          # tunable
        push_window_days=int(config.get("trend_regime.push_window_days")),      # tunable
    )


# ---------------------------------------------------------------------------
# Small arithmetic, all of it relative
# ---------------------------------------------------------------------------


def fit(values: list[float]) -> tuple[float, float]:
    """
    Least-squares slope of ``values`` against day number, RELATIVE, plus its R^2.

    The slope is divided by the window's own mean price, so it reads as a fraction
    per day and means the same thing at 2 USD and at 900 USD — the same reason every
    feature the meta layer compares is relative. R^2 comes back beside it because a
    slope without a fit quality is not a direction: a window that scatters produces a
    number that points somewhere and says nothing.

    Fewer than two points, a flat window or a non-positive mean: ``(0.0, 0.0)``.
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    if mean_y <= 0:
        return 0.0, 0.0
    sxx = sum((i - mean_x) ** 2 for i in range(n))
    sxy = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    if sxx <= 0:
        return 0.0, 0.0
    slope = sxy / sxx
    total = sum((v - mean_y) ** 2 for v in values)
    explained = slope * sxy
    r2 = explained / total if total > 0 else 0.0
    return slope / mean_y, max(0.0, min(1.0, r2))


def uniformity(values: list[float], direction: str) -> float:
    """
    How much of the movement inside a span went the span's own way, 0 .. 1.

    The sum of the day-to-day steps that agree with ``direction``, over the sum of
    all step sizes. 1.0 is a move that never gave anything back; 0.5 is a move that
    travelled as far against itself as with itself. Magnitude-weighted rather than
    counted, because ten flat days and one collapse are not ten agreements and one
    disagreement.
    """
    if len(values) < 2 or direction == FLAT:
        return 0.0
    steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    total = sum(abs(s) for s in steps)
    if total <= 0:
        return 0.0
    agreeing = sum(abs(s) for s in steps
                   if (s > 0 if direction == UP else s < 0))
    return agreeing / total


def shape_of(values: list[float], *, pauses: int) -> str:
    """
    The form of a move over its own span: linear, accelerating, decelerating, stepped.

    Two halves, two slopes, both relative. A second half steeper than the first by
    more than a quarter is ACCELERATING, flatter by as much is DECELERATING,
    everything between is LINEAR. A span that contains two or more pauses is STEPPED
    regardless of the halves — a staircase is a different animal from a straight
    line, and its two halves can easily average out to one.
    """
    if pauses >= 2:
        return STEPPED
    if len(values) < 4:
        return LINEAR
    middle = len(values) // 2
    first, _ = fit(values[:middle + 1])
    second, _ = fit(values[middle:])
    if abs(first) <= 0:
        return LINEAR
    ratio = second / first
    if ratio > 1.25:
        return ACCELERATING
    if ratio < 0.75:
        return DECELERATING
    return LINEAR


def _direction_of(regime: str) -> str:
    return {BULL: UP, BEAR: DOWN}.get(regime, FLAT)


def _majority(votes: list[str]) -> tuple[str, float]:
    """
    The regime most votes point at, and the share that agreed.

    A tie is SIDEWAYS: two signals pulling in opposite directions is not weak
    evidence for one of them, it is evidence that the day has no direction.
    """
    counts = {label: votes.count(label) for label in (BULL, BEAR, SIDEWAYS)}
    best = max(counts.values())
    winners = [label for label, count in counts.items() if count == best]
    if len(winners) > 1:
        return SIDEWAYS, counts[SIDEWAYS] / len(votes) if votes else 0.0
    return winners[0], best / len(votes) if votes else 0.0


# ---------------------------------------------------------------------------
# The four votes — the same rules for both outputs, different inputs
# ---------------------------------------------------------------------------


def slope_vote(slope_pct: float, r2: float, *, params: Params) -> str:
    """A direction only if the window actually held a line; otherwise sideways."""
    if r2 < params.slope_min_r2:
        return SIDEWAYS
    if slope_pct > params.flat_slope_pct:
        return BULL
    if slope_pct < -params.flat_slope_pct:
        return BEAR
    return SIDEWAYS


def drawdown_vote(drawdown_pct: float, *, params: Params) -> str:
    """Deep below the peak is a bear; close under it is a bull; between: neither."""
    if drawdown_pct <= -params.bear_drawdown_pct:
        return BEAR
    if drawdown_pct >= -params.near_peak_pct:
        return BULL
    return SIDEWAYS


def structure_vote(directions: list[str]) -> str:
    """Higher highs against lower lows, from the recent moves and nothing else."""
    if not directions:
        return SIDEWAYS
    ups = directions.count(UP)
    downs = directions.count(DOWN)
    if ups > downs:
        return BULL
    if downs > ups:
        return BEAR
    return SIDEWAYS


def ath_rate_vote(highs_in_window: int, days_since_ath: int, *,
                  params: Params) -> str:
    """
    New highs are a bull vote; a high that has gone stale is a bear vote.

    The asymmetry is deliberate: printing a high is an ACT, while not printing one
    is merely an absence, so silence only becomes evidence once it has lasted longer
    than ``ath_stale_days``. Anything in between abstains.
    """
    if highs_in_window > 0:
        return BULL
    if days_since_ath > params.ath_stale_days:
        return BEAR
    return SIDEWAYS


def _structure_label(directions: list[str]) -> str:
    if not directions:
        return NONE
    ups, downs = directions.count(UP), directions.count(DOWN)
    if ups and not downs:
        return HH_HL
    if downs and not ups:
        return LH_LL
    return MIXED


def confidence_of(agreement: float, days_in_regime: int, *, params: Params) -> float:
    """
    How much the label is worth at this point in its life, 0 .. 1.

    ``agreement * (0.5 + 0.5 * maturity)``, where maturity is the age of the regime
    against ``confirm_days``, capped at one. Both halves are needed and neither is
    enough: four signals agreeing on a regime that started yesterday is a guess that
    all four could be reading the same fresh move, and a three-month-old label with
    two signals against it is a description that has begun to fail. A fresh switch
    therefore halves its own confidence, which is exactly what the unconfirmed right
    edge deserves.
    """
    maturity = min(1.0, max(0, days_in_regime) / max(1, params.confirm_days))
    return agreement * (0.5 + 0.5 * maturity)


# ---------------------------------------------------------------------------
# OUTPUT 1 — the moment, strictly past-only
# ---------------------------------------------------------------------------


def moment_state(symbol: str, *, params: Params,
                 cache: Optional[meta_cache.MetaCache] = None,
                 config=None) -> list[dict[str, Any]]:
    """
    One row per trading day: the regime day x could read off its own past.

    EVERY input is taken from ``[0..x]``. The rolling peak is the highest close so
    far, not the highest of the series; the breakouts are the CAUSAL detection
    filtered by ``known_date <= x``, so a move is only structure once it was
    decidable; the all-time-high distance is the past-only primitive. Nothing here
    is clipped from :func:`structure` — the right edge is computed, day by day, from
    what stood to the left of it.

    The row keeps the four votes beside the label. A regime that cannot be taken
    apart cannot be calibrated, and a majority of four is worth little unless one
    can see which four.
    """
    own = cache is None
    store = cache or meta_cache.MetaCache(config)
    try:
        series = store.closes(symbol)
        if series is None or len(series) == 0:
            return []
        closes = [float(v) for v in series.to_numpy()]
        dates = [d.date().isoformat() for d in series.index]
        total = len(closes)

        distance = meta_cache.ath_distance(store, symbol)
        highs = {row["date"] for row in
                 [e for e in meta_cache.ath_events(store, symbol)]}
        causal_breaks = sorted(
            [b for b in meta_cache.causal_breakouts(store, symbol) if b["known_date"]],
            key=lambda b: (b["known_date"], b["start_date"]))
        causal_plateau = meta_cache.causal_plateaus(store, symbol)
        reaching = {name: scan["reaching"] for name, scan in causal_plateau.items()}
        in_progress = meta_cache.breakout_in_progress(store, symbol)

        rows: list[dict[str, Any]] = []
        peak = closes[0]
        known = 0                      # causal breakouts already decidable at x
        structure: list[str] = []      # their directions, newest last
        high_days: list[int] = []      # indices of the all-time highs so far
        previous_regime: Optional[str] = None
        age = 0

        for x in range(total):
            day = dates[x]
            peak = max(peak, closes[x])
            if day in highs:
                high_days.append(x)
            while known < len(causal_breaks) and causal_breaks[known]["known_date"] <= day:
                structure.append(causal_breaks[known]["direction"])
                known += 1

            window = closes[max(0, x - params.slope_window + 1):x + 1]
            slope_pct, r2 = fit(window)
            drawdown = (closes[x] - peak) / peak if peak > 0 else 0.0
            recent = structure[-params.structure_events:]
            reach = distance[x] if x < len(distance) else {}
            days_since_ath = int(reach.get("days_since_ath", 0))
            in_window = sum(1 for i in high_days if x - i < params.ath_window_days)

            votes = {
                "slope": slope_vote(slope_pct, r2, params=params),
                "drawdown": drawdown_vote(drawdown, params=params),
                "structure": structure_vote(recent),
                "ath_rate": ath_rate_vote(in_window, days_since_ath, params=params),
            }
            regime, agreement = _majority(list(votes.values()))
            age = age + 1 if regime == previous_regime else 1
            previous_regime = regime

            running = in_progress[x] if x < len(in_progress) else None
            resting = next((name for name in ("small", "large")
                            if name in reaching and x < len(reaching[name])
                            and reaching[name][x]), None)
            rows.append({
                "date": day,
                "regime": regime,
                "confidence": confidence_of(agreement, age, params=params),
                "agreement": agreement,
                "days_in_regime": age,
                "votes": votes,
                "slope_pct_per_day": slope_pct,
                "slope_r2": r2,
                "drawdown_from_peak_pct": drawdown,
                "structure": _structure_label(recent),
                "ath_highs_in_window": in_window,
                "ath_gap_pct": float(reach.get("gap_pct", 0.0)),
                "days_since_ath": days_since_ath,
                "blue_sky": bool(closes[x] >= peak),
                "resting_tier": resting,
                "move_now": running,
                "move_alignment": _alignment(running, regime),
            })
        return rows
    finally:
        if own:
            store.close()


def _alignment(running: Optional[str], regime: str) -> str:
    """Is the move happening right now going with the regime, against it, or neither?"""
    if running is None or regime == SIDEWAYS:
        return NONE
    return "with" if running == _direction_of(regime) else "against"


# ---------------------------------------------------------------------------
# OUTPUT 2 — the structure of the whole history
# ---------------------------------------------------------------------------


def _full_history_labels(closes: list[float], dates: list[str],
                         breaks: list[dict[str, Any]], high_days: list[int], *,
                         params: Params) -> list[str]:
    """
    The same four votes, read with the whole series in hand.

    Three things change and every one of them is forward information, which is
    allowed here and forbidden in output 1: the slope window is CENTERED on the day
    instead of ending at it, the peak is the highest close of the ENTIRE history,
    and a breakout counts as structure from the day it STARTED rather than from the
    day it became decidable. This is the description of what happened, not the view
    a day had of itself.
    """
    total = len(closes)
    if total == 0:
        return []
    peak = max(closes)
    half = max(1, params.slope_window // 2)
    starts = sorted((b["start_date"], b["direction"]) for b in breaks)
    labels: list[str] = []
    for x in range(total):
        window = closes[max(0, x - half):min(total, x + half + 1)]
        slope_pct, r2 = fit(window)
        drawdown = (closes[x] - peak) / peak if peak > 0 else 0.0
        recent = [d for start, d in starts if start <= dates[x]][-params.structure_events:]
        last_high = max([i for i in high_days if i <= x], default=None)
        days_since = x - last_high if last_high is not None else x + 1
        in_window = sum(1 for i in high_days if 0 <= x - i < params.ath_window_days)
        votes = [
            slope_vote(slope_pct, r2, params=params),
            drawdown_vote(drawdown, params=params),
            structure_vote(recent),
            ath_rate_vote(in_window, days_since, params=params),
        ]
        labels.append(_majority(votes)[0])
    return labels


def _stretches(labels: list[str]) -> list[tuple[int, int, str]]:
    """Consecutive equal labels as ``(first_index, last_index, label)``."""
    spans: list[tuple[int, int, str]] = []
    for i, label in enumerate(labels):
        if spans and spans[-1][2] == label:
            spans[-1] = (spans[-1][0], i, label)
        else:
            spans.append((i, i, label))
    return spans


def _absorb_short(spans: list[tuple[int, int, str]], *, minimum: int
                  ) -> list[tuple[int, int, str]]:
    """
    Stretches shorter than ``minimum`` days are not regimes and are absorbed.

    A three-day wobble inside a year of bull is a wobble. It is folded into the
    stretch BEFORE it (the regime it interrupted), because a regime ends when
    something replaces it, not every time it is briefly contradicted. Only a
    leading short stretch is folded forward instead — it has nothing behind it.
    """
    if not spans:
        return []
    kept: list[tuple[int, int, str]] = []
    for span in spans:
        start, end, label = span
        if end - start + 1 >= minimum or not kept:
            kept.append(span)
            continue
        previous = kept[-1]
        kept[-1] = (previous[0], end, previous[2])
    merged: list[tuple[int, int, str]] = []
    for span in kept:                      # absorbing can leave neighbours equal
        if merged and merged[-1][2] == span[2]:
            merged[-1] = (merged[-1][0], span[1], span[2])
        else:
            merged.append(span)
    return merged


def _pause_between(plateau_spans: list[tuple[int, int, str]], first: int,
                   second: int) -> tuple[bool, int]:
    """
    Was the gap between two same-direction segments a PAUSE, and how many were there?

    A pause is covered by SMALL plateaus. A LARGE plateau in the gap is not a
    breather but the move ending, and the two legs stay apart — the tier IS the
    distinction, which is why the classes are never merged into "a plateau".
    """
    inside = [(start, end, tier) for start, end, tier in plateau_spans
              if first <= start and end <= second]
    if any(tier != "small" for _s, _e, tier in inside):
        return False, len(inside)
    return bool(inside), len(inside)


def structure(symbol: str, *, params: Params,
              cache: Optional[meta_cache.MetaCache] = None,
              config=None) -> dict[str, Any]:
    """
    The whole history as regime segments, their features, and what followed what.

    Forward-informed on purpose (see :func:`_full_history_labels`) and therefore
    DESCRIPTIVE ONLY: nothing in here may reach a snapshot, a grouping key or a
    score. It is the material a later model can be fitted on, and the honest place
    to notice that a shape repeats before anyone claims it predicts.
    """
    own = cache is None
    store = cache or meta_cache.MetaCache(config)
    try:
        series = store.closes(symbol)
        if series is None or len(series) == 0:
            return {"symbol": symbol, "segments": [], "transitions": [],
                    "regime_days": {}, "hypotheses": {}}
        closes = [float(v) for v in series.to_numpy()]
        dates = [d.date().isoformat() for d in series.index]
        at = {day: i for i, day in enumerate(dates)}
        total = len(closes)

        breaks = meta_cache.breakouts(store, symbol)
        plateaus = meta_cache.plateaus(store, symbol)
        rolling = meta_cache.rolling_ath(store, symbol)
        volume = meta_cache.trailing_volume(store, symbol,
                                            window=int(params.slope_window))
        high_days = [at[e["date"]] for e in meta_cache.ath_events(store, symbol)
                     if e["date"] in at]

        labels = _full_history_labels(closes, dates, breaks, high_days, params=params)
        spans = _absorb_short(_stretches(labels), minimum=params.min_segment_days)

        plateau_spans = sorted((at[p["start_date"]], at[p["end_date"]], p["tier"])
                               for p in plateaus
                               if p["start_date"] in at and p["end_date"] in at)
        break_index = sorted(
            (at[b["start_date"]], at[b["end_date"]], b["direction"], b["tier"])
            for b in breaks if b["start_date"] in at and b["end_date"] in at)

        # ---- merge same-direction segments separated only by a small-plateau pause
        merged: list[dict[str, Any]] = []
        for start, end, label in spans:
            direction = _direction_of(label)
            if merged and merged[-1]["direction"] == direction and direction != FLAT:
                previous = merged[-1]
                is_pause, pauses = _pause_between(plateau_spans, previous["end_i"], start)
                gap = start - previous["end_i"]
                if is_pause and gap <= params.max_pause_days:
                    previous["end_i"] = end
                    previous["legs"] += 1
                    previous["pauses"] += pauses
                    continue
            merged.append({"start_i": start, "end_i": end, "direction": direction,
                           "regime": label, "legs": 1, "pauses": 0})

        segments = [_segment_features(entry, closes, dates, rolling, volume,
                                      plateau_spans, break_index, params=params)
                    for entry in merged]

        transitions = _transitions(merged, dates, break_index, high_days)
        regime_days: dict[str, int] = {}
        for label in labels:
            regime_days[label] = regime_days.get(label, 0) + 1

        return {
            "symbol": symbol,
            "days": total,
            "note": ("forward-informed, descriptive only — never a grouping key, "
                     "never a score, never clipped into output 1"),
            "regime_days": regime_days,
            "segments": segments,
            "transitions": transitions,
            "hypotheses": _hypotheses(segments, break_index, dates, rolling, closes,
                                      params=params),
        }
    finally:
        if own:
            store.close()


def _segment_features(entry: dict[str, Any], closes: list[float], dates: list[str],
                      rolling: list[float], volume: list[dict[str, Any]],
                      plateau_spans: list[tuple[int, int, str]],
                      break_index: list[tuple[int, int, str, str]], *,
                      params: Params) -> dict[str, Any]:
    """
    One segment as a comparable feature vector — every number relative, no price.

    Amplitude is measured from the segment's first close to its last, the slope is
    the relative fit over the whole span, and the volume signature is the mean
    trailing RATIO (already relative to the stock's own normal) with the direction
    that dominated. The all-time-high position says where the move lived: making
    new highs, pressed against the old one, or far below it.

    ``direction`` IS THE REGIME'S, NOT THE SIGN OF THE AMPLITUDE, and the two are
    allowed to disagree: a stretch deep under its peak with lower highs is a bear
    segment even if price happened to end three percent above where it began. Both
    numbers are in the vector, so the disagreement can be read instead of being
    resolved by whichever one the writer preferred.
    """
    start, end = entry["start_i"], entry["end_i"]
    values = closes[start:end + 1]
    first, last = values[0], values[-1]
    slope_pct, r2 = fit(values)
    inside_plateaus = [(s, e, tier) for s, e, tier in plateau_spans
                       if start <= s and e <= end]
    inside_breaks = [(s, e, direction, tier) for s, e, direction, tier in break_index
                     if start <= s <= end]
    ratios = [row["ratio"] for row in volume[start:end + 1] if row.get("ratio")]
    trends = [row["trend"] for row in volume[start:end + 1] if row.get("trend")]
    ceiling = rolling[end] if end < len(rolling) else last
    gap = (last - ceiling) / ceiling if ceiling > 0 else 0.0

    return {
        "start_date": dates[start],
        "end_date": dates[end],
        "regime": entry["regime"],
        "direction": entry["direction"],
        "length_days": end - start + 1,
        "amplitude_pct": (last - first) / first if first > 0 else 0.0,
        "slope_pct_per_day": slope_pct,
        "slope_r2": r2,
        "uniformity": uniformity(values, entry["direction"]),
        "shape": shape_of(values, pauses=len(inside_plateaus)),
        "legs": entry["legs"],
        "pauses_small": sum(1 for _s, _e, tier in inside_plateaus if tier == "small"),
        "pauses_large": sum(1 for _s, _e, tier in inside_plateaus if tier == "large"),
        "breakouts": {
            "up": sum(1 for _s, _e, d, _t in inside_breaks if d == UP),
            "down": sum(1 for _s, _e, d, _t in inside_breaks if d == DOWN),
            "by_tier": {tier: sum(1 for _s, _e, _d, t in inside_breaks if t == tier)
                        for tier in sorted({t for _s, _e, _d, t in inside_breaks})},
        },
        "volume_signature": {
            "mean_ratio": sum(ratios) / len(ratios) if ratios else None,
            "dominant_trend": max(set(trends), key=trends.count) if trends else None,
        },
        "ath_position": (BLUE_SKY if gap >= 0 else
                         AT_CEILING if gap >= -params.near_peak_pct else
                         UNDER_CEILING),
        "ath_gap_at_end_pct": gap,
    }


def _transitions(merged: list[dict[str, Any]], dates: list[str],
                 break_index: list[tuple[int, int, str, str]],
                 high_days: list[int]) -> list[dict[str, Any]]:
    """
    What followed what, and what sat at the boundary.

    One row per pair of neighbouring segments, carrying the trigger that was closest
    to the seam: the breakout that started within the joint, or a new all-time high
    on it, or nothing at all. Counting the pairs is the whole of the statistic —
    this module states frequencies and leaves prediction to whatever is built on
    top of it.
    """
    rows: list[dict[str, Any]] = []
    for previous, following in zip(merged, merged[1:]):
        seam = following["start_i"]
        trigger = next((f"breakout_{direction}_{tier}"
                        for s, _e, direction, tier in break_index
                        if previous["end_i"] <= s <= seam), None)
        if trigger is None and any(previous["end_i"] <= i <= seam for i in high_days):
            trigger = "new_ath"
        rows.append({
            "from": previous["regime"],
            "to": following["regime"],
            "at_date": dates[seam],
            "trigger": trigger or "none",
        })
    return rows


def _hypotheses(segments: list[dict[str, Any]],
                break_index: list[tuple[int, int, str, str]], dates: list[str],
                rolling: list[float], closes: list[float], *,
                params: Params) -> dict[str, Any]:
    """
    Couplings stated as RATES over many segments, never as a claim about today.

    The blue-sky push: of all up-segments, how many began within
    ``push_window_days`` of an upward breakout that ended at or above the running
    high, and how long those segments then lasted (median). The single live instance
    of the same coupling belongs in output 1, past-only, where it can be read on the
    day itself — here it is only ever a frequency.
    """
    at = {day: i for i, day in enumerate(dates)}
    pushes = [(end, start) for start, end, direction, _tier in break_index
              if direction == UP and end < len(rolling)
              and closes[end] >= rolling[end] * (1.0 - params.near_peak_pct)]
    ups = [s for s in segments if s["direction"] == UP]
    started_on_push = []
    for segment in ups:
        begin = at.get(segment["start_date"])
        if begin is None:
            continue
        if any(0 <= begin - end <= params.push_window_days for end, _s in pushes):
            started_on_push.append(segment)
    lengths = sorted(s["length_days"] for s in started_on_push)
    median = (lengths[len(lengths) // 2] if lengths else None)
    return {
        "blue_sky_push": {
            "up_segments": len(ups),
            "started_after_ath_breakout": len(started_on_push),
            "rate": len(started_on_push) / len(ups) if ups else None,
            "window_days": params.push_window_days,
            "median_length_days": median,
        },
    }


def analyse(symbol: str, *, params: Params,
            cache: Optional[meta_cache.MetaCache] = None,
            config=None) -> dict[str, Any]:
    """
    Both outputs of one symbol, side by side but never mixed.

    ``moment`` is past-only and snapshot-ready; ``structure`` is forward-informed and
    descriptive. They are returned together for convenience and are kept apart
    everywhere else — the keys are named after their truth regimes so that a caller
    that reaches for the wrong one has to type the wrong word to do it.
    """
    own = cache is None
    store = cache or meta_cache.MetaCache(config)
    try:
        return {
            "symbol": symbol,
            "moment": moment_state(symbol, params=params, cache=store),
            "structure": structure(symbol, params=params, cache=store),
        }
    finally:
        if own:
            store.close()
