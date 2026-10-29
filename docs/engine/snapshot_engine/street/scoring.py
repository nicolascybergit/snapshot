"""
STREET — the scorer: what one bundled outcome distribution is worth.

The analysis layer says WHAT HAPPENED after a situation — a set of return levels
with how often each occurred, how often the move went the group's way, and how the
average return accumulated over the days that followed. It deliberately computes no
statistics. This module is the other half: it turns those raw outcomes into one
comparable number.

    *** S = mu * w(p) * U * Z ***

  mu   the expected return of the distribution, in PERCENT.
  w(p) how much the direction agreement is worth. Below a coin flip it falls off
       steeply (``alpha_riskant``), above it rises gently (``alpha_sicher``) and is
       capped past ``p_cutoff`` — being surer than sure buys almost nothing.
  U    how much of the expectation survives its own spread, ``mu / (mu + sigma)``.
       A distribution that averages +10% by way of +100% once and nothing nine
       times is worth a quarter of one that returns +10% every time.
  Z    how fast it happened: ``ln(1 + R_total) / D``, where D is the return-weighted
       average day. Two identical gains are not identical trades if one took three
       days and the other seventeen.

Z CARRIES A UNIT, AND THE UNIT IS ARBITRARY. It is a growth rate per TRADING DAY,
and "per day" was a choice, not a fact: the same rate reads as a different number
per week, per month or per quarter. That choice sets the whole absolute scale of S,
which is why the scores come out around a hundredth while the moderator's radius is
an interval on ``[0, 1]``. The unit is therefore a tunable — ``scoring.time_scale``,
applied by the emitting layer as ``Z * time_scale`` — and because it multiplies
every score by the same constant it is a CHANGE OF UNITS and nothing else: the
ranking is untouched, and so is every allocation share, since the factor cancels in
``S_i / sum(S)``. It is not in this formula for exactly that reason — a constant
that cannot change any comparison has no business inside the definition.

A DOWNSIDE FILTER runs before any of that matters: if the weighted share of loss in
the distribution exceeds ``downside_threshold``, the score is zero. A wide spread
that happens to average positive is not a trade, and no amount of expectation is
allowed to argue it back in.

TWO PROBABILITIES, NEVER MIXED. ``p_i`` inside ``outcomes`` is how often one return
LEVEL occurred and sums to one; ``p_direction`` is how often the move went the
group's way at that offset. Only the second one enters ``w(p)``. Confusing them
would let a distribution with many small buckets pretend to be uncertain about its
direction.

RECENCY IS PRICED ON THE WAY IN, NOT INSIDE THE FORMULA. A group's observations
are spread across years, and a setup that behaved a certain way in 2017 says less
about tomorrow than the same setup last month. So each observation is weighted by
its AGE before mu and sigma exist — ``w(t) = floor + (1 - floor) * exp(-lambda * t)``
over trading days — and the level masses are renormalised to one, so what reaches
the formula is still an ordinary distribution. Deliberately a floor and not a decay
to zero: at the default floor of 0.5 the oldest observation still counts half as
much as today's, because thinning a history is not the same as discarding it, and
plain exp(-0.01 * t) discards everything past two years. ``lambda = 0`` or
``floor = 1`` turns it off. See :func:`recency_weight` and :func:`weight_by_age`.

SAMPLE SIZE ENTERS EXACTLY ONCE, AND NOT HERE. How many members stood behind a
distribution is a statement about the EVIDENCE, not about the trade the evidence
describes, so ``score_trade`` never looks at ``n``. It is applied afterwards, by the
caller, as ``C = n_t / (n_t + k_confidence)`` (:func:`sample_confidence`) — a
continuous discount that leaves a thin group's score small instead of throwing it
away. The reason it sits AFTER the formula rather than inside it is the shape of our
input: ``mu`` comes out of a DISTRIBUTION of observed levels, not out of
``p_raw * return``. There is no single probability to shrink toward a prior here —
shrinking ``p_direction`` would move ``w(p)`` while leaving ``mu`` untouched, which
prices the thinness in one factor and ignores it in the other. ``C`` discounts the
finished number, which is the only place the whole distribution is represented.
Two things follow and both are deliberate:

* there is NO minimum-n gate anywhere. A gate would discard the observation and
  invert the logic: a thin group would stop being a weak candidate and start being
  no candidate, which is a verdict the data cannot support.
* ``p_direction`` is NEVER shrunk toward a global prior before it enters ``w(p)``.
  That shrink and ``C`` measure the same thing — small n — and applying both would
  charge a thin group twice for its thinness, once inside the formula and once
  outside it. ``C`` is the one that stays, because it leaves this formula and its
  validation untouched.

TWO UNITS, NEVER MIXED. ``x_pct`` and ``mu`` are percent (15.0 means +15%);
``R_total`` and the path values are fractions (0.15 means the same thing). The
distinction is deliberate: ``Z`` needs ``ln(1 + R)``, which is only meaningful on a
fraction, while ``mu`` is a magnitude that should read as a percentage.

Parameter-free in the project's sense: every constant arrives through :class:`Params`,
resolved once by :func:`params_from_config`, and nothing here reads the config on its
own. Deterministic and pure — same inputs, same score, every time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

SOURCE = "street.scoring"


@dataclass(frozen=True)
class Params:
    """The scoring tunables."""

    alpha_riskant: float
    alpha_sicher: float
    p_cutoff: float
    cap_value: float
    downside_threshold: float
    # The half-credit sample size of the confidence discount. Defaulted to 0.0 (no
    # discount) so a caller that builds Params by hand keeps working; the config
    # stays the real source. ``score_trade`` never reads it — see the module
    # docstring and :func:`sample_confidence`.
    k_confidence: float = 0.0
    # The unit of Z: growth over this many trading days instead of over one. A pure
    # change of units, applied outside the formula by the emitting layer — 1.0 keeps
    # the per-day reading. Defaulted so a caller building Params by hand is
    # unaffected; the config stays the real source.
    time_scale: float = 1.0
    # Whether the discount is applied at all. Off means every group is credited in
    # full, whatever its size — kept as a switch so the effect of the discount can
    # be measured against its own absence rather than argued about.
    confidence_shrink_enabled: bool = True
    # RECENCY. The decay per trading day and the weight the curve lands on. Both
    # default to "off" (lambda 0, floor 1) so a caller building Params by hand, or a
    # bundle with no ages in it, scores exactly as before; the config is the real
    # source. See :func:`recency_weight`.
    recency_lambda: float = 0.0
    recency_floor: float = 1.0


def params_from_config(config) -> Params:
    """The ONE place the scoring tunables are read."""
    return Params(
        alpha_riskant=float(config.get("scoring.alpha_riskant")),            # tunable
        alpha_sicher=float(config.get("scoring.alpha_sicher")),              # tunable
        p_cutoff=float(config.get("scoring.p_cutoff")),                      # tunable
        cap_value=float(config.get("scoring.cap_value")),                    # tunable
        downside_threshold=float(config.get("scoring.downside_threshold")),  # tunable
        k_confidence=float(config.get("scoring.k_confidence")),              # tunable
        time_scale=float(config.get("scoring.time_scale")),                  # tunable
        confidence_shrink_enabled=bool(
            config.get("scoring.confidence_shrink_enabled")),                # tunable
        recency_lambda=float(config.get("scoring.recency_lambda")),          # tunable
        recency_floor=float(config.get("scoring.recency_floor")),            # tunable
    )


# ---------------------------------------------------------------------------
# The four factors, each on its own
# ---------------------------------------------------------------------------


def recency_weight(age_days: float, *, params: Params) -> float:
    """
    What one observation is worth for being ``age_days`` trading days old.

        w(t) = floor + (1 - floor) * exp(-lambda * t)

    A fresh observation (t = 0) is worth 1.0 and the curve falls toward ``floor``,
    which it approaches but never crosses. At the defaults (lambda 0.01, floor 0.5) a
    day from last week weighs ~0.995, one from a year ago ~0.52, one from five years
    ago ~0.50 — old evidence is halved, not erased.

    THE FLOOR IS THE WHOLE DESIGN. Plain exponential decay is the obvious form and
    the wrong one here: exp(-0.01 * 500) = 0.0067 deletes everything past two years,
    which would throw away the older half of exactly the histories that were
    collected to make groups big enough to mean anything. What we want to say is
    "older evidence counts, but less", and a curve that lands on a positive floor
    says that; a curve that lands on zero says something else.

    ``lambda = 0`` or ``floor = 1`` both mean "every age is worth the same", and both
    return 1.0 for every t. A negative age cannot happen (nothing is newer than the
    newest day) and is clamped to 0 rather than allowed to weigh more than the
    present.
    """
    if params.recency_lambda <= 0.0 or params.recency_floor >= 1.0:
        return 1.0
    floor = max(0.0, min(1.0, float(params.recency_floor)))
    return floor + (1.0 - floor) * math.exp(-params.recency_lambda * max(0.0, age_days))


def weight_by_age(outcomes: Sequence[dict[str, Any]], *, params: Params
                  ) -> list[dict[str, Any]]:
    """
    The same levels with ``p`` re-read as "how much recent evidence stands here".

    Each level arrives with the ages of the days that produced it. Every one of
    those days gets its :func:`recency_weight`, a level's mass becomes the SUM of
    its days' weights instead of their COUNT, and the masses are normalised back to
    one. Nothing else changes: the levels are the same levels, ``x_pct`` is
    untouched, and the result is still a distribution that sums to one — so
    everything downstream (mu, sigma, the downside filter) reads it exactly as
    before and needs to know nothing about ages.

    UNTOUCHED WHEN IT CANNOT BE DONE HONESTLY. If the weighting is off, or any level
    is missing its ages, the outcomes come back as they were. A bundle where only
    some levels know their age would otherwise be scored on a mixture of weighted and
    unweighted mass, which is not a distribution of anything.

    ``p_direction`` IS NOT TOUCHED HERE and is not weighted anywhere. It is the other
    probability (see the module docstring) and it enters the score through ``w(p)``,
    not through the distribution; whether it should age too is a real question and a
    separate one.
    """
    rows = list(outcomes)
    if params.recency_lambda <= 0.0 or params.recency_floor >= 1.0 or not rows:
        return rows
    if any("ages" not in row or not row["ages"] for row in rows):
        return rows

    masses = [sum(recency_weight(float(age), params=params) for age in row["ages"])
              for row in rows]
    total = sum(masses)
    if total <= 0:
        return rows
    return [{**row, "p": mass / total} for row, mass in zip(rows, masses)]


def recency_span(outcomes: Sequence[dict[str, Any]], *, params: Params
                 ) -> tuple[float, float]:
    """
    The weight of the OLDEST and of the NEWEST day in a bundle, for the trace.

    Two numbers that say at a glance how hard the weighting pulled on this
    particular distribution — equal when every observation is the same age or the
    weighting is off, furthest apart when the group reaches across the whole
    history. ``(1.0, 1.0)`` when no level knows its age.
    """
    ages = [float(age) for row in outcomes for age in (row.get("ages") or [])]
    if not ages:
        return (1.0, 1.0)
    return (recency_weight(max(ages), params=params),
            recency_weight(min(ages), params=params))


def expectation(outcomes: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """
    The distribution's expected return and its spread, both in percent.

    ``mu = sum(p_i * x_i)`` and ``sigma = sqrt(sum(p_i * (x_i - mu)^2))`` — the
    population form, because the bundled outcomes ARE the population: they are every
    observation the group made, grouped by level, not a sample drawn from it.

    The weights are used as given rather than renormalised. A caller that hands over
    weights not summing to one has a bundling bug, and quietly rescaling here would
    hide it behind a plausible number.
    """
    if not outcomes:
        return 0.0, 0.0
    mu = sum(float(o["p"]) * float(o["x_pct"]) for o in outcomes)
    variance = sum(float(o["p"]) * (float(o["x_pct"]) - mu) ** 2 for o in outcomes)
    return mu, math.sqrt(max(variance, 0.0))


def confidence_weight(p_direction: float, *, params: Params) -> float:
    """
    What the direction agreement is worth, as a multiplier around 1.0 at a coin flip.

    Three regimes, and the asymmetry between the first two is the whole point:

    * below 0.5 the weight decays with ``alpha_riskant``, steeply — a situation that
      went the other way more often than not is not a slightly worse trade, it is a
      different trade.
    * from 0.5 to ``p_cutoff`` it grows with ``alpha_sicher``, gently, because
      agreement is evidence but never proof.
    * past ``p_cutoff`` it is flat at ``cap_value``. Beyond that point the extra
      agreement is far more likely to be a small sample than a real edge, and a
      formula that kept rewarding it would reward exactly the groups with the
      fewest occurrences.
    """
    if p_direction < 0.5:
        return math.exp(params.alpha_riskant * (p_direction - 0.5))
    if p_direction <= params.p_cutoff:
        return math.exp(params.alpha_sicher * (p_direction - 0.5))
    return params.cap_value


def uncertainty_factor(mu: float, sigma: float) -> float:
    """
    How much of the expectation survives its own spread: ``mu / (mu + sigma)``.

    1.0 when every observation returned the same thing, and falling as the outcomes
    scatter. A non-positive expectation scores 0.0 outright — there is nothing left
    to discount, and the ratio would otherwise turn a loss into a positive number.
    """
    if mu <= 0:
        return 0.0
    total = mu + sigma
    return mu / total if total > 0 else 0.0


def duration_and_total(path: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """
    The return-weighted average day of a cumulative path, and where it ended.

    ``path`` is ``[(day, cumulative_return_fraction), ...]`` in ascending day order.
    Each step contributes ``delta = value_i - value_(i-1)`` (starting from 0), and
    the duration is ``sum(day_i * delta_i) / R_total`` — the day by which the gain
    had, on average, arrived. A path that earns most of its return early therefore
    has a shorter duration than one that earns it at the end, even when both finish
    in the same place on the same day.

    A single point is simply its own day. An empty path, or one that ends at zero,
    has no meaningful duration and reports 0.0.
    """
    if not path:
        return 0.0, 0.0
    total = float(path[-1][1])
    if len(path) == 1:
        return float(path[0][0]), total
    if total == 0:
        return 0.0, total
    weighted = 0.0
    previous = 0.0
    for day, value in path:
        weighted += float(day) * (float(value) - previous)
        previous = float(value)
    return weighted / total, total


def time_factor(path: Sequence[tuple[float, float]]) -> float:
    """
    How fast the return arrived: ``ln(1 + R_total) / D``.

    The logarithm is what makes two gains comparable across sizes — it is the
    continuously compounded return, so dividing by the duration gives a rate per day
    rather than a raw percentage per day. A duration of zero (no path, or nothing
    gained) leaves the score untouched at 1.0 instead of dividing by nothing.
    """
    duration, total = duration_and_total(path)
    if duration == 0:
        return 1.0
    if total <= -1:
        return 0.0                       # a total loss has no growth rate
    return math.log(1.0 + total) / duration


def sample_confidence(n: int, k: float) -> float:
    """
    How much of a score survives the size of the sample behind it: ``n / (n + k)``.

    The ONE place sample size acts on a score, and deliberately outside
    ``score_trade``: the formula scores the distribution it is handed, this asks how
    much that distribution is to be believed. ``k`` is the half-credit size — at
    ``n == k`` exactly half the score is credited — so the discount is severe where
    the evidence is thin, easing off smoothly as members accumulate and never
    reaching 1.0.

    It is a DISCOUNT, never a gate. A group of two keeps a small score and stays in
    the ranking, where the radius can refuse it on its merits; it is not deleted for
    being small, because "too little evidence to rank highly" and "no candidate at
    all" are different statements and only the first one is true.

    ``k = 0`` disables the discount (every n scores 1.0). A non-positive ``n`` has no
    evidence at all and scores 0.0.
    """
    if n <= 0:
        return 0.0
    total = float(n) + float(k)
    return float(n) / total if total > 0 else 0.0


def scale_time(z: float, time_scale: float) -> float:
    """
    Z read in a different unit: growth over ``time_scale`` days instead of over one.

    A linear rescaling and deliberately nothing more. Z is a rate with the unit
    ``1/day``; expressing it per week or per quarter multiplies it by a constant, and
    a constant that multiplies EVERY score cancels out of every comparison — the
    ranking is identical and each candidate's share of the total is identical,
    because the factor divides out of ``S_i / sum(S)``. What it does change is the
    absolute number, which is the only thing the moderator's radius can read.

    It lives here, next to the factor it rescales, but it is applied OUTSIDE
    :func:`score_trade`: putting it in the formula would make a unit look like a
    term, and would move the Z-values the formula is validated on.
    """
    return z * float(time_scale)


def downside_share(outcomes: Sequence[dict[str, Any]]) -> float:
    """
    The weighted share of the distribution's movement that is loss.

    ``sum(p_i * |x_i|)`` over the negative levels, divided by the same sum over all
    of them. Magnitude-weighted rather than count-weighted on purpose: seven small
    losses against three large gains is a different proposition from seven large
    losses against three small gains, and counting alone cannot tell them apart.
    A distribution that never moves has no downside and reports 0.0.
    """
    absolute = sum(float(o["p"]) * abs(float(o["x_pct"])) for o in outcomes)
    if absolute <= 0:
        return 0.0
    negative = sum(float(o["p"]) * abs(float(o["x_pct"]))
                   for o in outcomes if float(o["x_pct"]) < 0)
    return negative / absolute


# ---------------------------------------------------------------------------
# The score itself
# ---------------------------------------------------------------------------


def score_trade(outcomes: Sequence[dict[str, Any]], p_direction: float,
                path: Optional[Sequence[tuple[float, float]]] = None, *,
                params: Params) -> dict[str, Any]:
    """
    Score one bundled outcome distribution: ``S = mu * w(p) * U * Z``.

    Returns the score together with every component that produced it, because a
    score nobody can take apart is a score nobody can calibrate. ``rejected`` says
    whether the downside filter fired; when it did, ``score`` is 0.0 and the
    components are still filled in, so the rejection can be read rather than guessed.

    The caller decides what to do with a direction: this function scores what it is
    handed and never looks at whether the group went up or down. Restricting
    emission to upward groups belongs where decisions are made, not here.

    THE FIRST THING THAT HAPPENS IS THE RECENCY WEIGHTING, and it happens exactly
    once, on the way in (:func:`weight_by_age`). Every factor below then reads the
    same re-weighted distribution — mu, sigma and the downside filter alike — which
    is the point: a distribution is one object, and pricing recency into the
    expectation while the loss filter still counted old losses at full strength
    would be two different distributions wearing one name. The formula itself is
    untouched; what changed is which numbers go into it.
    """
    aged = weight_by_age(outcomes, params=params)
    oldest_weight, newest_weight = recency_span(outcomes, params=params)

    mu, sigma = expectation(aged)
    weight = confidence_weight(p_direction, params=params)
    factor = uncertainty_factor(mu, sigma)
    speed = time_factor(path or ())
    duration, total = duration_and_total(path or ())
    downside = downside_share(aged)

    rejected = downside > params.downside_threshold
    score = 0.0 if rejected else mu * weight * factor * speed

    return {
        "score": score,
        "rejected": rejected,
        "components": {
            "mu_pct": mu,
            "sigma_pct": sigma,
            "w": weight,
            "u": factor,
            "z": speed,
            "duration_days": duration,
            "r_total_frac": total,
            "downside_share": downside,
            "p_direction": float(p_direction),
            "n_outcomes": len(outcomes),
            "recency_lambda": float(params.recency_lambda),
            "recency_floor": float(params.recency_floor),
            "recency_weight_oldest": oldest_weight,
            "recency_weight_newest": newest_weight,
        },
    }
