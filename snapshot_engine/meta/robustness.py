"""
META — robustness: which snapshot groups are worth believing?

The snapshot layer answers "which days looked alike". That alone is not worth
much: any tolerance loose enough will find look-alikes, and any run of adjacent
days looks alike by construction. This layer asks the harder question —

    *** does the SITUATION carry the OUTCOME, or did the group get lucky? ***

Three ideas, in this order:

1. TEMPORAL COLLAPSE. Neighbouring days with the same Block A are not independent
   evidence: they share one future, so counting them separately manufactures
   uniformity out of nothing. A run of near-in-time days with near-identical A
   collapses into ONE occurrence, represented by its FIRST day — the day the
   situation first existed. Days whose parameters differ never collapse, however
   adjacent they are.

2. TWO TOLERANCE TIERS. The same collapsed snapshots are grouped twice: once at
   the BROAD tolerances, once at NARROW ones (every continuous tolerance times
   ``narrow_factor``; the hard states must match exactly either way). Narrow is
   strictly stricter, so every narrow group sits inside exactly one broad group —
   the same region of situation-space, seen at two resolutions.

3. THE COMPARISON THAT MATTERS. Take a narrow group's Block-B uniformity and its
   broad container's. If sharpening the situation sharpens the future
   (``improvement >= 0``), the parameters are carrying the prediction. If it does
   NOT — if the broad set contains plenty of same-looking days whose futures went
   everywhere — then the narrow group's tidy outcome is a coincidence of who
   happened to fall inside the tighter box, no matter how tidy it looks alone.
   That single comparison is the whole point of this module.

Block B is compared as a TRAJECTORY, not as a single endpoint: every snapshot
carries a forward profile sampled at trading-day offsets from its own day, and a
group's uniformity is measured offset by offset along that path
(:func:`profile_uniformity`). What survives the four gates also carries the honest
spread behind it -- the p10 / p50 / p90 of the forward returns and the median time
to each extreme -- and never a manufactured probability.

A group is called ROBUST only when all four hold: the narrow group's B is uniform
(``min_uniformity``), narrowing did not hurt it (``min_improvement``), it happened
often enough as INDEPENDENT occurrences (``min_independent_occurrences``), and
those occurrences are spread across time (``min_time_spread_days``) rather than
huddled in one episode. The last two are what separate a pattern from an anecdote:
the same situation in different months or years leading to the same outcome.

Everything else keeps running in the background — every snapshot, every group, at
both tiers. Only the robust ones are worth SHOWING, and that is all the verdict
does: ``robust`` is a LABEL on the record, never a filter on the way out
(:func:`all_groups` hands over every group, :func:`robust_groups` is the display
view). A group that missed a condition is weaker evidence, not absent evidence, and
what weak evidence is worth is a question for the scorer and the radius — a place
where it can be answered by a number instead of by a yes or a no.

Block A is untouched here and stays strictly past-only (the causal recompute in
:mod:`snapshot_engine.meta.snapshot`). Primitive values come only through
:mod:`snapshot_engine.meta.cache`. Deterministic and pure: same series, same params, same
groups, same verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from statistics import median, pstdev
from typing import Any, NamedTuple, Optional

from . import cache as meta_cache
from . import snapshot as snapshot_module

SOURCE = "meta.robustness"

UP, DOWN = "up", "down"


@dataclass(frozen=True)
class Params:
    """The robustness tunables, plus the snapshot params they build on."""

    snapshot: snapshot_module.Params
    collapse_window_days: int
    narrow_factor: float
    min_uniformity: float
    min_improvement: float
    min_independent_occurrences: int
    min_time_spread_days: int
    # Calendar days two kept occurrences must lie apart. Defaulted so a caller that
    # builds Params by hand keeps working; the config stays the real source.
    min_independent_gap_days: int = 90
    # The outcome-bundling tunables, defaulted for the same reason.
    bundle_population: str = "independent"
    bundle_tol_pct: float = 1.5
    # Arithmetic protection only: below two members an offset has no sigma and no
    # meaningful p_direction. Not a robustness threshold -- see `bundle_outcomes`.
    min_members_at_offset: int = 2


def params_from_config(config) -> Params:
    """The ONE place the robustness tunables are read."""
    return Params(
        snapshot=snapshot_module.params_from_config(config),
        collapse_window_days=int(config.get("snapshot.collapse_window_days")),   # tunable
        narrow_factor=float(config.get("snapshot.narrow_factor")),               # tunable
        min_uniformity=float(config.get("snapshot.min_uniformity")),             # tunable
        min_improvement=float(config.get("snapshot.min_improvement")),           # tunable
        min_independent_occurrences=int(
            config.get("snapshot.min_independent_occurrences")),                 # tunable
        min_time_spread_days=int(config.get("snapshot.min_time_spread_days")),   # tunable
        min_independent_gap_days=int(
            config.get("snapshot.min_independent_gap_days")),                    # tunable
        bundle_population=str(config.get("snapshot.bundle_population")),         # tunable
        bundle_tol_pct=float(config.get("snapshot.bundle_tol_pct")),             # tunable
        min_members_at_offset=int(
            config.get("snapshot.min_members_at_offset")),                       # tunable
    )


def narrowed(params: snapshot_module.Params, factor: float) -> snapshot_module.Params:
    """
    The same similarity params with every CONTINUOUS tolerance tightened.

    The hard states are categories and stay exactly as they are — there is no
    "narrower" version of a state matching. Only the three relative tolerances
    shrink, which is what makes a narrow group a strict subset of a broad one.
    """
    return replace(
        params,
        sim_tol_ath_gap=params.sim_tol_ath_gap * factor,
        sim_tol_vol_ratio=params.sim_tol_vol_ratio * factor,
        sim_tol_days_ath=max(1, int(round(params.sim_tol_days_ath * factor))),
    )


@dataclass
class RobustGroup:
    """A group whose uniform future survived every test this module can put to it."""

    group_id: int
    member_dates: list[str]          # the representative (collapsed) knowledge days
    independent_occurrences: int
    time_spread_days: int
    uniformity_narrow: float
    uniformity_broad: float
    improvement: float
    dominant_direction: str          # "up" | "down"
    # The evidence behind the verdict. Defaulted so an existing caller that builds
    # a RobustGroup by hand keeps working.
    independent_dates: list[str] = field(default_factory=list)   # the members that counted
    distribution: dict[str, Any] = field(default_factory=dict)   # n + quantiles + timing
    profile_distance_mean: Optional[float] = None                # how alike the paths were
    bundle: dict[str, Any] = field(default_factory=dict)         # outcomes per offset
    # Did this group pass all four conditions? A LABEL, not a filter: the dashboard
    # shows the robust ones, the street scores every group and lets the score say
    # what a weak one is worth. Defaulted True so a caller that builds a RobustGroup
    # by hand keeps the old meaning.
    robust: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "group_id": self.group_id,
            "member_dates": list(self.member_dates),
            "independent_occurrences": self.independent_occurrences,
            "time_spread_days": self.time_spread_days,
            "uniformity_narrow": self.uniformity_narrow,
            "uniformity_broad": self.uniformity_broad,
            "improvement": self.improvement,
            "dominant_direction": self.dominant_direction,
            "independent_dates": list(self.independent_dates),
            "distribution": dict(self.distribution),
            "profile_distance_mean": self.profile_distance_mean,
            "bundle": dict(self.bundle),
            "robust": self.robust,
        }


class Uniformity(NamedTuple):
    """How alike a group's futures were, in the two ways that differ."""

    dir_fraction: float    # share of members that went the dominant way
    path_spread: float     # dispersion ALONG the path, not of the endpoint
    score: float           # the two combined, 0 .. 1
    direction: str         # the dominant way itself


def _days_between(first: str, second: str) -> int:
    return abs((date.fromisoformat(second) - date.fromisoformat(first)).days)


def _forward(snapshot: snapshot_module.Snapshot) -> float:
    return float(snapshot.block_b.get("forward", {}).get("forward_return_pct", 0.0))


def uniformity(snapshots: list[snapshot_module.Snapshot]) -> Uniformity:
    """
    How uniform the futures of these days were.

    ``path_spread`` here IS the endpoint dispersion, because this scalar form has no
    path to look along. :func:`profile_uniformity` fills the same slot with the
    per-offset dispersion instead — same name, same role in the formula, different
    measurement, and neither is the sigma the scorer uses.

    Two different things can go wrong with a group's outcomes and they need
    separate measurement: the members can disagree about the DIRECTION, or agree
    on it but disagree wildly about the SIZE. ``dir_fraction`` catches the first,
    the spread of returns catches the second.

    They combine as ``dir_fraction / (1 + cv)``, where ``cv`` is the spread
    relative to the group's own average move. Using a relative spread rather than
    an absolute one keeps the score comparable between a calm stock and a violent
    one — the same reason every feature in Block A is relative. A group that all
    went the same way by the same amount scores 1.0; disagreement in either sense
    pulls it down. Returns of exactly zero count for neither direction: a day that
    went nowhere is not evidence for either outcome.
    """
    if not snapshots:
        return Uniformity(0.0, 0.0, 0.0, UP)
    returns = [_forward(s) for s in snapshots]
    ups = sum(1 for r in returns if r > 0)
    downs = sum(1 for r in returns if r < 0)
    direction = UP if ups >= downs else DOWN
    dir_fraction = max(ups, downs) / len(returns)

    spread = float(pstdev(returns)) if len(returns) > 1 else 0.0
    mean_abs = sum(abs(r) for r in returns) / len(returns)
    if mean_abs > 0:
        cv = spread / mean_abs
    else:
        cv = 1.0 if spread > 0 else 0.0
    return Uniformity(dir_fraction, spread, dir_fraction / (1.0 + cv), direction)


MIN_MEMBERS_PER_OFFSET = 2      # a dispersion needs two values; this is maths, not a knob


def independent_members(snapshots: list[snapshot_module.Snapshot], *,
                        min_gap_days: int) -> list[snapshot_module.Snapshot]:
    """
    The members of a group that own SEPARATE futures, kept greedily.

    Collapsing already folded runs of adjacent same-situation days. This is the
    other half of the same worry and it is not the same half: two occurrences can
    be weeks apart, never fold into one, hold genuinely different situations -- and
    still share most of their forward window, because the window is long. Counted
    as two, they turn one stretch of future into two confirmations.

    So the members are walked oldest first, the earliest is always kept, and every
    later one closer than ``min_gap_days`` CALENDAR days to the last kept member is
    dropped. Greedy from the oldest rather than from the best keeps the answer
    independent of how the group happened to be ordered, and keeps the day a
    situation FIRST occurred -- the only one of an overlapping pair that was news.

    ``min_gap_days`` is measured in calendar days while horizons are measured in
    trading days, so the configured value must sit near ``max_horizon_days * 7/5``
    for two kept futures to be truly disjoint. Raising the horizon without raising
    the gap quietly lets overlap back in, which is why the two are documented
    together in ``parameters.yaml``.
    """
    if not snapshots:
        return []
    ordered = sorted(snapshots, key=lambda s: s.date)
    kept = [ordered[0]]
    for candidate in ordered[1:]:
        if _days_between(kept[-1].date, candidate.date) >= min_gap_days:
            kept.append(candidate)
    return kept


def _profile_of(snapshot: snapshot_module.Snapshot) -> dict[str, Any]:
    return snapshot.block_b.get("forward", {}).get("profile") or {}


def _offset_columns(snapshots: list[snapshot_module.Snapshot]
                    ) -> dict[int, list[float]]:
    """
    The group's returns gathered per offset: one column of values per age.

    A column is what makes a trajectory comparison possible at all -- every entry in
    it was measured the same number of trading days after its own situation, so the
    spread of a column is disagreement about the future rather than about the
    calendar. Offsets a member never reached contribute nothing to its column.
    """
    columns: dict[int, list[float]] = {}
    for snapshot in snapshots:
        profile = _profile_of(snapshot)
        offsets = profile.get("offsets") or []
        returns = profile.get("return_pct") or []
        for i, offset in enumerate(offsets):
            if i < len(returns) and returns[i] is not None:
                columns.setdefault(int(offset), []).append(float(returns[i]))
    return columns


def profile_uniformity(snapshots: list[snapshot_module.Snapshot]) -> Uniformity:
    """
    How alike the group's futures were along the WHOLE path, not just at the end.

    Same spirit as :func:`uniformity` and the same formula shape --
    ``dir_fraction / (1 + spread / mean_abs)`` -- but the spread is now measured
    across the trajectory: for every offset the members share, the dispersion of
    their returns at that age; those dispersions are then averaged into one number,
    and ``mean_abs`` averages the typical move at each age so the ratio stays
    relative and comparable between a calm stock and a violent one.

    An offset that fewer than ``MIN_MEMBERS_PER_OFFSET`` members reached is skipped:
    a dispersion of one value is zero, and letting it in would reward a group for
    the offsets most of it never saw. If no offset survives -- a group whose
    snapshots carry no profile at all -- the scalar :func:`uniformity` answers
    instead, which is exactly what this measurement degrades to.

    Direction still comes from the endpoint return: the path says how alike the
    futures were, the endpoint still says which way they went.
    """
    if not snapshots:
        return Uniformity(0.0, 0.0, 0.0, UP)
    returns = [_forward(s) for s in snapshots]
    ups = sum(1 for r in returns if r > 0)
    downs = sum(1 for r in returns if r < 0)
    direction = UP if ups >= downs else DOWN
    dir_fraction = max(ups, downs) / len(returns)

    columns = {offset: values for offset, values in _offset_columns(snapshots).items()
               if len(values) >= MIN_MEMBERS_PER_OFFSET}
    if not columns:
        return uniformity(snapshots)

    spreads = [float(pstdev(values)) for values in columns.values()]
    sizes = [sum(abs(v) for v in values) / len(values) for values in columns.values()]
    spread = sum(spreads) / len(spreads)
    mean_abs = sum(sizes) / len(sizes)
    if mean_abs > 0:
        cv = spread / mean_abs
    else:
        cv = 1.0 if spread > 0 else 0.0
    return Uniformity(dir_fraction, spread, dir_fraction / (1.0 + cv), direction)


def _quantile(values: list[float], q: float) -> Optional[float]:
    """
    The q-quantile by linear interpolation, or ``None`` for an empty list.

    Written out rather than imported so the answer cannot shift with a library
    version: the same members must always produce the same quantile.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def outcome_distribution(snapshots: list[snapshot_module.Snapshot]) -> dict[str, Any]:
    """
    The honest spread behind a group, as numbers rather than as a verdict.

    Carries how many members it rests on, the p10 / p50 / p90 of their forward
    returns, and the median number of trading days each took to reach its best and
    its worst point. That is the material a probability statement would have to be
    built from later; this function deliberately does not build one. With a handful
    of occurrences a two-branch "p=..." reads like knowledge and is arithmetic on
    noise, so what gets stored is the distribution and the ``n`` it came from.
    """
    returns = [_forward(s) for s in snapshots]
    up_days = [p["days_to_max_up"] for p in map(_profile_of, snapshots)
               if p.get("days_to_max_up") is not None]
    down_days = [p["days_to_max_down"] for p in map(_profile_of, snapshots)
                 if p.get("days_to_max_down") is not None]
    return {
        "n": len(snapshots),
        "return_p10": _quantile(returns, 0.10),
        "return_p50": _quantile(returns, 0.50),
        "return_p90": _quantile(returns, 0.90),
        "median_days_to_max_up": float(median(up_days)) if up_days else None,
        "median_days_to_max_down": float(median(down_days)) if down_days else None,
    }


INDEPENDENT, ALL_MEMBERS = "independent", "all_members"


def _cluster(values: list[float], tol: float) -> list[list[float]]:
    """
    Group numbers that lie within ``tol`` of each other into levels.

    Walked from the smallest upward, and every value is compared against the value
    that OPENED its level rather than against its neighbour. Comparing neighbours
    would let a level creep across an arbitrary distance one small step at a time --
    the same drift the snapshot collapse avoids by comparing against a run's
    representative, and the same reason it is avoided here.

    A tolerance of zero puts every distinct value in its own level, which is the
    honest limit case: no merging, no variance quietly removed.
    """
    return [[value for value, _ in level]
            for level in _cluster_pairs([(v, None) for v in values], tol)]


def _cluster_pairs(pairs: list[tuple[float, Any]], tol: float
                   ) -> list[list[tuple[float, Any]]]:
    """
    :func:`_cluster` with a companion value carried along each number.

    Same walk, same comparison against the value that OPENED the level — the second
    element is never looked at, it just stays attached. It exists so a level can say
    not only WHICH returns fell into it but which days produced them, without
    clustering twice or matching values back up afterwards (equal returns from
    different days would be indistinguishable if one tried).
    """
    levels: list[list[tuple[float, Any]]] = []
    for value, mark in sorted(pairs, key=lambda pair: pair[0]):
        if levels and value - levels[-1][0][0] <= tol:
            levels[-1].append((value, mark))
        else:
            levels.append([(value, mark)])
    return levels


def bundle_outcomes(snapshots: list[snapshot_module.Snapshot], *, direction: str,
                    tol_pct: float, min_members: int,
                    population: str = "independent",
                    day_index: Optional[dict[str, int]] = None) -> dict[str, Any]:
    """
    A group's futures as a distribution of return LEVELS, one offset at a time.

    This is the raw material a score is computed FROM, and it is deliberately raw:
    what came out, how often, and when. No mean, no dispersion, no verdict -- those
    belong to :mod:`snapshot_engine.street.scoring`, and computing them here would scatter
    one formula across two layers.

    PER OFFSET, NOT PER HORIZON. Members of a group reach different distances into
    the future, so the only place their futures can be compared is at an offset they
    all actually saw. Each offset of the profile ladder is therefore bundled on its
    own, over exactly the members defined there. The largest offset that survives is
    how far ahead this group can speak at all -- typically far short of the horizon,
    which is a fact about the data and not a defect.

    ``min_members`` IS ARITHMETIC PROTECTION, NOT A QUALITY FILTER. A single member
    at an offset has no dispersion to measure (sigma is undefined on one observation)
    and a ``p_direction`` of exactly 0 or 1 that means nothing, so such an offset is
    skipped because it cannot be COMPUTED, not because it is judged too thin. Small
    but computable groups pass through here on purpose and are discounted later, by
    the scorer's ``U`` and by the sample confidence -- continuously, where a number
    can express how thin they are.

    TWO PROBABILITIES COME OUT AND THEY ARE NOT THE SAME. ``p`` inside ``outcomes``
    is how often a return LEVEL occurred and sums to one across the levels;
    ``p_direction`` is how often the move went the group's way, counted over the same
    members. The scorer weights its confidence with the second and would be wrong to
    use the first -- a distribution split into many small levels is not thereby
    unsure about its direction.

    HOW OLD EACH OBSERVATION IS TRAVELS WITH IT, AS A FACT AND NOT AS A WEIGHT.
    Given a ``day_index`` (date -> trading-day ordinal of the reading), every level
    also carries ``ages``: the distance in TRADING days from each contributing day to
    the newest day of that index, in the same order as nothing in particular — a
    level is a bag. What those ages are WORTH is a scoring question and is decided in
    :mod:`snapshot_engine.street.scoring`, which owns the decay and its floor; measuring the
    age here and pricing it there is the same split as everywhere else in this file.
    Without a ``day_index`` no ``ages`` key appears and every consumer treats the
    level as ageless, which is what a synthetic bundle in a test should be.

    Returns are relative throughout: levels in PERCENT (a magnitude to read),
    ``mean_return_frac`` and the path in FRACTIONS (what a growth rate needs). No
    absolute price enters at any point.
    """
    newest = max(day_index.values()) if day_index else None
    columns: dict[int, list[tuple[float, Optional[int]]]] = {}
    for snapshot in snapshots:
        profile = _profile_of(snapshot)
        offsets = profile.get("offsets") or []
        returns = profile.get("return_pct") or []
        ordinal = day_index.get(snapshot.date) if day_index else None
        age = (newest - ordinal) if (newest is not None and ordinal is not None) else None
        for i, offset in enumerate(offsets):
            if i < len(returns) and returns[i] is not None:
                columns.setdefault(int(offset), []).append((float(returns[i]), age))

    usable = sorted(offset for offset, values in columns.items()
                    if len(values) >= min_members)
    tol = float(tol_pct) / 100.0        # the tunable is percentage points
    records: list[dict[str, Any]] = []
    path: list[list[float]] = []
    for offset in usable:
        pairs = columns[offset]
        values = [value for value, _ in pairs]
        count = len(values)
        outcomes = []
        for level in _cluster_pairs(pairs, tol):
            entry: dict[str, Any] = {
                "x_pct": (sum(v for v, _ in level) / len(level)) * 100.0,
                "p": len(level) / count,
            }
            ages = [a for _, a in level if a is not None]
            if len(ages) == len(level):
                entry["ages"] = ages
            outcomes.append(entry)
        if direction == DOWN:
            agreeing = sum(1 for v in values if v < 0)
        else:
            agreeing = sum(1 for v in values if v > 0)
        mean_return = sum(values) / count
        records.append({
            "offset_t": offset,
            "n_t": count,
            "outcomes": outcomes,
            "p_direction": agreeing / count,
            "mean_return_frac": mean_return,
        })
        path.append([float(offset), mean_return])

    return {
        "direction": direction,
        "population": population,
        "tol_pct": float(tol_pct),
        "min_members_at_offset": int(min_members),
        "max_usable_offset": usable[-1] if usable else 0,
        "offsets": records,
        "path": path,
    }


def population_for(members: list[snapshot_module.Snapshot],
                   kept: list[snapshot_module.Snapshot], *,
                   population: str) -> list[snapshot_module.Snapshot]:
    """
    Which members the distribution is built from.

    ``independent`` (the default) uses only what survived the overlap thinning, so
    no forward window is counted twice; ``all_members`` uses every folded day, which
    yields more outcomes at the price of letting neighbouring futures vote more than
    once. An unknown word is a configuration error and is refused here rather than
    silently treated as one of the two.
    """
    if population == ALL_MEMBERS:
        return members
    if population != INDEPENDENT:
        raise ValueError(f"unknown bundle_population: {population!r} "
                         f"({INDEPENDENT}|{ALL_MEMBERS})")
    return kept


def mean_profile_distance(snapshots: list[snapshot_module.Snapshot], *,
                          params: snapshot_module.Params) -> Optional[float]:
    """
    The average distance between the group's futures, over comparable pairs only.

    A pair whose profiles share too few offsets is incomparable and is left out
    rather than counted as a match (see
    :func:`snapshot_engine.meta.snapshot.profile_distance`). ``None`` means no pair in the
    group could be compared at all -- not that the group was uniform.
    """
    distances = [d for i, a in enumerate(snapshots) for b in snapshots[i + 1:]
                 if (d := snapshot_module.profile_distance(
                     _profile_of(a), _profile_of(b), params=params)) is not None]
    return sum(distances) / len(distances) if distances else None


def collapse_snapshots(snapshots: list[snapshot_module.Snapshot], *,
                       collapse_window_days: int,
                       narrow_tols: snapshot_module.Params
                       ) -> list[snapshot_module.Snapshot]:
    """
    Fold runs of near-in-time, same-situation days into ONE snapshot each.

    Two adjacent days with the same Block A are not two pieces of evidence. Their
    forward windows overlap almost completely, so they will agree about the future
    whatever the future does — count them separately and a single quiet fortnight
    turns into ten "independent" confirmations. Each run therefore becomes one
    occurrence, represented by its FIRST day: the day the situation first existed,
    and the only one of them that is news.

    A day joins the current run when it is within ``collapse_window_days``
    CALENDAR days of the previous member AND still similar to the run's
    representative under the NARROW tolerances — same-situation, not merely
    same-ish. Comparing against the representative rather than the neighbour stops
    a run from drifting across the whole tolerance band one small step at a time.
    Days whose parameters genuinely differ never collapse, however adjacent.

    The representative keeps its OWN Block B (the futures overlap; one of them
    counts once) and records the fold in ``block_b["collapse"]`` — how many days
    it stands for and the span they covered. Block A is untouched.
    """
    if not snapshots:
        return []

    collapsed: list[snapshot_module.Snapshot] = []
    run: list[snapshot_module.Snapshot] = [snapshots[0]]
    for current in snapshots[1:]:
        near = _days_between(run[-1].date, current.date) <= collapse_window_days
        same = snapshot_module.similar(run[0], current, params=narrow_tols)
        if near and same:
            run.append(current)
            continue
        collapsed.append(_representative(run))
        run = [current]
    collapsed.append(_representative(run))
    return collapsed


def _representative(run: list[snapshot_module.Snapshot]) -> snapshot_module.Snapshot:
    """The first day of a run, carrying the fold it stands for."""
    first = run[0]
    block_b = dict(first.block_b)
    block_b["collapse"] = {
        "collapsed_count": len(run),
        "from_date": first.date,
        "to_date": run[-1].date,
    }
    return snapshot_module.Snapshot(symbol=first.symbol, date=first.date,
                                    block_a=first.block_a, block_b=block_b)


def _by_date(snapshots: list[snapshot_module.Snapshot]
             ) -> dict[str, snapshot_module.Snapshot]:
    return {s.date: s for s in snapshots}


def score_groups(collapsed: list[snapshot_module.Snapshot], *, params: Params,
                 day_index: Optional[dict[str, int]] = None
                 ) -> list[dict[str, Any]]:
    """
    Group the collapsed snapshots at both tiers and score every narrow group
    against the broad group that contains it.

    Both tiers are thinned to their INDEPENDENT members first
    (:func:`independent_members`), and every number after that -- uniformity,
    occurrence count, spread, distribution -- is measured on what survived. Scoring
    a group on days whose futures overlap is what made a long horizon look like
    more evidence instead of the same evidence stretched, so the thinning happens
    before the counting rather than as a filter after it. ``member_days`` keeps the
    count before thinning beside it, so the difference stays visible.

    ``day_index`` maps a date to its trading-day ordinal in the FULL series, not in
    the collapsed list: the ages it produces have to be distances on the calendar the
    market actually traded, and the collapsed list has holes in it by construction.
    It only ever reaches :func:`bundle_outcomes`, and only to record how old each
    observation is. Omit it and the bundle simply carries no ages.

    Returns one record per narrow group -- robust or not -- so a caller can see
    what was rejected and why. :func:`robust_groups` is the filtered view.
    """
    broad_tols = params.snapshot
    narrow_tols = narrowed(broad_tols, params.narrow_factor)

    broad = snapshot_module.group_similar(collapsed, params=broad_tols)
    narrow = snapshot_module.group_similar(collapsed, params=narrow_tols)
    index = _by_date(collapsed)

    broad_of_date: dict[str, int] = {}
    broad_members: dict[int, list[snapshot_module.Snapshot]] = {}
    for group in broad:
        broad_members[group.group_id] = [index[d] for d in group.member_dates]
        for day in group.member_dates:
            broad_of_date[day] = group.group_id

    records: list[dict[str, Any]] = []
    gap = params.min_independent_gap_days
    for group in narrow:
        members = [index[d] for d in group.member_dates]
        kept = independent_members(members, min_gap_days=gap)
        containers = {broad_of_date.get(d) for d in group.member_dates}
        container = next(iter(containers)) if len(containers) == 1 else None

        narrow_score = profile_uniformity(kept)
        if container is not None:
            broad_score = profile_uniformity(
                independent_members(broad_members[container], min_gap_days=gap))
        else:
            broad_score = narrow_score
        spread = (_days_between(kept[0].date, kept[-1].date) if len(kept) > 1 else 0)
        occurrences = len(kept)

        record = {
            "narrow_group_id": group.group_id,
            "broad_group_id": container,
            "member_dates": list(group.member_dates),
            "independent_dates": [s.date for s in kept],
            "member_days": len(members),      # before thinning, for the honesty check
            "independent_occurrences": occurrences,
            "time_spread_days": spread,
            "uniformity_narrow": narrow_score.score,
            "uniformity_broad": broad_score.score,
            "improvement": narrow_score.score - broad_score.score,
            "dir_fraction": narrow_score.dir_fraction,
            "path_spread": narrow_score.path_spread,
            "dominant_direction": narrow_score.direction,
            "distribution": outcome_distribution(kept),
            "profile_distance_mean": mean_profile_distance(
                kept, params=params.snapshot),
            "bundle": bundle_outcomes(
                population_for(members, kept, population=params.bundle_population),
                direction=narrow_score.direction,
                tol_pct=params.bundle_tol_pct,
                min_members=params.min_members_at_offset,
                population=params.bundle_population,
                day_index=day_index),
        }
        record["robust"] = bool(
            record["uniformity_narrow"] >= params.min_uniformity
            and record["improvement"] >= params.min_improvement
            and occurrences >= params.min_independent_occurrences
            and spread >= params.min_time_spread_days
        )
        records.append(record)
    return records


def all_groups(symbol: str, *, params: Params,
               cache: Optional[meta_cache.MetaCache] = None,
               config=None) -> list[RobustGroup]:
    """
    EVERY group of ``symbol``, robust or not, each carrying its verdict as a flag.

    Collapsed, grouped at both tiers, scored — and then nothing is thrown away. A
    group that failed a condition is not an error and not an absence: it is a weaker
    piece of evidence, and the layers above are built to say what weak evidence is
    worth (the scorer's ``U``, the sample discount, the moderator's radius). Dropping
    it here would replace all of that with a yes/no answer the data cannot give.

    :func:`robust_groups` is the filtered view, for the one consumer that wants to
    SHOW groups rather than weigh them.
    """
    own = cache is None
    store = cache or meta_cache.MetaCache(config)
    try:
        snapshots = snapshot_module.build_snapshots(symbol, params=params.snapshot,
                                                    cache=store)
        if not snapshots:
            return []
        collapsed = collapse_snapshots(
            snapshots, collapse_window_days=params.collapse_window_days,
            narrow_tols=narrowed(params.snapshot, params.narrow_factor))
        # Trading-day ordinals off the FULL snapshot list — one per traded day,
        # before any folding — so an age is a count of sessions and not of
        # surviving group members.
        day_index = {snap.date: i for i, snap in enumerate(snapshots)}
        records = sorted(score_groups(collapsed, params=params, day_index=day_index),
                         key=lambda r: r["member_dates"][0])

        return [
            RobustGroup(
                group_id=i,
                member_dates=r["member_dates"],
                independent_occurrences=r["independent_occurrences"],
                time_spread_days=r["time_spread_days"],
                uniformity_narrow=r["uniformity_narrow"],
                uniformity_broad=r["uniformity_broad"],
                improvement=r["improvement"],
                dominant_direction=r["dominant_direction"],
                independent_dates=r["independent_dates"],
                distribution=r["distribution"],
                profile_distance_mean=r["profile_distance_mean"],
                bundle=r["bundle"],
                robust=bool(r["robust"]),
            )
            for i, r in enumerate(records)
        ]
    finally:
        if own:
            store.close()


def robust_groups(symbol: str, *, params: Params,
                  cache: Optional[meta_cache.MetaCache] = None,
                  config=None) -> list[RobustGroup]:
    """
    The groups of ``symbol`` worth SHOWING: :func:`all_groups`, kept down to the ones
    that passed all four conditions, renumbered so the ids are contiguous.

    Everything upstream still runs — every day gets a snapshot, every group is formed
    and scored at both tiers. This is the display view; it is not a precondition for
    being scored (see :func:`all_groups`).
    """
    kept = [g for g in all_groups(symbol, params=params, cache=cache, config=config)
            if g.robust]
    for i, group in enumerate(kept):
        group.group_id = i
    return kept
