"""
META — snapshot: one BLOCK per day, and the days that look alike.

A snapshot is everything the primitives say about ONE trading day x, assembled
into a single object and split in two by a rule that is easy to state and easy to
get wrong:

    *** THE SPLIT IS BY TIME, NOT BY PRIMITIVE. ***

  BLOCK A — what day x KNEW. Strictly past-only: every value in it is computed
    from days <= x. A plateau that ENDED on or before x is in A, because it was a
    finished fact by then. A breakout still running past x is NOT — what a day
    can know about it is only that something is breaking out right now, which is
    the ``in_breakout_now`` flag, measured from days <= x and true even if the
    move later fails.
  BLOCK B — what the days AFTER x shaped. The finished events that overlap x but
    end later, the ones that start later, and the forward price path. It is
    stored for later comparison and NEVER used to group.

The same primitive therefore lands in A or in B depending on WHEN its event
ended, which is why the split cannot be made by naming primitives.

Grouping compares BLOCK A ONLY, and only in RELATIVE terms — percentages, ratios,
counts of days and states. No absolute price ever enters a comparison: a day at
8 USD and a day at 800 USD with the same picture are the same picture. Two tiers
are never averaged either; a small plateau and a large one are different facts.

Similarity is two-tier:
  * HARD STATES must match EXACTLY — volume direction, in-plateau, in-breakout
    (including its direction). No tolerance: these are categories. The trend
    primitive's consistency state used to lead this list and was REMOVED: it was a
    classification that flipped with noise, so it fragmented groups without saying
    anything true about the day (see
    :func:`snapshot_engine.meta.cache.trend_classification_enabled`).
  * CONTINUOUS RELATIVE features must be within tolerance — the gap to the
    all-time high, the volume ratio, the days since the high.
Groups are the connected components of that relation, and only groups of at least
``min_group_size`` are kept; a day with no partner stays a singleton and is not
drawn.

Block B is stored twice over: as the three scalars it always had
(``forward_return_pct``, ``max_up_pct``, ``max_down_pct``) and as a PROFILE -- the
same forward path sampled at a grid of trading-day offsets anchored on x, with the
running envelopes, the days it took to reach each extreme and which extreme came
first. The scalars say where the future ended; the profile says what shape it had
on the way, which is what :func:`profile_distance` compares.

Every primitive value is obtained through :mod:`snapshot_engine.meta.cache` — this module
calls no primitive and redoes no primitive's maths. Deterministic and pure: same
series, same params, same snapshots and same groups, every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from . import cache as meta_cache

SOURCE = "meta.snapshot"


DEFAULT_PROFILE_OFFSETS = (1, 2, 3, 5, 8, 13, 21, 34, 45, 60)


@dataclass(frozen=True)
class Params:
    """The snapshot tunables. Tight by default: near-identical A, or no group."""

    max_horizon_days: int
    trail_window: int
    sim_tol_ath_gap: float
    sim_tol_vol_ratio: float
    sim_tol_days_ath: int
    min_group_size: int
    # The forward-profile tunables. They carry defaults so a caller that builds
    # Params by hand keeps working; the config stays the real source.
    profile_offsets: tuple[int, ...] = DEFAULT_PROFILE_OFFSETS
    min_profile_overlap: int = 4
    profile_w_path: float = 1.0
    profile_w_envelope: float = 0.5
    profile_w_timing: float = 0.25


def params_from_config(config) -> Params:
    """The ONE place the snapshot tunables are read."""
    offsets = config.get("snapshot.profile_offsets") or DEFAULT_PROFILE_OFFSETS  # tunable
    return Params(
        max_horizon_days=int(config.get("snapshot.max_horizon_days")),      # tunable
        trail_window=int(config.get("snapshot.trail_window")),              # tunable
        sim_tol_ath_gap=float(config.get("snapshot.sim_tol_ath_gap")),      # tunable
        sim_tol_vol_ratio=float(config.get("snapshot.sim_tol_vol_ratio")),  # tunable
        sim_tol_days_ath=int(config.get("snapshot.sim_tol_days_ath")),      # tunable
        min_group_size=int(config.get("snapshot.min_group_size")),          # tunable
        profile_offsets=tuple(sorted({int(t) for t in offsets if int(t) > 0})),
        min_profile_overlap=int(config.get("snapshot.min_profile_overlap")),  # tunable
        profile_w_path=float(config.get("snapshot.profile_w_path")),          # tunable
        profile_w_envelope=float(config.get("snapshot.profile_w_envelope")),  # tunable
        profile_w_timing=float(config.get("snapshot.profile_w_timing")),      # tunable
    )


@dataclass
class Snapshot:
    """One day, as a block of everything known then and everything that followed."""

    symbol: str
    date: str            # day x
    block_a: dict[str, Any]   # past-only: what x knew
    block_b: dict[str, Any]   # future-shaped: what came after x

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "symbol": self.symbol,
            "date": self.date,
            "block_a": self.block_a,
            "block_b": self.block_b,
        }


@dataclass
class SnapshotGroup:
    """Days whose Block A are mutually similar — one drawn ring colour."""

    group_id: int
    member_dates: list[str]
    size: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "group_id": self.group_id,
            "member_dates": list(self.member_dates),
            "size": self.size,
        }


# ---------------------------------------------------------------------------
# Building the blocks
# ---------------------------------------------------------------------------

# The Block-A features compared as CATEGORIES: they must match exactly.
#
# ``trend_state_at_x`` USED TO SIT FIRST HERE and no longer exists. The trend
# primitive's classification was deactivated (see
# :func:`snapshot_engine.meta.cache.trend_classification_enabled`): as a hard state it split
# days on a label that flipped with noise, and a grouping key that flips costs
# members twice — it fragments ``n`` and it separates by something that was never a
# fact about the day. Volume's direction stays: it comes from the VOLUME primitive,
# measured, not classified.
HARD_STATES = ("volume_trend_at_x", "in_plateau_now",
               "in_breakout_now", "breakout_direction_now")
# The Block-A features compared with a tolerance, all of them relative.
CONTINUOUS = ("ath_gap_pct", "volume_ratio_at_x", "days_since_ath")


def _event_ref(event: dict[str, Any], kind: str) -> dict[str, Any]:
    """
    A compact, tier-preserving reference to one detected event.

    Tier is carried through untouched: a small plateau and a large one are
    different statements about the same days, and collapsing them would destroy
    the very distinction the size classes exist for.
    """
    ref = {
        "kind": kind,
        "start_date": event["start_date"],
        "end_date": event["end_date"],
        "tier": event.get("tier"),
    }
    if "direction" in event:
        ref["direction"] = event["direction"]
    return ref


def _plateau_ref(plateau: dict) -> dict:
    """
    The past-only plateau that reaches a day, as a compact reference.

    Only the recomputed shape travels — band, mean, length, tier — never the
    symbol (the snapshot already knows it) and never a volume measurement (a
    past-only scan takes no volume series). Keeping it symbol-free also keeps
    Block A comparable across symbols, which is the whole point of a relative
    comparison.
    """
    return {
        "kind": "plateau",
        "start_date": plateau["start_date"],
        "end_date": plateau["end_date"],
        "tier": plateau.get("tier"),
        "band_low": plateau["band_low"],
        "band_high": plateau["band_high"],
        "mean": plateau["mean"],
        "duration_days": plateau["duration_days"],
    }


def _profile(prices: list[float], x: int, horizon: int,
             offsets: tuple[int, ...]) -> dict[str, Any]:
    """
    The forward path of day x, sampled at a grid of trading-day offsets.

    WHY A GRID AND NOT A WINDOW. Two days that sit close together in the calendar
    share almost the same forward window, so comparing their futures as windows
    compares one stretch of time with itself and finds an agreement that was never
    there. Anchoring the samples on each day's OWN x and comparing them
    offset-for-offset -- day 8 after A1 against day 8 after A2 -- turns two futures
    into two SHAPES that can be held side by side no matter when they happened.

    That makes them comparable. It does NOT make them independent evidence: two
    overlapping futures still overlap after being reshaped. Independence is a
    separate question, enforced in :mod:`snapshot_engine.meta.robustness` by keeping only
    occurrences that lie far enough apart to own separate futures.

    Everything is relative to ``close[x]``; no absolute price is stored. The three
    per-offset lists run parallel to ``offsets``, and an offset the data cannot
    reach (a day near the right edge of the CSV) stays ``None`` rather than being
    padded -- a value that was never observed must not be invented. The envelopes
    span ``[x .. x+t]`` INCLUSIVE of x, so ``run_max_up`` starts at 0.0 and only
    rises while ``run_max_down`` starts at 0.0 and only falls; the scalar
    ``max_up_pct`` beside them is measured over ``[x+1 .. end]`` and may therefore
    be negative. The two answer different questions and are deliberately not merged.
    """
    base = prices[x]
    last = len(prices) - 1
    sampled = [int(t) for t in offsets if 0 < int(t) <= horizon]
    returns: list[Optional[float]] = []
    up_envelope: list[Optional[float]] = []
    down_envelope: list[Optional[float]] = []
    for offset in sampled:
        target = x + offset
        if target > last or base <= 0:
            returns.append(None)
            up_envelope.append(None)
            down_envelope.append(None)
            continue
        window = prices[x:target + 1]
        returns.append((prices[target] - base) / base)
        up_envelope.append((max(window) - base) / base)
        down_envelope.append((min(window) - base) / base)

    end = min(last, x + horizon)
    if end > x and base > 0:
        path = prices[x + 1:end + 1]
        days_to_max_up: Optional[int] = 1 + path.index(max(path))
        days_to_max_down: Optional[int] = 1 + path.index(min(path))
        up_first: Optional[bool] = days_to_max_up < days_to_max_down
    else:
        days_to_max_up = days_to_max_down = None
        up_first = None

    return {
        "offsets": sampled,
        "return_pct": returns,
        "run_max_up": up_envelope,
        "run_max_down": down_envelope,
        "days_to_max_up": days_to_max_up,
        "days_to_max_down": days_to_max_down,
        "up_first": up_first,
    }


def _forward_summary(prices: list[float], x: int, horizon: int,
                     offsets: tuple[int, ...] = ()) -> dict[str, Any]:
    """
    Where price went over the Block-B horizon, purely relative to close[x].

    The three scalars are the original summary and keep their exact meaning --
    ``firing.py`` and the street read them. ``profile`` is the same future told as
    a shape (see :func:`_profile`), added beside them, never in place of them.
    """
    base = prices[x]
    end = min(len(prices) - 1, x + horizon)
    profile = _profile(prices, x, horizon, offsets)
    if end <= x or base <= 0:
        return {"horizon_days": 0, "forward_return_pct": 0.0,
                "max_up_pct": 0.0, "max_down_pct": 0.0, "profile": profile}
    path = prices[x + 1:end + 1]
    return {
        "horizon_days": end - x,
        "forward_return_pct": (prices[end] - base) / base,
        "max_up_pct": (max(path) - base) / base,
        "max_down_pct": (min(path) - base) / base,
        "profile": profile,
    }


def build_snapshots(symbol: str, *, params: Params,
                    cache: Optional[meta_cache.MetaCache] = None,
                    config=None) -> list[Snapshot]:
    """
    One snapshot per trading day of ``symbol``.

    Every primitive value comes from the cache; this function only assembles and
    splits. The Block-B horizon is read FROM THE DATA — it runs as far as the
    longest event still in flight at x reaches — and is then capped at
    ``max_horizon_days`` so one endless event cannot make a snapshot look
    arbitrarily far ahead. A day with nothing in flight takes the full cap; that is
    the common case now that trend segments no longer count as events, and it makes
    the horizons LONGER and more uniform, not shorter.
    """
    own = cache is None
    store = cache or meta_cache.MetaCache(config)
    try:
        series = store.closes(symbol)
        if series is None or len(series) == 0:
            return []
        prices = [float(v) for v in series.to_numpy()]
        dates = [pd.Timestamp(d).date().isoformat() for d in series.index]
        position = {d: i for i, d in enumerate(dates)}
        total = len(prices)

        distance = meta_cache.ath_distance(store, symbol)
        vol = meta_cache.trailing_volume(store, symbol, window=params.trail_window)
        breaking = meta_cache.breakout_in_progress(store, symbol)

        # Block A's plateau and breakout facts come from the PAST-ONLY scans:
        # every one of them is recomputed with the day in question as the right
        # edge. The forward-informed detections below are for Block B only.
        causal_plateau = meta_cache.causal_plateaus(store, symbol)
        causal_breaks = meta_cache.causal_breakouts(store, symbol)
        reaching_by_tier = {name: scan["reaching"]
                            for name, scan in causal_plateau.items()}
        known_events = sorted(
            [dict(_event_ref(p, "plateau"), known_date=p["known_date"])
             for scan in causal_plateau.values() for p in scan["completed"]
             if p["known_date"]]
            + [dict(_event_ref(b, "breakout"), known_date=b["known_date"])
               for b in causal_breaks if b["known_date"]],
            key=lambda e: (e["known_date"], e["start_date"]))
        # PLATEAU AND BREAKOUT ONLY. Trend segments used to be a third kind here and
        # are gone with the classification they came from: they entered Block B's
        # event lists and, because a segment covers long stretches, they set the
        # forward horizon of most days. Both now rest on events that were actually
        # detected rather than on a label of the slope.
        events = ([_event_ref(p, "plateau") for p in meta_cache.plateaus(store, symbol)]
                  + [_event_ref(b, "breakout")
                     for b in meta_cache.breakouts(store, symbol)])

        # events sorted by the day they end on, so "finished by x" is one walk
        by_end = sorted(events, key=lambda e: (e["end_date"], e["start_date"]))
        by_start = sorted(events, key=lambda e: (e["start_date"], e["end_date"]))

        snapshots: list[Snapshot] = []
        known = 0                        # events already knowable at day x
        counts = {"plateau": 0, "breakout": 0}
        last_done: dict[str, Optional[dict[str, Any]]] = {
            "plateau": None, "breakout": None}

        for x in range(total):
            day = dates[x]
            while known < len(known_events) and known_events[known]["known_date"] <= day:
                event = known_events[known]
                counts[event["kind"]] += 1
                last_done[event["kind"]] = event
                known += 1

            direction = breaking[x] if x < len(breaking) else None
            measured = vol[x] if x < len(vol) else {"ratio": 1.0, "trend": "flat"}
            reach = distance[x] if x < len(distance) else {}
            # the tightest class that reaches x wins: a small plateau is the
            # stronger statement, and tiers are never merged into one
            reached = next(
                (reaching_by_tier[name][x] for name in ("small", "large")
                 if name in reaching_by_tier and x < len(reaching_by_tier[name])
                 and reaching_by_tier[name][x]), None)
            plateau_now = _plateau_ref(reached) if reached else None

            block_a = {
                "ath_gap_pct": float(reach.get("gap_pct", 0.0)),
                "days_since_ath": int(reach.get("days_since_ath", 0)),
                "volume_ratio_at_x": float(measured["ratio"]),
                "volume_trend_at_x": measured["trend"],
                "in_plateau_now": plateau_now is not None,
                "plateau_now": plateau_now,
                "in_breakout_now": direction is not None,
                "breakout_direction_now": direction,
                "completed_events": {
                    kind: {"count": counts[kind], "last": last_done[kind]}
                    for kind in counts
                },
            }

            # Block B: the horizon comes from the events still in flight at x
            running = [e for e in by_start
                       if e["start_date"] <= day < e["end_date"]]
            reach_idx = max([position.get(e["end_date"], x) for e in running] + [x])
            horizon = min(int(params.max_horizon_days), max(0, reach_idx - x))
            if horizon == 0:
                horizon = min(int(params.max_horizon_days), total - 1 - x)
            ahead = [e for e in by_start
                     if day < e["start_date"] <= dates[min(total - 1, x + horizon)]]
            block_b = {
                "forward": _forward_summary(prices, x, horizon,
                                            params.profile_offsets),
                "running_events": running,      # overlap x but end after it
                "starting_events": ahead,       # begin inside the horizon
            }
            snapshots.append(Snapshot(symbol=symbol, date=day,
                                      block_a=block_a, block_b=block_b))
        return snapshots
    finally:
        if own:
            store.close()


# ---------------------------------------------------------------------------
# Comparing two futures as shapes
# ---------------------------------------------------------------------------


def _defined(profile: dict[str, Any]) -> dict[int, int]:
    """Offset -> its index, for the offsets this profile actually observed."""
    offsets = profile.get("offsets") or []
    returns = profile.get("return_pct") or []
    return {int(t): i for i, t in enumerate(offsets)
            if i < len(returns) and returns[i] is not None}


def _span(profile: dict[str, Any], defined: dict[int, int]) -> int:
    """The furthest offset this profile actually reached -- its own time scale."""
    return max(defined) if defined else 0


def profile_distance(a: dict[str, Any], b: dict[str, Any], *,
                     params: Params) -> Optional[float]:
    """
    How far apart two forward profiles are, as one number comparable across symbols.

    Only offsets that BOTH profiles observed take part: an offset one of them could
    not reach (right edge of its CSV) is not evidence of anything and is skipped
    rather than filled in. If fewer than ``min_profile_overlap`` offsets are shared,
    the pair is INCOMPARABLE and ``None`` comes back -- a thin overlap that happens
    to agree is not a match, and a caller must skip such a pair instead of reading
    the number as a small distance.

    Three terms, each already relative and therefore free of the price level:

    * the PATH term -- the mean absolute difference of the two returns at each
      shared offset: how differently the two futures stood at the same age.
    * the ENVELOPE term -- the same for the running best and running worst levels.
      Two paths can end in the same place having travelled very differently, and
      this is the term that notices.
    * the TIMING term -- the difference in how far into their own horizon each one
      reached its extremes, plus a flat penalty of one when the two disagree about
      which extreme came first. Normalising by each profile's own span is what lets
      a 20-day and a 60-day profile still be compared without one dominating.

    The weights are tunables, so their relative size is a config decision, not a
    property of this function. A distance of 0.0 means the two paths were identical
    at every shared offset; there is no upper bound, because returns have none.
    """
    defined_a, defined_b = _defined(a), _defined(b)
    shared = sorted(set(defined_a) & set(defined_b))
    if len(shared) < params.min_profile_overlap:
        return None                  # too little in common to mean anything

    path_gap = 0.0
    envelope_gap = 0.0
    envelope_terms = 0
    for offset in shared:
        i, j = defined_a[offset], defined_b[offset]
        path_gap += abs(a["return_pct"][i] - b["return_pct"][j])
        for key in ("run_max_up", "run_max_down"):
            left = (a.get(key) or [None] * (i + 1))[i]
            right = (b.get(key) or [None] * (j + 1))[j]
            if left is not None and right is not None:
                envelope_gap += abs(left - right)
                envelope_terms += 1
    path_term = path_gap / len(shared)
    envelope_term = envelope_gap / envelope_terms if envelope_terms else 0.0

    span_a, span_b = _span(a, defined_a), _span(b, defined_b)
    timing_term = 0.0
    timing_parts = 0
    if span_a > 0 and span_b > 0:
        for key in ("days_to_max_up", "days_to_max_down"):
            left, right = a.get(key), b.get(key)
            if left is not None and right is not None:
                timing_term += abs(left / span_a - right / span_b)
                timing_parts += 1
    if timing_parts:
        timing_term /= timing_parts
    if (a.get("up_first") is not None and b.get("up_first") is not None
            and a["up_first"] != b["up_first"]):
        timing_term += 1.0           # a mirrored path is not a similar path

    return (params.profile_w_path * path_term
            + params.profile_w_envelope * envelope_term
            + params.profile_w_timing * timing_term)


# ---------------------------------------------------------------------------
# Grouping — Block A only, relative only
# ---------------------------------------------------------------------------


def _hard_key(block_a: dict[str, Any]) -> tuple:
    """The categorical part of a Block A. Equal keys are a precondition to match."""
    return tuple(block_a.get(name) for name in HARD_STATES)


def _within_tolerance(a: dict[str, Any], b: dict[str, Any], params: Params) -> bool:
    """The continuous part: every relative feature inside its own tolerance."""
    return (abs(a["ath_gap_pct"] - b["ath_gap_pct"]) <= params.sim_tol_ath_gap
            and abs(a["volume_ratio_at_x"] - b["volume_ratio_at_x"])
            <= params.sim_tol_vol_ratio
            and abs(a["days_since_ath"] - b["days_since_ath"])
            <= params.sim_tol_days_ath)


def similar(a: Snapshot, b: Snapshot, *, params: Params) -> bool:
    """
    Do two days look the same, as far as each of them could tell at the time?

    Hard states must be identical and every continuous feature within tolerance.
    Only Block A is read — Block B is what happened NEXT, and letting it into the
    comparison would group days by their outcome instead of by their situation,
    which is precisely the mistake this layer exists to avoid.
    """
    return (_hard_key(a.block_a) == _hard_key(b.block_a)
            and _within_tolerance(a.block_a, b.block_a, params))


def group_similar(snapshots: list[Snapshot], *, params: Params) -> list[SnapshotGroup]:
    """
    Group the snapshots whose Block A are mutually similar (connected components).

    Days are bucketed by their hard-state key first — days that differ there can
    never match, so the pairwise tolerance check only runs inside a bucket. Groups
    smaller than ``min_group_size`` are dropped: a day with no partner is a
    singleton, not a pattern, and is not drawn. Ids are handed out in order of
    each group's earliest day, so the same input always numbers the same way.
    """
    buckets: dict[tuple, list[int]] = {}
    for i, snap in enumerate(snapshots):
        buckets.setdefault(_hard_key(snap.block_a), []).append(i)

    parent = list(range(len(snapshots)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for members in buckets.values():
        for pos, i in enumerate(members):
            for j in members[pos + 1:]:
                if _within_tolerance(snapshots[i].block_a, snapshots[j].block_a, params):
                    union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(len(snapshots)):
        components.setdefault(find(i), []).append(i)

    groups: list[SnapshotGroup] = []
    for root in sorted(components, key=lambda r: min(components[r])):
        members = sorted(components[root])
        if len(members) < params.min_group_size:
            continue                     # a singleton is not a pattern
        groups.append(SnapshotGroup(
            group_id=len(groups),
            member_dates=[snapshots[i].date for i in members],
            size=len(members),
        ))
    return groups
