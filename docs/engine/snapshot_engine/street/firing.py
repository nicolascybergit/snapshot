"""
STREET — the firing interface: the analyses that actually emit a decision.

A FIRING analysis is one that has something to propose. It looks at a symbol and
returns zero or more decisions, each with

  * an INITIAL SCORE — how much it is worth before anyone else has looked at it,
    unbounded above (there is no score cap any more), and
  * a GRIP CONDITION — the parameter picture it waits for, travelling inside the
    decision so the moderator never has to reconstruct the trigger.

Today the only firing analysis is the snapshot/robustness pair: every group of a
symbol becomes a situation, and its Block-A picture becomes the grip condition — the
same features the meta layer already uses to decide that two days look alike, reused
unchanged as "wait until it looks like this again".

EVERY GROUP, NOT ONLY THE ROBUST ONES. Whether a group passed the four robustness
conditions is a label it carries, not permission to be scored. The question this
layer asks is "what is this worth", and that is a number: a thin or scattered group
answers it with a small one, through the scorer's ``U``, through the sample
confidence applied here, and finally through the moderator's radius, which refuses
it for the score it earned rather than for the box it fell in. A gate at this point
would throw the observation away and swap the question for a yes-or-no the data
cannot support.

The SCORE comes from :mod:`snapshot_engine.street.scoring` whenever the situation carries a
bundle; ``dummy_initial`` remains only for a provider that supplies pictures without
distributions.

Parameter-free like every analysis module: the cap, the dummy score, the grip
tolerance and the scoring params are resolved by the caller and passed in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..logging_setup import get_logger
from ..meta import cache as meta_cache
from ..meta import robustness as robustness_module
from ..meta import snapshot as snapshot_module
from . import scoring
from .model import BUY, SELL, Decision, GripCondition, floor_score

#: One decision per usable offset, or one for the group's furthest offset only.
ALL_OFFSETS, BOUNDARY_ONLY = "all_offsets", "boundary_only"

SOURCE = "street.firing"

log = get_logger(__name__)

#: The Block-A features that make up a grip picture. Exactly the ones the meta layer
#: compares days by — hard states first, then the continuous relative readings — so a
#: grip condition says "the situation is back", in the vocabulary that already means
#: that. Never an absolute price: a grip must be as price-independent as the group it
#: came from.
GRIP_KEYS: tuple[str, ...] = snapshot_module.HARD_STATES + snapshot_module.CONTINUOUS


@dataclass(frozen=True)
class Situation:
    """One situation worth waiting for: which way it went, what it looked like, what it did."""

    direction: str               # "up" | "down"
    params: dict[str, Any]       # the picture the decision waits for
    #: The bundled outcomes of the group this came from, offset by offset (see
    #: :func:`snapshot_engine.meta.robustness.bundle_outcomes`). Defaulted empty so a
    #: provider that only knows situations still works — such a situation cannot be
    #: scored and falls back on the placeholder.
    bundle: dict[str, Any] = field(default_factory=dict)
    #: Did the group pass all four robustness conditions? Carried along for the meta
    #: layer to use later; it changes NOTHING here. A situation is emitted on the
    #: strength of its distribution, and a weak one ends up with a low score rather
    #: than with no decision.
    robust: bool = True
    #: The pictures of the group's OTHER member days; any of them counts as the
    #: situation being back (see ``GripCondition.alternatives``).
    member_params: tuple[dict[str, Any], ...] = ()


def grip_params(block_a: dict[str, Any]) -> dict[str, Any]:
    """The grip-relevant part of a Block A, in a fixed key order."""
    return {key: block_a.get(key) for key in GRIP_KEYS}


def side_of(direction: str) -> str:
    """A group that went up is a buy; one that went down is a sell."""
    return BUY if direction == robustness_module.UP else SELL


class FiringAnalysis(ABC):
    """An analysis that emits decisions."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def emit(self, symbol: str) -> list[Decision]:
        """
        The decisions this analysis has for ``symbol`` right now.

        Each carries an initial score at or below the cap and a grip condition.
        Order must be deterministic — the street sorts by score, and equal scores
        fall back on names, so a wobbling emission order would still be visible in
        the audit trail.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.name!r})"


def group_situations(config) -> Callable[[str], list[Situation]]:
    """
    The real source: EVERY group of a symbol, as a situation — robust or not.

    Robustness is a label the group carries, not a permit to be scored. A group that
    missed one of the four conditions still made observations, and the honest way to
    handle weaker observations is to score them lower, which the layers below already
    do continuously (the spread through ``U``, the member count through the sample
    confidence, the whole thing through the moderator's radius). Filtering here would
    replace all of that with a verdict, and would silently answer a question the
    conditions were never asked: not "is this evidence strong" but "may this evidence
    speak at all".

    One primitive cache is shared by the group run and the snapshot rebuild, so the
    primitives are computed at most once. The representative day of a group (its
    first member — the day the situation first existed) supplies the picture.
    """

    def provider(symbol: str) -> list[Situation]:
        params = robustness_module.params_from_config(config)
        store = meta_cache.MetaCache(config)
        try:
            groups = robustness_module.all_groups(symbol, params=params, cache=store)
            if not groups:
                return []
            by_date = {s.date: s for s in snapshot_module.build_snapshots(
                symbol, params=params.snapshot, cache=store)}
        finally:
            store.close()

        situations: list[Situation] = []
        for group in groups:
            representative = by_date.get(group.member_dates[0])
            if representative is None:
                continue
            members = tuple(grip_params(by_date[d].block_a)
                            for d in group.member_dates[1:] if d in by_date)
            situations.append(Situation(group.dominant_direction,
                                        grip_params(representative.block_a),
                                        bundle=group.bundle,
                                        robust=group.robust,
                                        member_params=members))
        return situations

    return provider


def score_offsets(bundle: dict[str, Any], *, params: scoring.Params
                  ) -> list[dict[str, Any]]:
    """
    Every usable offset of one bundle, scored, with the whole chain kept apart.

    The SAME steps :meth:`SnapshotFiring.emit` applies — ``score_trade``, then the
    time unit, then the sample discount — and the same numbers come out; emit calls
    this and turns the rows into decisions. It exists as its own function so a
    REPORT can score a group without emitting anything, and so there is exactly one
    place where the chain is written down. A second implementation of the order
    ``raw x time_scale x shrink`` is precisely the thing that would drift.

    Rows come back for every offset, including the ones the downside filter refused
    (``rejected``) and the ones that score zero. Emission drops those; a dossier
    keeps them, because "this offset was refused and here is by how much" is a
    reading, not an absence.
    """
    rows: list[dict[str, Any]] = []
    path = bundle.get("path") or []
    for record in bundle.get("offsets") or []:
        offset = int(record["offset_t"])
        walked = [(day, value) for day, value in path if day <= offset]
        result = scoring.score_trade(record["outcomes"], record["p_direction"],
                                     walked, params=params)
        components = result["components"]
        members = int(record.get("n_t", 0))
        raw_score = float(result["score"])
        scaled_score = raw_score * float(params.time_scale)
        shrink = (scoring.sample_confidence(members, params.k_confidence)
                  if params.confidence_shrink_enabled else 1.0)
        rows.append({
            "offset_t": offset,
            "n_t": members,
            "rejected": bool(result["rejected"]),
            "final_score": scaled_score * shrink,
            "score_trace": {
                "mu_pct": components["mu_pct"],
                "sigma_pct": components["sigma_pct"],
                "w": components["w"],
                "u": components["u"],
                "z": components["z"],
                "time_scale": float(params.time_scale),
                "z_scaled": scoring.scale_time(components["z"], params.time_scale),
                "downside_share": components["downside_share"],
                "p_direction": components["p_direction"],
                "recency_lambda": components["recency_lambda"],
                "recency_floor": components["recency_floor"],
                "recency_weight_oldest": components["recency_weight_oldest"],
                "recency_weight_newest": components["recency_weight_newest"],
                "duration_days": components["duration_days"],
                "r_total_frac": components["r_total_frac"],
                "raw_score": raw_score,
                "scaled_score": scaled_score,
                "shrink": shrink,
                "k_confidence": float(params.k_confidence),
                "shrink_enabled": bool(params.confidence_shrink_enabled),
                "final_score": scaled_score * shrink,
            },
        })
    return rows


class SnapshotFiring(FiringAnalysis):
    """
    Fires on snapshot groups: one decision per usable offset of every group that
    went up, worth whatever its own distribution is worth.

    The ``situations`` provider is injected rather than hard-wired — that is what
    keeps the module testable without a CSV and what will let a different situation
    source (a goal catalogue) replace the robustness path without touching the
    emission logic.
    """

    def __init__(self, situations: Callable[[str], list[Situation]], *,
                 initial_score: float, grip_tolerance: float = 0.0,
                 name: str = "snapshot",
                 scoring_params: Optional[scoring.Params] = None,
                 emit_policy: str = ALL_OFFSETS,
                 grip_tolerances: Optional[dict[str, float]] = None,
                 k_factor: Optional[Callable[[str], float]] = None) -> None:
        super().__init__(name)
        self.k_factor = k_factor          # small companies: a milder sample discount (street.size_tilt)
        self.situations = situations
        self.initial_score = float(initial_score)
        self.grip_tolerance = float(grip_tolerance)
        self.scoring_params = scoring_params
        self.emit_policy = emit_policy
        self.grip_tolerances = dict(grip_tolerances or {})

    def _decision(self, symbol: str, situation: Situation, score: float, *,
                  source: str, trace: Optional[dict[str, Any]] = None) -> Decision:
        """One decision, gripped on the situation's Block-A picture."""
        return Decision(
            symbol=symbol,
            side=side_of(situation.direction),
            source=source,
            initial_score=floor_score(score),
            grip=GripCondition(symbol, dict(situation.params),
                               tolerance=self.grip_tolerance,
                               tolerances=dict(self.grip_tolerances),
                               alternatives=[dict(p) for p in situation.member_params]),
            score_trace=dict(trace or {}),
        )

    def emit(self, symbol: str) -> list[Decision]:
        """
        One decision per usable OFFSET of every situation worth buying.

        A group does not have one future, it has a future per distance: what it did
        after three days and what it did after thirty are different propositions with
        different expectations and different speeds, and the only way a short horizon
        can ever win is if it is scored as its own candidate. So each usable offset
        becomes its own decision, carrying the distribution measured there and the
        path up to there, and they compete on score like anything else.

        Three things are refused rather than emitted at a low number: a group that
        went DOWN (this layer buys), a distribution the downside filter rejected, and
        one that scores at or below zero. A decision worth nothing is not a decision
        the analysis has.

        THE SAMPLE SIZE IS THE ONE THING APPLIED AFTER THE SCORE. Each offset's score
        is multiplied by ``n_t / (n_t + k_confidence)`` before the cap, so an offset
        four members reached is worth a fraction of the same distribution seen forty
        times. It is the only place ``n`` acts, it is a discount and not a threshold,
        and the count itself stays visible in the bundle. Nothing else about a thin
        group is treated differently: it is emitted, ranked, and left for the radius
        to refuse on the strength of the number it earned.
        ``scoring.confidence_shrink_enabled`` turns the discount off (shrink = 1.0)
        so its effect can be measured against its own absence; the numbers are
        logged and carried apart on the decision (``score_trace``: ``raw_score``,
        ``z``, ``time_scale``, ``z_scaled``, ``scaled_score``, ``shrink``,
        ``final_score``), never silently multiplied into one.

        BEFORE the discount comes the TIME UNIT. ``scoring.time_scale`` rescales Z
        from "per trading day" to "per that many trading days" — a change of units
        applied to the finished score, so the absolute numbers land on a scale the
        moderator's radius can read. Being one constant across every emission it
        cannot move a ranking or an allocation share; only where the whole field
        sits. Order is ``raw x time_scale x shrink``, and nothing caps the result —
        ``initial_score`` IS ``score_trace["final_score"]``.

        A situation without a bundle AT ALL — a provider that supplies pictures only —
        falls back to the placeholder score, the one remaining use of
        ``street.dummy_initial``. A situation that HAS a bundle which reached no
        computable offset is a different case and emits nothing: the group exists but
        never had two members at the same distance, so there is no distribution to
        score and a placeholder would be a number standing in for no evidence.
        """
        decisions: list[Decision] = []
        for situation in self.situations(symbol):
            bundle = situation.bundle or {}
            if not bundle or self.scoring_params is None:
                decisions.append(self._decision(symbol, situation,
                                                self.initial_score, source=self.name))
                continue
            if situation.direction != robustness_module.UP:
                continue                      # this layer emits buys
            if not (bundle.get("offsets") or []):
                continue     # a bundle that reached no computable offset says nothing
            sp = self.scoring_params
            kf = self.k_factor(symbol) if self.k_factor else 1.0
            if kf != 1.0:
                from dataclasses import replace as _replace
                sp = _replace(sp, k_confidence=sp.k_confidence * kf)
            scored = score_offsets(bundle, params=sp)
            chosen = scored if self.emit_policy == ALL_OFFSETS else scored[-1:]
            for row in chosen:
                if row["rejected"] or row["final_score"] <= 0:
                    continue
                trace = row["score_trace"]
                log.debug("%s @%dd: raw=%.6f z=%.6f x time_scale=%.1f -> %.6f "
                          "shrink=%.4f (n=%d) final=%.6f",
                          symbol, row["offset_t"], trace["raw_score"], trace["z"],
                          trace["time_scale"], trace["scaled_score"],
                          trace["shrink"], row["n_t"], row["final_score"])
                decisions.append(self._decision(
                    symbol, situation, row["final_score"],
                    source=f"{self.name}@{row['offset_t']}d",
                    trace=dict(trace, n_t=row["n_t"])))
        return decisions


def grip_tolerances(config) -> dict[str, float]:
    """The snapshot's own similarity tolerances, per continuous Block-A key."""
    snap = snapshot_module.params_from_config(config)
    return {"ath_gap_pct": snap.sim_tol_ath_gap,
            "volume_ratio_at_x": snap.sim_tol_vol_ratio,
            "days_since_ath": snap.sim_tol_days_ath}


def down_signal(config, symbol: str, picture: Optional[dict[str, Any]], *,
                situations: Optional[Callable[[str], list[Situation]]] = None
                ) -> Optional[dict[str, Any]]:
    """
    Does today's picture belong to a group whose direction is DOWN?

    The mirror of grabbing a buy: today's Block A must match the group (any member,
    same tolerances). No threshold on HOW FAR it fell — a forecast of falling is a
    reason to sell whether the group lost 5 % or 60 %. Only a group seen at least
    twice at some offset counts (one day is no forecast). Returns the clearest such
    group (highest share of days that fell), or None. The ACTIVE sell reason.
    """
    if not picture:
        return None
    tolerances = grip_tolerances(config)
    best: Optional[dict[str, Any]] = None
    for situation in (situations or group_situations(config))(symbol):
        if situation.direction == robustness_module.UP:
            continue
        grip = GripCondition(symbol, dict(situation.params), tolerances=tolerances,
                             alternatives=[dict(p) for p in situation.member_params])
        if not grip.is_met(picture):
            continue
        for record in (situation.bundle or {}).get("offsets") or []:
            if int(record.get("n_t", 0)) < 2:
                continue
            p = float(record.get("p_direction", 0.0))
            if best is None or p > best["p_direction"]:
                best = {"p_direction": p, "offset_t": int(record["offset_t"]),
                        "n_t": int(record["n_t"])}
    return best


def current_pictures(config, symbols: list[str]) -> dict[str, dict[str, Any]]:
    """
    Today's grip picture per symbol: the Block A of each symbol's newest day.

    Block A is past-only, so the newest day's picture is exactly what could be
    known now. A symbol without snapshots gets no picture and its decisions wait.
    """
    params = robustness_module.params_from_config(config)
    store = meta_cache.MetaCache(config)
    pictures: dict[str, dict[str, Any]] = {}
    try:
        for symbol in symbols:
            snaps = snapshot_module.build_snapshots(symbol, params=params.snapshot,
                                                    cache=store)
            if snaps:
                pictures[symbol] = grip_params(snaps[-1].block_a)
    finally:
        store.close()
    return pictures
