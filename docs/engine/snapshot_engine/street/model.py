"""
STREET — the data model: what travels down the street, and what it waits for.

Two objects, and the whole layer is built on them.

A :class:`GripCondition` is the parameter picture a decision waits for. It travels
INSIDE the decision, so a decision is never a bare "buy AMD" that someone has to
re-derive a trigger for: it carries its own trigger with it. ``is_met`` is the
replaceable hook — right now a plain equality/tolerance match against the current
picture; real matching (bands, staleness, partial pictures) lands here later without
anything above it changing.

A :class:`Decision` is what a FIRING analysis emits. It knows where it came from
(``source``), what it started at (``initial_score``), what it is worth right now
(``score``, which starts equal to the initial score) and — once it has walked the
street — every factor that changed it, in order (``factors``). The audit trail is
not decoration: a decision that ranks high must be able to say which station put it
there.

No maths beyond a floor lives here, and no threshold is hard-coded. Scores are
UNBOUNDED above: the ceiling that used to sit at ``street.score_cap`` was removed
because it flattened the top of the ranking into ties, which is where the ranking
matters most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SOURCE = "street.model"

BUY, SELL = "buy", "sell"
SIDES = (BUY, SELL)


def floor_score(value: float) -> float:
    """
    A score, held at or above zero. NO CEILING — and that is the point.

    There used to be one (``street.score_cap`` = 1.0), and once Z was rescaled it
    started doing damage: every strong candidate landed on exactly 1.0, so the top
    of the list — precisely the part the moderator selects from — was a block of
    ties. A ceiling cannot say which of two good trades is better, and that is the
    only question the ranking exists to answer.

    Nothing needs the upper bound. The radius is an INTERVAL, so it reads
    ``lo <= score``; sorting reads ``>``; allocation reads ``S_i / sum(S)``. All
    three work on an unbounded positive number, and each of them works better on
    one. What remains is the floor: a negative score is not a weaker proposal but a
    proposal to lose money, and those are refused at emission rather than ranked.
    """
    return max(0.0, float(value))


def _matches(wanted: Any, actual: Any, tolerance: float) -> bool:
    """
    Does one parameter of the current picture satisfy what the decision waits for?

    Booleans and strings are states and must be equal — there is no "nearly
    break_down". Numbers are compared inside ``tolerance`` (0.0 = exact), because a
    volume ratio of 1.0000001 is the same reading as 1.0. ``bool`` is checked first
    on purpose: it is a subclass of ``int`` and would otherwise be compared as a
    number, making ``True`` match ``1.0``.
    """
    if isinstance(wanted, bool) or isinstance(actual, bool):
        return wanted is actual or wanted == actual
    if isinstance(wanted, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(wanted) - float(actual)) <= tolerance
    return wanted == actual


@dataclass
class GripCondition:
    """
    The exact parameter picture a decision is waiting for.

    ``params`` is a plain dict so any firing analysis can describe its own trigger in
    its own vocabulary; the snapshot firing fills it with the Block-A features it
    already compares days by, but nothing here depends on that choice.
    """

    symbol: str
    params: dict[str, Any]
    tolerance: float = 0.0
    #: Per-key tolerance, overriding ``tolerance`` for the keys it names. The snapshot
    #: firing fills it with the SAME tolerances the meta layer groups days by
    #: (``snapshot.sim_tol_*``), so "the situation is back" means exactly "today would
    #: have joined this group" — one rule for grouping and for grabbing.
    tolerances: dict[str, float] = field(default_factory=dict)
    #: Further pictures that count as "the situation is back" — the other member
    #: days of the group. Groups are chains (A~B, B~C), so a day that resembles ANY
    #: member would have joined the group; matching only the first day would demand
    #: more than the grouping itself ever did.
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    def _picture_met(self, wanted_params: dict[str, Any],
                     current_params: dict[str, Any]) -> bool:
        for key, wanted in wanted_params.items():
            if key not in current_params:
                return False
            tolerance = float(self.tolerances.get(key, self.tolerance))
            if not _matches(wanted, current_params[key], tolerance):
                return False
        return True

    def is_met(self, current_params: dict[str, Any]) -> bool:
        """
        Is the awaited picture the picture right now?

        Met when the current picture matches ``params`` or any of the
        ``alternatives``: every key named must be present and match (states exactly,
        numbers within their tolerance). Keys the current picture carries in addition
        are ignored — a decision waits for what it named, not for the world to hold
        still. An empty grip is met by any picture; a missing key is not met. Bands
        and staleness of a waiting decision are TODO #44.
        """
        return any(self._picture_met(p, current_params)
                   for p in [self.params, *self.alternatives])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GripCondition":
        return cls(d["symbol"], dict(d.get("params") or {}), float(d.get("tolerance", 0.0)),
                   dict(d.get("tolerances") or {}), [dict(p) for p in d.get("alternatives") or []])

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict."""
        return {
            "symbol": self.symbol,
            "params": dict(self.params),
            "tolerance": self.tolerance,
            "tolerances": dict(self.tolerances),
            "alternatives": [dict(p) for p in self.alternatives],
        }


@dataclass
class Decision:
    """One emitted decision, walking the street and collecting factors."""

    symbol: str
    side: str                 # "buy" | "sell"
    source: str               # the firing analysis that emitted it
    initial_score: float      # what the firing analysis thought it was worth, uncapped
    grip: GripCondition
    score: Optional[float] = None            # current score; starts = initial_score
    factors: list[dict[str, Any]] = field(default_factory=list)
    #: How the initial score came about, kept in its PARTS: ``raw_score`` (what the
    #: formula said), ``shrink`` (the sample-size discount) and ``final_score``
    #: (their product, before the cap), plus the ``n_t`` and ``k_confidence`` behind
    #: the discount. Three numbers multiplied into one cannot be calibrated
    #: afterwards, so they travel apart. Empty for a placeholder emission.
    score_trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is None:
            self.score = float(self.initial_score)

    def reset(self) -> None:
        """
        Back to the emitted state — no factors applied, score at the start value.

        ``score_trace`` survives a reset: it describes how the decision was EMITTED,
        not what happened to it on the walk, and re-walking does not change how it
        was born.
        """
        self.score = float(self.initial_score)
        self.factors = []

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "source": self.source,
            "initial_score": self.initial_score,
            "score": self.score,
            "grip": self.grip.to_dict(),
            "factors": [dict(f) for f in self.factors],
            "score_trace": dict(self.score_trace),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Decision":
        """Back from :meth:`to_dict` — a stored decision can walk the street again."""
        return cls(symbol=d["symbol"], side=d["side"], source=d["source"],
                   initial_score=float(d["initial_score"]),
                   grip=GripCondition.from_dict(d["grip"]), score=d.get("score"),
                   factors=[dict(f) for f in d.get("factors") or []],
                   score_trace=dict(d.get("score_trace") or {}))
