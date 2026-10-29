"""
TODAY — the snapshot part of a stock's trade score, read off its newest day.

The engine emits one decision per usable offset of every snapshot group that
went up (``street.firing.SnapshotFiring``), each waiting for its grip condition:
"today would have joined this group". This module asks the same firing for the
same decisions and keeps the ones whose grip is met by today's Block A. Their best
score is the stock's snapshot score; a stock whose newest day resembles no scored
group simply has none today.
"""

from __future__ import annotations

from typing import Any

from .config.loader import Config
from .meta import cache as meta_cache
from .meta import robustness as robustness_module
from .meta import snapshot as snapshot_module
from .street import firing, scoring
from .street.model import GripCondition


def today(config: Config, symbol: str, store: meta_cache.MetaCache) -> dict[str, Any]:
    """Today's picture, the groups it belongs to and the snapshot score it earns."""
    params = robustness_module.params_from_config(config)
    snaps = snapshot_module.build_snapshots(symbol, params=params.snapshot, cache=store)
    if not snaps:
        return {"date": None, "picture": {}, "matches": [], "score": None}
    picture = firing.grip_params(snaps[-1].block_a)
    tolerances = firing.grip_tolerances(config)
    sp = scoring.params_from_config(config)

    by_date = {s.date: s for s in snaps}
    matches = []
    for group in robustness_module.all_groups(symbol, params=params, cache=store):
        # The same grip firing.group_situations builds: the first member's picture,
        # any other member's picture as an alternative.
        pictures = [firing.grip_params(by_date[d].block_a)
                    for d in group.member_dates if d in by_date]
        if not pictures:
            continue
        grip = GripCondition(symbol, pictures[0], tolerances=tolerances,
                             alternatives=pictures[1:])
        if not grip.is_met(picture):
            continue
        rows = firing.score_offsets(group.bundle or {}, params=sp)
        usable = [r for r in rows if not r["rejected"] and r["final_score"] > 0]
        best = max(usable, key=lambda r: r["final_score"], default=None)
        matches.append({
            "group_id": group.group_id,
            "direction": group.dominant_direction,
            "robust": bool(group.robust),
            "members": len(group.member_dates),
            "independent": group.independent_occurrences,
            "first_date": group.member_dates[0],
            # Down groups emit no buy; their score is not part of the stock's score.
            "score": best["final_score"] if best and group.dominant_direction == "up" else None,
            "offsets": [{"offset_t": r["offset_t"], "n_t": r["n_t"], "rejected": r["rejected"],
                         "final_score": r["final_score"],
                         "mu_pct": r["score_trace"]["mu_pct"],
                         "p_direction": r["score_trace"]["p_direction"]} for r in rows],
        })
    scores = [m["score"] for m in matches if m["score"] is not None]
    return {
        "date": snaps[-1].date,
        "picture": picture,
        "matches": matches,
        "score": max(scores) if scores else None,
        "radius_lo": float(config.get("street.default_radius_lo")),
    }
