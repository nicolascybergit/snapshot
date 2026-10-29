"""
META — ATH-META: what all-time highs have historically had to do with breakouts,
volume and plateaus.

Above the all-time high nobody is holding at a loss. There is no overhang of
people waiting to get out at break-even, so a move into blue sky meets less
resistance than the same move under an old high — and that is supposed to show
up as bigger, cleaner moves. A new high made on heavy volume means many hands
took part and are now in profit above it; a new high made on thin volume means
almost nobody did, and the level tends not to hold. Heavy participation at a move
also makes a following plateau more likely: a lot of holders, few sellers.

This module does not assume any of that. It MEASURES it over the whole history:

  ath_in_breakout_rate  — how often a new high was actually made BY a breakout
  blue_sky_strength     — breakouts that cross the old high vs those that stay
                          under it: size, cleanliness, and how often a plateau
                          follows (the gain being held)
  ath_volume_split      — new highs bucketed by the volume at the event, and what
                          happened next: held, plateaued, or fell back through
  post_ath_outcome      — the outcome distribution after every new high
  approach_before_break — how many times price came up to the ceiling before it
                          finally went through

FULL HISTORY, forward information allowed. This is a historical statistic, not a
day's view of itself: it uses the STANDARD detections, not the strictly past-only
ones the snapshot needs.

STANDALONE. It neither reads from nor writes to the snapshot meta, and the
snapshot knows nothing about it. The two are combined only much later, by the
Moderator. Every primitive value comes through :mod:`snapshot_engine.meta.cache`; nothing
here recomputes a primitive. Tiers stay apart — a small breakout and a large one
are different statements and are never averaged into one number.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Optional

from . import cache as meta_cache

SOURCE = "meta.ath_meta"

HIGH, LOW, MID = "high", "low", "mid"
CONTINUED, HELD, FELL_BACK = "continued_up", "held", "fell_back"


@dataclass(frozen=True)
class Params:
    """The tunables this meta needs."""

    near_days: int
    follow_days: int
    vol_high: float
    vol_low: float
    approach_pct: float


def params_from_config(config) -> Params:
    """The ONE place the ATH-meta tunables are read."""
    return Params(
        near_days=int(config.get("ath_meta.near_days")),            # tunable
        follow_days=int(config.get("ath_meta.follow_days")),        # tunable
        vol_high=float(config.get("ath_meta.vol_high")),            # tunable
        vol_low=float(config.get("ath_meta.vol_low")),              # tunable
        approach_pct=float(config.get("ath_meta.approach_pct")),    # tunable
    )


@dataclass
class AthMeta:
    """Everything this meta measured for one symbol."""

    symbol: str
    ath_count: int
    ath_in_breakout_rate: dict[str, Any]
    blue_sky_strength: dict[str, Any]
    ath_volume_split: dict[str, Any]
    post_ath_outcome: dict[str, Any]
    approach_before_break: dict[str, Any]
    events: list[dict[str, Any]]        # per new high, for the chart

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (dates already ISO strings)."""
        return {
            "symbol": self.symbol,
            "ath_count": self.ath_count,
            "ath_in_breakout_rate": self.ath_in_breakout_rate,
            "blue_sky_strength": self.blue_sky_strength,
            "ath_volume_split": self.ath_volume_split,
            "post_ath_outcome": self.post_ath_outcome,
            "approach_before_break": self.approach_before_break,
            "events": list(self.events),
        }


def _median(values: list[float]) -> Optional[float]:
    return float(median(values)) if values else None


def _share(part: int, whole: int) -> Optional[float]:
    return part / whole if whole else None


def _bucket(ratio: Optional[float], params: Params) -> str:
    """Heavy, thin, or neither — the volume at the moment of the high."""
    if ratio is None:
        return MID
    if ratio >= params.vol_high:
        return HIGH
    if ratio <= params.vol_low:
        return LOW
    return MID


def analyse(symbol: str, *, params: Params,
            cache: Optional[meta_cache.MetaCache] = None,
            config=None) -> Optional[AthMeta]:
    """
    Measure the historical relationships around this symbol's all-time highs.

    Everything comes from the cache: the new-high events, the standard breakout
    and plateau detections of both size classes, the running high and the daily
    volume ratio. ``None`` when the symbol has no data.
    """
    own = cache is None
    store = cache or meta_cache.MetaCache(config)
    try:
        series = store.closes(symbol)
        if series is None or len(series) == 0:
            return None
        closes = [float(v) for v in series.to_numpy()]
        dates = [d.date().isoformat() for d in series.index]
        at = {day: i for i, day in enumerate(dates)}
        total = len(closes)

        events = meta_cache.ath_events(store, symbol)
        breakouts = meta_cache.breakouts(store, symbol)
        plateaus = meta_cache.plateaus(store, symbol)
        rolling = meta_cache.rolling_ath(store, symbol)
        volume_window = int(store.config.get("volume.baseline_window"))       # tunable
        volume = meta_cache.trailing_volume(store, symbol, window=volume_window)

        ups = [b for b in breakouts if b["direction"] == "up"]
        tiers = sorted({b["tier"] for b in breakouts})
        plateau_starts = sorted((at[p["start_date"]], p["tier"])
                                for p in plateaus if p["start_date"] in at)

        def plateau_within(start: int, days: int) -> Optional[str]:
            """The tier of the first plateau starting inside the window, if any."""
            for index, tier in plateau_starts:
                if start < index <= start + days:
                    return tier
            return None

        def covers(breakout: dict[str, Any], index: int) -> bool:
            first = at.get(breakout["start_date"])
            last = at.get(breakout["end_date"])
            if first is None or last is None:
                return False
            return first - params.near_days <= index <= last + params.near_days

        # ---------------------------------------------------------- events
        rows: list[dict[str, Any]] = []
        for event in events:
            index = at.get(event["date"])
            if index is None:
                continue
            in_tiers = {tier: any(covers(b, index) for b in ups if b["tier"] == tier)
                        for tier in tiers}
            ratio = volume[index]["ratio"] if index < len(volume) else None
            end = min(total - 1, index + params.follow_days)
            window = closes[index + 1:end + 1]
            price = float(event["price"])

            # "continued" means the market went on to make ANOTHER new high inside
            # the window — not merely that some day closed a cent above this one
            higher = any(index < at.get(e["date"], -1) <= end for e in events)
            if not window:
                outcome = HELD
            elif closes[end] < price:
                outcome = FELL_BACK
            elif higher:
                outcome = CONTINUED
            else:
                outcome = HELD
            fell_through = bool(window) and min(window) < price

            rows.append({
                "date": event["date"],
                "price": price,
                "step_pct": event["step_pct"],
                "in_breakout": any(in_tiers.values()),
                "in_breakout_by_tier": in_tiers,
                "volume_ratio": ratio,
                "volume_bucket": _bucket(ratio, params),
                "outcome": outcome,
                "fell_through": fell_through,
                "plateau_after": plateau_within(index, params.follow_days),
            })

        # ------------------------------------------- a high made by a breakout
        in_breakout_rate = {
            "any": _share(sum(1 for r in rows if r["in_breakout"]), len(rows)),
            "by_tier": {tier: _share(sum(1 for r in rows
                                         if r["in_breakout_by_tier"].get(tier)), len(rows))
                        for tier in tiers},
            "n": len(rows),
        }

        # ------------------------------------------------------- blue sky
        blue_sky: dict[str, Any] = {}
        for tier in tiers:
            crossing: list[dict[str, Any]] = []
            below: list[dict[str, Any]] = []
            for b in (x for x in ups if x["tier"] == tier):
                start = at.get(b["start_date"])
                end = at.get(b["end_date"])
                if start is None or end is None:
                    continue
                ceiling = rolling[start] if start < len(rolling) else None
                target = crossing if (ceiling is not None
                                      and b["price_to"] > ceiling) else below
                target.append(b)

            def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
                held = sum(1 for b in group
                           if plateau_within(at[b["end_date"]], params.follow_days))
                return {
                    "n": len(group),
                    "median_delta_pct": _median([abs(b["delta_pct"]) for b in group]),
                    "median_move_r2": _median([b["move_r2"] for b in group]),
                    "plateau_after_rate": _share(held, len(group)),
                }

            blue_sky[tier] = {"crossing_ath": describe(crossing),
                              "below_ath": describe(below)}

        # ------------------------------------------------- volume at the high
        volume_split: dict[str, Any] = {}
        for name in (HIGH, MID, LOW):
            bucket = [r for r in rows if r["volume_bucket"] == name]
            volume_split[name] = {
                "n": len(bucket),
                "held_rate": _share(sum(1 for r in bucket
                                        if r["outcome"] != FELL_BACK), len(bucket)),
                "fell_back_rate": _share(sum(1 for r in bucket
                                             if r["outcome"] == FELL_BACK), len(bucket)),
                "fell_through_rate": _share(sum(1 for r in bucket
                                                if r["fell_through"]), len(bucket)),
                "plateau_after_rate": _share(sum(1 for r in bucket
                                                 if r["plateau_after"]), len(bucket)),
            }

        outcomes = {name: sum(1 for r in rows if r["outcome"] == name)
                    for name in (CONTINUED, HELD, FELL_BACK)}
        post_outcome = {
            "counts": outcomes,
            "shares": {name: _share(count, len(rows)) for name, count in outcomes.items()},
            "follow_days": params.follow_days,
        }

        # ------------------------------------------- knocking on the ceiling
        approaches: dict[str, Any] = {}
        for tier in tiers:
            counts: list[int] = []
            for b in (x for x in ups if x["tier"] == tier):
                start = at.get(b["start_date"])
                if start is None or start >= len(rolling):
                    continue
                ceiling = rolling[start]
                if ceiling is None or ceiling <= 0 or b["price_to"] <= ceiling:
                    continue                       # only moves that went through
                # back to the day the ceiling was set: how often did price knock?
                since = start
                while since > 0 and rolling[since - 1] == ceiling:
                    since -= 1
                episodes, inside = 0, False
                for i in range(since, start):
                    near = (ceiling - closes[i]) / ceiling <= params.approach_pct
                    if near and not inside:
                        episodes += 1
                    inside = near
                counts.append(episodes)
            approaches[tier] = {"n": len(counts), "median_approaches": _median(counts)}

        return AthMeta(
            symbol=symbol,
            ath_count=len(rows),
            ath_in_breakout_rate=in_breakout_rate,
            blue_sky_strength=blue_sky,
            ath_volume_split=volume_split,
            post_ath_outcome=post_outcome,
            approach_before_break=approaches,
            events=rows,
        )
    finally:
        if own:
            store.close()
