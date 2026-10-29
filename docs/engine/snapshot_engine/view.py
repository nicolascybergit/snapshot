"""
VIEW — every analysis of one symbol, projected into the chart schema.

The page draws, this module asks
the analyses and passes their results on as pure geometry (dates, prices, spans,
directions). No figure is derived here and none is derived on the page.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from .applied_data import ath, breakout, plateau, volume
from .config.loader import Config
from .meta import ath_meta as ath_meta_module
from .meta import cache as meta_cache
from .meta import robustness as robustness_module
from .meta import snapshot as snapshot_module
from .meta import trend_regime as trend_regime_module
from .raw_data import long_term_csv


def _series(config: Config, symbol: str):
    csv_dir, csv_map = config.get("data.csv_dir"), config.get("csv_map", {})
    return (long_term_csv.load(symbol, csv_dir, csv_map),
            long_term_csv.load_volume(symbol, csv_dir, csv_map))


def price_view(config: Config, symbol: str) -> dict[str, Any]:
    """
    Close series + every detected plateau and breakout, as pure geometry.

    The size class is deliberately NOT passed on: small and large are drawn
    identically. Breakouts carry ``direction`` — green for up, red for down.
    """
    series, volumes = _series(config, symbol)
    if series is None or series.empty:
        return {"prices": [], "plateaus": [], "breakouts": [], "volume": []}

    volume_settings = volume.settings_from_config(config)
    plateaus = plateau.detect_tiers(
        symbol, series, plateau.tiers_from_config(config),
        volumes=volumes, volume_settings=volume_settings,
    )
    breakouts = breakout.detect_tiers(
        symbol, series,
        tiers=breakout.breakout_tiers_from_config(config),
        plateaus=plateaus,
        volumes=volumes, volume_settings=volume_settings,
    )
    return {
        "prices": [{"date": idx.date().isoformat(), "close": float(value)}
                   for idx, value in series.items()],
        "plateaus": [{"start_date": p.start_date, "end_date": p.end_date,
                      "band_low": p.band_low, "band_high": p.band_high} for p in plateaus],
        "breakouts": [{"start_date": b.start_date, "end_date": b.end_date,
                       "direction": b.direction} for b in breakouts],
        # A missing Volume cell becomes 0.0 so the bars keep their 1:1 alignment.
        "volume": [{"date": idx.date().isoformat(),
                    "volume": float(value) if value == value else 0.0}
                   for idx, value in (volumes.items() if volumes is not None else [])],
    }


def ath_view(config: Config, symbol: str) -> dict[str, Any]:
    """The rolling all-time high (never the global maximum) and the days a new one was set."""
    series, _ = _series(config, symbol)
    if series is None or series.empty:
        return {"rolling_ath": [], "events": []}
    running = ath.rolling_ath(series)
    return {
        "rolling_ath": [{"date": idx.date().isoformat(), "value": value}
                        for idx, value in zip(series.index, running)],
        "events": [{"date": e.date, "price": e.price}
                   for e in ath.ath_events(series, symbol=symbol)],
    }


def snapshot_view(config: Config, symbol: str, store: meta_cache.MetaCache) -> dict[str, Any]:
    """
    Every snapshot group, each member day as a ring carrying its group's verdict.

    The ring's POSITION says which day, its COLOUR which group, its WEIGHT whether
    the group passed all four robustness gates.
    """
    series = store.closes(symbol)
    if series is None or series.empty:
        return {"rings": [], "robust_groups": 0, "groups": 0}
    close_of = {idx.date().isoformat(): float(value) for idx, value in series.items()}
    params = robustness_module.params_from_config(config)
    groups = robustness_module.all_groups(symbol, params=params, cache=store)
    snaps = {s.date: s for s in snapshot_module.build_snapshots(
        symbol, params=params.snapshot, cache=store)}
    horizon_of = {d: int(s.block_b.get("forward", {}).get("horizon_days") or 0)
                  for d, s in snaps.items()}
    # a group is only drawn when its Block Bs agree: enough of them went the dominant
    # way and the median move points that way too, so the page never says "went up
    # by -4 %"
    min_agreement = float(config.get("core.min_agreement"))

    def agrees(group) -> bool:
        returns = [robustness_module._forward(snaps[d])
                   for d in (group.independent_dates or group.member_dates) if d in snaps]
        p50 = (group.distribution or {}).get("return_p50")
        if not returns or p50 is None:
            return False
        up = group.dominant_direction == "up"
        share = sum(1 for r in returns if (r > 0 if up else r < 0)) / len(returns)
        return share >= min_agreement and (p50 > 0 if up else p50 < 0)

    shown = [g for g in groups if agrees(g)]
    rings = [
        {"date": day, "price": close_of[day], "group_id": group.group_id,
         "direction": group.dominant_direction, "robust": bool(group.robust)}
        for group in shown for day in group.member_dates if day in close_of
    ]
    rings.sort(key=lambda r: r["date"])
    # how alike each group's futures were, and the median move over its median horizon
    outcome = {}
    for group in shown:
        days = group.independent_dates or group.member_dates
        spans = [horizon_of[d] for d in days if horizon_of.get(d)]
        outcome[str(group.group_id)] = {
            "uniformity": group.uniformity_narrow,
            "return_p50": (group.distribution or {}).get("return_p50"),
            "horizon_days": median(spans) if spans else None,
        }
    return {"rings": rings, "robust_groups": sum(1 for g in groups if g.robust),
            "groups": len(groups), "shown_groups": len(shown), "outcome": outcome,
            "min_uniformity": params.min_uniformity}


def ath_meta_view(config: Config, symbol: str, store: meta_cache.MetaCache) -> dict[str, Any]:
    """One mark per new high: blue sky or fell back, heavy or thin volume."""
    found = ath_meta_module.analyse(
        symbol, params=ath_meta_module.params_from_config(config), cache=store)
    if found is None:
        return {"events": []}
    return {"events": [
        {"date": row["date"], "price": row["price"],
         "blue_sky": bool(row["in_breakout"] and row["outcome"] != "fell_back"),
         "volume": row["volume_bucket"], "in_breakout": bool(row["in_breakout"])}
        for row in found.events
    ]}


def trend_meta_view(config: Config, symbol: str, store: meta_cache.MetaCache) -> dict[str, Any]:
    """Bull / bear / sideways stretches over the whole history (descriptive, forward-informed)."""
    series = store.closes(symbol)
    if series is None or series.empty:
        return {"regimes": []}
    found = trend_regime_module.structure(
        symbol, params=trend_regime_module.params_from_config(config), cache=store)
    return {"regimes": [{"start_date": seg["start_date"], "end_date": seg["end_date"],
                         "regime": seg["regime"]} for seg in found.get("segments", [])]}
