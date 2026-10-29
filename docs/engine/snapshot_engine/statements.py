"""
STATEMENTS — a pattern, once it is consistent enough, written as one plain sentence.

Two kinds of statement need nothing but prices:

* SNAPSHOT: a group whose futures agreed, e.g. "When Micron trades about 20 % below
  its all-time high on rising volume, the price rose in 83 % of 12 independent
  cases within 21 trading days."
* RECOVERY: after a fall of X % from the all-time high, did the stock get back to
  it, and within how long at most?

Thresholds (``core.*``) decide what is consistent enough; below them nothing is
written. Each snapshot sentence carries the id of the group it describes, so the
page can show it exactly where that group is drawn.
"""

from __future__ import annotations

from datetime import date
from statistics import median
from typing import Any, Optional

from .config.loader import Config
from .meta import snapshot as snapshot_module


def _pct(p: float) -> str:
    return f"{round(p * 100):d} %"


def _days(t: int) -> str:
    return "one trading day" if t == 1 else f"{t} trading days"


def _best_offset(bundle: dict[str, Any], min_n: int) -> Optional[dict[str, Any]]:
    rows = [r for r in bundle.get("offsets") or [] if r.get("n_t", 0) >= min_n]
    return max(rows, key=lambda r: (r["p_direction"], r["n_t"]), default=None)


def snapshot_statements(name: str, groups: list, snaps: list[snapshot_module.Snapshot],
                        config: Config) -> list[dict[str, Any]]:
    """One sentence per group whose outcome agreed often enough, keyed by group id."""
    min_p = float(config.get("core.min_agreement"))
    min_n = int(config.get("core.min_snapshot_cases"))
    block_of = {s.date: s.block_a for s in snaps}
    out = []
    for g in groups:
        rec = _best_offset(g.bundle or {}, min_n)
        if not rec or rec["p_direction"] < min_p:
            continue
        blocks = [block_of[d] for d in g.member_dates if d in block_of]
        if not blocks:
            continue
        gap = median(-float(b.get("ath_gap_pct") or 0.0) for b in blocks)
        where = ("at its all-time high" if gap < 0.02
                 else f"about {round(gap * 100)} % below its all-time high")
        trends = [b.get("volume_trend_at_x") for b in blocks]
        vol = max(set(trends), key=trends.count)
        vol_en = {"rising": " on rising volume", "falling": " on falling volume",
                  "flat": " on steady volume"}.get(vol, "")
        extra = ""
        if all(b.get("in_plateau_now") for b in blocks):
            extra = " inside a sideways phase"
        elif all(b.get("in_breakout_now") for b in blocks):
            extra = " in the middle of a breakout"
        move = "rose" if (g.bundle or {}).get("direction") == "up" else "fell"
        text = (f"When {name} trades {where}{extra}{vol_en}, the price {move} in "
                f"{_pct(rec['p_direction'])} of {rec['n_t']} independent cases within "
                f"{_days(int(rec['offset_t']))}.")
        out.append({"kind": "snapshot", "group_id": g.group_id, "p": rec["p_direction"],
                    "n": rec["n_t"], "text": text})
    return sorted(out, key=lambda s: (-s["p"], -s["n"]))


def drawdown_episodes(dates: list[str], closes: list[float], depth: float) -> list[dict]:
    """Falls of at least ``depth`` from the running high, and when the old high came back."""
    out, high, open_ep = [], closes[0], None
    for i, c in enumerate(closes):
        if open_ep is None:
            if c > high:
                high = c
            elif c <= high * (1 - depth):
                open_ep = {"high": high, "start": dates[i]}
        elif c >= open_ep["high"]:
            open_ep["back"] = dates[i]
            out.append(open_ep)
            open_ep, high = None, c
    if open_ep:
        open_ep["back"] = None
        out.append(open_ep)
    return out


def recovery_statements(name: str, dates: list[str], closes: list[float],
                        config: Config) -> list[dict[str, Any]]:
    """After a fall of X % from the high: back every time, and how long at most."""
    out = []
    min_eps = int(config.get("core.min_drawdowns"))
    min_p = float(config.get("core.min_agreement"))
    for depth in (0.2, 0.3, 0.4, 0.5):
        eps = drawdown_episodes(dates, closes, depth)
        closed = [e for e in eps if e["back"]]
        judged = closed + [e for e in eps if not e["back"]
                           and (date.fromisoformat(dates[-1])
                                - date.fromisoformat(e["start"])).days > 365]
        if len(judged) < min_eps or not closed:
            continue
        share = len(closed) / len(judged)
        if share < min_p:
            continue
        months = max((date.fromisoformat(e["back"]) - date.fromisoformat(e["start"])).days
                     for e in closed) / 30.4
        if months > float(config.get("core.max_recovery_months", 36)):
            continue                       # back eventually, but too slowly to be a trait
        m = max(1, round(months))
        d = f"{round(depth * 100)} %"
        if share == 1.0:
            text = (f"After a fall of {d} from its all-time high, {name} came back every time: "
                    f"in all {len(closed)} cases since {dates[0][:4]} the old high was reached "
                    f"again within {m} months at most.")
        else:
            text = (f"After a fall of {d} from its all-time high, {name} reached the old high "
                    f"again in {len(closed)} of {len(judged)} cases since {dates[0][:4]}, "
                    f"within {m} months at most.")
        out.append({"kind": "recovery", "depth": depth, "p": share, "n": len(judged),
                    "text": text})
    if len(out) > 2:                       # the shallowest and the deepest that still speak
        out = [out[0], out[-1]]
    return out
