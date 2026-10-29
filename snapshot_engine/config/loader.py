"""
Config loader — typed access to ``config/parameters.yaml``.

Parameters come in two shapes in the YAML:

  * TYPED nodes: a mapping with a ``type`` key (``fixed`` or ``variable``) and a
    ``value``; ``variable`` nodes additionally carry a ``range: [min, max]`` the
    Loop may tune within. Example::

        cash_buffer:
          type: variable
          value: 0.30
          range: [0.20, 0.50]

  * PLAIN nodes: ordinary scalars / lists / maps (hosts, ports, symbol maps …).

On load the tree is *resolved* to plain values (typed node -> its ``value``) and
every ``variable`` value is validated to lie within its range; a breach raises
``ConfigError``. The original type/range metadata is kept separately so the Loop
can later ask "what may I tune, and between which bounds?".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigError(ValueError):
    """Raised when the configuration is structurally invalid or out of range."""


def _is_typed_node(node: Any) -> bool:
    return isinstance(node, dict) and "type" in node and "value" in node


def _validate_typed(path: str, node: dict[str, Any]) -> Any:
    node_type = node["type"]
    value = node["value"]
    if node_type not in ("fixed", "variable"):
        raise ConfigError(f"{path}: unknown parameter type {node_type!r} (fixed|variable)")
    if node_type == "variable":
        rng = node.get("range")
        if rng is not None:
            if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
                raise ConfigError(f"{path}: variable range must be [min, max]")
            lo, hi = rng
            # A list value (a grid of offsets, say) is in range when every one of
            # its entries is: the range bounds one element, not the whole list.
            entries = value if isinstance(value, (list, tuple)) else [value]
            for entry in entries:
                if not (lo <= entry <= hi):
                    raise ConfigError(
                        f"{path}: value {entry} outside allowed range [{lo}, {hi}]"
                    )
    return value


def _resolve(node: Any, path: str, meta: dict[str, dict[str, Any]]) -> Any:
    """Recursively resolve typed nodes to plain values, collecting metadata."""
    if _is_typed_node(node):
        meta[path] = {
            "type": node["type"],
            "range": node.get("range"),
            "seal": node.get("seal"),            # how surely the value is measured (#163)
        }
        return _validate_typed(path, node)
    if isinstance(node, dict):
        return {k: _resolve(v, f"{path}.{k}" if path else k, meta) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, f"{path}[{i}]", meta) for i, v in enumerate(node)]
    return node


class Config:
    """Resolved configuration with dotted-path access and range metadata."""

    def __init__(self, data: dict[str, Any], meta: dict[str, dict[str, Any]], source: Path):
        self.data = data
        self.meta = meta
        self.source = source

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        from ..fsretry import read_text
        raw = yaml.safe_load(read_text(p)) or {}            # waits while a sync client holds the file
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: top-level YAML must be a mapping")
        meta: dict[str, dict[str, Any]] = {}
        resolved = _resolve(raw, "", meta)
        return cls(resolved, meta, p)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a resolved value by dotted path, e.g. ``get('broker.port')``."""
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def require(self, dotted: str) -> Any:
        """Like :meth:`get` but raises if the key is missing."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required config key: {dotted}")
        return value

    def fingerprint(self) -> tuple[str, str]:
        """
        ``(hash, json)`` of the WHOLE resolved parameter state.

        The hash names one exact state of ``parameters.yaml``: every analysis and
        every trade stores it, so a later review can tell which parameters produced
        what, and compare two states value by value. Sorted keys, so the same values
        always give the same hash regardless of file order or comments.
        """
        import hashlib
        import json

        blob = json.dumps(self.data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], blob

    def range_of(self, dotted: str) -> Optional[list]:
        """Return the ``[min, max]`` range for a variable parameter, if any."""
        info = self.meta.get(dotted)
        return info.get("range") if info else None
