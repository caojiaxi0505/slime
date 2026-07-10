"""Default CC reward: binary resolved → {0,1}."""

from __future__ import annotations

from typing import Any


def compose(*, base_eval: dict[str, Any], sample: Any = None, args: Any = None) -> tuple[float, dict[str, Any]]:
    del sample, args
    resolved = bool(base_eval["resolved"])
    return (1.0 if resolved else 0.0), {"mode": "binary", "resolved": resolved}
