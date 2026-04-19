"""
JSON helper with optional orjson acceleration.

If the optional ``orjson`` package is installed (``pip install
aiomtec2mqtt[fast]``), its C-accelerated implementation is used. Otherwise
the stdlib :mod:`json` module is used transparently.

(c) 2026 by SukramJ
"""

from __future__ import annotations

import json as _stdlib_json
from typing import Any

__all__ = ["dumps", "loads"]

try:  # pragma: no cover - fast-path only available with optional extra
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None


def dumps(obj: Any, *, sort_keys: bool = False) -> str:  # kwonly: disable
    """Serialize ``obj`` to a JSON string (orjson if available).

    Mirrors the stdlib :func:`json.dumps` signature (positional ``obj``)
    so call sites can swap in this wrapper without churn.
    """
    if _orjson is not None:
        option = _orjson.OPT_SORT_KEYS if sort_keys else 0
        encoded: bytes = _orjson.dumps(obj, option=option)
        return encoded.decode("utf-8")
    return _stdlib_json.dumps(obj, sort_keys=sort_keys, default=str)


def loads(data: str | bytes) -> Any:  # kwonly: disable
    """Deserialize ``data`` (orjson if available).

    Mirrors the stdlib :func:`json.loads` signature (positional ``data``).
    """
    if _orjson is not None:
        return _orjson.loads(data)
    return _stdlib_json.loads(data)
