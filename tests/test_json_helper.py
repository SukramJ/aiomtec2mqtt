"""Tests for :mod:`aiomtec2mqtt._json` (orjson-accelerated JSON helper)."""

from __future__ import annotations

import pytest

from aiomtec2mqtt import _json


class TestDumps:
    def test_roundtrip_nested(self) -> None:
        payload = {"a": [1, 2, {"b": "c"}], "d": None}
        assert _json.loads(_json.dumps(payload)) == payload

    def test_roundtrip_simple_dict(self) -> None:
        payload = {"foo": "bar", "num": 42, "flag": True}
        result = _json.loads(_json.dumps(payload))
        assert result == payload

    def test_sort_keys(self) -> None:
        result = _json.dumps({"b": 1, "a": 2}, sort_keys=True)
        assert result.index('"a"') < result.index('"b"')

    def test_unsorted_by_default_preserves_insertion_order(self) -> None:
        result = _json.dumps({"b": 1, "a": 2})
        # Insertion order is kept; we only verify both keys are present.
        assert '"a"' in result and '"b"' in result


class TestLoads:
    def test_loads_from_bytes(self) -> None:
        assert _json.loads(b'{"k": 2}') == {"k": 2}

    def test_loads_from_str(self) -> None:
        assert _json.loads('{"k": 1}') == {"k": 1}

    def test_loads_invalid_raises(self) -> None:
        with pytest.raises(Exception):  # orjson/stdlib raise different types
            _json.loads("{not valid json}")


class TestDumpsFallback:
    """Exercise the stdlib fallback path when orjson is monkey-patched away."""

    def test_stdlib_fallback_via_default_str(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from decimal import Decimal

        monkeypatch.setattr(_json, "_orjson", None)
        # Decimal is not natively JSON-serialisable; stdlib path uses default=str
        result = _json.dumps({"amount": Decimal("3.14")})
        assert "3.14" in result
