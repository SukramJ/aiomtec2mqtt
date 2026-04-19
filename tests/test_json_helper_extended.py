"""Edge-case tests for :mod:`aiomtec2mqtt._json`.

Complements :mod:`tests.test_json_helper` with input-shape coverage
(unicode, datetime/date, floats, deep nesting) and sort-keys behaviour
on both the orjson and stdlib paths.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
import math

import pytest

from aiomtec2mqtt import _json

# ---------------------------------------------------------------------------
# Round trips on typical payload shapes
# ---------------------------------------------------------------------------


class TestRoundTripShapes:
    """Varied shapes that show up in MQTT topics."""

    def test_boolean_and_null(self) -> None:
        payload = {"on": True, "off": False, "none": None}
        assert _json.loads(_json.dumps(payload)) == payload

    def test_deeply_nested(self) -> None:
        payload: dict = {"l1": {"l2": {"l3": {"l4": [1, 2, 3]}}}}
        assert _json.loads(_json.dumps(payload)) == payload

    def test_empty_dict(self) -> None:
        assert _json.loads(_json.dumps({})) == {}

    def test_empty_list(self) -> None:
        assert _json.loads(_json.dumps([])) == []

    def test_mixed_numeric_types(self) -> None:
        payload = {"i": 1, "f": 1.5, "neg": -3, "big": 2**31}
        assert _json.loads(_json.dumps(payload)) == payload

    def test_unicode_strings(self) -> None:
        payload = {"name": "Energiebütler", "city": "Köln", "emoji": "⚡"}
        assert _json.loads(_json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# Sort keys
# ---------------------------------------------------------------------------


class TestSortKeys:
    """``sort_keys`` works on both code paths."""

    def test_sort_keys_alphabetical(self) -> None:
        result = _json.dumps({"z": 1, "a": 2, "m": 3}, sort_keys=True)
        a_pos = result.index('"a"')
        m_pos = result.index('"m"')
        z_pos = result.index('"z"')
        assert a_pos < m_pos < z_pos

    def test_sort_keys_does_not_affect_values(self) -> None:
        sorted_payload = _json.dumps({"a": [3, 2, 1]}, sort_keys=True)
        # Values inside arrays are NOT reordered.
        assert sorted_payload == '{"a":[3,2,1]}' or sorted_payload == '{"a": [3, 2, 1]}'

    def test_sort_keys_stable_for_already_sorted(self) -> None:
        first = _json.dumps({"a": 1, "b": 2}, sort_keys=True)
        second = _json.dumps({"b": 2, "a": 1}, sort_keys=True)
        assert first == second


# ---------------------------------------------------------------------------
# loads input flexibility
# ---------------------------------------------------------------------------


class TestLoadsInputs:
    """``loads`` accepts both bytes and str."""

    def test_loads_array(self) -> None:
        assert _json.loads("[1, 2, 3]") == [1, 2, 3]

    def test_loads_empty_string_raises(self) -> None:
        with pytest.raises(Exception):  # orjson/stdlib raise different types
            _json.loads("")

    def test_loads_scalar_int(self) -> None:
        assert _json.loads("42") == 42

    def test_loads_scalar_string(self) -> None:
        assert _json.loads('"hello"') == "hello"

    def test_loads_unicode_bytes(self) -> None:
        raw = '{"city": "Köln"}'.encode()
        assert _json.loads(raw) == {"city": "Köln"}


# ---------------------------------------------------------------------------
# Stdlib fallback path
# ---------------------------------------------------------------------------


class TestStdlibFallback:
    """All behaviour holds when orjson is unavailable."""

    def test_fallback_date_via_default_str(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_json, "_orjson", None)
        result = _json.dumps({"d": date(2026, 4, 17)})
        assert "2026-04-17" in result

    def test_fallback_datetime_via_default_str(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_json, "_orjson", None)
        ts = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)
        result = _json.dumps({"ts": ts})
        assert "2026-04-17" in result

    def test_fallback_decimal_via_default_str(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_json, "_orjson", None)
        result = _json.dumps({"amount": Decimal("99.99")})
        assert '"99.99"' in result

    def test_fallback_loads_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_json, "_orjson", None)
        assert _json.loads(b'{"k": 1}') == {"k": 1}

    def test_fallback_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_json, "_orjson", None)
        payload = {"a": [1, 2, 3], "b": "x"}
        assert _json.loads(_json.dumps(payload)) == payload

    def test_fallback_sort_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_json, "_orjson", None)
        result = _json.dumps({"z": 1, "a": 2}, sort_keys=True)
        assert result.index('"a"') < result.index('"z"')


# ---------------------------------------------------------------------------
# Float edge cases
# ---------------------------------------------------------------------------


class TestFloats:
    """Floats are the most common payload type — pin precision behaviour."""

    def test_large_float(self) -> None:
        payload = {"x": 1e9}
        assert _json.loads(_json.dumps(payload))["x"] == pytest.approx(1e9)

    def test_nan_round_trip_via_stdlib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # orjson rejects NaN by default; stdlib accepts it. Pin stdlib behaviour
        # since the fallback is what callers see when [fast] extra is missing.
        monkeypatch.setattr(_json, "_orjson", None)
        result = _json.dumps({"x": math.nan})
        assert "NaN" in result

    def test_negative_float(self) -> None:
        payload = {"x": -2.5}
        assert _json.loads(_json.dumps(payload)) == payload

    def test_round_to_three_decimals(self) -> None:
        payload = {"x": 1.234}
        assert _json.loads(_json.dumps(payload))["x"] == pytest.approx(1.234)

    def test_zero(self) -> None:
        payload = {"x": 0, "y": 0.0}
        decoded = _json.loads(_json.dumps(payload))
        assert decoded == payload
