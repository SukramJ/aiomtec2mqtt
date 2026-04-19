"""
Assertion helpers that produce readable failure messages for common patterns.

Hand-written ``assert`` statements work, but they tend to dump opaque diffs
when MQTT publish lists or register snapshots get long. The helpers here
encode the intent (``assert_register_published`` rather than
``assert <complex list comprehension>``), so a failing test points at the
real problem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiomtec2mqtt.testing import FakeMqttClient

__all__ = [
    "assert_mqtt_topic_seen",
    "assert_register_published",
]


def assert_mqtt_topic_seen(
    client: FakeMqttClient,
    topic_suffix: str,
    *,
    times: int | None = None,
) -> None:
    """Assert that ``client`` published a message matching ``topic_suffix``.

    If ``times`` is given, assert an exact match count.
    """
    matches = [m for m in client.published_messages if m[0].endswith(topic_suffix)]
    if times is None:
        if not matches:
            seen = sorted({m[0] for m in client.published_messages})
            raise AssertionError(
                f"No publish to a topic ending with {topic_suffix!r}. Topics seen: {seen}"
            )
    elif len(matches) != times:
        raise AssertionError(
            f"Expected {times} publishes to *{topic_suffix}, got {len(matches)}"
        )


def assert_register_published(
    client: FakeMqttClient,
    register_name: str,
    expected_value: object,
    *,
    topic_suffix: str | None = None,
) -> None:
    """Assert that a publish carries ``register_name`` with ``expected_value``.

    Optionally filtered to topics ending in ``topic_suffix``. Performs a
    substring match on the JSON payload; this works for the standard
    ``"name": value`` formatting used by the coordinator and avoids pulling in
    a JSON parser for what is meant to be a quick assertion.
    """
    candidates = client.published_messages
    if topic_suffix is not None:
        candidates = [m for m in candidates if m[0].endswith(topic_suffix)]
    needle = f'"{register_name}"'
    for _topic, payload, *_rest in candidates:
        if needle in payload and str(expected_value) in payload:
            return
    raise AssertionError(
        f"No publish carried {register_name}={expected_value!r}"
        + (f" on topic *{topic_suffix}" if topic_suffix else "")
    )
