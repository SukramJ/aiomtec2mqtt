"""Pytest fixtures and fakes for aiomtec2mqtt tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import types
from typing import Any

import pytest


@dataclass
class FakePahoMessage:
    """A minimal fake of paho.mqtt.client.MQTTMessage for callbacks."""

    topic: str
    payload: bytes


class FakePahoClient:
    """
    A lightweight stand-in for paho.mqtt.client.Client.

    It records published/subscribed topics and supports the minimal surface that
    our wrapper uses. Network operations are no-ops.
    """

    def __init__(
        self,
        client_id: str | None = None,
        protocol: Any | None = None,
        clean_session: bool | None = None,
    ) -> None:  # noqa: D401 - paho-compatible signature
        """Initialize."""
        self._connected = False
        self.on_connect: Callable[..., None] | None = None
        self.on_message: Callable[..., None] | None = None
        self.on_subscribe: Callable[..., None] | None = None
        self.on_disconnect: Callable[..., None] | None = None
        self._published: list[tuple[str, str, int, bool]] = []
        self._subscribed: list[str] = []
        self._unsubscribed: list[str] = []
        self._will: tuple[str, str, bool] | None = None
        self._reconnect_delay: tuple[int, int] | None = None
        self._logger = None

    @property
    def published(self) -> list[tuple[str, str, int, bool]]:
        """Return the published messages."""
        return list(self._published)

    @property
    def subscribed(self) -> list[str]:
        """Return the topics that were subscribed to."""
        return list(self._subscribed)

    @property
    def unsubscribed(self) -> list[str]:
        """Return the topics that were unsubscribed from."""
        return list(self._unsubscribed)

    def connect_async(self, *, host: str, port: int, keepalive: int) -> None:
        """Connect asynchronously."""
        # simulate that connection will succeed later
        self._host, self._port, self._keepalive = host, port, keepalive

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        self._connected = False
        if self.on_disconnect:
            self.on_disconnect(self, None, 0)

    def enable_logger(self, logger: Any) -> None:
        """Set the logger."""
        self._logger = logger

    def loop_start(self) -> None:
        """Start the loop."""
        # Immediately trigger on_connect success to simplify tests
        self._connected = True
        if self.on_connect:
            self.on_connect(self, None, types.SimpleNamespace(), 0)

    def loop_stop(self) -> None:
        """Stop the loop."""
        self._connected = False

    def max_inflight_messages_set(self, value: int) -> None:  # noqa: D401 - compat only
        """Set the maximum number of in-flight messages."""
        self._max_inflight = value

    def max_queued_messages_set(self, value: int) -> None:  # noqa: D401 - compat only
        """Set the maximum number of messages that can be queued."""
        self._max_queued = value

    def publish(self, topic: str, payload: str, qos: int, retain: bool) -> None:
        """Publish a message."""
        self._published.append((topic, str(payload), qos, retain))

    def reconnect_delay_set(self, min_delay: int, max_delay: int) -> None:
        """Set the reconnect delay."""
        self._reconnect_delay = (min_delay, max_delay)

    def subscribe(self, topic: str) -> None:
        """Subscribe to a topic."""
        if topic not in self._subscribed:
            self._subscribed.append(topic)

    def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic."""
        if topic not in self._unsubscribed:
            self._unsubscribed.append(topic)
        if topic in self._subscribed:
            self._subscribed.remove(topic)

    def username_pw_set(self, username: str, password: str) -> None:
        """Set the username and password."""
        self._username = username
        self._password = password

    def will_set(self, topic: str, payload: str, retain: bool) -> None:
        """Set the will."""
        self._will = (topic, payload, retain)


@pytest.fixture
def fake_paho(monkeypatch: pytest.MonkeyPatch) -> FakePahoClient:
    """
    Patch paho.mqtt.client.Client to use our FakePahoClient and return an instance.

    Returns the last created FakePahoClient so tests can make assertions.
    """
    from aiomtec2mqtt import mqtt_client as _mqtt_client

    holder: dict[str, Any] = {}

    def factory(*args: Any, **kwargs: Any) -> FakePahoClient:
        client = FakePahoClient(*args, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(_mqtt_client.mqtt, "Client", factory, raising=True)
    return holder  # type: ignore[return-value]
