"""Pytest fixtures and fakes for aiomtec2mqtt tests."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from unittest.mock import patch

import pytest


@dataclass
class FakePahoMessage:
    """A minimal fake of paho.mqtt.client.MQTTMessage for callbacks."""

    topic: str
    payload: bytes


@pytest.fixture(autouse=True)
def _stop_all_patches() -> Generator[None, None, None]:
    """Ensure ``unittest.mock.patch`` state is cleaned up after each test.

    Tests occasionally forget to stop manually started patches (``patch(...).start()``).
    Leftover patches leak into the next test, causing flaky failures that are hard to
    diagnose. This autouse fixture calls ``patch.stopall()`` on teardown so every test
    starts from a clean slate.
    """
    yield
    patch.stopall()
