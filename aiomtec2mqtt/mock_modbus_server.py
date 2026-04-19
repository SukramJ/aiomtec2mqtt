"""
In-process Modbus transport fake for integration tests.

The production :class:`~aiomtec2mqtt.async_modbus_client.AsyncModbusClient`
wraps :class:`pymodbus.client.AsyncModbusTcpClient`. Tests that want to drive
the *real* ``AsyncModbusClient`` code paths (register clustering, retries,
circuit breaker) need a fake that replaces the low-level pymodbus transport.

:class:`MockModbusTransport` stands in for ``AsyncModbusTcpClient``. It
implements exactly the surface that ``AsyncModbusClient`` calls and nothing
else, so it stays light-weight. Behaviour:

- Keeps a dict ``address -> list[int]`` of register values.
- ``read_holding_registers(address, count, slave)`` returns a response object
  with ``registers`` and ``isError()`` — the same shape pymodbus returns.
- Scriptable failure modes: ``fail_connect``, ``fail_read_n_times``,
  ``fail_writes``.
- Records every call into ``calls`` so tests can assert traffic patterns.

Use with ``unittest.mock.patch`` to replace the pymodbus transport for the
duration of a test:

.. code-block:: python

    from aiomtec2mqtt.mock_modbus_server import MockModbusTransport

    transport = MockModbusTransport(register_values={10100: [0x1234, 0x5678]})
    with patch("aiomtec2mqtt.async_modbus_client.AsyncModbusTcpClient", lambda **kw: transport):
        client = AsyncModbusClient(config=..., register_map=..., register_groups=[...])
        async with client.connection():
            ...

(c) 2026 by SukramJ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["MockModbusResponse", "MockModbusTransport"]


@dataclass(slots=True)
class MockModbusResponse:
    """Minimal stand-in for ``ReadHoldingRegistersResponse``."""

    registers: list[int]
    _is_error: bool = False

    def isError(self) -> bool:  # noqa: N802 — pymodbus API
        """Return ``True`` if this response represents a Modbus error."""
        return self._is_error


@dataclass(slots=True)
class _Call:
    kind: str  # "connect" | "close" | "read" | "write"
    address: int | None = None
    count: int | None = None
    values: list[int] | None = None
    slave: int | None = None


@dataclass
class MockModbusTransport:
    """
    In-process replacement for ``pymodbus.client.AsyncModbusTcpClient``.

    Attributes:
        register_values: ``{address: [word, word, ...]}``. Any address not in
            the dict returns ``[0] * count`` (acts like a freshly booted
            inverter with zero values).
        fail_connect: If ``True``, :meth:`connect` returns ``False``.
        fail_read_n_times: Next ``N`` reads return an error response, then
            succeed. Counter decrements on every read attempt.
        fail_writes: If ``True``, :meth:`write_registers` returns an error
            response.
        default_register_count: Fallback count for unknown addresses.
        calls: Ordered history of transport calls, useful for assertions.
    """

    register_values: dict[int, list[int]] = field(default_factory=dict)
    fail_connect: bool = False
    fail_read_n_times: int = 0
    fail_writes: bool = False
    default_register_count: int = 1
    calls: list[_Call] = field(default_factory=list)
    _connected: bool = False

    # --- pymodbus surface -------------------------------------------------

    @property
    def connected(self) -> bool:
        """Return the current connection state."""
        return self._connected

    def close(self) -> None:
        """Simulate TCP close. Safe to call repeatedly."""
        self.calls.append(_Call(kind="close"))
        self._connected = False

    async def connect(self) -> bool:
        """Simulate TCP connect. Returns ``False`` when ``fail_connect`` is set."""
        self.calls.append(_Call(kind="connect"))
        if self.fail_connect:
            self._connected = False
            return False
        self._connected = True
        return True

    def read_calls(self) -> list[_Call]:
        """Return only the read calls recorded so far."""
        return [c for c in self.calls if c.kind == "read"]

    async def read_holding_registers(  # kwonly: disable
        self,
        address: int,
        count: int = 1,
        slave: int = 0,
        **_: Any,
    ) -> MockModbusResponse:
        """Simulate a Modbus read of ``count`` words starting at ``address``."""
        self.calls.append(_Call(kind="read", address=address, count=count, slave=slave))
        if self.fail_read_n_times > 0:
            self.fail_read_n_times -= 1
            return MockModbusResponse(registers=[], _is_error=True)

        if (values := self.register_values.get(address)) is None:
            values = [0] * count
        # Pad with zeros if the caller asked for more words than we scripted.
        elif len(values) < count:
            values = [*values, *([0] * (count - len(values)))]
        else:
            values = values[:count]
        return MockModbusResponse(registers=list(values))

    def reset(self) -> None:
        """Clear call history and disconnect."""
        self.calls.clear()
        self._connected = False

    def write_calls(self) -> list[_Call]:
        """Return only the write calls recorded so far."""
        return [c for c in self.calls if c.kind == "write"]

    async def write_registers(  # kwonly: disable
        self,
        address: int,
        values: list[int],
        slave: int = 0,
        **_: Any,
    ) -> MockModbusResponse:
        """Simulate a Modbus write and persist ``values`` into the register map."""
        self.calls.append(_Call(kind="write", address=address, values=list(values), slave=slave))
        if self.fail_writes:
            return MockModbusResponse(registers=[], _is_error=True)
        self.register_values[address] = list(values)
        return MockModbusResponse(registers=list(values))
