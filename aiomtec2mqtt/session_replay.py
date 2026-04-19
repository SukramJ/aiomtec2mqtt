"""
Replay framework for captured Modbus sessions.

A *session* is a chronologically ordered list of *frames*. Each frame is a
snapshot of the register values seen during one polling cycle of the live
device (``{address: [word, ...]}``). Sessions are persisted as a single ZIP
archive containing one ``manifest.json`` entry, which keeps recordings small
and self-contained.

Two main components live here:

- :class:`SessionRecorder` — wraps a real pymodbus client and captures every
  successful ``read_holding_registers`` into the current frame. Call
  :meth:`SessionRecorder.commit_frame` at the end of each cycle to close the
  frame, and :meth:`SessionRecorder.save` to write the ZIP. Recording is
  opt-in and intended for short capture runs against a real inverter, not for
  production use.

- :class:`SessionPlayer` — pymodbus-compatible transport stand-in that hands
  out frames in order. Plug it into ``AsyncModbusClient`` the same way as
  :class:`~aiomtec2mqtt.mock_modbus_server.MockModbusTransport`:

  .. code-block:: python

      player = SessionPlayer.from_zip(Path("session.zip"))
      with patch(
          "aiomtec2mqtt.async_modbus_client.AsyncModbusTcpClient",
          lambda **kw: player,
      ):
          ...

  The player advances frames automatically when it has read every address in
  the current frame at least once (so one full coordinator cycle = one
  frame). Looping is configurable.

The on-disk format is intentionally tiny:

.. code-block:: json

    {
      "version": 1,
      "metadata": {"recorded_at": "...", "device_serial": "...", ...},
      "frames": [
        {"ts": 0.0, "registers": {"10100": [4660, 22136], ...}},
        {"ts": 5.1, "registers": {...}}
      ]
    }

Addresses are stored as JSON object keys (strings) because JSON has no
integer keys; they round-trip back to ``int`` on load.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import io
import json
from pathlib import Path
import time
from typing import Any, Final, Self
import zipfile

from aiomtec2mqtt.mock_modbus_server import MockModbusResponse

__all__ = ["Frame", "SessionMetadata", "SessionPlayer", "SessionRecorder"]

_MANIFEST_NAME: Final = "manifest.json"
_FORMAT_VERSION: Final = 1


@dataclass(slots=True)
class SessionMetadata:
    """Identifying metadata recorded alongside a session."""

    recorded_at: str
    device_serial: str | None = None
    firmware_version: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:  # kwonly: disable
        """Build from a JSON dict, tolerating missing optional fields."""
        return cls(
            recorded_at=str(data.get("recorded_at", "")),
            device_serial=data.get("device_serial"),
            firmware_version=data.get("firmware_version"),
            notes=data.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict, omitting unset fields."""
        out: dict[str, Any] = {"recorded_at": self.recorded_at}
        if self.device_serial is not None:
            out["device_serial"] = self.device_serial
        if self.firmware_version is not None:
            out["firmware_version"] = self.firmware_version
        if self.notes is not None:
            out["notes"] = self.notes
        return out


@dataclass(slots=True)
class Frame:
    """One polling-cycle snapshot of register values."""

    ts: float
    registers: dict[int, list[int]] = field(default_factory=dict)


class SessionRecorder:
    """
    Capture a live Modbus session into an in-memory list of frames.

    Wraps any object exposing the pymodbus async client surface
    (``connect``, ``close``, ``connected``, ``read_holding_registers``,
    ``write_registers``). Reads that succeed are recorded into the current
    frame; errors and writes are passed through untouched.

    Frame boundaries are explicit: call :meth:`commit_frame` whenever a
    coordinator polling cycle ends. A frame is only persisted if it contains
    at least one register; back-to-back commits are no-ops.
    """

    def __init__(  # kwonly: disable
        self,
        inner: Any,
        *,
        metadata: SessionMetadata | None = None,
    ) -> None:
        """Wrap ``inner`` (a pymodbus-compatible client) for capture."""
        self._inner = inner
        self._metadata = metadata or SessionMetadata(
            recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._frames: list[Frame] = []
        self._current: dict[int, list[int]] = {}
        self._t0: float | None = None

    @property
    def connected(self) -> bool:
        """Mirror the inner client's connection state."""
        return bool(getattr(self._inner, "connected", False))

    @property
    def frames(self) -> list[Frame]:
        """All committed frames so far."""
        return list(self._frames)

    @property
    def metadata(self) -> SessionMetadata:
        """The metadata that will be written into the ZIP manifest."""
        return self._metadata

    def close(self) -> None:
        """Forward to the inner client."""
        self._inner.close()

    def commit_frame(self) -> None:
        """Close the current frame and append it to the session if non-empty."""
        if not self._current:
            return
        ts = (time.monotonic() - self._t0) if self._t0 is not None else 0.0
        self._frames.append(Frame(ts=ts, registers=dict(self._current)))
        self._current = {}

    async def connect(self) -> bool:
        """Forward to the inner client; mark recording start time."""
        if self._t0 is None:
            self._t0 = time.monotonic()
        return bool(await self._inner.connect())

    async def read_holding_registers(  # kwonly: disable
        self,
        address: int,
        count: int = 1,
        slave: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Read through the inner client; capture successful responses."""
        response = await self._inner.read_holding_registers(
            address,
            count,
            slave=slave,
            **kwargs,
        )
        if not response.isError():
            self._current[address] = list(response.registers)
        return response

    def save(self, path: Path) -> None:  # kwonly: disable
        """Persist the captured session as a ZIP archive at ``path``."""
        # Auto-commit any pending reads so we never lose a half-cycle.
        self.commit_frame()
        payload = {
            "version": _FORMAT_VERSION,
            "metadata": self._metadata.to_dict(),
            "frames": [
                {
                    "ts": frame.ts,
                    "registers": {str(addr): regs for addr, regs in frame.registers.items()},
                }
                for frame in self._frames
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                _MANIFEST_NAME,
                json.dumps(payload, indent=2, sort_keys=True),
            )

    async def write_registers(  # kwonly: disable
        self,
        address: int,
        values: list[int],
        slave: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Forward writes without recording (writes are not part of replay)."""
        return await self._inner.write_registers(
            address,
            values,
            slave=slave,
            **kwargs,
        )


class SessionPlayer:
    """
    Drop-in replacement for :class:`pymodbus.client.AsyncModbusTcpClient`.

    Returns register values from a recorded session. Mirrors the surface of
    :class:`~aiomtec2mqtt.mock_modbus_server.MockModbusTransport` so it
    plugs into ``AsyncModbusClient`` via the same patch mechanism.

    Frame advance: the player keeps track of which addresses have been read
    in the current frame. Once *every* address in the current frame has been
    served at least once, the next read advances to the following frame. If
    ``loop`` is ``True`` the player wraps from the last frame back to the
    first; otherwise it stays on the final frame.
    """

    def __init__(  # kwonly: disable
        self,
        frames: list[Frame],
        *,
        metadata: SessionMetadata | None = None,
        loop: bool = True,
    ) -> None:
        """Initialise the player with at least one frame."""
        if not frames:
            raise ValueError("SessionPlayer requires at least one frame")
        self._frames = frames
        self._metadata = metadata
        self._loop = loop
        self._frame_idx = 0
        self._served: set[int] = set()
        self._connected = False

    @classmethod
    def _from_payload(cls, payload: dict[str, Any], *, loop: bool) -> Self:  # kwonly: disable
        if (version := payload.get("version")) != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported session format version: {version!r} (expected {_FORMAT_VERSION})"
            )
        metadata = SessionMetadata.from_dict(payload.get("metadata") or {})
        frames = [
            Frame(
                ts=float(raw.get("ts", 0.0)),
                registers={int(addr): list(regs) for addr, regs in raw["registers"].items()},
            )
            for raw in payload.get("frames", [])
        ]
        if not frames:
            raise ValueError("Session contains no frames")
        return cls(frames, metadata=metadata, loop=loop)

    @classmethod
    def from_bytes(cls, data: bytes, *, loop: bool = True) -> Self:  # kwonly: disable
        """Load a session from an in-memory ZIP byte buffer."""
        with (
            zipfile.ZipFile(io.BytesIO(data), mode="r") as archive,
            archive.open(_MANIFEST_NAME) as fp,
        ):
            payload = json.load(fp)
        return cls._from_payload(payload, loop=loop)

    @classmethod
    def from_zip(cls, path: Path, *, loop: bool = True) -> Self:  # kwonly: disable
        """Load a session from the ZIP archive at ``path``."""
        with zipfile.ZipFile(path, mode="r") as archive, archive.open(_MANIFEST_NAME) as fp:
            payload = json.load(fp)
        return cls._from_payload(payload, loop=loop)

    @property
    def connected(self) -> bool:
        """Current connection state (toggled by connect/close)."""
        return self._connected

    @property
    def current_frame_index(self) -> int:
        """Zero-based index of the frame currently being served."""
        return self._frame_idx

    @property
    def frame_count(self) -> int:
        """Total number of frames in the session."""
        return len(self._frames)

    @property
    def metadata(self) -> SessionMetadata | None:
        """Metadata loaded from the session manifest, if any."""
        return self._metadata

    def close(self) -> None:
        """Mark the player as disconnected."""
        self._connected = False

    async def connect(self) -> bool:
        """Mark the player as connected. Always succeeds."""
        self._connected = True
        return True

    async def read_holding_registers(  # kwonly: disable
        self,
        address: int,
        count: int = 1,
        slave: int = 0,  # noqa: ARG002 — matches pymodbus signature
        **_: Any,
    ) -> MockModbusResponse:
        """Return register values from the current frame, then maybe advance."""
        frame = self._frames[self._frame_idx]
        if (registers := frame.registers.get(address)) is None:
            response_values: list[int] = [0] * count
        elif len(registers) < count:
            response_values = [*registers, *([0] * (count - len(registers)))]
        else:
            response_values = list(registers[:count])

        # Track served addresses; advance when the frame has been fully read.
        self._served.add(address)
        if self._served >= set(frame.registers.keys()) and frame.registers:
            self._advance_frame()

        return MockModbusResponse(registers=response_values)

    async def write_registers(  # kwonly: disable
        self,
        address: int,  # noqa: ARG002 — matches pymodbus signature
        values: list[int],
        slave: int = 0,  # noqa: ARG002 — matches pymodbus signature
        **_: Any,
    ) -> MockModbusResponse:
        """Accept writes silently (replay is read-only by design)."""
        return MockModbusResponse(registers=list(values))

    def _advance_frame(self) -> None:
        if self._frame_idx + 1 < len(self._frames):
            self._frame_idx += 1
        elif self._loop:
            self._frame_idx = 0
        # else: stay on the final frame
        self._served.clear()
