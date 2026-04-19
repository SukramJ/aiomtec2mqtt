"""Tests for :mod:`aiomtec2mqtt.session_replay`."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from aiomtec2mqtt.mock_modbus_server import MockModbusResponse, MockModbusTransport
from aiomtec2mqtt.session_replay import Frame, SessionMetadata, SessionPlayer, SessionRecorder


def _make_session(tmp_path: Path) -> Path:
    """Build a small two-frame session ZIP and return its path."""
    frame1 = Frame(ts=0.0, registers={10100: [4660, 22136], 10102: [0, 100]})
    frame2 = Frame(ts=5.0, registers={10100: [4661, 22137], 10102: [0, 105]})
    metadata = SessionMetadata(
        recorded_at="2026-04-19T12:00:00+00:00",
        device_serial="TEST-001",
        firmware_version="V27.52.4.0",
    )
    payload = {
        "version": 1,
        "metadata": metadata.to_dict(),
        "frames": [
            {"ts": frame1.ts, "registers": {str(a): r for a, r in frame1.registers.items()}},
            {"ts": frame2.ts, "registers": {str(a): r for a, r in frame2.registers.items()}},
        ],
    }
    path = tmp_path / "session.zip"
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(payload))
    return path


class TestSessionMetadata:
    def test_from_dict_tolerates_missing_optionals(self) -> None:
        m = SessionMetadata.from_dict({"recorded_at": "2026-04-19T00:00:00+00:00"})
        assert m.device_serial is None
        assert m.notes is None

    def test_round_trip_full(self) -> None:
        m = SessionMetadata(
            recorded_at="2026-04-19T00:00:00+00:00",
            device_serial="X",
            firmware_version="Y",
            notes="hello",
        )
        assert SessionMetadata.from_dict(m.to_dict()) == m

    def test_to_dict_omits_unset_fields(self) -> None:
        m = SessionMetadata(recorded_at="2026-04-19T00:00:00+00:00")
        assert m.to_dict() == {"recorded_at": "2026-04-19T00:00:00+00:00"}


class TestSessionPlayerLoading:
    def test_constructor_rejects_empty_frames(self) -> None:
        with pytest.raises(ValueError, match="at least one frame"):
            SessionPlayer(frames=[])

    def test_from_bytes(self, tmp_path: Path) -> None:
        path = _make_session(tmp_path)
        player = SessionPlayer.from_bytes(path.read_bytes())
        assert player.frame_count == 2

    def test_from_zip(self, tmp_path: Path) -> None:
        path = _make_session(tmp_path)
        player = SessionPlayer.from_zip(path)
        assert player.frame_count == 2
        assert player.metadata is not None
        assert player.metadata.device_serial == "TEST-001"

    def test_rejects_empty_session(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, mode="w") as archive:
            archive.writestr(
                "manifest.json",
                json.dumps({"version": 1, "frames": []}),
            )
        with pytest.raises(ValueError, match="Session contains no frames"):
            SessionPlayer.from_zip(path)

    def test_rejects_unknown_version(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.zip"
        with zipfile.ZipFile(path, mode="w") as archive:
            archive.writestr(
                "manifest.json",
                json.dumps({"version": 999, "frames": [{"ts": 0.0, "registers": {}}]}),
            )
        with pytest.raises(ValueError, match="Unsupported session format version"):
            SessionPlayer.from_zip(path)


class TestSessionPlayerReads:
    @pytest.mark.asyncio
    async def test_connect_close_toggle(self) -> None:
        player = SessionPlayer([Frame(ts=0.0, registers={100: [1]})])
        assert player.connected is False
        assert await player.connect() is True
        assert player.connected is True
        player.close()
        assert player.connected is False

    @pytest.mark.asyncio
    async def test_pads_short_registers(self) -> None:
        player = SessionPlayer([Frame(ts=0.0, registers={100: [42]})])
        response = await player.read_holding_registers(100, count=4)
        assert response.registers == [42, 0, 0, 0]

    @pytest.mark.asyncio
    async def test_returns_recorded_values(self) -> None:
        frame = Frame(ts=0.0, registers={10100: [4660, 22136]})
        player = SessionPlayer([frame])
        response = await player.read_holding_registers(10100, count=2)
        assert isinstance(response, MockModbusResponse)
        assert response.isError() is False
        assert response.registers == [4660, 22136]

    @pytest.mark.asyncio
    async def test_truncates_long_registers(self) -> None:
        player = SessionPlayer([Frame(ts=0.0, registers={100: [1, 2, 3, 4, 5]})])
        response = await player.read_holding_registers(100, count=2)
        assert response.registers == [1, 2]

    @pytest.mark.asyncio
    async def test_unknown_address_returns_zeros(self) -> None:
        player = SessionPlayer([Frame(ts=0.0, registers={100: [1]})])
        response = await player.read_holding_registers(999, count=3)
        assert response.registers == [0, 0, 0]


class TestSessionPlayerAdvance:
    @pytest.mark.asyncio
    async def test_advances_after_full_frame_read(self) -> None:
        f1 = Frame(ts=0.0, registers={100: [1], 200: [2]})
        f2 = Frame(ts=1.0, registers={100: [10], 200: [20]})
        player = SessionPlayer([f1, f2], loop=False)
        assert player.current_frame_index == 0
        await player.read_holding_registers(100)
        assert player.current_frame_index == 0
        await player.read_holding_registers(200)
        # After reading every address in frame 0, next read serves frame 1.
        assert player.current_frame_index == 1
        response = await player.read_holding_registers(100)
        assert response.registers == [10]

    @pytest.mark.asyncio
    async def test_loop_wraps(self) -> None:
        f1 = Frame(ts=0.0, registers={100: [1]})
        f2 = Frame(ts=1.0, registers={100: [10]})
        player = SessionPlayer([f1, f2], loop=True)
        await player.read_holding_registers(100)  # advance past f1
        assert player.current_frame_index == 1
        await player.read_holding_registers(100)  # advance past f2 → wrap
        assert player.current_frame_index == 0

    @pytest.mark.asyncio
    async def test_no_loop_pins_to_last_frame(self) -> None:
        f1 = Frame(ts=0.0, registers={100: [1]})
        f2 = Frame(ts=1.0, registers={100: [10]})
        player = SessionPlayer([f1, f2], loop=False)
        await player.read_holding_registers(100)
        await player.read_holding_registers(100)
        # Already on the last frame; further reads must keep returning it.
        for _ in range(3):
            response = await player.read_holding_registers(100)
            assert response.registers == [10]
        assert player.current_frame_index == 1

    @pytest.mark.asyncio
    async def test_writes_are_accepted_silently(self) -> None:
        player = SessionPlayer([Frame(ts=0.0, registers={100: [1]})])
        response = await player.write_registers(100, [9, 9])
        assert response.isError() is False
        assert response.registers == [9, 9]


class TestSessionRecorder:
    @pytest.mark.asyncio
    async def test_captures_successful_reads(self) -> None:
        transport = MockModbusTransport(register_values={100: [1, 2], 200: [3]})
        recorder = SessionRecorder(transport)
        await recorder.connect()
        await recorder.read_holding_registers(100, count=2)
        await recorder.read_holding_registers(200, count=1)
        recorder.commit_frame()
        assert len(recorder.frames) == 1
        assert recorder.frames[0].registers == {100: [1, 2], 200: [3]}

    @pytest.mark.asyncio
    async def test_commit_no_op_when_empty(self) -> None:
        recorder = SessionRecorder(MockModbusTransport())
        await recorder.connect()
        recorder.commit_frame()
        recorder.commit_frame()
        assert recorder.frames == []

    @pytest.mark.asyncio
    async def test_connected_property_mirrors_inner(self) -> None:
        transport = MockModbusTransport()
        recorder = SessionRecorder(transport)
        assert recorder.connected is False
        await recorder.connect()
        assert recorder.connected is True
        recorder.close()
        assert recorder.connected is False

    @pytest.mark.asyncio
    async def test_save_writes_loadable_zip(self, tmp_path: Path) -> None:
        transport = MockModbusTransport(register_values={100: [7], 200: [8]})
        recorder = SessionRecorder(
            transport,
            metadata=SessionMetadata(
                recorded_at="2026-04-19T00:00:00+00:00",
                device_serial="ABC",
            ),
        )
        await recorder.connect()
        await recorder.read_holding_registers(100)
        await recorder.read_holding_registers(200)
        # Note: no manual commit_frame — save() must auto-commit.
        path = tmp_path / "out.zip"
        recorder.save(path)

        player = SessionPlayer.from_zip(path)
        assert player.frame_count == 1
        assert player.metadata is not None
        assert player.metadata.device_serial == "ABC"
        assert (await player.read_holding_registers(100)).registers == [7]
        assert (await player.read_holding_registers(200)).registers == [8]

    @pytest.mark.asyncio
    async def test_skips_failed_reads(self) -> None:
        transport = MockModbusTransport(
            register_values={100: [1]},
            fail_read_n_times=1,
        )
        recorder = SessionRecorder(transport)
        await recorder.connect()
        # First read fails (counter→0) → not captured.
        first = await recorder.read_holding_registers(100)
        assert first.isError() is True
        # Second read succeeds and is captured.
        await recorder.read_holding_registers(100)
        recorder.commit_frame()
        assert recorder.frames[0].registers == {100: [1]}

    @pytest.mark.asyncio
    async def test_writes_pass_through_without_capture(self) -> None:
        transport = MockModbusTransport()
        recorder = SessionRecorder(transport)
        await recorder.connect()
        await recorder.write_registers(100, [42])
        recorder.commit_frame()
        assert recorder.frames == []
        assert transport.write_calls()[0].values == [42]


@pytest.mark.integration
class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_record_then_replay_two_cycles(self, tmp_path: Path) -> None:
        # Phase 1: record two cycles with values that change between cycles.
        transport = MockModbusTransport(register_values={100: [1], 200: [2]})
        recorder = SessionRecorder(transport)
        await recorder.connect()

        # Cycle 1
        r1a = await recorder.read_holding_registers(100)
        r1b = await recorder.read_holding_registers(200)
        recorder.commit_frame()

        # Mutate device state and record cycle 2.
        transport.register_values[100] = [11]
        transport.register_values[200] = [22]
        r2a = await recorder.read_holding_registers(100)
        r2b = await recorder.read_holding_registers(200)
        recorder.commit_frame()

        path = tmp_path / "session.zip"
        recorder.save(path)

        # Phase 2: replay and verify identical values in the same order.
        player = SessionPlayer.from_zip(path, loop=False)
        assert player.frame_count == 2

        # Cycle 1 of playback
        p1a = await player.read_holding_registers(100)
        p1b = await player.read_holding_registers(200)
        assert p1a.registers == r1a.registers
        assert p1b.registers == r1b.registers

        # Cycle 2 of playback
        p2a = await player.read_holding_registers(100)
        p2b = await player.read_holding_registers(200)
        assert p2a.registers == r2a.registers
        assert p2b.registers == r2b.registers
