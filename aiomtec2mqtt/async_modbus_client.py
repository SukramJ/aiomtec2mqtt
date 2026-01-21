"""
Async Modbus client for reading M-TEC inverter data via Modbus TCP.

This module provides an asynchronous Modbus client implementation using pymodbus 3.x
AsyncModbusTcpClient. It supports circuit breaker pattern, exponential backoff, health
monitoring, and concurrent register reads.

Key Features:
- Non-blocking I/O with asyncio
- Concurrent register group reads with asyncio.gather()
- Circuit breaker for connection management
- Connection state machine
- Health check integration
- Typed exceptions
- Register clustering for optimal Modbus traffic

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import Any, Final, cast

from pymodbus.client import AsyncModbusTcpClient
from pymodbus.framer import FramerType
from pymodbus.pdu.register_message import ReadHoldingRegistersResponse

from aiomtec2mqtt.const import DEFAULT_FRAMER, Config, Register, RegisterGroup
from aiomtec2mqtt.exceptions import (
    ModbusConnectionError,
    ModbusReadError,
    ModbusTimeoutError,
    ModbusWriteError,
)
from aiomtec2mqtt.health import HealthCheck
from aiomtec2mqtt.resilience import (
    BackoffConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    ConnectionState,
    ConnectionStateMachine,
    ExponentialBackoff,
)

_LOGGER: Final = logging.getLogger(__name__)


class AsyncModbusClient:
    """Async Modbus TCP client for M-TEC inverter."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        register_map: dict[str, dict[str, Any]],
        register_groups: list[str],
        health_check: HealthCheck | None = None,
    ) -> None:
        """
        Initialize async Modbus client.

        Args:
            config: Configuration dictionary
            register_map: Register definitions
            register_groups: List of register group names
            health_check: Optional health check manager

        """
        self._config = config
        self._register_map: Final = register_map
        self._register_groups: Final = register_groups
        self._health_check = health_check

        # Connection parameters (matching sync client)
        self._modbus_framer: Final[str] = config.get(Config.MODBUS_FRAMER, DEFAULT_FRAMER)
        self._modbus_host: Final[str] = config[Config.MODBUS_IP]
        self._modbus_port: Final[int] = config[Config.MODBUS_PORT]
        self._modbus_retries: Final[int] = config.get(Config.MODBUS_RETRIES, 3)
        self._modbus_slave: Final[int] = config[Config.MODBUS_SLAVE]
        self._modbus_timeout: Final[int] = config[Config.MODBUS_TIMEOUT]

        # Modbus client
        self._client: AsyncModbusTcpClient | None = None
        self._error_count: int = 0

        # Register clustering cache
        self._cluster_cache: Final[dict[tuple[int, ...], list[dict[str, Any]]]] = {}

        # Precompute frequently used lookups (matching sync client)
        # Numeric registers used when reading "all" registers
        self._all_numeric_registers: Final[list[str]] = [r for r in register_map if r.isnumeric()]
        # Mapping from MQTT name to numeric register string for quick writes
        self._mqtt_name_to_register: Final[dict[str, str]] = {
            cast(str, item.get(Register.MQTT)): reg
            for reg, item in register_map.items()
            if reg.isnumeric() and item.get(Register.MQTT)
        }

        # Resilience patterns
        self._circuit_breaker: Final = CircuitBreaker(
            name="async_modbus",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=30.0,
            ),
        )
        self._state_machine: Final = ConnectionStateMachine(name="async_modbus")
        self._backoff: Final = ExponentialBackoff(
            config=BackoffConfig(
                initial_delay=1.0,
                max_delay=60.0,
                multiplier=2.0,
                jitter=True,
            )
        )

        # Register with health check
        if self._health_check:
            self._health_check.register_component(name="async_modbus")

    @property
    def error_count(self) -> int:
        """Get error count."""
        return self._error_count

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._state_machine.state == ConnectionState.CONNECTED

    @property
    def register_groups(self) -> list[str]:
        """Return the register groups."""
        return self._register_groups

    @property
    def register_map(self) -> dict[str, dict[str, Any]]:
        """Return the register map."""
        return self._register_map

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state_machine.state

    async def connect(self) -> bool:
        """
        Connect to Modbus server.

        Returns:
            True if connected successfully

        Raises:
            ModbusConnectionError: If connection fails

        """
        # Update state machine
        if self._state_machine.state == ConnectionState.CONNECTED:
            self._state_machine.transition_to(new_state=ConnectionState.RECONNECTING)
        else:
            self._state_machine.transition_to(new_state=ConnectionState.CONNECTING)

        self._error_count = 0
        _LOGGER.debug(
            "Connecting to Modbus server %s:%i (framer=%s)",
            self._modbus_host,
            self._modbus_port,
            self._modbus_framer,
        )

        try:
            self._client = AsyncModbusTcpClient(
                host=self._modbus_host,
                port=self._modbus_port,
                framer=FramerType(self._modbus_framer),
                timeout=self._modbus_timeout,
                retries=self._modbus_retries,
            )

            # Connect with timeout
            connected = await asyncio.wait_for(
                self._client.connect(),
                timeout=self._modbus_timeout,
            )

            if connected:
                _LOGGER.info(
                    "Successfully connected to Modbus server %s:%i",
                    self._modbus_host,
                    self._modbus_port,
                )
                self._state_machine.transition_to(new_state=ConnectionState.CONNECTED)
                if self._health_check:
                    self._health_check.record_success(name="async_modbus")
                self._backoff.reset()
                return True
            # Connection failed
            error_msg = (
                f"Couldn't connect to Modbus server {self._modbus_host}:{self._modbus_port}"
            )
            _LOGGER.error(error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.ERROR, error=error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_modbus", error=error_msg)
            return False  # noqa: TRY300

        except TimeoutError as ex:
            error_msg = f"Connection timeout to {self._modbus_host}:{self._modbus_port}"
            _LOGGER.exception(error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.ERROR, error=error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_modbus", error=error_msg)
            raise ModbusConnectionError(
                message=error_msg,
                details={"host": self._modbus_host, "port": self._modbus_port},
            ) from ex
        except Exception as ex:
            error_msg = f"Connection error to {self._modbus_host}:{self._modbus_port}: {ex}"
            _LOGGER.exception(error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.ERROR, error=error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_modbus", error=error_msg)
            raise ModbusConnectionError(
                message=error_msg,
                details={"host": self._modbus_host, "port": self._modbus_port},
            ) from ex

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncModbusClient]:
        """
        Async context manager for connection lifecycle.

        Usage:
            async with client.connection():
                data = await client.read_register_group(group_name="BASE")

        """
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    async def disconnect(self) -> None:
        """Disconnect from Modbus server."""
        if self._client:
            try:
                self._client.close()
                _LOGGER.info("Disconnected from Modbus server")
                self._state_machine.transition_to(new_state=ConnectionState.DISCONNECTED)
            except Exception as ex:
                error_msg = f"Error disconnecting from Modbus server: {ex}"
                _LOGGER.warning(error_msg)
                if self._health_check:
                    self._health_check.record_failure(name="async_modbus", error=error_msg)

    def get_register_list(self, *, group: RegisterGroup) -> list[str]:
        """Get a list of all registers which belong to a given group."""
        registers: list[str] = []
        for register, item in self._register_map.items():
            if item[Register.GROUP] == group:
                registers.append(register)

        if len(registers) == 0:
            _LOGGER.error("Unknown or empty register group: %s", group)
            return []
        return registers

    async def read_holding_registers(
        self,
        *,
        address: int,
        count: int,
    ) -> ReadHoldingRegistersResponse:
        """
        Read holding registers from Modbus server.

        Args:
            address: Starting register address
            count: Number of registers to read

        Returns:
            Modbus response with register values

        Raises:
            ModbusReadError: If read fails
            ModbusTimeoutError: If read times out

        """
        if not self._client:
            error_msg = "Not connected to Modbus server"
            _LOGGER.error(error_msg)
            raise ModbusReadError(
                message=error_msg,
                address=address,
                slave_id=self._modbus_slave,
            )

        async def _do_read() -> ReadHoldingRegistersResponse:
            """Perform the read operation."""
            assert self._client is not None  # Type narrowing for mypy

            try:
                result = cast(
                    ReadHoldingRegistersResponse,
                    await asyncio.wait_for(
                        self._client.read_holding_registers(
                            address=address,
                            count=count,
                            device_id=self._modbus_slave,
                        ),
                        timeout=self._modbus_timeout,
                    ),
                )

                # Check for errors
                if result.isError():
                    error_msg = f"Modbus error reading address {address}, count {count}"
                    _LOGGER.error(error_msg)
                    raise ModbusReadError(
                        message=error_msg,
                        address=address,
                        slave_id=self._modbus_slave,
                        details={"count": count},
                    )

                return result  # noqa: TRY300

            except TimeoutError as ex:
                error_msg = f"Timeout reading address {address}, count {count}"
                _LOGGER.error(error_msg)
                self._error_count += 1
                raise ModbusTimeoutError(
                    message=error_msg,
                    address=address,
                    slave_id=self._modbus_slave,
                ) from ex

        try:
            # Execute read operation
            result: ReadHoldingRegistersResponse = await _do_read()
        except ModbusReadError:
            raise
        except ModbusTimeoutError:
            raise
        except Exception as ex:
            error_msg = f"Unexpected error reading address {address}, count {count}: {ex}"
            _LOGGER.exception(error_msg)
            self._error_count += 1
            if self._health_check:
                self._health_check.record_failure(name="async_modbus", error=error_msg)
            raise ModbusReadError(
                message=error_msg,
                address=address,
                slave_id=self._modbus_slave,
            ) from ex
        else:
            if self._health_check:
                self._health_check.record_success(name="async_modbus")
            return result

    async def read_register(
        self,
        *,
        register: str,
    ) -> dict[str, Any] | None:
        """
        Read a single register and return its value with metadata.

        Args:
            register: Register address as string

        Returns:
            Dictionary with NAME, VALUE, UNIT keys or None if not found/error

        """
        # Lookup register
        if not (item := self._register_map.get(str(register))):
            _LOGGER.error("Unknown register: %s", register)
            return None

        # Read as a cluster of one
        if not (clusters := self._get_register_clusters(registers=[register])):
            return None

        try:
            if (cluster_data := await self._read_cluster(cluster=clusters[0])) and (
                reg_name := item.get(Register.NAME, register)
            ) in cluster_data:
                return {
                    Register.NAME: reg_name,
                    Register.VALUE: cluster_data[reg_name],
                    Register.UNIT: item.get(Register.UNIT, ""),
                }
        except Exception as ex:
            _LOGGER.error("Failed to read register %s: %s", register, ex)

        return None

    async def read_register_group(
        self,
        *,
        group_name: str,
    ) -> dict[str, Any]:
        """
        Read all registers in a group.

        Args:
            group_name: Name of register group

        Returns:
            Dictionary of register values

        Raises:
            ModbusReadError: If read fails

        """
        # Get registers for this group
        group_registers = [
            reg_name
            for reg_name, reg_info in self._register_map.items()
            if reg_info.get(Register.GROUP) == group_name
        ]

        if not group_registers:
            _LOGGER.warning("No registers found for group: %s", group_name)
            return {}

        # Get register clusters for optimized reads
        clusters = self._get_register_clusters(registers=group_registers)

        # Read all clusters concurrently
        cluster_results = await asyncio.gather(
            *[self._read_cluster(cluster=cluster) for cluster in clusters],
            return_exceptions=True,
        )

        # Combine results
        result: dict[str, Any] = {}
        for cluster_data in cluster_results:
            if isinstance(cluster_data, Exception):
                _LOGGER.error("Cluster read failed: %s", cluster_data)
                continue
            # Type narrowing: after Exception check, cluster_data must be dict
            assert isinstance(cluster_data, dict)  # Type narrowing for mypy
            result.update(cluster_data)

        return result

    async def write_register(self, *, register: str, value: Any) -> bool:
        """
        Write a value to a register.

        Args:
            register: Register address as string
            value: Value to write

        Returns:
            True if successful, False otherwise

        Raises:
            ModbusWriteError: On write errors

        """
        # Check if connected
        if self._client is None:
            _LOGGER.error("Can't write register - not connected")
            return False

        # Lookup register
        if not (item := self._register_map.get(str(register), None)):
            _LOGGER.error("Can't write unknown register: %s", register)
            return False
        if item.get(Register.WRITABLE, False) is False:
            _LOGGER.error("Can't write register which is marked read-only: %s", register)
            return False

        # check value
        try:
            if isinstance(value, str):
                value = float(value) if "." in value else int(value)
        except Exception:
            _LOGGER.error("Invalid numeric value: %s", value)
            return False

        # adjust scale
        if (item_scale := int(item.get(Register.SCALE, 1))) > 1:
            value *= item_scale

        try:
            result = await asyncio.wait_for(
                self._client.write_register(
                    address=int(register),
                    value=int(value),
                    device_id=self._modbus_slave,
                ),
                timeout=self._modbus_timeout,
            )

            if result.isError():
                error_msg = f"Error writing register {register} value {value}"
                _LOGGER.error(error_msg)
                if self._health_check:
                    self._health_check.record_failure(name="async_modbus", error=error_msg)
                raise ModbusWriteError(
                    message=error_msg,
                    address=int(register),
                    slave_id=self._modbus_slave,
                    details={"value": value},
                )

        except TimeoutError as ex:
            error_msg = f"Timeout writing register {register} value {value}"
            _LOGGER.error(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_modbus", error=error_msg)
            raise ModbusWriteError(
                message=error_msg,
                address=int(register),
                slave_id=self._modbus_slave,
                details={"value": value},
            ) from ex
        except ModbusWriteError:
            # Re-raise our typed exception
            raise
        except Exception as ex:
            error_msg = f"Unexpected error writing register {register}: {ex}"
            _LOGGER.exception(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="async_modbus", error=error_msg)
            raise ModbusWriteError(
                message=error_msg,
                address=int(register),
                slave_id=self._modbus_slave,
                details={"value": value},
            ) from ex
        else:
            if self._health_check:
                self._health_check.record_success(name="async_modbus")
            return True

    async def write_register_by_name(self, *, name: str, value: Any) -> bool:
        """Write a value to a register with a given name."""
        if (register := self._mqtt_name_to_register.get(name)) is None:
            _LOGGER.error("Can't write unknown register with name: %s", name)
            return False
        item = self._register_map[register]
        if value_items := item.get(Register.VALUE_ITEMS):
            for value_modbus, value_display in value_items.items():
                if value_display == value:
                    value = value_modbus
                    break
        return await self.write_register(register=register, value=value)

    def _decode_register(
        self,
        *,
        response: ReadHoldingRegistersResponse,
        offset: int,
        reg_info: dict[str, Any],
    ) -> Any:
        """
        Decode register value based on type and length.

        Args:
            response: Modbus response
            offset: Offset in response
            reg_info: Register information

        Returns:
            Decoded value

        """
        item_type = str(reg_info.get(Register.TYPE, "U16"))
        item_length = int(reg_info.get(Register.LENGTH, 1))

        # Bounds check
        if offset < 0 or item_length <= 0 or offset + item_length > len(response.registers):
            _LOGGER.error(
                "Decoding bounds error (type=%s, offset=%s, length=%s, available=%s)",
                item_type,
                offset,
                item_length,
                len(response.registers),
            )
            return None

        val: Any = None

        if item_type == "U16":
            val = int(response.registers[offset])
        elif item_type in ("S16", "I16"):
            raw = int(response.registers[offset])
            val = raw - 65536 if raw > 32767 else raw
        elif item_type == "U32":
            reg = response.registers[offset : offset + 2]
            val = (int(reg[0]) << 16) + int(reg[1])
        elif item_type in ("S32", "I32"):
            reg = response.registers[offset : offset + 2]
            raw = (int(reg[0]) << 16) + int(reg[1])
            val = raw - 4294967296 if raw > 2147483647 else raw
        elif item_type == "BYTE":
            if item_length == 1:
                reg1 = int(response.registers[offset])
                val = f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}"
            elif item_length == 2:
                reg1 = int(response.registers[offset])
                reg2 = int(response.registers[offset + 1])
                val = f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}  {reg2 >> 8:02d} {reg2 & 0xFF:02d}"
            elif item_length == 4:
                reg1 = int(response.registers[offset])
                reg2 = int(response.registers[offset + 1])
                reg3 = int(response.registers[offset + 2])
                reg4 = int(response.registers[offset + 3])
                val = (
                    f"{reg1 >> 8:02d} {reg1 & 0xFF:02d} {reg2 >> 8:02d} {reg2 & 0xFF:02d}  "
                    f"{reg3 >> 8:02d} {reg3 & 0xFF:02d} {reg4 >> 8:02d} {reg4 & 0xFF:02d}"
                )
            else:
                _LOGGER.error("Unsupported BYTE length: %s", item_length)
                return None
        elif item_type == "BIT":
            if item_length == 1:
                reg1 = int(response.registers[offset])
                val = f"{reg1:016b}"
            elif item_length == 2:
                reg1 = int(response.registers[offset])
                reg2 = int(response.registers[offset + 1])
                val = f"{reg1:016b} {reg2:016b}"
            else:
                bits = [f"{int(response.registers[offset + i]):016b}" for i in range(item_length)]
                val = " ".join(bits)
        elif item_type == "DAT":
            if offset + 3 > len(response.registers):
                _LOGGER.error("DAT requires 3 registers but not enough data available")
                return None
            reg1 = int(response.registers[offset])
            reg2 = int(response.registers[offset + 1])
            reg3 = int(response.registers[offset + 2])
            val = (
                f"{reg1 >> 8:02d}-{reg1 & 0xFF:02d}-{reg2 >> 8:02d} "
                f"{reg2 & 0xFF:02d}:{reg3 >> 8:02d}:{reg3 & 0xFF:02d}"
            )
        elif item_type == "STR":
            # item_length defines number of 16-bit registers to read
            reg = response.registers[offset : offset + item_length]
            # Convert registers to bytes (each register = 2 bytes, big-endian)
            raw_bytes = b"".join(r.to_bytes(2, byteorder="big") for r in reg)
            # Decode as UTF-8 string
            try:
                sval = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                sval = raw_bytes.decode("latin-1")
            # Strip trailing null bytes and spaces (same as sync client)
            val = sval.rstrip(" ").rstrip("\x00").rstrip(" ")
        else:
            _LOGGER.error("Unknown type %s to decode", item_type)
            return None

        # Apply scaling to numeric values
        item_scale = int(reg_info.get(Register.SCALE, 1))
        if item_scale > 1 and isinstance(val, (int, float)):
            val = val / item_scale

        return val

    def _generate_register_clusters(
        self,
        *,
        registers: list[str],
    ) -> list[dict[str, Any]]:
        """
        Generate register clusters considering register lengths.

        Args:
            registers: List of register names

        Returns:
            List of cluster definitions

        """
        # Sort registers numerically and get their lengths
        reg_infos: list[tuple[int, int]] = []  # (address, length)
        for r in registers:
            if r.isnumeric() and r in self._register_map:
                reg_info = self._register_map[r]
                addr = int(r)
                length = int(reg_info.get(Register.LENGTH, 1))
                reg_infos.append((addr, length))

        if not reg_infos:
            return []

        # Sort by address
        reg_infos.sort(key=lambda x: x[0])

        clusters: list[dict[str, Any]] = []
        first_addr, first_length = reg_infos[0]
        current_cluster: dict[str, Any] = {
            "start": first_addr,
            "count": first_length,
            "registers": [str(first_addr)],
        }

        # Build clusters (gap <= 10 registers)
        for addr, length in reg_infos[1:]:
            cluster_end = current_cluster["start"] + current_cluster["count"]
            gap = addr - cluster_end

            if gap <= 10:  # noqa: PLR2004 # pylint: disable=consider-using-assignment-expr
                # Extend current cluster to include this register and its length
                new_end = addr + length
                current_cluster["count"] = new_end - current_cluster["start"]
                current_cluster["registers"].append(str(addr))
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = {
                    "start": addr,
                    "count": length,
                    "registers": [str(addr)],
                }

        # Add final cluster
        clusters.append(current_cluster)

        return clusters

    def _get_register_clusters(
        self,
        *,
        registers: list[str],
    ) -> list[dict[str, Any]]:
        """
        Cluster registers for optimal Modbus traffic.

        Args:
            registers: List of register names

        Returns:
            List of cluster definitions

        """
        # Normalize key: use sorted unique numeric registers
        key_tuple: tuple[int, ...] = tuple(
            sorted({int(r) for r in registers if r.isnumeric() and r in self._register_map})
        )

        if key_tuple not in self._cluster_cache:
            # Simple cache size guard
            if len(self._cluster_cache) > 256:
                self._cluster_cache.clear()
            self._cluster_cache[key_tuple] = self._generate_register_clusters(registers=registers)

        return self._cluster_cache[key_tuple]

    async def _read_cluster(
        self,
        *,
        cluster: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Read a cluster of registers.

        Args:
            cluster: Cluster definition with start, count, registers

        Returns:
            Dictionary of decoded register values

        """
        start_addr = cluster["start"]
        count = cluster["count"]
        registers = cluster["registers"]

        # Read cluster
        response = await self.read_holding_registers(address=start_addr, count=count)

        # Decode registers
        result: dict[str, Any] = {}
        for reg_name in registers:
            reg_info = self._register_map[reg_name]
            offset = int(reg_name) - start_addr
            decoded = self._decode_register(response=response, offset=offset, reg_info=reg_info)
            result[reg_info[Register.NAME]] = decoded

        return result
