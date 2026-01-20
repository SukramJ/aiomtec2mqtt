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
from pymodbus.pdu.register_message import ReadHoldingRegistersResponse

from aiomtec2mqtt.const import Config, Register
from aiomtec2mqtt.exceptions import ModbusConnectionError, ModbusReadError, ModbusTimeoutError
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
        self._register_map = register_map
        self._register_groups = register_groups
        self._health_check = health_check

        # Connection parameters
        self._modbus_host: Final[str] = config[Config.MODBUS_IP]
        self._modbus_port: Final[int] = config[Config.MODBUS_PORT]
        self._modbus_slave: Final[int] = config[Config.MODBUS_SLAVE]
        self._modbus_timeout: Final[int] = config[Config.MODBUS_TIMEOUT]

        # Modbus client
        self._client: AsyncModbusTcpClient | None = None
        self._error_count: int = 0

        # Register clustering cache
        self._cluster_cache: dict[tuple[int, ...], list[dict[str, Any]]] = {}

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
            "Connecting to Modbus server %s:%i",
            self._modbus_host,
            self._modbus_port,
        )

        try:
            self._client = AsyncModbusTcpClient(
                host=self._modbus_host,
                port=self._modbus_port,
                timeout=self._modbus_timeout,
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

    def _decode_register(
        self,
        *,
        response: ReadHoldingRegistersResponse,
        offset: int,
        reg_info: dict[str, Any],
    ) -> Any:
        """
        Decode register value.

        Args:
            response: Modbus response
            offset: Offset in response
            reg_info: Register information

        Returns:
            Decoded value

        """
        # Get raw value
        raw_value = response.registers[offset]

        # Apply scale
        scale = reg_info.get(Register.SCALE, 1)
        return raw_value / scale if scale > 1 else raw_value

    def _generate_register_clusters(
        self,
        *,
        registers: list[str],
    ) -> list[dict[str, Any]]:
        """
        Generate register clusters.

        Args:
            registers: List of register names

        Returns:
            List of cluster definitions

        """
        # Sort registers numerically
        sorted_regs = sorted(
            [int(r) for r in registers if r.isnumeric() and r in self._register_map]
        )

        if not sorted_regs:
            return []

        clusters: list[dict[str, Any]] = []
        current_cluster: dict[str, Any] = {
            "start": sorted_regs[0],
            "count": 1,
            "registers": [str(sorted_regs[0])],
        }

        # Build clusters (gap <= 10 registers)
        for reg in sorted_regs[1:]:
            gap = reg - (current_cluster["start"] + current_cluster["count"])

            if gap <= 10:  # noqa: PLR2004 # pylint: disable=consider-using-assignment-expr
                # Extend current cluster
                current_cluster["count"] = reg - current_cluster["start"] + 1
                current_cluster["registers"].append(str(reg))
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = {
                    "start": reg,
                    "count": 1,
                    "registers": [str(reg)],
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
