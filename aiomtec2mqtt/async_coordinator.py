"""
Async coordinator for polling M-TEC Energybutler via Modbus and publishing to MQTT.

This module provides an asynchronous implementation of the coordinator pattern using
asyncio.TaskGroup for managing concurrent polling tasks. It replaces the synchronous
blocking architecture with non-blocking I/O operations.

Key Features:
- Concurrent register reads with asyncio.gather()
- Separate tasks for BASE, EXTENDED, and STATS polling
- Health check monitoring loop
- Graceful shutdown with task cancellation
- Event-driven architecture ready
- Full backward compatibility via sync wrapper

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Final

from aiomtec2mqtt import hass_int
from aiomtec2mqtt.async_modbus_client import AsyncModbusClient
from aiomtec2mqtt.async_mqtt_client import AsyncMqttClient
from aiomtec2mqtt.config import init_config, init_register_map
from aiomtec2mqtt.const import (
    REFRESH_DEFAULTS,
    SECONDARY_REGISTER_GROUPS,
    UTF8,
    Config,
    Register,
    RegisterGroup,
)
from aiomtec2mqtt.health import HealthCheck

_LOGGER: Final = logging.getLogger(__name__)

PVDATA_TYPE = dict[str, dict[str, Any] | int | float | str | bool]


class AsyncMtecCoordinator:
    """Async coordinator for M-TEC to MQTT data flow."""

    def __init__(self) -> None:
        """Initialize the async coordinator."""
        self._config = init_config()
        self._register_map, register_groups = init_register_map()

        # Health check manager
        self._health_check: Final = HealthCheck(stale_threshold=300.0)

        # Initialize components
        self._hass: Final = (
            hass_int.HassIntegration(
                hass_base_topic=self._config[Config.HASS_BASE_TOPIC],
                register_map=self._register_map,
            )
            if self._config[Config.HASS_ENABLE]
            else None
        )

        self._modbus_client: Final = AsyncModbusClient(
            config=self._config,
            register_map=self._register_map,
            register_groups=register_groups,
            health_check=self._health_check,
        )

        self._mqtt_client: Final = AsyncMqttClient(
            config=self._config,
            on_message=self._on_mqtt_message if self._hass else None,
            health_check=self._health_check,
        )

        # Cache register lists per group
        self._registers_by_group: dict[RegisterGroup, list[str]] = {}
        for grp_name in register_groups:
            grp = RegisterGroup(grp_name)
            self._registers_by_group[grp] = [
                reg_name
                for reg_name, reg_info in self._register_map.items()
                if reg_info.get(Register.GROUP) == grp_name
            ]

        # Configuration
        # Get float format and normalize it (e.g., "{:.3f}" or ":.3f" -> ".3f")
        mqtt_float_format_raw = self._config.get(Config.MQTT_FLOAT_FORMAT, ".3f")
        # Strip braces and leading colon to get clean format spec
        mqtt_float_format_clean = mqtt_float_format_raw.strip("{}").lstrip(":")
        self._mqtt_float_format: Final[str] = mqtt_float_format_clean
        self._mqtt_refresh_config: Final[int] = self._config.get(
            Config.REFRESH_CONFIG, REFRESH_DEFAULTS[Config.REFRESH_CONFIG]
        )
        self._mqtt_refresh_day: Final[int] = self._config.get(
            Config.REFRESH_DAY, REFRESH_DEFAULTS[Config.REFRESH_DAY]
        )
        self._mqtt_refresh_now: Final[int] = self._config.get(
            Config.REFRESH_NOW, REFRESH_DEFAULTS[Config.REFRESH_NOW]
        )
        self._mqtt_refresh_static: Final[int] = self._config.get(
            Config.REFRESH_STATIC, REFRESH_DEFAULTS[Config.REFRESH_STATIC]
        )
        self._mqtt_refresh_total: Final[int] = self._config.get(
            Config.REFRESH_TOTAL, REFRESH_DEFAULTS[Config.REFRESH_TOTAL]
        )
        self._mqtt_topic: Final[str] = self._config[Config.MQTT_TOPIC]
        self._hass_status_topic: Final[str] = f"{self._config[Config.HASS_BASE_TOPIC]}/status"
        self._hass_birth_gracetime: Final[int] = self._config.get(Config.HASS_BIRTH_GRACETIME, 15)

        # Runtime state
        self._shutdown_event: Final = asyncio.Event()
        self._secondary_group_index: int = 0
        self._hass_discovery_sent: bool = False

        if self._config[Config.DEBUG]:
            logging.getLogger().setLevel(logging.DEBUG)

        _LOGGER.info("Async coordinator initialized")

    async def run(self) -> None:
        """
        Run the async coordinator main loop.

        This method manages all concurrent tasks using asyncio.TaskGroup.
        """
        _LOGGER.info("Starting async coordinator")

        try:
            async with asyncio.TaskGroup() as task_group:  # noqa: SIM117
                # Connect to Modbus and MQTT
                async with self._modbus_client.connection():
                    async with self._mqtt_client.connection():
                        # Wait for HASS birth message
                        if self._hass:
                            await self._wait_for_hass_birth()

                        # Create polling tasks
                        task_group.create_task(
                            self._poll_base_registers(),
                            name="poll_base",
                        )
                        task_group.create_task(
                            self._poll_secondary_registers(),
                            name="poll_secondary",
                        )
                        task_group.create_task(
                            self._poll_statistics(),
                            name="poll_statistics",
                        )
                        task_group.create_task(
                            self._health_check_loop(),
                            name="health_check",
                        )

                        # Wait for shutdown signal
                        await self._shutdown_event.wait()

                        _LOGGER.info("Shutdown signal received, cancelling tasks")

        except* Exception as eg:
            for exc in eg.exceptions:
                _LOGGER.exception("Task failed: %s", exc)
            raise

        _LOGGER.info("Async coordinator stopped")

    def shutdown(self) -> None:
        """Signal shutdown to all tasks."""
        _LOGGER.info("Shutdown requested")
        self._shutdown_event.set()

    def _format_payload(self, *, data: dict[str, Any]) -> str:
        """
        Format data as JSON payload.

        Args:
            data: Dictionary of register values

        Returns:
            JSON string

        """
        # Format floats according to configuration
        formatted = {}
        for key, value in data.items():
            if isinstance(value, float):
                formatted[key] = f"{value:{self._mqtt_float_format}}"
            else:
                formatted[key] = value

        return json.dumps(formatted, ensure_ascii=False)

    async def _health_check_loop(self) -> None:
        """Periodic health check monitoring."""
        _LOGGER.info("Starting health check loop (interval: 60s)")

        while not self._shutdown_event.is_set():
            try:
                # Perform health check
                system_health = self._health_check.check_health()

                # Log health status
                if system_health.is_healthy():
                    _LOGGER.debug("System health: %s", system_health.message)
                else:
                    _LOGGER.warning("System health: %s", system_health.message)

                    # Log unhealthy components
                    for name, component in system_health.components.items():
                        if component.is_unhealthy():
                            _LOGGER.warning(
                                "Component %s unhealthy: %s (errors: %d)",
                                name,
                                component.message,
                                component.error_count,
                            )

                # Wait for next check
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                _LOGGER.info("Health check loop cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in health check loop: %s", ex)
                await asyncio.sleep(60)

    def _on_mqtt_message(self, message: Any) -> None:  # kwonly: disable
        """Handle incoming MQTT messages."""
        # Handle HASS birth message
        if message.topic == self._hass_status_topic and message.payload.decode(UTF8) == "online":
            _LOGGER.info("Home Assistant came online")
            # Trigger discovery resend
            self._hass_discovery_sent = False

    async def _poll_base_registers(self) -> None:
        """Poll BASE register group continuously."""
        _LOGGER.info("Starting BASE registers polling (interval: %ds)", self._mqtt_refresh_now)

        while not self._shutdown_event.is_set():
            try:
                # Read BASE registers
                data = await self._modbus_client.read_register_group(group_name=RegisterGroup.BASE)

                if data:
                    # Format and publish
                    topic = f"{self._mqtt_topic}/{RegisterGroup.BASE}"
                    payload = self._format_payload(data=data)
                    await self._mqtt_client.publish(topic=topic, payload=payload)

                    _LOGGER.debug("Published BASE data: %d values", len(data))

                # Wait for next poll
                await asyncio.sleep(self._mqtt_refresh_now)

            except asyncio.CancelledError:
                _LOGGER.info("BASE polling task cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in BASE polling: %s", ex)
                await asyncio.sleep(self._mqtt_refresh_now)

    async def _poll_secondary_registers(self) -> None:
        """Poll secondary register groups (GRID, INVERTER, etc.) in round-robin."""
        _LOGGER.info(
            "Starting secondary registers polling (interval: %ds)", self._mqtt_refresh_now
        )

        while not self._shutdown_event.is_set():
            try:
                # Get next secondary group
                group = SECONDARY_REGISTER_GROUPS.get(self._secondary_group_index)
                if not group:  # pylint: disable=consider-using-assignment-expr
                    self._secondary_group_index = 0
                    group = SECONDARY_REGISTER_GROUPS[0]

                # Read group
                data = await self._modbus_client.read_register_group(group_name=group)

                if data:  # pylint: disable=consider-using-assignment-expr
                    # Format and publish
                    topic = f"{self._mqtt_topic}/{group}"
                    payload = self._format_payload(data=data)
                    await self._mqtt_client.publish(topic=topic, payload=payload)

                    _LOGGER.debug("Published %s data: %d values", group, len(data))

                # Move to next group
                self._secondary_group_index = (self._secondary_group_index + 1) % len(
                    SECONDARY_REGISTER_GROUPS
                )

                # Wait for next poll
                await asyncio.sleep(self._mqtt_refresh_now)

            except asyncio.CancelledError:
                _LOGGER.info("Secondary polling task cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in secondary polling: %s", ex)
                await asyncio.sleep(self._mqtt_refresh_now)

    async def _poll_statistics(self) -> None:
        """Poll DAY and TOTAL statistics periodically."""
        _LOGGER.info("Starting statistics polling (interval: %ds)", self._mqtt_refresh_day)

        while not self._shutdown_event.is_set():
            try:
                # Read DAY statistics
                day_data = await self._modbus_client.read_register_group(
                    group_name=RegisterGroup.DAY
                )
                if day_data:
                    topic = f"{self._mqtt_topic}/{RegisterGroup.DAY}"
                    payload = self._format_payload(data=day_data)
                    await self._mqtt_client.publish(topic=topic, payload=payload)
                    _LOGGER.debug("Published DAY statistics")

                # Read TOTAL statistics
                total_data = await self._modbus_client.read_register_group(
                    group_name=RegisterGroup.TOTAL
                )
                if total_data:
                    topic = f"{self._mqtt_topic}/{RegisterGroup.TOTAL}"
                    payload = self._format_payload(data=total_data)
                    await self._mqtt_client.publish(topic=topic, payload=payload)
                    _LOGGER.debug("Published TOTAL statistics")

                # Wait for next poll
                await asyncio.sleep(self._mqtt_refresh_day)

            except asyncio.CancelledError:
                _LOGGER.info("Statistics polling task cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in statistics polling: %s", ex)
                await asyncio.sleep(self._mqtt_refresh_day)

    async def _send_hass_discovery(self) -> None:
        """Send Home Assistant discovery messages."""
        if not self._hass:
            return

        _LOGGER.info("Sending Home Assistant discovery messages")

        try:
            # Get device info from first Modbus read
            static_data = await self._modbus_client.read_register_group(
                group_name=RegisterGroup.STATIC
            )

            if static_data:
                # Extract serial number, firmware, and equipment info
                serial_no = static_data.get("serial_no", "unknown")
                firmware = static_data.get("firmware_version", "unknown")
                equipment = static_data.get("equipment_info", "unknown")

                # Initialize HASS integration (generates _devices_array)
                # Pass None for mqtt since we'll publish discovery messages ourselves
                self._hass.initialize(
                    mqtt=None,
                    serial_no=serial_no,
                    firmware_version=firmware,
                    equipment_info=equipment,
                )

                # Publish discovery messages from _devices_array
                for topic, payload_str, _command_topic in self._hass._devices_array:  # noqa: SLF001 # pylint: disable=protected-access
                    await self._mqtt_client.publish(topic=topic, payload=payload_str, retain=True)

                _LOGGER.info("Sent %d discovery messages", len(self._hass._devices_array))  # pylint: disable=protected-access
                self._hass_discovery_sent = True

        except Exception as ex:
            _LOGGER.error("Failed to send HASS discovery: %s", ex)

    async def _wait_for_hass_birth(self) -> None:
        """Wait for Home Assistant birth message."""
        if not self._hass:
            return

        _LOGGER.info(
            "Waiting %ds for Home Assistant birth message",
            self._hass_birth_gracetime,
        )

        # Subscribe to HASS status topic
        try:
            await self._mqtt_client.subscribe(topic=self._hass_status_topic)
        except Exception as ex:
            _LOGGER.warning("Failed to subscribe to HASS status: %s", ex)

        # Wait for birth message or timeout
        await asyncio.sleep(self._hass_birth_gracetime)

        # Send discovery messages
        if not self._hass_discovery_sent:
            await self._send_hass_discovery()


def main() -> None:
    """Run async coordinator main loop."""
    coordinator = AsyncMtecCoordinator()

    try:
        asyncio.run(coordinator.run())
    except KeyboardInterrupt:
        _LOGGER.info("Received keyboard interrupt")
        coordinator.shutdown()
    except Exception:
        _LOGGER.exception("Fatal error")
        raise


if __name__ == "__main__":
    main()
