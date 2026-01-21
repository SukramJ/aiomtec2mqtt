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
from datetime import datetime
import logging
from typing import Any, Final

from aiomtec2mqtt import hass_int
from aiomtec2mqtt.async_modbus_client import AsyncModbusClient
from aiomtec2mqtt.async_mqtt_client import AsyncMqttClient
from aiomtec2mqtt.config import init_config, init_register_map
from aiomtec2mqtt.const import (
    EQUIPMENT,
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
                mqtt_topic=self._config[Config.MQTT_TOPIC],
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
        self._serial_no: str | None = None
        self._topic_base: str | None = None
        self._pending_write_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

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
                            self._poll_config_registers(),
                            name="poll_config",
                        )
                        task_group.create_task(
                            self._poll_secondary_registers(),
                            name="poll_secondary",
                        )
                        task_group.create_task(
                            self._poll_day_statistics(),
                            name="poll_day_statistics",
                        )
                        task_group.create_task(
                            self._poll_total_statistics(),
                            name="poll_total_statistics",
                        )
                        task_group.create_task(
                            self._poll_static_registers(),
                            name="poll_static",
                        )
                        task_group.create_task(
                            self._health_check_loop(),
                            name="health_check",
                        )
                        task_group.create_task(
                            self._modbus_watchdog(),
                            name="modbus_watchdog",
                        )
                        task_group.create_task(
                            self._process_write_queue(),
                            name="write_queue",
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

    def _convert_code(self, *, value: int | str, value_items: dict[int, str]) -> str:
        """Convert code register value to string."""
        if isinstance(value, int):
            return value_items.get(value, "Unknown")

        faults: list[str] = []
        try:
            value_no = int(f"0b{str(value).replace(' ', '')}", 2)
            for no, fault in value_items.items():
                if (value_no & (1 << no)) > 0:
                    faults.append(fault)
        except (ValueError, TypeError):
            pass

        if not faults:
            faults.append("OK")
        return ", ".join(faults)

    def _format_value(self, *, value: Any) -> str:
        """
        Format a single value as payload string.

        Args:
            value: The value to format

        Returns:
            Formatted string

        """
        if isinstance(value, float):
            return f"{value:{self._mqtt_float_format}}"
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

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

    async def _initialize_from_static_data(self) -> bool:
        """
        Initialize serial_no and topic_base from STATIC register data.

        Returns:
            True if initialization succeeded, False otherwise.

        """
        if self._serial_no is not None:
            return True  # Already initialized

        _LOGGER.info("Reading STATIC data to initialize serial number")

        try:
            static_data = await self._modbus_client.read_register_group(
                group_name=RegisterGroup.STATIC
            )

            if not static_data:
                _LOGGER.warning("No STATIC data received")
                return False

            # Extract serial number from the data
            # The async_modbus_client returns {Register.NAME: value}
            serial_no = static_data.get("Inverter serial number")
            firmware = static_data.get("Firmware version")
            equipment = static_data.get("Equipment info")

            if not serial_no:
                _LOGGER.warning("Serial number not found in STATIC data")
                return False

            self._serial_no = str(serial_no)
            self._topic_base = f"{self._mqtt_topic}/{self._serial_no}"

            _LOGGER.info(
                "Initialized: serial_no=%s, topic_base=%s",
                self._serial_no,
                self._topic_base,
            )

            # Initialize HASS if enabled
            if self._hass and not self._hass.is_initialized:
                self._hass.initialize(
                    mqtt=None,
                    serial_no=self._serial_no,
                    firmware_version=str(firmware) if firmware else "unknown",
                    equipment_info=str(equipment) if equipment else "unknown",
                )

        except Exception as ex:
            _LOGGER.error("Failed to initialize from STATIC data: %s", ex)
            return False
        else:
            return True

    async def _modbus_watchdog(self) -> None:
        """
        Monitor Modbus connection and reconnect if needed.

        Matches sync coordinator behavior: reconnect after 10 consecutive errors.
        """
        _LOGGER.info("Starting Modbus watchdog (error threshold: 10)")
        max_errors = 10

        while not self._shutdown_event.is_set():
            try:
                # Check error count
                if self._modbus_client.error_count > max_errors:
                    _LOGGER.warning(
                        "Modbus error count (%d) exceeded threshold (%d), reconnecting...",
                        self._modbus_client.error_count,
                        max_errors,
                    )
                    await self._reconnect_modbus()

                # Check every 5 seconds
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                _LOGGER.info("Modbus watchdog cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in Modbus watchdog: %s", ex)
                await asyncio.sleep(5)

    def _on_mqtt_message(self, message: Any) -> None:  # kwonly: disable
        """Handle incoming MQTT messages."""
        topic = str(message.topic)
        payload = message.payload.decode(UTF8) if message.payload else ""

        # Handle HASS birth message
        if topic == self._hass_status_topic and payload == "online":
            _LOGGER.info("Home Assistant came online")
            # Trigger discovery resend
            self._hass_discovery_sent = False
            return

        # Handle command topics for writable registers
        # Expected format: {topic_base}/{group}/{mqtt_key}/set
        if self._topic_base and topic.startswith(self._topic_base) and topic.endswith("/set"):
            # Extract mqtt_key from topic
            # Topic: MTEC/serial/group/mqtt_key/set
            parts = topic.split("/")
            if len(parts) >= 4:
                mqtt_key = parts[-2]  # The part before /set
                _LOGGER.info("Received write command for %s: %s", mqtt_key, payload)
                # Queue the write request for async processing
                try:
                    self._pending_write_queue.put_nowait((mqtt_key, payload))
                except asyncio.QueueFull:
                    _LOGGER.error("Write queue full, dropping command for %s", mqtt_key)

    async def _poll_base_registers(self) -> None:
        """Poll BASE register group continuously."""
        _LOGGER.info("Starting BASE registers polling (interval: %ds)", self._mqtt_refresh_now)

        # Ensure initialization before polling
        while not self._shutdown_event.is_set() and not await self._initialize_from_static_data():
            _LOGGER.warning("Waiting for initialization... retrying in 10s")
            await asyncio.sleep(10)

        while not self._shutdown_event.is_set():
            try:
                # Read BASE registers
                data = await self._modbus_client.read_register_group(group_name=RegisterGroup.BASE)

                if data:
                    await self._publish_register_data(data=data, group=RegisterGroup.BASE)
                    _LOGGER.debug("Published BASE data: %d values", len(data))

                # Wait for next poll
                await asyncio.sleep(self._mqtt_refresh_now)

            except asyncio.CancelledError:
                _LOGGER.info("BASE polling task cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in BASE polling: %s", ex)
                await asyncio.sleep(self._mqtt_refresh_now)

    async def _poll_config_registers(self) -> None:
        """Poll CONFIG register group periodically (contains writable entities)."""
        _LOGGER.info(
            "Starting CONFIG registers polling (interval: %ds)", self._mqtt_refresh_config
        )

        # Wait for initialization (poll until topic_base is set)
        while not self._shutdown_event.is_set() and self._topic_base is None:  # noqa: ASYNC110
            await asyncio.sleep(1)

        while not self._shutdown_event.is_set():
            try:
                # Read CONFIG registers
                data = await self._modbus_client.read_register_group(
                    group_name=RegisterGroup.CONFIG
                )

                if data:
                    await self._publish_register_data(data=data, group=RegisterGroup.CONFIG)
                    _LOGGER.debug("Published CONFIG data: %d values", len(data))

                # Wait for next poll
                await asyncio.sleep(self._mqtt_refresh_config)

            except asyncio.CancelledError:
                _LOGGER.info("CONFIG polling task cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in CONFIG polling: %s", ex)
                await asyncio.sleep(self._mqtt_refresh_config)

    async def _poll_day_statistics(self) -> None:
        """Poll DAY statistics periodically (matching sync: separate from TOTAL)."""
        _LOGGER.info("Starting DAY statistics polling (interval: %ds)", self._mqtt_refresh_day)

        # Wait for initialization (poll until topic_base is set)
        while not self._shutdown_event.is_set() and self._topic_base is None:  # noqa: ASYNC110
            await asyncio.sleep(1)

        while not self._shutdown_event.is_set():
            try:
                # Read DAY statistics
                day_data = await self._modbus_client.read_register_group(
                    group_name=RegisterGroup.DAY
                )
                if day_data:
                    await self._publish_register_data(data=day_data, group=RegisterGroup.DAY)
                    _LOGGER.debug("Published DAY statistics")

                # Wait for next poll
                await asyncio.sleep(self._mqtt_refresh_day)

            except asyncio.CancelledError:
                _LOGGER.info("DAY statistics polling task cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in DAY statistics polling: %s", ex)
                await asyncio.sleep(self._mqtt_refresh_day)

    async def _poll_secondary_registers(self) -> None:
        """Poll secondary register groups (GRID, INVERTER, etc.) in round-robin."""
        _LOGGER.info(
            "Starting secondary registers polling (interval: %ds)", self._mqtt_refresh_now
        )

        # Wait for initialization (poll until topic_base is set)
        while not self._shutdown_event.is_set() and self._topic_base is None:  # noqa: ASYNC110
            await asyncio.sleep(1)

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
                    await self._publish_register_data(data=data, group=group)
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

    async def _poll_static_registers(self) -> None:
        """Poll STATIC register group periodically."""
        _LOGGER.info(
            "Starting STATIC registers polling (interval: %ds)", self._mqtt_refresh_static
        )

        # Wait for initialization (poll until topic_base is set)
        while not self._shutdown_event.is_set() and self._topic_base is None:  # noqa: ASYNC110
            await asyncio.sleep(1)

        while not self._shutdown_event.is_set():
            try:
                # Read STATIC registers
                data = await self._modbus_client.read_register_group(
                    group_name=RegisterGroup.STATIC
                )

                if data:
                    await self._publish_register_data(data=data, group=RegisterGroup.STATIC)
                    _LOGGER.debug("Published STATIC data: %d values", len(data))

                # Wait for next poll
                await asyncio.sleep(self._mqtt_refresh_static)

            except asyncio.CancelledError:
                _LOGGER.info("STATIC polling task cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in STATIC polling: %s", ex)
                await asyncio.sleep(self._mqtt_refresh_static)

    async def _poll_total_statistics(self) -> None:
        """Poll TOTAL statistics periodically (matching sync: separate from DAY)."""
        _LOGGER.info("Starting TOTAL statistics polling (interval: %ds)", self._mqtt_refresh_total)

        # Wait for initialization (poll until topic_base is set)
        while not self._shutdown_event.is_set() and self._topic_base is None:  # noqa: ASYNC110
            await asyncio.sleep(1)

        while not self._shutdown_event.is_set():
            try:
                # Read TOTAL statistics
                total_data = await self._modbus_client.read_register_group(
                    group_name=RegisterGroup.TOTAL
                )
                if total_data:
                    await self._publish_register_data(data=total_data, group=RegisterGroup.TOTAL)
                    _LOGGER.debug("Published TOTAL statistics")

                # Wait for next poll
                await asyncio.sleep(self._mqtt_refresh_total)

            except asyncio.CancelledError:
                _LOGGER.info("TOTAL statistics polling task cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in TOTAL statistics polling: %s", ex)
                await asyncio.sleep(self._mqtt_refresh_total)

    def _process_register_value(
        self, *, register_addr: str, value: Any, reg_info: dict[str, Any]
    ) -> Any:
        """
        Process a register value (special formatting, enum conversion).

        Args:
            register_addr: The register address as string
            value: The raw value
            reg_info: Register information

        Returns:
            Processed value

        """
        # Firmware version formatting (register 10011)
        if register_addr == "10011" and isinstance(value, str) and "  " in value:
            try:
                fw0, fw1 = str(value).split("  ")
                return f"V{fw0.replace(' ', '.')}-V{fw1.replace(' ', '.')}"
            except (ValueError, AttributeError):
                pass

        # Equipment info lookup (register 10008)
        if register_addr == "10008" and isinstance(value, str) and " " in value:
            try:
                upper, lower = value.split(" ")
                return EQUIPMENT.get(int(upper), {}).get(int(lower), "unknown")
            except (ValueError, AttributeError):
                pass

        # Enum conversion for device_class == "enum"
        if reg_info.get(Register.DEVICE_CLASS) == "enum" and (
            value_items := reg_info.get(Register.VALUE_ITEMS)
        ):
            return self._convert_code(value=value, value_items=value_items)

        return value

    async def _process_write_queue(self) -> None:
        """Process write requests from the queue."""
        _LOGGER.info("Starting write queue processor")

        while not self._shutdown_event.is_set():
            try:
                # Wait for a write request (with timeout to check shutdown)
                try:
                    mqtt_key, value = await asyncio.wait_for(
                        self._pending_write_queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                # Process the write request
                _LOGGER.debug("Processing write request: %s = %s", mqtt_key, value)
                try:
                    success = await self._modbus_client.write_register_by_name(
                        name=mqtt_key, value=value
                    )
                    if success:
                        _LOGGER.info("Successfully wrote %s = %s", mqtt_key, value)
                    else:
                        _LOGGER.error("Failed to write %s = %s", mqtt_key, value)
                except Exception as ex:
                    _LOGGER.error("Error writing %s = %s: %s", mqtt_key, value, ex)

                self._pending_write_queue.task_done()

            except asyncio.CancelledError:
                _LOGGER.info("Write queue processor cancelled")
                raise
            except Exception as ex:
                _LOGGER.error("Error in write queue processor: %s", ex)

    async def _publish_pseudo_registers(
        self,
        *,
        processed_data: dict[str, Any],
        raw_data: dict[str, Any],
        group: RegisterGroup,
    ) -> None:
        """
        Calculate and publish pseudo-registers.

        Args:
            processed_data: Processed data with mqtt_keys
            raw_data: Raw data with Register.NAME keys
            group: The register group

        """
        if not self._topic_base:
            return

        base = f"{self._topic_base}/{group}"
        pseudo_values: dict[str, Any] = {}

        # Map Register.NAME to value for calculations
        name_to_value: dict[str, float] = {}
        for name, value in raw_data.items():
            if isinstance(value, (int, float)):
                name_to_value[name] = float(value)

        if group == RegisterGroup.BASE:
            # consumption = Inverter AC power (11016) - Grid power (11000)
            inverter_ac = name_to_value.get("Inverter AC power", 0)
            grid_power = name_to_value.get("Grid power", 0)
            pseudo_values["consumption"] = max(0.0, inverter_ac - grid_power)

            # api_date = current timestamp (mqtt key from registers.yaml)
            pseudo_values["api_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        elif group == RegisterGroup.DAY:
            # consumption_day = PV + Grid purchase + Battery discharge - Grid injection - Battery charge
            # Register names from registers.yaml
            pv_gen = name_to_value.get("PV energy generated (day)", 0)
            grid_purchase = name_to_value.get("Grid purchased energy (day)", 0)
            batt_discharge = name_to_value.get("Battery discharge energy (day)", 0)
            grid_injection = name_to_value.get("Grid injection energy (day)", 0)
            batt_charge = name_to_value.get("Battery charge energy (day)", 0)

            consumption_day = (
                pv_gen + grid_purchase + batt_discharge - grid_injection - batt_charge
            )
            pseudo_values["consumption_day"] = max(0.0, consumption_day)

            # autarky_rate_day = 100 * (1 - grid_purchase / consumption_day)
            if consumption_day > 0:
                pseudo_values["autarky_rate_day"] = 100 * (1 - grid_purchase / consumption_day)
            else:
                pseudo_values["autarky_rate_day"] = 0

            # own_consumption_day = 100 * (1 - grid_injection / pv_gen)
            if pv_gen > 0:
                pseudo_values["own_consumption_day"] = 100 * (1 - grid_injection / pv_gen)
            else:
                pseudo_values["own_consumption_day"] = 0

        elif group == RegisterGroup.TOTAL:
            # consumption_total
            pv_gen = name_to_value.get("PV energy generated (total)", 0)
            grid_purchase = name_to_value.get("Grid energy purchased (total)", 0)
            batt_discharge = name_to_value.get("Battery energy discharged (total)", 0)
            grid_injection = name_to_value.get("Grid energy injected (total)", 0)
            batt_charge = name_to_value.get("Battery energy charged (total)", 0)

            consumption_total = (
                pv_gen + grid_purchase + batt_discharge - grid_injection - batt_charge
            )
            pseudo_values["consumption_total"] = max(0.0, consumption_total)

            # autarky_rate_total
            if consumption_total > 0:
                pseudo_values["autarky_rate_total"] = 100 * (1 - grid_purchase / consumption_total)
            else:
                pseudo_values["autarky_rate_total"] = 0

            # own_consumption_total
            if pv_gen > 0:
                pseudo_values["own_consumption_total"] = 100 * (1 - grid_injection / pv_gen)
            else:
                pseudo_values["own_consumption_total"] = 0

        # Publish pseudo-registers
        for mqtt_key, value in pseudo_values.items():
            # Correct negative values (matching sync: both int and float)
            if isinstance(value, (int, float)) and value < 0:
                value = 0

            topic = f"{base}/{mqtt_key}/state"
            payload = self._format_value(value=value)
            await self._mqtt_client.publish(topic=topic, payload=payload)

    async def _publish_register_data(self, *, data: dict[str, Any], group: RegisterGroup) -> None:
        """
        Publish register data to MQTT in the correct format.

        Publishes each value to: {topic_base}/{group}/{mqtt_key}/state

        Args:
            data: Dictionary of {Register.NAME: value} from modbus client
            group: The register group

        """
        if not self._topic_base:
            _LOGGER.warning("Cannot publish: topic_base not initialized")
            return

        # Build reverse lookups
        name_to_mqtt: dict[str, str] = {}
        name_to_addr: dict[str, str] = {}
        name_to_info: dict[str, dict[str, Any]] = {}
        for reg_addr, reg_info in self._register_map.items():
            name = reg_info.get(Register.NAME)
            mqtt_key = reg_info.get(Register.MQTT)
            if name:
                if mqtt_key:
                    name_to_mqtt[name] = mqtt_key
                name_to_addr[name] = reg_addr
                name_to_info[name] = reg_info

        base = f"{self._topic_base}/{group}"
        processed_data: dict[str, Any] = {}

        for name, value in data.items():
            if not (mqtt_key := name_to_mqtt.get(name)):
                _LOGGER.debug("No mqtt_key for register name: %s", name)
                continue

            reg_addr = name_to_addr.get(name, "")
            reg_info = name_to_info.get(name, {})

            # Process the value (special formatting, enum conversion)
            processed_value = self._process_register_value(
                register_addr=reg_addr, value=value, reg_info=reg_info
            )

            # Note: Negative value correction only for pseudo-registers (matching sync behavior)
            # Regular registers CAN have negative values (e.g., grid_power when exporting)

            processed_data[mqtt_key] = processed_value

            topic = f"{base}/{mqtt_key}/state"
            payload = self._format_value(value=processed_value)
            await self._mqtt_client.publish(topic=topic, payload=payload)

        # Calculate and publish pseudo-registers for specific groups
        await self._publish_pseudo_registers(
            processed_data=processed_data, raw_data=data, group=group
        )

    async def _reconnect_modbus(self) -> None:
        """Reconnect to Modbus server (matching sync coordinator behavior)."""
        _LOGGER.info("Attempting Modbus reconnection...")

        try:
            # Disconnect first
            await self._modbus_client.disconnect()

            # Wait before reconnecting (matching sync: 10 seconds)
            await asyncio.sleep(10)

            # Reconnect
            if await self._modbus_client.connect():
                _LOGGER.info("Modbus reconnection successful")
            else:
                _LOGGER.error("Modbus reconnection failed")

        except Exception as ex:
            _LOGGER.error("Error during Modbus reconnection: %s", ex)

    async def _send_hass_discovery(self) -> None:
        """Send Home Assistant discovery messages."""
        if not self._hass:
            return

        # Ensure we're initialized first
        if not await self._initialize_from_static_data():
            _LOGGER.error("Cannot send HASS discovery: not initialized")
            return

        _LOGGER.info("Sending Home Assistant discovery messages")

        try:
            # Publish discovery messages and subscribe to command topics
            command_topics_subscribed = 0
            for topic, payload_str, command_topic in self._hass._devices_array:  # noqa: SLF001 # pylint: disable=protected-access
                await self._mqtt_client.publish(topic=topic, payload=payload_str, retain=True)

                # Subscribe to command topic for writable entities (matching sync behavior)
                if command_topic:
                    try:
                        await self._mqtt_client.subscribe(topic=command_topic)
                        command_topics_subscribed += 1
                        _LOGGER.debug("Subscribed to command topic: %s", command_topic)
                    except Exception as ex:
                        _LOGGER.warning("Failed to subscribe to %s: %s", command_topic, ex)

            _LOGGER.info(
                "Sent %d discovery messages, subscribed to %d command topics",
                len(self._hass._devices_array),  # pylint: disable=protected-access
                command_topics_subscribed,
            )
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
