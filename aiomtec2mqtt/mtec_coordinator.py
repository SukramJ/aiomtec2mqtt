"""
Coordinator/orchestrator for polling M-TEC Energybutler via Modbus and publishing values to MQTT (and Home Assistant discovery when enabled).

This module manages the main lifecycle: configuration load, Modbus connection
handling (including reconnects), scheduling periodic reads for different
register groups, publishing to the configured MQTT topic tree, and graceful
shutdown on OS signals.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
import logging
import threading
import time
from typing import Any, Final

from paho.mqtt import client as paho

from aiomtec2mqtt import hass_int, modbus_client, mqtt_client
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
from aiomtec2mqtt.shutdown import get_shutdown_manager

_LOGGER: Final = logging.getLogger(__name__)

PVDATA_TYPE = dict[str, dict[str, Any] | int | float | str | bool]


class MtecCoordinator:
    """MTEC MQTT Coordinator."""

    def __init__(self) -> None:
        """Initialize the coordinator."""
        config = init_config()
        self._register_map, register_groups = init_register_map()
        self._hass: Final = (
            hass_int.HassIntegration(
                hass_base_topic=config[Config.HASS_BASE_TOPIC],
                mqtt_topic=config[Config.MQTT_TOPIC],
                register_map=self._register_map,
            )
            if config[Config.HASS_ENABLE]
            else None
        )
        self._modbus_client: Final = modbus_client.MTECModbusClient(
            config=config,
            register_map=self._register_map,
            register_groups=register_groups,
        )
        self._mqtt_client: Final = mqtt_client.MqttClient(
            config=config, on_mqtt_message=self._on_mqtt_message, hass=self._hass
        )

        # Cache register lists per group to avoid recomputing on every poll
        self._registers_by_group: dict[RegisterGroup, list[str]] = {}
        try:
            for grp in self._modbus_client.register_groups:
                # Ensure keys are of type RegisterGroup for consistent lookups
                self._registers_by_group[RegisterGroup(grp)] = (
                    self._modbus_client.get_register_list(group=RegisterGroup(grp))
                )
        except Exception:
            # Fallback: compute lazily if anything unexpected happens
            self._registers_by_group = {}

        # Get float format and normalize it (e.g., "{:.3f}" or ":.3f" -> ".3f")
        mqtt_float_format_raw = config.get(Config.MQTT_FLOAT_FORMAT, ".3f")
        # Strip braces and leading colon to get clean format spec
        self._mqtt_float_format: Final[str] = mqtt_float_format_raw.strip("{}").lstrip(":")
        self._mqtt_refresh_config: Final[int] = config.get(
            Config.REFRESH_CONFIG, REFRESH_DEFAULTS[Config.REFRESH_CONFIG]
        )
        self._mqtt_refresh_day: Final[int] = config.get(
            Config.REFRESH_DAY, REFRESH_DEFAULTS[Config.REFRESH_DAY]
        )
        self._mqtt_refresh_now: Final[int] = config.get(
            Config.REFRESH_NOW, REFRESH_DEFAULTS[Config.REFRESH_NOW]
        )
        self._mqtt_refresh_static: Final[int] = config.get(
            Config.REFRESH_STATIC, REFRESH_DEFAULTS[Config.REFRESH_STATIC]
        )
        self._mqtt_refresh_total: Final[int] = config.get(
            Config.REFRESH_TOTAL, REFRESH_DEFAULTS[Config.REFRESH_TOTAL]
        )
        self._mqtt_topic: Final[str] = config[Config.MQTT_TOPIC]
        self._hass_birth_gracetime: Final[int] = config.get(Config.HASS_BIRTH_GRACETIME, 15)
        self._hass_status_topic: Final[str] = f"{config[Config.HASS_BASE_TOPIC]}/status"
        self._hass_birth_timer: threading.Timer | None = None

        if config[Config.DEBUG] is True:
            logging.getLogger().setLevel(level=logging.DEBUG)
        _LOGGER.info("Starting")

    def read_mtec_data(self, *, group: RegisterGroup) -> PVDATA_TYPE:
        """Read data from MTEC modbus."""
        _LOGGER.info("Reading registers for group: %s", group)
        if (registers := self._registers_by_group.get(group)) is None:
            # Lazy compute and cache if not present
            registers = self._modbus_client.get_register_list(group=group)
            self._registers_by_group[group] = registers
        # Only compute current time if needed (for api-date)
        now_str: str | None = None
        data = self._modbus_client.read_modbus_data(registers=registers)
        pvdata: PVDATA_TYPE = {}
        try:  # assign all data  # pylint: disable=too-many-nested-blocks
            RV = Register.VALUE
            RMQTT = Register.MQTT
            RDEV = Register.DEVICE_CLASS
            RVITEMS = Register.VALUE_ITEMS
            rdata = data  # local alias
            reg_map = self._register_map  # local alias to reduce attribute lookups
            for register in registers:
                item = reg_map[register]
                if mqtt_key := item[RMQTT]:
                    if register.isdigit():
                        entry = rdata[register]
                        value = entry[RV]
                        if register == "10011":
                            fw0, fw1 = str(value).split("  ")
                            entry[RV] = f"V{fw0.replace(' ', '.')}-V{fw1.replace(' ', '.')}"
                        elif register == "10008":
                            entry[RV] = _get_equipment_info(value=value)
                        elif item.get(RDEV) == "enum" and (value_items := item.get(RVITEMS)):
                            entry[RV] = _convert_code(value=value, value_items=value_items)

                        pvdata[mqtt_key] = entry
                    else:  # non-numeric registers are deemed to be calculated pseudo-registers
                        # use locals to avoid repeated dict lookups
                        if register == "consumption":
                            pvdata[mqtt_key] = rdata["11016"][RV] - rdata["11000"][RV]
                        elif register == "consumption-day":
                            pvdata[mqtt_key] = (
                                rdata["31005"][RV]
                                + rdata["31001"][RV]
                                + rdata["31004"][RV]
                                - rdata["31000"][RV]
                                - rdata["31003"][RV]
                            )
                        elif register == "autarky-day":
                            cons_day = pvdata.get("consumption_day")
                            pvdata[mqtt_key] = (
                                100 * (1 - (rdata["31001"][RV] / cons_day))
                                if isinstance(cons_day, (float, int)) and float(cons_day) > 0
                                else 0
                            )
                        elif register == "ownconsumption-day":
                            gen_day = rdata["31005"][RV]
                            pvdata[mqtt_key] = (
                                100 * (1 - rdata["31000"][RV] / gen_day) if gen_day > 0 else 0
                            )
                        elif register == "consumption-total":
                            pvdata[mqtt_key] = (
                                rdata["31112"][RV]
                                + rdata["31104"][RV]
                                + rdata["31110"][RV]
                                - rdata["31102"][RV]
                                - rdata["31108"][RV]
                            )
                        elif register == "autarky-total":
                            cons_total = pvdata.get("consumption_total")
                            pvdata[mqtt_key] = (
                                100 * (1 - (rdata["31104"][RV] / cons_total))
                                if isinstance(cons_total, (float, int)) and float(cons_total) > 0
                                else 0
                            )
                        elif register == "ownconsumption-total":
                            gen_total = rdata["31112"][RV]
                            pvdata[mqtt_key] = (
                                100 * (1 - rdata["31102"][RV] / gen_total) if gen_total > 0 else 0
                            )
                        elif register == "api-date":
                            if now_str is None:
                                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            pvdata[mqtt_key] = now_str
                        else:
                            _LOGGER.warning("Unknown calculated pseudo-register: %s", register)

                        # Avoid to report negative values, which might occur in some edge cases
                        val = pvdata[mqtt_key]
                        if isinstance(val, float) and val < 0:
                            pvdata[mqtt_key] = 0

        except Exception as ex:
            _LOGGER.warning("Retrieved Modbus data is incomplete: %s", ex)
            return {}
        return pvdata

    def run(self) -> None:
        """Run the coordinator."""
        next_read_config = datetime.now()
        next_read_day = datetime.now()
        next_read_total = datetime.now()
        next_read_static = datetime.now()
        now_ext_idx = 0
        sec_groups_len = len(SECONDARY_REGISTER_GROUPS)

        self._modbus_client.connect()

        # Initialize
        pv_config: PVDATA_TYPE | None = None
        while not pv_config:
            if not (pv_config := self.read_mtec_data(group=RegisterGroup.STATIC)):
                _LOGGER.warning("Can't retrieve initial config - retry in 10 s")
                time.sleep(10)

        serial_no = pv_config[Register.SERIAL_NO][Register.VALUE]  # type: ignore[index]
        firmware_version = pv_config[Register.FIRMWARE_VERSION][Register.VALUE]  # type: ignore[index]
        equipment_info = pv_config[Register.EQUIPMENT_INFO][Register.VALUE]  # type: ignore[index]
        topic_base = f"{self._mqtt_topic}/{serial_no}"
        if self._hass and not self._hass.is_initialized:
            self._hass.initialize(
                mqtt=self._mqtt_client,
                serial_no=serial_no,
                firmware_version=firmware_version,
                equipment_info=equipment_info,
            )

        # Main loop - exit on shutdown signal only
        shutdown_manager = get_shutdown_manager()
        while not shutdown_manager.should_shutdown():
            # check if modbus is alive and reconnect if necessary
            if self._modbus_client.error_count > 10:
                self._reconnect_modbus()

            now = datetime.now()

            # Now base
            if pvdata := self.read_mtec_data(group=RegisterGroup.BASE):
                self.write_to_mqtt(pvdata=pvdata, topic_base=topic_base, group=RegisterGroup.BASE)

            # Config
            if next_read_config <= now and (
                pvdata := self.read_mtec_data(group=RegisterGroup.CONFIG)
            ):
                self.write_to_mqtt(
                    pvdata=pvdata, topic_base=topic_base, group=RegisterGroup.CONFIG
                )
                next_read_config = now + timedelta(seconds=self._mqtt_refresh_config)

            # Now extended - read groups in a round-robin - one per loop
            if sec_groups_len:
                if (group := SECONDARY_REGISTER_GROUPS.get(now_ext_idx)) and (
                    pvdata := self.read_mtec_data(group=group)
                ):
                    self.write_to_mqtt(pvdata=pvdata, topic_base=topic_base, group=group)

                # advance round-robin index efficiently without magic numbers
                now_ext_idx = (now_ext_idx + 1) % sec_groups_len

            # Day
            if next_read_day <= now and (pvdata := self.read_mtec_data(group=RegisterGroup.DAY)):
                self.write_to_mqtt(pvdata=pvdata, topic_base=topic_base, group=RegisterGroup.DAY)
                next_read_day = now + timedelta(seconds=self._mqtt_refresh_day)

            # Total
            if next_read_total <= now and (
                pvdata := self.read_mtec_data(group=RegisterGroup.TOTAL)
            ):
                self.write_to_mqtt(pvdata=pvdata, topic_base=topic_base, group=RegisterGroup.TOTAL)
                next_read_total = now + timedelta(seconds=self._mqtt_refresh_total)

            # Static
            if next_read_static <= now and (
                pvdata := self.read_mtec_data(group=RegisterGroup.STATIC)
            ):
                self.write_to_mqtt(
                    pvdata=pvdata, topic_base=topic_base, group=RegisterGroup.STATIC
                )
                next_read_static = now + timedelta(seconds=self._mqtt_refresh_static)

            _LOGGER.debug("Sleep %ss", self._mqtt_refresh_now)
            time.sleep(self._mqtt_refresh_now)

    def stop(self) -> None:
        """Stop the coordinator."""
        # clean up
        # if self._hass:
        #    hass.send_unregister_info()
        if self._hass_birth_timer is not None:
            with contextlib.suppress(Exception):
                self._hass_birth_timer.cancel()
            self._hass_birth_timer = None
        self._modbus_client.disconnect()
        self._mqtt_client.stop()
        _LOGGER.info("Stopping clients")

    def write_to_mqtt(self, *, pvdata: PVDATA_TYPE, topic_base: str, group: RegisterGroup) -> None:
        """Write data to MQTT."""
        fmt = self._mqtt_float_format
        publish = self._mqtt_client.publish
        RV = Register.VALUE
        base = f"{topic_base}/{group}"
        for param, data in pvdata.items():
            topic = f"{base}/{param}/state"
            value = data[RV] if isinstance(data, dict) else data

            if isinstance(value, float):
                payload = f"{value:{fmt}}"
            elif isinstance(value, bool):
                payload = "1" if value else "0"
            else:
                payload = str(value)
            publish(topic=topic, payload=payload)

    def _on_mqtt_message(  # kwonly: disable
        self,
        client: Any,
        userdata: Any,
        message: paho.MQTTMessage,
    ) -> None:
        """Handle received message."""
        try:
            msg = message.payload.decode(UTF8)
            if (topic := message.topic) == self._hass_status_topic:
                if msg == "online" and self._hass is not None:
                    gracetime = self._hass_birth_gracetime
                    _LOGGER.info(
                        "Received HASS online message. Scheduling discovery info in %i sec",
                        gracetime,
                    )
                    # Avoid blocking the MQTT network thread; schedule delayed discovery
                    if self._hass_birth_timer is not None:
                        with contextlib.suppress(Exception):
                            self._hass_birth_timer.cancel()
                    self._hass_birth_timer = threading.Timer(
                        interval=gracetime, function=self._send_hass_discovery
                    )
                    self._hass_birth_timer.daemon = True
                    self._hass_birth_timer.start()
                elif msg == "offline":
                    _LOGGER.info("Received HASS offline message.")
            elif (topic_parts := message.topic.split("/")) is not None and len(topic_parts) >= 4:
                register_name = topic_parts[3]
                self._modbus_client.write_register_by_name(name=register_name, value=msg)
            else:
                _LOGGER.warning("Received topic %s is not usable.", topic)
        except Exception as ex:
            _LOGGER.warning("Error while handling MQTT message: %s", ex)

    def _reconnect_modbus(self) -> None:
        """Reconnect to modbus/mqtt."""
        _LOGGER.info("Reconnecting modbus client.")
        self._modbus_client.disconnect()
        time.sleep(10)
        self._modbus_client.connect()
        _LOGGER.info("Reconnected modbus client.")

    def _send_hass_discovery(self) -> None:
        """Send Home Assistant discovery info after grace period (timer callback)."""
        try:
            if self._hass is not None:
                self._hass.send_discovery_info()
        except Exception as ex:  # defensive
            _LOGGER.warning("Failed to send HASS discovery info: %s", ex)
        finally:
            # clear the timer reference
            self._hass_birth_timer = None


def _convert_code(*, value: int | str, value_items: dict[int, str]) -> str:
    """Convert bms fault code register value."""
    if isinstance(value, int):
        return value_items.get(value, "Unknown")

    faults: list[str] = []
    value_no = int(f"0b{value.replace(' ', '')}", 2)
    for no, fault in value_items.items():
        if _has_bit(val=value_no, idx=no):
            faults.append(fault)

    if not faults:
        faults.append("OK")
    return ", ".join(faults)


def _get_equipment_info(*, value: str) -> str:
    """Extract the Equipment info from code."""
    upper, lower = value.split(" ")
    return EQUIPMENT.get(int(upper), {}).get(int(lower), "unknown")


def _has_bit(*, val: int, idx: int) -> bool:
    """Return true if idx bit is set."""
    return (val & (1 << idx)) > 0


# ==========================================
def main() -> None:
    """Start mtec mqtt."""
    # Initialize shutdown manager and register signal handlers
    shutdown_manager = get_shutdown_manager()
    shutdown_manager.register_signal_handlers()

    coordinator = MtecCoordinator()
    coordinator.run()
    coordinator.stop()


if __name__ == "__main__":
    main()
