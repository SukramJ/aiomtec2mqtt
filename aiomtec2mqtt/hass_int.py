"""
Home Assistant MQTT discovery integration.

Constructs and publishes MQTT discovery payloads for sensors, binary sensors,
number/select/switch entities, and optional custom automations. It uses the
register map to determine which entities should be exposed and subscribes to
command topics for controllable entities.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from aiomtec2mqtt import mqtt_client
from aiomtec2mqtt.const import HA, MTEC_PREFIX, HAPlatform, Register

_LOGGER: Final = logging.getLogger(__name__)


class HassIntegration:
    """HA integration."""

    # Custom automations
    # Tuple format: (name, unique_id, payload_press)
    buttons: list[tuple[str, str, str]] = [
        # ("Set general mode", "MTEC_load_battery_btn", "load_battery_from_grid"),
    ]

    def __init__(
        self, *, hass_base_topic: str, mqtt_topic: str, register_map: dict[str, dict[str, Any]]
    ) -> None:
        """Init hass integration."""
        self._hass_base_topic: Final = hass_base_topic
        self._mqtt_topic: Final = mqtt_topic
        self._register_map: Final = register_map
        self._mqtt: mqtt_client.MqttClient | None = None
        self._serial_no: str | None = None
        self._is_initialized = False
        # Store: (config_topic, serialized_payload, command_topic_or_none)
        self._devices_array: Final[list[tuple[str, str, str | None]]] = []
        self._device_info: dict[str, Any] = {}

    @property
    def is_initialized(self) -> bool:
        """Return True if hass integration initialized."""
        return self._is_initialized

    def initialize(
        self,
        *,
        mqtt: mqtt_client.MqttClient | None,
        serial_no: str,
        firmware_version: str,
        equipment_info: str,
    ) -> None:
        """
        Initialize.

        Args:
            mqtt: MQTT client (can be None if caller will publish manually)
            serial_no: Device serial number
            firmware_version: Firmware version
            equipment_info: Equipment information

        """
        self._mqtt = mqtt
        self._serial_no = serial_no
        self._device_info = {
            HA.IDENTIFIERS: [serial_no],
            HA.MANUFACTURER: "M-TEC",
            HA.MODEL: "Energy-Butler",
            HA.MODEL_ID: equipment_info,
            HA.NAME: "MTEC EnergyButler",
            HA.SERIAL_NUMBER: serial_no,
            HA.SW_VERSION: firmware_version,
        }
        self._devices_array.clear()
        self._build_devices_array()
        self._build_automation_array()
        # Only auto-send if mqtt client is provided
        if self._mqtt is not None:
            self.send_discovery_info()
        self._is_initialized = True

    def send_discovery_info(self) -> None:
        """Send discovery info."""
        if self._mqtt is None:
            _LOGGER.warning("Cannot send discovery info: MQTT client is None")
            return

        _LOGGER.info("Sending home assistant discovery info")
        publish = self._mqtt.publish
        subscribe = self._mqtt.subscribe_to_topic
        for topic, payload_str, command_topic in self._devices_array:
            publish(topic=topic, payload=payload_str, retain=True)
            if command_topic:
                subscribe(topic=command_topic)

    def send_unregister_info(self) -> None:
        """Send unregister info."""
        if self._mqtt is None:
            _LOGGER.warning("Cannot send unregister info: MQTT client is None")
            return

        _LOGGER.info("Sending info to unregister from home assistant")
        for topic, _, _ in self._devices_array:
            self._mqtt.publish(topic=topic, payload="")

    def _append_binary_sensor(self, *, item: dict[str, Any]) -> None:
        name = item[Register.NAME]
        group = item[Register.GROUP]
        mqtt = item[Register.MQTT]
        unique_id = f"{MTEC_PREFIX}{mqtt}"
        state_topic = f"{self._mqtt_topic}/{self._serial_no}/{group}/{mqtt}/state"

        data_item = {
            HA.DEVICE: self._device_info,
            HA.ENABLED_BY_DEFAULT: True,
            HA.NAME: name,
            HA.STATE_TOPIC: state_topic,
            HA.UNIQUE_ID: unique_id,
        }

        if hass_device_class := item.get(Register.DEVICE_CLASS):
            data_item[HA.DEVICE_CLASS] = hass_device_class
        if hass_payload_on := item.get(Register.PAYLOAD_ON):
            data_item[HA.PAYLOAD_ON] = hass_payload_on
        if hass_payload_off := item.get(Register.PAYLOAD_OFF):
            data_item[HA.PAYLOAD_OFF] = hass_payload_off

        topic = f"{self._hass_base_topic}/{HAPlatform.BINARY_SENSOR}/{unique_id}/config"
        self._devices_array.append((topic, json.dumps(data_item), None))

    def _append_number(self, *, item: dict[str, Any]) -> None:
        group = item[Register.GROUP]
        mqtt = item[Register.MQTT]
        name = item[Register.NAME]
        unit = item[Register.UNIT]
        unique_id = f"{MTEC_PREFIX}{mqtt}"
        mtec_topic = f"{self._mqtt_topic}/{self._serial_no}/{group}/{mqtt}"
        command_topic = f"{mtec_topic}/set"
        state_topic = f"{mtec_topic}/state"
        data_item = {
            HA.COMMAND_TOPIC: command_topic,
            HA.DEVICE: self._device_info,
            HA.ENABLED_BY_DEFAULT: False,
            HA.MODE: "box",
            HA.NAME: name,
            HA.STATE_TOPIC: state_topic,
            HA.UNIQUE_ID: unique_id,
            HA.UNIT_OF_MEASUREMENT: unit,
        }

        if hass_device_class := item.get(Register.DEVICE_CLASS):
            data_item[HA.DEVICE_CLASS] = hass_device_class

        topic = f"{self._hass_base_topic}/{HAPlatform.NUMBER}/{unique_id}/config"
        self._devices_array.append((topic, json.dumps(data_item), command_topic))

    def _append_select(self, *, item: dict[str, Any]) -> None:
        options = item[Register.VALUE_ITEMS]
        group = item[Register.GROUP]
        mqtt = item[Register.MQTT]
        name = item[Register.NAME]
        unique_id = f"{MTEC_PREFIX}{mqtt}"
        mtec_topic = f"{self._mqtt_topic}/{self._serial_no}/{group}/{mqtt}"
        command_topic = f"{mtec_topic}/set"
        state_topic = f"{mtec_topic}/state"
        data_item = {
            HA.COMMAND_TOPIC: command_topic,
            HA.DEVICE: self._device_info,
            HA.ENABLED_BY_DEFAULT: False,
            HA.NAME: name,
            HA.OPTIONS: list(options.values()),
            HA.STATE_TOPIC: state_topic,
            HA.UNIQUE_ID: unique_id,
        }

        topic = f"{self._hass_base_topic}/{HAPlatform.SELECT}/{unique_id}/config"
        self._devices_array.append((topic, json.dumps(data_item), command_topic))

    def _append_sensor(self, *, item: dict[str, Any]) -> None:
        name = item[Register.NAME]
        group = item[Register.GROUP]
        mqtt = item[Register.MQTT]
        unit = item.get(Register.UNIT, "")
        unique_id = f"{MTEC_PREFIX}{mqtt}"
        state_topic = f"{self._mqtt_topic}/{self._serial_no}/{group}/{mqtt}/state"

        data_item = {
            HA.DEVICE: self._device_info,
            HA.ENABLED_BY_DEFAULT: True,
            HA.NAME: name,
            HA.STATE_TOPIC: state_topic,
            HA.UNIQUE_ID: unique_id,
            HA.UNIT_OF_MEASUREMENT: unit,
        }
        if hass_device_class := item.get(Register.DEVICE_CLASS):
            data_item[HA.DEVICE_CLASS] = hass_device_class
        if hass_value_template := item.get(Register.VALUE_TEMPLATE):
            data_item[HA.VALUE_TEMPLATE] = hass_value_template
        if hass_state_class := item.get(Register.STATE_CLASS):
            data_item[HA.STATE_CLASS] = hass_state_class

        topic = f"{self._hass_base_topic}/{HAPlatform.SENSOR}/{unique_id}/config"
        self._devices_array.append((topic, json.dumps(data_item), None))

    def _append_switch(self, *, item: dict[str, Any]) -> None:
        group = item[Register.GROUP]
        mqtt = item[Register.MQTT]
        name = item[Register.NAME]
        unique_id = f"{MTEC_PREFIX}{mqtt}"
        mtec_topic = f"{self._mqtt_topic}/{self._serial_no}/{group}/{mqtt}"
        command_topic = f"{mtec_topic}/set"
        state_topic = f"{mtec_topic}/state"
        data_item = {
            HA.COMMAND_TOPIC: command_topic,
            HA.DEVICE: self._device_info,
            HA.ENABLED_BY_DEFAULT: False,
            HA.NAME: name,
            HA.STATE_TOPIC: state_topic,
            HA.UNIQUE_ID: unique_id,
        }

        if hass_device_class := item.get(Register.DEVICE_CLASS):
            data_item[HA.DEVICE_CLASS] = hass_device_class
        if hass_payload_on := item.get(Register.PAYLOAD_ON):
            data_item[HA.PAYLOAD_ON] = hass_payload_on
        if hass_payload_off := item.get(Register.PAYLOAD_OFF):
            data_item[HA.PAYLOAD_OFF] = hass_payload_off

        topic = f"{self._hass_base_topic}/{HAPlatform.SWITCH}/{unique_id}/config"
        self._devices_array.append((topic, json.dumps(data_item), command_topic))

    def _build_automation_array(self) -> None:
        # Buttons
        for name, unique_id, payload_press in self.buttons:
            command_topic = f"{self._mqtt_topic}/{self._serial_no}/automations/command"
            data_item = {
                HA.COMMAND_TOPIC: command_topic,
                HA.DEVICE: self._device_info,
                HA.NAME: name,
                HA.PAYLOAD_PRESS: payload_press,
                HA.UNIQUE_ID: unique_id,
            }
            topic = f"{self._hass_base_topic}/button/{unique_id}/config"
            self._devices_array.append((topic, json.dumps(data_item), command_topic))

    def _build_devices_array(self) -> None:
        """Build discovery data for devices."""
        # Keys that indicate HA exposure when present in register config
        hass_keys = (
            Register.COMPONENT_TYPE,
            Register.DEVICE_CLASS,
            Register.PAYLOAD_OFF,
            Register.PAYLOAD_ON,
            Register.STATE_CLASS,
            Register.VALUE_ITEMS,
            Register.VALUE_TEMPLATE,
        )
        for item in self._register_map.values():
            # Do registration if there is at least one specific hass_* config entry
            do_hass_registration = any(k in item for k in hass_keys)

            if item[Register.GROUP] and do_hass_registration:
                component_type = item.get(Register.COMPONENT_TYPE, HAPlatform.SENSOR)
                if component_type == HAPlatform.SENSOR:
                    self._append_sensor(item=item)
                elif component_type == HAPlatform.BINARY_SENSOR:
                    self._append_binary_sensor(item=item)
                elif component_type == HAPlatform.NUMBER:
                    self._append_number(item=item)
                    self._append_sensor(item=item)
                elif component_type == HAPlatform.SELECT:
                    self._append_select(item=item)
                    self._append_sensor(item=item)
                elif component_type == HAPlatform.SWITCH:
                    self._append_switch(item=item)
                    self._append_binary_sensor(item=item)
