"""
aiomtec2mqtt constants.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

CLIENT_ID: Final = "M-TEC-MQTT"
CONFIG_FILE: Final = "config.yaml"
CONFIG_PATH: Final = "aiomtec2mqtt"
CONFIG_ROOT: Final = ".config"
CONFIG_TEMPLATE: Final = "config-template.yaml"
DEFAULT_FRAMER: Final = "rtu"
MTEC_TOPIC_ROOT: Final = "MTEC"
MTEC_PREFIX: Final = "MTEC_"
UTF8: Final = "utf-8"
ENV_XDG_CONFIG_HOME: Final = "XDG_CONFIG_HOME"
ENV_APPDATA: Final = "APPDATA"
FILE_REGISTERS: Final = "registers.yaml"


class Config(StrEnum):
    """enum with config qualifiers."""

    DEBUG = "DEBUG"
    HASS_BASE_TOPIC = "HASS_BASE_TOPIC"
    HASS_BIRTH_GRACETIME = "HASS_BIRTH_GRACETIME"
    HASS_ENABLE = "HASS_ENABLE"
    MODBUS_FRAMER = "MODBUS_FRAMER"
    MODBUS_IP = "MODBUS_IP"
    MODBUS_PORT = "MODBUS_PORT"
    MODBUS_RETRIES = "MODBUS_RETRIES"
    MODBUS_SLAVE = "MODBUS_SLAVE"
    MODBUS_TIMEOUT = "MODBUS_TIMEOUT"
    MQTT_FLOAT_FORMAT = "MQTT_FLOAT_FORMAT"
    MQTT_LOGIN = "MQTT_LOGIN"
    MQTT_PASSWORD = "MQTT_PASSWORD"
    MQTT_PORT = "MQTT_PORT"
    MQTT_SERVER = "MQTT_SERVER"
    MQTT_TOPIC = "MQTT_TOPIC"
    REFRESH_CONFIG = "REFRESH_CONFIG"
    REFRESH_DAY = "REFRESH_DAY"
    REFRESH_NOW = "REFRESH_NOW"
    REFRESH_STATIC = "REFRESH_STATIC"
    REFRESH_TOTAL = "REFRESH_TOTAL"


REFRESH_DEFAULTS: Final = {
    Config.REFRESH_CONFIG: 30,
    Config.REFRESH_DAY: 300,
    Config.REFRESH_NOW: 10,
    Config.REFRESH_STATIC: 3600,
    Config.REFRESH_TOTAL: 300,
}


class HA(StrEnum):
    """Enum with HA qualifiers."""

    COMMAND_TOPIC = "command_topic"
    DEVICE = "device"
    DEVICE_CLASS = "device_class"
    ENABLED_BY_DEFAULT = "enabled_by_default"
    IDENTIFIERS = "identifiers"
    MANUFACTURER = "manufacturer"
    MODE = "mode"
    MODEL = "model"
    MODEL_ID = "model_id"
    NAME = "name"
    OPTIONS = "options"
    PAYLOAD_OFF = "payload_off"
    PAYLOAD_ON = "payload_on"
    PAYLOAD_PRESS = "payload_press"
    SERIAL_NUMBER = "serial_number"
    STATE_CLASS = "state_class"
    STATE_TOPIC = "state_topic"
    UNIQUE_ID = "unique_id"
    SW_VERSION = "sw_version"
    UNIT_OF_MEASUREMENT = "unit_of_measurement"
    VALUE_TEMPLATE = "value_template"


class HAPlatform(StrEnum):
    """Enum with HA platform."""

    BINARY_SENSOR = "binary_sensor"
    NUMBER = "number"
    SELECT = "select"
    SENSOR = "sensor"
    SWITCH = "switch"


class Register(StrEnum):
    """Enum with Register qualifiers."""

    COMPONENT_TYPE = "hass_component_type"
    DEVICE_CLASS = "hass_device_class"
    EQUIPMENT_INFO = "equipment_info"
    FIRMWARE_VERSION = "firmware_version"
    GROUP = "group"
    LENGTH = "length"
    MQTT = "mqtt"
    NAME = "name"
    PAYLOAD_OFF = "hass_payload_off"
    PAYLOAD_ON = "hass_payload_on"
    SCALE = "scale"
    SERIAL_NO = "serial_no"
    STATE_CLASS = "hass_state_class"
    TYPE = "type"
    UNIT = "unit"
    VALUE = "value"
    VALUE_ITEMS = "hass_value_items"
    VALUE_TEMPLATE = "hass_value_template"
    WRITABLE = "writable"


class RegisterGroup(StrEnum):
    """Enum with Register group qualifiers."""

    BACKUP = "now-backup"
    BASE = "now-base"
    BATTERY = "now-battery"
    CONFIG = "config"
    DAY = "day"
    GRID = "now-grid"
    INVERTER = "now-inverter"
    PV = "now-pv"
    TOTAL = "total"
    STATIC = "static"


SECONDARY_REGISTER_GROUPS: Final = {
    0: RegisterGroup.GRID,
    1: RegisterGroup.INVERTER,
    2: RegisterGroup.BACKUP,
    3: RegisterGroup.BATTERY,
    4: RegisterGroup.PV,
}

MANDATORY_PARAMETERS: Final = [Register.NAME]

OPTIONAL_PARAMETERS: Final = {
    Register.LENGTH: None,
    Register.TYPE: None,
    Register.UNIT: "",
    Register.SCALE: 1,
    Register.WRITABLE: False,
    Register.MQTT: None,
    Register.GROUP: None,
}

EQUIPMENT: Final = {
    30: {
        0: "4.0K-25A-3P",
        1: "5.0K-25A-3P",
        2: "6.0K-25A-3P",
        3: "8.0K-25A-3P",
        4: "10K-25A-3P",
        5: "12K-25A-3P",
        6: "10K-40A-3P",
        7: "12K-40A-3P",
        8: "15K-40A-3P",
        9: "20K-40A-3P",
    },
    31: {
        0: "3.0K-30A-1P",
        1: "3.6K-30A-1P",
        2: "4.2K-30A-1P",
        3: "4.6K-30A-1P",
        4: "5.0K-30A-1P",
        5: "6.0K-30A-1P",
        6: "7.0K-30A-1P",
        7: "8.0K-30A-1P",
        8: "3.0K-30A-1P-S",
        9: "3.6K-30A-1P-S",
    },
    32: {
        0: "25K-100A-3P",
        1: "30K-100A-3P",
        2: "36K-100A-3P",
        3: "40K-100A-3P",
        4: "50K-100A-3P",
    },
}
