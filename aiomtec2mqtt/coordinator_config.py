"""
Builder-pattern configuration for :class:`AsyncMtecCoordinator`.

This module exposes a typed, field-based config object plus a fluent builder.
Historically, the coordinator consumed a plain ``dict[str, Any]`` populated by
YAML + env-var overrides. That works for YAML-driven deployments but is
awkward for:

- programmatic setups (tests, integrations, scripts)
- IDE autocompletion
- partial overrides where only one field should change

``CoordinatorConfig`` gives callers a concrete type with validated fields,
while still producing the legacy dict via :meth:`CoordinatorConfig.as_dict`,
so the coordinator can stay dict-based internally until it is refactored.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from aiomtec2mqtt.config_schema import validate_config
from aiomtec2mqtt.const import Config

__all__ = ["CoordinatorConfig", "CoordinatorConfigBuilder"]


@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
    """
    Immutable, validated configuration bundle for :class:`AsyncMtecCoordinator`.

    Prefer :class:`CoordinatorConfigBuilder` or :meth:`from_dict` over the
    direct constructor — they run schema validation and apply defaults.
    """

    modbus_ip: str
    modbus_port: int
    modbus_slave: int
    modbus_timeout: int
    modbus_framer: str
    modbus_retries: int
    mqtt_server: str
    mqtt_port: int
    mqtt_login: str
    mqtt_password: str
    mqtt_topic: str
    mqtt_float_format: str
    hass_enable: bool
    hass_base_topic: str
    hass_birth_gracetime: int
    refresh_now: int
    refresh_config: int
    refresh_day: int
    refresh_static: int
    refresh_total: int
    debug: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:  # kwonly: disable
        """Build a :class:`CoordinatorConfig` from a raw (or validated) dict."""
        validated = validate_config(raw)
        return cls(
            modbus_ip=validated["MODBUS_IP"],
            modbus_port=validated["MODBUS_PORT"],
            modbus_slave=validated["MODBUS_SLAVE"],
            modbus_timeout=validated["MODBUS_TIMEOUT"],
            modbus_framer=validated["MODBUS_FRAMER"],
            modbus_retries=validated["MODBUS_RETRIES"],
            mqtt_server=validated["MQTT_SERVER"],
            mqtt_port=validated["MQTT_PORT"],
            mqtt_login=validated["MQTT_LOGIN"],
            mqtt_password=validated["MQTT_PASSWORD"],
            mqtt_topic=validated["MQTT_TOPIC"],
            mqtt_float_format=validated["MQTT_FLOAT_FORMAT"],
            hass_enable=validated["HASS_ENABLE"],
            hass_base_topic=validated["HASS_BASE_TOPIC"],
            hass_birth_gracetime=validated["HASS_BIRTH_GRACETIME"],
            refresh_now=validated["REFRESH_NOW"],
            refresh_config=validated["REFRESH_CONFIG"],
            refresh_day=validated["REFRESH_DAY"],
            refresh_static=validated["REFRESH_STATIC"],
            refresh_total=validated["REFRESH_TOTAL"],
            debug=validated["DEBUG"],
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise to the legacy ``dict[str, Any]`` shape (``Config``-enum keyed)."""
        return {
            Config.MODBUS_IP: self.modbus_ip,
            Config.MODBUS_PORT: self.modbus_port,
            Config.MODBUS_SLAVE: self.modbus_slave,
            Config.MODBUS_TIMEOUT: self.modbus_timeout,
            Config.MODBUS_FRAMER: self.modbus_framer,
            Config.MODBUS_RETRIES: self.modbus_retries,
            Config.MQTT_SERVER: self.mqtt_server,
            Config.MQTT_PORT: self.mqtt_port,
            Config.MQTT_LOGIN: self.mqtt_login,
            Config.MQTT_PASSWORD: self.mqtt_password,
            Config.MQTT_TOPIC: self.mqtt_topic,
            Config.MQTT_FLOAT_FORMAT: self.mqtt_float_format,
            Config.HASS_ENABLE: self.hass_enable,
            Config.HASS_BASE_TOPIC: self.hass_base_topic,
            Config.HASS_BIRTH_GRACETIME: self.hass_birth_gracetime,
            Config.REFRESH_NOW: self.refresh_now,
            Config.REFRESH_CONFIG: self.refresh_config,
            Config.REFRESH_DAY: self.refresh_day,
            Config.REFRESH_STATIC: self.refresh_static,
            Config.REFRESH_TOTAL: self.refresh_total,
            Config.DEBUG: self.debug,
        }


class CoordinatorConfigBuilder:
    """
    Fluent builder for :class:`CoordinatorConfig`.

    Instances start empty. Each ``with_*`` method returns the builder so calls
    can be chained. :meth:`build` runs full schema validation and returns an
    immutable :class:`CoordinatorConfig`. If a seed dict is provided via
    :meth:`from_dict` or :meth:`from_env`, its values are merged in; subsequent
    ``with_*`` calls take precedence.

    Example:
        >>> cfg = (
        ...     CoordinatorConfigBuilder()
        ...     .with_modbus(ip="192.168.1.20", port=502, slave=1, timeout=5)
        ...     .with_mqtt(server="localhost", port=1883, topic="MTEC")
        ...     .build()
        ... )
    """

    def __init__(self, seed: dict[str, Any] | None = None) -> None:  # kwonly: disable
        """Start with an optional ``seed`` dict (typically validated YAML)."""
        self._state: dict[str, Any] = dict(seed) if seed else {}

    @classmethod
    def from_config(cls, cfg: CoordinatorConfig) -> CoordinatorConfigBuilder:  # kwonly: disable
        """Start a builder pre-populated with values from an existing config."""
        return cls(seed={str(k): v for k, v in cfg.as_dict().items()})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CoordinatorConfigBuilder:  # kwonly: disable
        """Start a builder pre-populated with values from ``raw``."""
        return cls(seed=raw)

    def build(self) -> CoordinatorConfig:
        """Validate accumulated state and return an immutable config.

        Raises:
            ConfigValidationError: if required fields are missing or any field
                violates its schema constraints.
        """
        return CoordinatorConfig.from_dict(self._state)

    def with_debug(self, enabled: bool) -> Self:  # kwonly: disable
        """Toggle debug mode."""
        self._state["DEBUG"] = enabled
        return self

    def with_home_assistant(
        self,
        *,
        enabled: bool | None = None,
        base_topic: str | None = None,
        birth_gracetime: int | None = None,
    ) -> Self:
        """Configure the Home Assistant integration."""
        self._maybe_set("HASS_ENABLE", enabled)
        self._maybe_set("HASS_BASE_TOPIC", base_topic)
        self._maybe_set("HASS_BIRTH_GRACETIME", birth_gracetime)
        return self

    def with_modbus(
        self,
        *,
        ip: str | None = None,
        port: int | None = None,
        slave: int | None = None,
        timeout: int | None = None,
        framer: str | None = None,
        retries: int | None = None,
    ) -> Self:
        """Configure the Modbus client."""
        self._maybe_set("MODBUS_IP", ip)
        self._maybe_set("MODBUS_PORT", port)
        self._maybe_set("MODBUS_SLAVE", slave)
        self._maybe_set("MODBUS_TIMEOUT", timeout)
        self._maybe_set("MODBUS_FRAMER", framer)
        self._maybe_set("MODBUS_RETRIES", retries)
        return self

    def with_mqtt(
        self,
        *,
        server: str | None = None,
        port: int | None = None,
        login: str | None = None,
        password: str | None = None,
        topic: str | None = None,
        float_format: str | None = None,
    ) -> Self:
        """Configure the MQTT client."""
        self._maybe_set("MQTT_SERVER", server)
        self._maybe_set("MQTT_PORT", port)
        self._maybe_set("MQTT_LOGIN", login)
        self._maybe_set("MQTT_PASSWORD", password)
        self._maybe_set("MQTT_TOPIC", topic)
        self._maybe_set("MQTT_FLOAT_FORMAT", float_format)
        return self

    def with_refresh(
        self,
        *,
        now: int | None = None,
        config: int | None = None,
        day: int | None = None,
        static: int | None = None,
        total: int | None = None,
    ) -> Self:
        """Configure polling intervals (seconds)."""
        self._maybe_set("REFRESH_NOW", now)
        self._maybe_set("REFRESH_CONFIG", config)
        self._maybe_set("REFRESH_DAY", day)
        self._maybe_set("REFRESH_STATIC", static)
        self._maybe_set("REFRESH_TOTAL", total)
        return self

    def _maybe_set(self, key: str, value: Any) -> None:  # kwonly: disable
        if value is not None:
            self._state[key] = value
