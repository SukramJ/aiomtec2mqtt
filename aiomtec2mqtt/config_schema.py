"""
Pydantic schema for aiomtec2mqtt configuration.

Centralizes validation for every field in ``config.yaml`` and env-var overrides.
Validation failures produce a single aggregated error message instead of
``KeyError`` deep inside the coordinator.

The module keeps the runtime surface dict-based: ``validate_config()`` returns a
plain dict so callers can continue to use ``config[Config.MODBUS_IP]`` style
access. The Pydantic model is an internal guardrail, not a public type.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aiomtec2mqtt.const import Config

__all__ = ["ConfigSchema", "ConfigValidationError", "validate_config"]


_ALLOWED_FRAMERS: Final[frozenset[str]] = frozenset({"rtu", "socket", "ascii", "tls"})


class ConfigValidationError(ValueError):
    """Raised when ``config.yaml`` fails schema validation."""


class ConfigSchema(BaseModel):
    """Validated configuration model."""

    model_config = ConfigDict(extra="allow", populate_by_name=True, use_enum_values=True)

    # Modbus
    MODBUS_IP: str = Field(min_length=1)
    MODBUS_PORT: int = Field(ge=1, le=65535)
    MODBUS_SLAVE: int = Field(ge=0, le=247)
    MODBUS_TIMEOUT: int = Field(ge=1, le=600)
    MODBUS_FRAMER: str = Field(default="rtu")
    MODBUS_RETRIES: int = Field(default=3, ge=0, le=20)

    # MQTT
    MQTT_SERVER: str = Field(min_length=1)
    MQTT_PORT: int = Field(ge=1, le=65535)
    MQTT_LOGIN: str = Field(default="")
    MQTT_PASSWORD: str = Field(default="")
    MQTT_TOPIC: str = Field(min_length=1)
    MQTT_FLOAT_FORMAT: str = Field(default=".3f")

    # Home Assistant
    HASS_ENABLE: bool = Field(default=False)
    HASS_BASE_TOPIC: str = Field(default="homeassistant")
    HASS_BIRTH_GRACETIME: int = Field(default=15, ge=0, le=600)

    # Refresh intervals (seconds)
    REFRESH_NOW: int = Field(default=10, ge=1, le=3600)
    REFRESH_CONFIG: int = Field(default=30, ge=1, le=3600)
    REFRESH_DAY: int = Field(default=300, ge=1, le=86400)
    REFRESH_STATIC: int = Field(default=3600, ge=1, le=86400)
    REFRESH_TOTAL: int = Field(default=300, ge=1, le=86400)

    # Misc
    DEBUG: bool = Field(default=False)

    @field_validator("MQTT_FLOAT_FORMAT")
    @classmethod
    def _validate_float_format(cls, value: str) -> str:  # kwonly: disable
        # Accept both ``.3f`` and ``{:.3f}`` template styles; coordinator strips braces.
        cleaned = value.strip("{}").lstrip(":")
        try:
            _ = format(1.5, cleaned)
        except (TypeError, ValueError) as err:
            raise ValueError(f"MQTT_FLOAT_FORMAT is not a valid format spec: {value!r}") from err
        return value

    @field_validator("MODBUS_FRAMER")
    @classmethod
    def _validate_framer(cls, value: str) -> str:  # kwonly: disable
        if value not in _ALLOWED_FRAMERS:
            raise ValueError(
                f"MODBUS_FRAMER must be one of {sorted(_ALLOWED_FRAMERS)}, got {value!r}"
            )
        return value


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:  # kwonly: disable
    """Validate ``raw`` and return a dict with defaults applied.

    Extra keys are preserved (``extra="allow"``) so forward-compatible additions
    do not break validation. Keys required by :class:`ConfigSchema` but missing
    from the input raise :class:`ConfigValidationError`.
    """
    try:
        model = ConfigSchema.model_validate(raw)
    except ValidationError as err:
        # Re-raise as a project-specific error so callers do not depend on pydantic.
        raise ConfigValidationError(str(err)) from err

    validated = model.model_dump()
    # Restore enum-keyed access pattern: merge extras and enforce Config-enum keys.
    for key in list(raw.keys()):
        if isinstance(key, Config):
            validated[key.value] = validated.get(str(key), validated.get(key.value))
    return validated
