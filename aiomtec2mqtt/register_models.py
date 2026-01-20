"""
Pydantic models for register configuration validation.

This module provides validated models for register definitions, calculated
registers, and Home Assistant configurations using Pydantic v2.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RegisterDataType(str, Enum):
    """Register data types."""

    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    FLOAT32 = "float32"
    STRING = "string"


class RegisterGroup(str, Enum):
    """Register groups for polling intervals."""

    BASE = "BASE"
    EXTENDED = "EXTENDED"
    GRID = "GRID"
    INVERTER = "INVERTER"
    BACKUP = "BACKUP"
    BATTERY = "BATTERY"
    PV = "PV"
    DAY = "DAY"
    TOTAL = "TOTAL"
    STATIC = "STATIC"
    CONFIG = "CONFIG"


class HassDeviceClass(str, Enum):
    """Home Assistant device classes."""

    POWER = "power"
    ENERGY = "energy"
    VOLTAGE = "voltage"
    CURRENT = "current"
    FREQUENCY = "frequency"
    TEMPERATURE = "temperature"
    BATTERY = "battery"
    POWER_FACTOR = "power_factor"


class HassStateClass(str, Enum):
    """Home Assistant state classes."""

    MEASUREMENT = "measurement"
    TOTAL = "total"
    TOTAL_INCREASING = "total_increasing"


class HassConfig(BaseModel):
    """Home Assistant auto-discovery configuration."""

    enabled: bool = Field(default=True, description="Enable HA discovery")
    device_class: HassDeviceClass | None = Field(
        default=None,
        description="HA device class",
    )
    state_class: HassStateClass | None = Field(
        default=None,
        description="HA state class",
    )
    icon: str | None = Field(default=None, description="MDI icon")
    entity_category: str | None = Field(
        default=None,
        description="Entity category (config, diagnostic)",
    )

    model_config = {"use_enum_values": True}


class RegisterDefinition(BaseModel):
    """
    Register definition with validation.

    Validates register addresses, data types, and required fields.
    """

    address: int = Field(ge=0, le=65535, description="Modbus register address")
    name: str = Field(min_length=1, max_length=100, description="Register name")
    unit: str | None = Field(default=None, max_length=20, description="Unit of measurement")
    group: RegisterGroup = Field(description="Register group for polling")
    scale: int | float = Field(default=1, gt=0, description="Scale factor")
    data_type: RegisterDataType = Field(
        default=RegisterDataType.UINT16,
        description="Data type",
    )
    writable: bool = Field(default=False, description="Register is writable")
    min_value: int | float | None = Field(default=None, description="Minimum value")
    max_value: int | float | None = Field(default=None, description="Maximum value")
    hass: HassConfig | None = Field(default=None, description="Home Assistant config")
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Human-readable description",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:  # kwonly: disable
        """Validate name contains only allowed characters."""
        if not v.replace("_", "").replace("-", "").isalnum():
            msg = "Name must contain only alphanumeric characters, underscores, and hyphens"
            raise ValueError(msg)
        return v

    @field_validator("min_value", "max_value")
    @classmethod
    def validate_range(  # kwonly: disable
        cls,
        v: int | float | None,
        info: Any,
    ) -> int | float | None:
        """Validate min/max range."""
        if v is None:
            return v

        # Check that min < max if both are set
        if info.field_name == "max_value":
            data = info.data
            min_val = data.get("min_value")
            if min_val is not None and v <= min_val:
                msg = "max_value must be greater than min_value"
                raise ValueError(msg)

        return v

    model_config = {"use_enum_values": True}


class CalculatedRegister(BaseModel):
    """
    Calculated register definition.

    Defines registers computed from formulas using other register values.
    """

    name: str = Field(min_length=1, max_length=100, description="Register name")
    formula: str = Field(min_length=1, max_length=500, description="Calculation formula")
    unit: str | None = Field(default=None, max_length=20, description="Unit of measurement")
    group: RegisterGroup = Field(
        default=RegisterGroup.BASE,
        description="Register group",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required register names",
    )
    hass: HassConfig | None = Field(default=None, description="Home Assistant config")
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Human-readable description",
    )

    @field_validator("formula")
    @classmethod
    def validate_formula_syntax(cls, v: str) -> str:  # kwonly: disable
        """Validate formula syntax for dangerous patterns."""
        # Check for dangerous patterns
        dangerous = ["__", "import", "eval", "exec", "compile", "open", "file"]
        v_lower = v.lower()
        for pattern in dangerous:
            if pattern in v_lower:
                msg = f"Formula contains dangerous pattern: {pattern}"
                raise ValueError(msg)

        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:  # kwonly: disable
        """Validate name contains only allowed characters."""
        if not v.replace("_", "").replace("-", "").isalnum():
            msg = "Name must contain only alphanumeric characters, underscores, and hyphens"
            raise ValueError(msg)
        return v

    model_config = {"use_enum_values": True}


class RegisterMap(BaseModel):
    """
    Complete register map with validation.

    Contains all register definitions and calculated registers.
    """

    registers: dict[str, RegisterDefinition] = Field(
        default_factory=dict,
        description="Register definitions by address",
    )
    calculated: dict[str, CalculatedRegister] = Field(
        default_factory=dict,
        description="Calculated register definitions",
    )
    version: str = Field(default="1.0", description="Schema version")

    @field_validator("calculated")
    @classmethod
    def validate_calculated_dependencies(  # kwonly: disable
        cls,
        v: dict[str, CalculatedRegister],
        info: Any,
    ) -> dict[str, CalculatedRegister]:
        """Validate calculated register dependencies exist."""
        # Get all available register names
        data = info.data
        registers = data.get("registers", {})
        available_names = {reg.name for reg in registers.values()}
        available_names.update(v.keys())  # Include other calculated registers

        # Check each calculated register's dependencies
        for calc_name, calc_reg in v.items():
            for dep in calc_reg.dependencies:
                if dep not in available_names:
                    msg = f"Calculated register '{calc_name}' depends on unknown register '{dep}'"
                    raise ValueError(msg)

        # Check for circular dependencies
        cls._check_circular_dependencies(calculated=v)

        return v

    @field_validator("registers")
    @classmethod
    def validate_unique_names(  # kwonly: disable
        cls,
        v: dict[str, RegisterDefinition],
    ) -> dict[str, RegisterDefinition]:
        """Validate register names are unique."""
        names = [reg.name for reg in v.values()]
        if len(names) != len(set(names)):
            msg = "Register names must be unique"
            raise ValueError(msg)
        return v

    @staticmethod
    def _check_circular_dependencies(
        *,
        calculated: dict[str, CalculatedRegister],
    ) -> None:
        """Check for circular dependencies in calculated registers."""

        def has_cycle(
            *,
            name: str,
            visited: set[str],
            rec_stack: set[str],
        ) -> bool:
            visited.add(name)
            rec_stack.add(name)

            if name in calculated:
                for dep in calculated[name].dependencies:
                    if dep not in visited:
                        if has_cycle(name=dep, visited=visited, rec_stack=rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(name)
            return False

        visited: set[str] = set()
        for name in calculated:
            if name not in visited and has_cycle(name=name, visited=visited, rec_stack=set()):
                msg = f"Circular dependency detected involving '{name}'"
                raise ValueError(msg)

    model_config = {"use_enum_values": True}
