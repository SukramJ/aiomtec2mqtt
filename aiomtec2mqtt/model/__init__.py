"""
Model subpackage — register + configuration Pydantic models.

Re-exports validated data models under a single namespace. See
:mod:`aiomtec2mqtt.register_models`, :mod:`aiomtec2mqtt.register_processors`,
:mod:`aiomtec2mqtt.formula_evaluator`, and :mod:`aiomtec2mqtt.config_schema`
for the authoritative definitions.

(c) 2026 by SukramJ
"""

from __future__ import annotations

from aiomtec2mqtt.config_schema import ConfigSchema, ConfigValidationError, validate_config
from aiomtec2mqtt.coordinator_config import CoordinatorConfig, CoordinatorConfigBuilder
from aiomtec2mqtt.formula_evaluator import FormulaEvaluator
from aiomtec2mqtt.register_models import (
    CalculatedRegister,
    HassConfig,
    HassDeviceClass,
    HassStateClass,
    RegisterDataType,
    RegisterDefinition,
    RegisterGroup,
    RegisterMap,
)
from aiomtec2mqtt.register_processors import (
    DefaultProcessor,
    EnergyProcessor,
    EquipmentProcessor,
    PercentageProcessor,
    PowerProcessor,
    RegisterProcessorRegistry,
    TemperatureProcessor,
)

__all__ = [
    "CalculatedRegister",
    "ConfigSchema",
    "ConfigValidationError",
    "CoordinatorConfig",
    "CoordinatorConfigBuilder",
    "DefaultProcessor",
    "EnergyProcessor",
    "EquipmentProcessor",
    "FormulaEvaluator",
    "HassConfig",
    "HassDeviceClass",
    "HassStateClass",
    "PercentageProcessor",
    "PowerProcessor",
    "RegisterDataType",
    "RegisterDefinition",
    "RegisterGroup",
    "RegisterMap",
    "RegisterProcessorRegistry",
    "TemperatureProcessor",
    "validate_config",
]
