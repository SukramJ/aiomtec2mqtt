"""
Register processor registry for type conversions and value transformations.

This module provides a registry of processors that can transform raw register
values into meaningful data (e.g., equipment codes to names, temperatures, etc.).

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import logging
from typing import Any, Final

from aiomtec2mqtt.const import EQUIPMENT
from aiomtec2mqtt.protocols import RegisterProcessorProtocol

_LOGGER: Final = logging.getLogger(__name__)


class TemperatureProcessor:
    """Processor for temperature values."""

    def can_process(self, *, register_name: str, register_info: dict[str, Any]) -> bool:
        """Check if this processor can handle the register."""
        unit = register_info.get("unit", "").lower()
        return "temperature" in register_name.lower() or unit in ("°c", "c", "celsius")

    def process(
        self,
        *,
        register_name: str,
        raw_value: int | float,
        register_info: dict[str, Any],
    ) -> float:
        """
        Process temperature value.

        Args:
            register_name: Register name
            raw_value: Raw register value
            register_info: Register metadata

        Returns:
            Processed temperature value

        """
        # Apply scale if present
        scale = register_info.get("scale", 1)
        temp = raw_value / scale if scale > 1 else raw_value

        # Validate reasonable range (-50 to 150°C)
        if not -50 <= temp <= 150:
            _LOGGER.warning(
                "Temperature %s has unusual value: %.1f°C",
                register_name,
                temp,
            )

        return round(temp, 1)


class EquipmentProcessor:
    """Processor for equipment code values."""

    def __init__(self) -> None:
        """Initialize with equipment mapping."""
        # Flatten the nested EQUIPMENT dict for easier lookup
        self._equipment_map: dict[int, str] = {}
        for category_dict in EQUIPMENT.values():
            self._equipment_map.update(category_dict)

    def can_process(self, *, register_name: str, register_info: dict[str, Any]) -> bool:
        """Check if this processor can handle the register."""
        return "equipment" in register_name.lower()

    def process(
        self,
        *,
        register_name: str,
        raw_value: int | float,
        register_info: dict[str, Any],
    ) -> str:
        """
        Process equipment code to name.

        Args:
            register_name: Register name
            raw_value: Raw register value (equipment code)
            register_info: Register metadata

        Returns:
            Equipment name string

        """
        code = int(raw_value)
        equipment_name = self._equipment_map.get(code, f"Unknown({code})")

        _LOGGER.debug(
            "Equipment code %d -> %s",
            code,
            equipment_name,
        )

        return equipment_name


class PercentageProcessor:
    """Processor for percentage values."""

    def can_process(self, *, register_name: str, register_info: dict[str, Any]) -> bool:
        """Check if this processor can handle the register."""
        unit = register_info.get("unit", "").lower()
        return unit in ("%", "percent") or "soc" in register_name.lower()

    def process(
        self,
        *,
        register_name: str,
        raw_value: int | float,
        register_info: dict[str, Any],
    ) -> int:
        """
        Process percentage value.

        Args:
            register_name: Register name
            raw_value: Raw register value
            register_info: Register metadata

        Returns:
            Percentage as integer 0-100

        """
        # Apply scale if present
        scale = register_info.get("scale", 1)
        percent = raw_value / scale if scale > 1 else raw_value

        # Clamp to 0-100
        percent = max(0, min(100, percent))

        return int(percent)


class PowerProcessor:
    """Processor for power values."""

    def can_process(self, *, register_name: str, register_info: dict[str, Any]) -> bool:
        """Check if this processor can handle the register."""
        unit = register_info.get("unit", "").lower()
        return unit in ("w", "watt", "kw", "kilowatt") or "power" in register_name.lower()

    def process(
        self,
        *,
        register_name: str,
        raw_value: int | float,
        register_info: dict[str, Any],
    ) -> float:
        """
        Process power value.

        Args:
            register_name: Register name
            raw_value: Raw register value
            register_info: Register metadata

        Returns:
            Power value in watts

        """
        # Apply scale if present
        scale = register_info.get("scale", 1)
        power = raw_value / scale if scale > 1 else raw_value

        # Round to 1 decimal place
        return round(power, 1)


class EnergyProcessor:
    """Processor for energy values."""

    def can_process(self, *, register_name: str, register_info: dict[str, Any]) -> bool:
        """Check if this processor can handle the register."""
        unit = register_info.get("unit", "").lower()
        return unit in ("wh", "kwh", "mwh") or "energy" in register_name.lower()

    def process(
        self,
        *,
        register_name: str,
        raw_value: int | float,
        register_info: dict[str, Any],
    ) -> float:
        """
        Process energy value.

        Args:
            register_name: Register name
            raw_value: Raw register value
            register_info: Register metadata

        Returns:
            Energy value in kWh

        """
        # Apply scale if present
        scale = register_info.get("scale", 1)
        energy = raw_value / scale if scale > 1 else raw_value

        # Round to 2 decimal places
        return round(energy, 2)


class DefaultProcessor:
    """Default processor for generic values."""

    def can_process(self, *, register_name: str, register_info: dict[str, Any]) -> bool:
        """Return True for all registers as fallback processor."""
        return True

    def process(
        self,
        *,
        register_name: str,
        raw_value: int | float,
        register_info: dict[str, Any],
    ) -> int | float:
        """
        Process value with scale only.

        Args:
            register_name: Register name
            raw_value: Raw register value
            register_info: Register metadata

        Returns:
            Scaled value

        """
        # Apply scale if present
        scale = register_info.get("scale", 1)
        return raw_value / scale if scale > 1 else raw_value


class RegisterProcessorRegistry:
    """
    Registry for register processors.

    Manages multiple processors and dispatches to the appropriate one based
    on register metadata.
    """

    def __init__(self) -> None:
        """Initialize the registry with default processors."""
        self._processors: list[RegisterProcessorProtocol] = []

        # Register default processors (order matters - first match wins)
        self.register_processor(processor=EquipmentProcessor())
        self.register_processor(processor=TemperatureProcessor())
        self.register_processor(processor=PercentageProcessor())
        self.register_processor(processor=PowerProcessor())
        self.register_processor(processor=EnergyProcessor())
        self.register_processor(processor=DefaultProcessor())  # Fallback

        _LOGGER.debug("Initialized with %d processors", len(self._processors))

    def process(
        self,
        *,
        register_name: str,
        raw_value: int | float,
        register_info: dict[str, Any],
    ) -> Any:
        """
        Process register value using appropriate processor.

        Args:
            register_name: Register name
            raw_value: Raw register value
            register_info: Register metadata

        Returns:
            Processed value

        """
        # Find first matching processor
        for processor in self._processors:
            if processor.can_process(register_name=register_name, register_info=register_info):
                try:
                    return processor.process(
                        register_name=register_name,
                        raw_value=raw_value,
                        register_info=register_info,
                    )
                except Exception as ex:
                    _LOGGER.error(
                        "Processor %s failed for %s: %s",
                        type(processor).__name__,
                        register_name,
                        ex,
                    )
                    # Fall through to next processor
                    continue

        # Should never reach here due to DefaultProcessor
        return raw_value

    def process_batch(
        self,
        *,
        values: dict[str, int | float],
        register_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Process multiple register values.

        Args:
            values: Dictionary of register name -> raw value
            register_map: Register metadata map

        Returns:
            Dictionary of register name -> processed value

        """
        results: dict[str, Any] = {}

        for name, raw_value in values.items():
            if name in register_map:
                register_info = register_map[name]
                results[name] = self.process(
                    register_name=name, raw_value=raw_value, register_info=register_info
                )
            else:
                # No metadata, return raw
                results[name] = raw_value

        return results

    def register_processor(self, *, processor: RegisterProcessorProtocol) -> None:
        """
        Register a new processor.

        Args:
            processor: Processor instance

        """
        self._processors.append(processor)
        _LOGGER.debug("Registered processor: %s", type(processor).__name__)
