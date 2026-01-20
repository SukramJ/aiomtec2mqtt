"""Tests for register processors."""

from __future__ import annotations

import pytest

from aiomtec2mqtt.register_processors import (
    DefaultProcessor,
    EnergyProcessor,
    EquipmentProcessor,
    PercentageProcessor,
    PowerProcessor,
    RegisterProcessorRegistry,
    TemperatureProcessor,
)


class TestTemperatureProcessor:
    """Test TemperatureProcessor class."""

    def test_can_process_temperature_by_name(self) -> None:
        """Test can process register with 'temperature' in name."""
        processor = TemperatureProcessor()

        assert processor.can_process(
            register_name="battery_temperature",
            register_info={"unit": "°C"},
        )

    def test_can_process_temperature_by_unit(self) -> None:
        """Test can process register with temperature unit."""
        processor = TemperatureProcessor()

        assert processor.can_process(register_name="sensor_1", register_info={"unit": "°C"})
        assert processor.can_process(register_name="sensor_2", register_info={"unit": "C"})
        assert processor.can_process(register_name="sensor_3", register_info={"unit": "celsius"})

    def test_process_warns_on_unusual_value(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning on unusual temperature value."""
        processor = TemperatureProcessor()

        # Temperature way out of normal range
        processor.process(
            register_name="test_temp",
            raw_value=200,
            register_info={"unit": "°C", "scale": 1},
        )

        assert "unusual value" in caplog.text

    def test_process_with_scale(self) -> None:
        """Test processing temperature with scale."""
        processor = TemperatureProcessor()

        result = processor.process(
            register_name="battery_temperature",
            raw_value=250,
            register_info={"unit": "°C", "scale": 10},
        )

        assert result == 25.0

    def test_process_without_scale(self) -> None:
        """Test processing temperature without scale."""
        processor = TemperatureProcessor()

        result = processor.process(
            register_name="ambient_temperature",
            raw_value=22,
            register_info={"unit": "°C", "scale": 1},
        )

        assert result == 22.0


class TestEquipmentProcessor:
    """Test EquipmentProcessor class."""

    def test_can_process_equipment(self) -> None:
        """Test can process register with 'equipment' in name."""
        processor = EquipmentProcessor()

        assert processor.can_process(register_name="equipment_info", register_info={})
        assert processor.can_process(register_name="equipment_code", register_info={})

    def test_process_known_equipment_code(self) -> None:
        """Test processing known equipment code."""
        processor = EquipmentProcessor()

        # Code 8 is typically an inverter type
        result = processor.process(
            register_name="equipment_code",
            raw_value=8,
            register_info={},
        )

        assert isinstance(result, str)

    def test_process_unknown_equipment_code(self) -> None:
        """Test processing unknown equipment code."""
        processor = EquipmentProcessor()

        result = processor.process(
            register_name="equipment_code",
            raw_value=99999,
            register_info={},
        )

        assert "Unknown" in result
        assert "99999" in result


class TestPercentageProcessor:
    """Test PercentageProcessor class."""

    def test_can_process_by_unit(self) -> None:
        """Test can process by percentage unit."""
        processor = PercentageProcessor()

        assert processor.can_process(register_name="value", register_info={"unit": "%"})
        assert processor.can_process(register_name="value", register_info={"unit": "percent"})

    def test_can_process_soc(self) -> None:
        """Test can process SOC registers."""
        processor = PercentageProcessor()

        assert processor.can_process(register_name="battery_soc", register_info={})
        assert processor.can_process(register_name="soc_value", register_info={})

    def test_process_clamps_to_0(self) -> None:
        """Test percentage is clamped to 0."""
        processor = PercentageProcessor()

        result = processor.process(
            register_name="battery_soc",
            raw_value=-10,
            register_info={"unit": "%", "scale": 1},
        )

        assert result == 0

    def test_process_clamps_to_100(self) -> None:
        """Test percentage is clamped to 100."""
        processor = PercentageProcessor()

        result = processor.process(
            register_name="battery_soc",
            raw_value=150,
            register_info={"unit": "%", "scale": 1},
        )

        assert result == 100

    def test_process_with_scale(self) -> None:
        """Test processing percentage with scale."""
        processor = PercentageProcessor()

        result = processor.process(
            register_name="battery_soc",
            raw_value=850,
            register_info={"unit": "%", "scale": 10},
        )

        assert result == 85


class TestPowerProcessor:
    """Test PowerProcessor class."""

    def test_can_process_by_name(self) -> None:
        """Test can process by 'power' in name."""
        processor = PowerProcessor()

        assert processor.can_process(register_name="grid_power", register_info={})
        assert processor.can_process(register_name="solar_power", register_info={})

    def test_can_process_by_unit(self) -> None:
        """Test can process by power unit."""
        processor = PowerProcessor()

        assert processor.can_process(register_name="value", register_info={"unit": "W"})
        assert processor.can_process(register_name="value", register_info={"unit": "watt"})
        assert processor.can_process(register_name="value", register_info={"unit": "kW"})

    def test_process_with_scale(self) -> None:
        """Test processing power with scale."""
        processor = PowerProcessor()

        result = processor.process(
            register_name="grid_power",
            raw_value=15000,
            register_info={"unit": "W", "scale": 10},
        )

        assert result == 1500.0


class TestEnergyProcessor:
    """Test EnergyProcessor class."""

    def test_can_process_by_name(self) -> None:
        """Test can process by 'energy' in name."""
        processor = EnergyProcessor()

        assert processor.can_process(register_name="daily_energy", register_info={})
        assert processor.can_process(register_name="total_energy", register_info={})

    def test_can_process_by_unit(self) -> None:
        """Test can process by energy unit."""
        processor = EnergyProcessor()

        assert processor.can_process(register_name="value", register_info={"unit": "Wh"})
        assert processor.can_process(register_name="value", register_info={"unit": "kWh"})
        assert processor.can_process(register_name="value", register_info={"unit": "MWh"})

    def test_process_with_scale(self) -> None:
        """Test processing energy with scale."""
        processor = EnergyProcessor()

        result = processor.process(
            register_name="daily_energy",
            raw_value=12345,
            register_info={"unit": "kWh", "scale": 10},
        )

        assert result == 1234.50


class TestDefaultProcessor:
    """Test DefaultProcessor class."""

    def test_can_process_anything(self) -> None:
        """Test default processor accepts anything."""
        processor = DefaultProcessor()

        assert processor.can_process(register_name="anything", register_info={})
        assert processor.can_process(register_name="", register_info={})

    def test_process_with_scale(self) -> None:
        """Test processing with scale."""
        processor = DefaultProcessor()

        result = processor.process(
            register_name="generic",
            raw_value=1000,
            register_info={"scale": 10},
        )

        assert result == 100.0

    def test_process_without_scale(self) -> None:
        """Test processing without scale."""
        processor = DefaultProcessor()

        result = processor.process(
            register_name="generic",
            raw_value=42,
            register_info={"scale": 1},
        )

        assert result == 42


class TestRegisterProcessorRegistry:
    """Test RegisterProcessorRegistry class."""

    def test_initialization_registers_default_processors(self) -> None:
        """Test registry initializes with default processors."""
        registry = RegisterProcessorRegistry()

        # Should have processors registered
        # Test by processing different types
        assert isinstance(
            registry.process(
                register_name="battery_temperature",
                raw_value=250,
                register_info={"unit": "°C", "scale": 10},
            ),
            float,
        )

    def test_process_batch(self) -> None:
        """Test batch processing of multiple registers."""
        registry = RegisterProcessorRegistry()

        values = {
            "battery_soc": 85,
            "battery_temperature": 250,
            "grid_power": 15000,
        }

        register_map = {
            "battery_soc": {"unit": "%", "scale": 1},
            "battery_temperature": {"unit": "°C", "scale": 10},
            "grid_power": {"unit": "W", "scale": 10},
        }

        results = registry.process_batch(values=values, register_map=register_map)

        assert results["battery_soc"] == 85  # Percentage clamped
        assert results["battery_temperature"] == 25.0  # Scaled temp
        assert results["grid_power"] == 1500.0  # Scaled power

    def test_process_batch_handles_missing_metadata(self) -> None:
        """Test batch processing with missing register metadata."""
        registry = RegisterProcessorRegistry()

        values = {
            "known": 100,
            "unknown": 200,
        }

        register_map = {
            "known": {"unit": "W", "scale": 1},
        }

        results = registry.process_batch(values=values, register_map=register_map)

        assert results["known"] == 100.0  # Processed
        assert results["unknown"] == 200  # Raw value returned

    def test_processor_order_matters(self) -> None:
        """Test first matching processor wins."""
        registry = RegisterProcessorRegistry()

        # Temperature processor should match before default
        result = registry.process(
            register_name="battery_temperature",
            raw_value=250,
            register_info={"unit": "°C", "scale": 10},
        )

        # Should be rounded (temperature processor) not just scaled (default)
        assert result == 25.0
        assert isinstance(result, float)

    def test_register_custom_processor(self) -> None:
        """Test registering custom processor."""
        # Create empty registry to avoid default processors
        registry = RegisterProcessorRegistry()

        class CustomProcessor:
            """Custom processor for testing."""

            def can_process(
                self,
                register_name: str,
                register_info: dict,
            ) -> bool:
                """Check if can process."""
                return "custom" in register_name

            def process(
                self,
                register_name: str,
                raw_value: int | float,
                register_info: dict,
            ) -> str:
                """Process value."""
                return f"custom_{raw_value}"

        # Clear default processors and add only custom
        registry._processors.clear()  # noqa: SLF001
        custom = CustomProcessor()
        registry.register_processor(processor=custom)

        result = registry.process(register_name="custom_value", raw_value=123, register_info={})

        assert result == "custom_123"
