"""Tests for register models."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from aiomtec2mqtt.register_models import CalculatedRegister, RegisterDefinition, RegisterMap


class TestRegisterDefinition:
    """Test RegisterDefinition model."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        reg = RegisterDefinition(
            address=10100,
            name="test",
            group="BASE",
        )

        assert reg.scale == 1
        assert reg.data_type == "uint16"
        assert reg.writable is False
        assert reg.min_value is None
        assert reg.max_value is None

    def test_empty_name_raises_error(self) -> None:
        """Test empty name raises validation error."""
        with pytest.raises(ValidationError):
            RegisterDefinition(
                address=10100,
                name="",
                group="BASE",
            )

    def test_invalid_address_raises_error(self) -> None:
        """Test invalid address raises validation error."""
        with pytest.raises(ValidationError):
            RegisterDefinition(
                address=-1,  # Invalid
                name="test",
                group="BASE",
            )

        with pytest.raises(ValidationError):
            RegisterDefinition(
                address=70000,  # Too large
                name="test",
                group="BASE",
            )

    def test_invalid_name_raises_error(self) -> None:
        """Test invalid name raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterDefinition(
                address=10100,
                name="invalid name!",  # Special characters
                group="BASE",
            )

        assert "alphanumeric" in str(exc_info.value)

    def test_invalid_range_raises_error(self) -> None:
        """Test invalid range (min >= max) raises error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterDefinition(
                address=10100,
                name="test",
                group="BASE",
                min_value=100,
                max_value=50,  # Invalid: less than min
            )

        assert "greater than" in str(exc_info.value).lower()

    def test_min_max_validation(self) -> None:
        """Test min/max value validation."""
        # Valid range
        reg = RegisterDefinition(
            address=10100,
            name="test",
            group="BASE",
            min_value=0,
            max_value=100,
        )

        assert reg.min_value == 0
        assert reg.max_value == 100

    def test_valid_register_definition(self) -> None:
        """Test creating valid register definition."""
        reg = RegisterDefinition(
            address=10100,
            name="battery_soc",
            unit="%",
            group="BASE",
            scale=1,
        )

        assert reg.address == 10100
        assert reg.name == "battery_soc"
        assert reg.unit == "%"
        assert reg.group == "BASE"
        assert reg.scale == 1


class TestCalculatedRegister:
    """Test CalculatedRegister model."""

    def test_dangerous_formula_raises_error(self) -> None:
        """Test dangerous patterns in formula raise error."""
        dangerous_patterns = [
            "__import__('os')",
            "eval('malicious')",
            "exec('code')",
            "import sys",
            "open('file')",
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(ValidationError) as exc_info:
                CalculatedRegister(
                    name="dangerous",
                    formula=pattern,
                )

            assert "dangerous pattern" in str(exc_info.value).lower()

    def test_default_values(self) -> None:
        """Test default values."""
        calc = CalculatedRegister(
            name="test",
            formula="a + b",
        )

        assert calc.group == "BASE"
        assert calc.dependencies == []
        assert calc.unit is None

    def test_invalid_name_raises_error(self) -> None:
        """Test invalid name raises validation error."""
        with pytest.raises(ValidationError):
            CalculatedRegister(
                name="invalid name!",
                formula="a + b",
            )

    def test_valid_calculated_register(self) -> None:
        """Test creating valid calculated register."""
        calc = CalculatedRegister(
            name="total_power",
            formula="grid_power + solar_power",
            unit="W",
            dependencies=["grid_power", "solar_power"],
        )

        assert calc.name == "total_power"
        assert calc.formula == "grid_power + solar_power"
        assert calc.unit == "W"
        assert calc.dependencies == ["grid_power", "solar_power"]


class TestRegisterMap:
    """Test RegisterMap model."""

    def test_circular_dependency_raises_error(self) -> None:
        """Test circular dependencies raise error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterMap(
                registers={},
                calculated={
                    "calc_a": CalculatedRegister(
                        name="calc_a",
                        formula="calc_b + 1",
                        dependencies=["calc_b"],
                    ),
                    "calc_b": CalculatedRegister(
                        name="calc_b",
                        formula="calc_a + 1",
                        dependencies=["calc_a"],
                    ),
                },
            )

        assert "circular dependency" in str(exc_info.value).lower()

    def test_complex_dependency_chain(self) -> None:
        """Test complex but valid dependency chain."""
        reg_map = RegisterMap(
            registers={
                "10100": RegisterDefinition(
                    address=10100,
                    name="base",
                    group="BASE",
                ),
            },
            calculated={
                "level1": CalculatedRegister(
                    name="level1",
                    formula="base * 2",
                    dependencies=["base"],
                ),
                "level2": CalculatedRegister(
                    name="level2",
                    formula="level1 + 10",
                    dependencies=["level1"],
                ),
                "level3": CalculatedRegister(
                    name="level3",
                    formula="level2 * level1",
                    dependencies=["level2", "level1"],
                ),
            },
        )

        # Should not raise error
        assert len(reg_map.calculated) == 3

    def test_duplicate_names_raises_error(self) -> None:
        """Test duplicate register names raise error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterMap(
                registers={
                    "10100": RegisterDefinition(
                        address=10100,
                        name="same_name",
                        group="BASE",
                    ),
                    "10101": RegisterDefinition(
                        address=10101,
                        name="same_name",  # Duplicate
                        group="BASE",
                    ),
                },
            )

        assert "unique" in str(exc_info.value).lower()

    def test_missing_dependency_raises_error(self) -> None:
        """Test missing dependency raises error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterMap(
                registers={
                    "10100": RegisterDefinition(
                        address=10100,
                        name="battery_soc",
                        group="BASE",
                    ),
                },
                calculated={
                    "bad_calc": CalculatedRegister(
                        name="bad_calc",
                        formula="unknown_register * 2",
                        dependencies=["unknown_register"],  # Doesn't exist
                    ),
                },
            )

        assert "unknown register" in str(exc_info.value).lower()

    def test_self_referencing_dependency_raises_error(self) -> None:
        """Test self-referencing dependency raises error."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterMap(
                registers={},
                calculated={
                    "bad": CalculatedRegister(
                        name="bad",
                        formula="bad + 1",
                        dependencies=["bad"],  # Self-reference
                    ),
                },
            )

        assert "circular dependency" in str(exc_info.value).lower()

    def test_valid_register_map(self) -> None:
        """Test creating valid register map."""
        reg_map = RegisterMap(
            registers={
                "10100": RegisterDefinition(
                    address=10100,
                    name="battery_soc",
                    unit="%",
                    group="BASE",
                ),
                "10101": RegisterDefinition(
                    address=10101,
                    name="battery_voltage",
                    unit="V",
                    group="BASE",
                ),
            },
            calculated={
                "avg_voltage": CalculatedRegister(
                    name="avg_voltage",
                    formula="battery_voltage / 2",
                    dependencies=["battery_voltage"],
                ),
            },
        )

        assert len(reg_map.registers) == 2
        assert len(reg_map.calculated) == 1
