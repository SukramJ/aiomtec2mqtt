"""Tests for formula evaluator."""

from __future__ import annotations

import pytest

from aiomtec2mqtt.formula_evaluator import (
    CalculatedRegisterProcessor,
    FormulaEvaluationError,
    FormulaEvaluator,
)


class TestFormulaEvaluator:
    """Test FormulaEvaluator class."""

    def test_evaluate_abs_function(self) -> None:
        """Test abs() function."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="abs(a)", context={"a": -15})

        assert result == 15

    def test_evaluate_complex_expression(self) -> None:
        """Test complex mathematical expression."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(
            formula="(a + b) * c / d", context={"a": 10, "b": 20, "c": 3, "d": 2}
        )

        assert result == 45.0

    def test_evaluate_division(self) -> None:
        """Test division formula."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="a / b", context={"a": 100, "b": 4})

        assert result == 25.0

    def test_evaluate_division_by_zero_raises_error(self) -> None:
        """Test division by zero raises error."""
        evaluator = FormulaEvaluator()

        with pytest.raises(FormulaEvaluationError):
            evaluator.evaluate(formula="a / b", context={"a": 10, "b": 0})

    def test_evaluate_invalid_syntax_raises_error(self) -> None:
        """Test invalid syntax raises error."""
        evaluator = FormulaEvaluator()

        with pytest.raises(FormulaEvaluationError):
            evaluator.evaluate(formula="a +* b", context={"a": 10, "b": 20})

    def test_evaluate_max_function(self) -> None:
        """Test max() function."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="max(a, b, c)", context={"a": 10, "b": 5, "c": 15})

        assert result == 15

    def test_evaluate_min_function(self) -> None:
        """Test min() function."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="min(a, b, c)", context={"a": 10, "b": 5, "c": 15})

        assert result == 5

    def test_evaluate_modulo(self) -> None:
        """Test modulo operation."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="a % b", context={"a": 17, "b": 5})

        assert result == 2

    def test_evaluate_multiplication(self) -> None:
        """Test multiplication formula."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="a * b", context={"a": 5, "b": 7})

        assert result == 35

    def test_evaluate_nested_functions(self) -> None:
        """Test nested function calls."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="abs(min(a, b))", context={"a": -10, "b": -5})

        assert result == 10

    def test_evaluate_power(self) -> None:
        """Test power operation."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="a ** b", context={"a": 2, "b": 3})

        assert result == 8

    def test_evaluate_round_function(self) -> None:
        """Test round() function."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="round(a, 2)", context={"a": 3.14159})

        assert result == 3.14

    def test_evaluate_simple_addition(self) -> None:
        """Test simple addition formula."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="a + b", context={"a": 10, "b": 20})

        assert result == 30

    def test_evaluate_simple_subtraction(self) -> None:
        """Test simple subtraction formula."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="a - b", context={"a": 100, "b": 30})

        assert result == 70

    def test_evaluate_unary_minus(self) -> None:
        """Test unary minus operation."""
        evaluator = FormulaEvaluator()
        result = evaluator.evaluate(formula="-a", context={"a": 42})

        assert result == -42

    def test_evaluate_undefined_variable_raises_error(self) -> None:
        """Test undefined variable raises error."""
        evaluator = FormulaEvaluator()

        with pytest.raises(FormulaEvaluationError) as exc_info:
            evaluator.evaluate(formula="a + b", context={"a": 10})

        assert "Undefined variable" in str(exc_info.value)

    def test_get_dependencies(self) -> None:
        """Test extracting dependencies from formula."""
        evaluator = FormulaEvaluator()

        deps = evaluator.get_dependencies(formula="a + b * c")
        assert deps == {"a", "b", "c"}

        deps = evaluator.get_dependencies(formula="abs(x - y)")
        assert deps == {"x", "y"}

        deps = evaluator.get_dependencies(formula="(power + voltage) / 1000")
        assert deps == {"power", "voltage"}

    def test_get_dependencies_with_functions(self) -> None:
        """Test dependencies don't include function names."""
        evaluator = FormulaEvaluator()

        deps = evaluator.get_dependencies(formula="min(a, b) + max(c, d)")
        assert deps == {"a", "b", "c", "d"}
        assert "min" not in deps
        assert "max" not in deps

    def test_validate_formula_invalid_syntax(self) -> None:
        """Test validating invalid syntax."""
        evaluator = FormulaEvaluator()

        with pytest.raises(FormulaEvaluationError):
            evaluator.validate_formula(formula="a + (b * ")

    def test_validate_formula_valid(self) -> None:
        """Test validating valid formula."""
        evaluator = FormulaEvaluator()

        assert evaluator.validate_formula(formula="a + b * c")
        assert evaluator.validate_formula(formula="abs(a - b)")
        assert evaluator.validate_formula(formula="(a + b) / max(c, d)")


class TestCalculatedRegisterProcessor:
    """Test CalculatedRegisterProcessor class."""

    def test_add_calculated_register(self) -> None:
        """Test adding calculated register."""
        processor = CalculatedRegisterProcessor()

        processor.add_calculated_register(
            name="total_power",
            formula="grid_power + solar_power",
            dependencies=["grid_power", "solar_power"],
        )

    def test_add_invalid_formula_raises_error(self) -> None:
        """Test adding invalid formula raises error."""
        processor = CalculatedRegisterProcessor()

        with pytest.raises(FormulaEvaluationError):
            processor.add_calculated_register(
                name="bad",
                formula="a + (b * ",
                dependencies=["a", "b"],
            )

    def test_execution_order_respects_dependencies(self) -> None:
        """Test execution order respects dependencies."""
        processor = CalculatedRegisterProcessor()

        # Add in reverse dependency order
        processor.add_calculated_register(
            name="final",
            formula="intermediate * 2",
            dependencies=["intermediate"],
        )

        processor.add_calculated_register(
            name="intermediate",
            formula="base + 10",
            dependencies=["base"],
        )

        register_values = {"base": 5}

        results = processor.process_all(register_values=register_values)

        assert results["intermediate"] == 15
        assert results["final"] == 30

    def test_process_all_handles_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test processing handles formula errors gracefully."""
        processor = CalculatedRegisterProcessor()

        processor.add_calculated_register(
            name="bad_calc",
            formula="a / b",
            dependencies=["a", "b"],
        )

        # Missing dependency will cause error
        register_values = {"a": 100}

        results = processor.process_all(register_values=register_values)

        # Should return 0.0 for failed calculation
        assert "bad_calc" in results
        assert results["bad_calc"] == 0.0
        assert "Failed to calculate" in caplog.text

    def test_process_all_simple(self) -> None:
        """Test processing calculated registers."""
        processor = CalculatedRegisterProcessor()

        processor.add_calculated_register(
            name="total_power",
            formula="grid_power + solar_power",
            dependencies=["grid_power", "solar_power"],
        )

        register_values = {
            "grid_power": 1000,
            "solar_power": 2500,
        }

        results = processor.process_all(register_values=register_values)

        assert "total_power" in results
        assert results["total_power"] == 3500

    def test_process_all_with_dependencies(self) -> None:
        """Test processing with dependent calculated registers."""
        processor = CalculatedRegisterProcessor()

        # First calculated register
        processor.add_calculated_register(
            name="total_power",
            formula="grid_power + solar_power",
            dependencies=["grid_power", "solar_power"],
        )

        # Second depends on first
        processor.add_calculated_register(
            name="avg_power",
            formula="total_power / 2",
            dependencies=["total_power"],
        )

        register_values = {
            "grid_power": 1000,
            "solar_power": 2000,
        }

        results = processor.process_all(register_values=register_values)

        assert "total_power" in results
        assert results["total_power"] == 3000
        assert "avg_power" in results
        assert results["avg_power"] == 1500
