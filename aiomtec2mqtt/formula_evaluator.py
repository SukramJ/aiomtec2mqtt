"""
Safe formula evaluator for calculated registers.

This module provides a safe AST-based formula evaluator that supports basic
arithmetic operations and common mathematical functions without eval() security risks.

(c) 2024 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import ast
from collections.abc import Callable
import logging
import operator
from typing import Any, Final

_LOGGER: Final = logging.getLogger(__name__)

# Allowed operators
OPERATORS: dict[type[ast.operator | ast.unaryop], Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed functions
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
}


class FormulaEvaluationError(Exception):
    """Formula evaluation error."""


class FormulaEvaluator:
    """
    Safe formula evaluator using AST.

    Evaluates mathematical formulas with variable substitution without
    using eval() to avoid security risks.
    """

    def __init__(self) -> None:
        """Initialize the formula evaluator."""
        self._operators = OPERATORS.copy()
        self._functions = FUNCTIONS.copy()

    def evaluate(
        self,
        *,
        formula: str,
        context: dict[str, Any],
    ) -> float | int:
        """
        Evaluate formula with given context.

        Args:
            formula: Mathematical formula string
            context: Variable values dictionary

        Returns:
            Evaluated result

        Raises:
            FormulaEvaluationError: If evaluation fails

        """
        try:
            # Parse formula to AST
            tree = ast.parse(formula, mode="eval")

            # Evaluate AST
            result = self._eval_node(node=tree.body, context=context)

            # Ensure numeric result
            if not isinstance(result, (int, float)):
                msg = f"Formula must return a number, got {type(result).__name__}"
                raise FormulaEvaluationError(msg)

            return result  # noqa: TRY300

        except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as ex:
            msg = f"Formula evaluation failed: {ex}"
            raise FormulaEvaluationError(msg) from ex

    def get_dependencies(self, *, formula: str) -> set[str]:
        """
        Extract variable dependencies from formula.

        Args:
            formula: Formula string

        Returns:
            Set of variable names used in formula

        """
        try:
            tree = ast.parse(formula, mode="eval")
            return self._extract_names(node=tree.body)

        except SyntaxError:
            return set()

    def validate_formula(self, *, formula: str) -> bool:
        """
        Validate formula syntax.

        Args:
            formula: Formula string to validate

        Returns:
            True if valid

        Raises:
            FormulaEvaluationError: If formula is invalid

        """
        try:
            # Try to parse
            tree = ast.parse(formula, mode="eval")

            # Check all nodes are allowed
            self._validate_ast(node=tree.body)

            return True  # noqa: TRY300

        except SyntaxError as ex:
            msg = f"Formula syntax error: {ex}"
            raise FormulaEvaluationError(msg) from ex

    def _eval_node(
        self,
        *,
        node: ast.expr,
        context: dict[str, Any],
    ) -> Any:
        """
        Recursively evaluate AST node.

        Args:
            node: AST node to evaluate
            context: Variable context

        Returns:
            Evaluated value

        Raises:
            FormulaEvaluationError: If node type is not allowed

        """
        # Literal number
        if isinstance(node, ast.Constant):
            return node.value

        # Variable name
        if isinstance(node, ast.Name):
            var_name = node.id
            if var_name not in context:  # pylint: disable=consider-using-assignment-expr
                msg = f"Undefined variable: {var_name}"
                raise FormulaEvaluationError(msg)
            return context[var_name]

        # Binary operation
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._operators:  # pylint: disable=consider-using-assignment-expr
                msg = f"Unsupported operator: {op_type.__name__}"
                raise FormulaEvaluationError(msg)

            left = self._eval_node(node=node.left, context=context)
            right = self._eval_node(node=node.right, context=context)
            return self._operators[op_type](left, right)

        # Unary operation
        if isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in self._operators:  # pylint: disable=consider-using-assignment-expr
                msg = f"Unsupported unary operator: {unary_op_type.__name__}"
                raise FormulaEvaluationError(msg)

            operand = self._eval_node(node=node.operand, context=context)
            return self._operators[unary_op_type](operand)

        # Function call
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                msg = "Only named functions are allowed"
                raise FormulaEvaluationError(msg)

            func_name = node.func.id
            if func_name not in self._functions:  # pylint: disable=consider-using-assignment-expr
                msg = f"Unsupported function: {func_name}"
                raise FormulaEvaluationError(msg)

            # Evaluate arguments
            args = [self._eval_node(node=arg, context=context) for arg in node.args]

            # Call function
            return self._functions[func_name](*args)

        # Comparison (for min/max with conditions)
        if isinstance(node, ast.Compare):
            msg = "Comparison operators not yet supported"
            raise FormulaEvaluationError(msg)

        # Not allowed
        msg = f"Unsupported node type: {type(node).__name__}"
        raise FormulaEvaluationError(msg)

    def _extract_names(self, *, node: ast.AST) -> set[str]:
        """
        Recursively extract variable names from AST.

        Args:
            node: AST node

        Returns:
            Set of variable names

        """
        names: set[str] = set()

        if isinstance(node, ast.Name):
            # Don't include function names
            names.add(node.id)

        elif isinstance(node, ast.Call):
            # Don't include the function name itself
            for arg in node.args:
                names.update(self._extract_names(node=arg))

        else:
            # Recursively process children
            for child in ast.iter_child_nodes(node):
                names.update(self._extract_names(node=child))

        return names

    def _validate_ast(self, *, node: ast.AST) -> None:
        """
        Recursively validate AST nodes.

        Args:
            node: AST node to validate

        Raises:
            FormulaEvaluationError: If node type is not allowed

        """
        # Skip context nodes (Load, Store, Del)
        if isinstance(node, (ast.Load, ast.Store, ast.Del)):
            return

        # Skip operator nodes (they are checked in _eval_node)
        if isinstance(
            node,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
                ast.USub,
                ast.UAdd,
            ),
        ):
            return

        # Check allowed expression node types
        allowed_nodes = (
            ast.Constant,
            ast.Name,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
        )

        if not isinstance(node, allowed_nodes):
            msg = f"Unsupported node type: {type(node).__name__}"
            raise FormulaEvaluationError(msg)

        # Recursively validate children
        for child in ast.iter_child_nodes(node):
            self._validate_ast(node=child)


class CalculatedRegisterProcessor:
    """
    Processor for calculated registers.

    Evaluates formulas and manages dependencies between calculated registers.
    """

    def __init__(self, *, evaluator: FormulaEvaluator | None = None) -> None:
        """
        Initialize the processor.

        Args:
            evaluator: Formula evaluator (creates default if None)

        """
        self._evaluator = evaluator or FormulaEvaluator()
        self._calculated_registers: dict[str, Any] = {}

    def add_calculated_register(
        self,
        *,
        name: str,
        formula: str,
        dependencies: list[str],
    ) -> None:
        """
        Add a calculated register definition.

        Args:
            name: Register name
            formula: Calculation formula
            dependencies: Required register names

        """
        # Validate formula
        self._evaluator.validate_formula(formula=formula)

        self._calculated_registers[name] = {
            "formula": formula,
            "dependencies": dependencies,
        }

        _LOGGER.debug(
            "Added calculated register '%s' with %d dependencies",
            name,
            len(dependencies),
        )

    def process_all(
        self,
        *,
        register_values: dict[str, Any],
    ) -> dict[str, float | int]:
        """
        Process all calculated registers.

        Args:
            register_values: Current register values

        Returns:
            Dictionary of calculated values

        """
        results: dict[str, float | int] = {}

        # Build dependency order
        ordered = self._get_execution_order()

        # Evaluate in order
        for name in ordered:
            calc_info = self._calculated_registers[name]
            formula = calc_info["formula"]

            # Build context with all values
            context = {**register_values, **results}

            try:
                value = self._evaluator.evaluate(formula=formula, context=context)
                results[name] = value
                _LOGGER.debug("Calculated %s = %s", name, value)

            except FormulaEvaluationError as ex:
                _LOGGER.error("Failed to calculate '%s': %s", name, ex)
                results[name] = 0.0

        return results

    def _get_execution_order(self) -> list[str]:
        """
        Get execution order for calculated registers (topological sort).

        Returns:
            List of register names in dependency order

        """
        # Build dependency graph
        graph: dict[str, set[str]] = {}
        for name, info in self._calculated_registers.items():
            graph[name] = {
                dep for dep in info["dependencies"] if dep in self._calculated_registers
            }

        # Topological sort
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)

            for dep in graph.get(node, set()):
                visit(dep)

            ordered.append(node)

        for name in self._calculated_registers:
            visit(name)

        return ordered
