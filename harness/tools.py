"""Tool executors available to the agent. Deterministic, local, no network.

The tool a task uses is declared in task.context["tool"] = {"name": ..., "input_schema": {...}};
tasks without it default to `lookup`. The tool *description* always comes from the tool_policy slot.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from shared.schemas import Task

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


MAX_ABS = 1e30  # bound on every intermediate value; nested powers would otherwise allocate GBs


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression without exec/eval. Raises ValueError on anything else."""
    if len(expr) > 200:
        raise ValueError("expression too long")
    tree = ast.parse(expr.strip(), mode="eval")

    def bounded(x: float) -> float:
        if abs(x) > MAX_ABS:
            raise ValueError("value too large")
        return x

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return bounded(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("exponent too large")
            return bounded(_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:40]}")

    return ev(tree)


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() and abs(x) < 1e15 else f"{x:.10g}"


def execute_tool(task: Task, name: str, inp: Any) -> tuple[str, bool]:
    """Returns (content, is_error)."""
    spec = tool_spec(task)
    if name != spec["name"]:
        return f"unknown tool {name!r}", True
    if not isinstance(inp, dict):
        return "tool input must be an object", True
    if spec["name"] == "lookup":
        field = str(inp.get("field", ""))
        table: dict[str, str] = task.context.get("lookup", {})
        return (table[field], False) if field in table else (f"no metadata recorded for field {field!r}", True)
    if spec["name"] == "calculator":
        expr = str(inp.get("expression", ""))
        try:
            return _fmt(safe_eval(expr)), False
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
            return f"calculator error: {e}", True
    return f"no executor for tool {spec['name']!r}", True


def tool_spec(task: Task) -> dict[str, Any]:
    """Tool name + input schema for this task (description is filled from the tool_policy slot)."""
    declared = task.context.get("tool")
    if declared:
        return {"name": declared["name"], "input_schema": declared["input_schema"]}
    fields = task.context.get("lookup_fields") or sorted(task.context.get("lookup", {}).keys())
    return {"name": "lookup", "input_schema": {
        "type": "object", "properties": {"field": {"type": "string", "enum": fields}},
        "required": ["field"], "additionalProperties": False}}


def task_input_text(task: Task) -> str:
    return task.context.get("input_text") or task.context.get("notebook_text", "")


def input_label(task: Task) -> str:
    return task.context.get("input_label", "Notebook entry")
