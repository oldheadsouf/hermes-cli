"""Mathematical calculation tools."""

import math
from hermes_cli.tools import tool


@tool(
    name="calculate",
    description="Perform mathematical calculations. Supports basic arithmetic, sqrt, abs, min, max, pow, and common math functions.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '15 * 23', 'pow(2, 8)')"
            }
        },
        "required": ["expression"]
    },
    builtin=True
)
def calculate(expression: str) -> dict:
    """Execute a mathematical calculation safely.

    Args:
        expression: Math expression string

    Returns:
        Dict with 'result' key or 'error' key
    """
    # Safe namespace with limited functions
    safe_dict = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "round": round,
        "sum": sum,
        "pi": math.pi,
        "e": math.e,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    try:
        result = eval(expression, safe_dict, {})
        return {"result": result}
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}
