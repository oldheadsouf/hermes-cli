"""Tool registry and execution system for Hermes CLI."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Callable, Optional


def tool(name: str, description: str, parameters: Dict[str, Any], builtin: bool = False):
    """Decorator to register a function as a tool.

    Args:
        name: Tool name (used in API calls)
        description: Human-readable description of what the tool does
        parameters: JSON Schema for tool parameters
        builtin: Whether this is a built-in tool

    Usage:
        @tool(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
            builtin=True
        )
        def calculate(expression: str) -> dict:
            return {"result": eval(expression)}
    """
    def decorator(func):
        func.__tool_name__ = name
        func.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        func.__builtin__ = builtin
        return func
    return decorator


class ToolRegistry:
    """Manages the tool library and selection."""

    def __init__(self):
        """Initialize tool registry and load all available tools."""
        self.tools: Dict[str, Callable] = {}
        self._load_builtin_tools()

    def _load_builtin_tools(self):
        """Load built-in tools from builtin_tools/ directory."""
        try:
            from hermes_cli.builtin_tools import (
                calculate,
                execute_shell_command,
                read_file,
                write_file,
                web_search
            )

            for tool_func in [calculate, execute_shell_command, read_file, write_file, web_search]:
                self.tools[tool_func.__tool_name__] = tool_func
        except ImportError as e:
            print(f"Warning: Failed to load built-in tools: {e}", file=sys.stderr)

    def select_tools(self, tool_spec: str) -> Dict[str, Callable]:
        """Select tools based on specification.

        Args:
            tool_spec: Comma-separated tool names, or 'all' for all available

        Returns:
            Dictionary of {tool_name: tool_function}

        Raises:
            ValueError: If any specified tool is not found
        """
        selected = {}

        if tool_spec == "all":
            selected.update(self.tools)
        else:
            tool_names = [name.strip() for name in tool_spec.split(",")]
            for name in tool_names:
                if name not in self.tools:
                    available = ", ".join(sorted(self.tools.keys()))
                    raise ValueError(
                        f"Unknown tool: '{name}'\n"
                        f"Available tools: {available}\n"
                        f"Use 'hermes tools list' to see all tools"
                    )
                selected[name] = self.tools[name]

        return selected

    def get_tool_schemas(self, selected_tools: Dict[str, Callable]) -> list[dict]:
        """Get OpenAI-compatible schemas for selected tools.

        Args:
            selected_tools: Dict of selected tool functions

        Returns:
            List of tool schemas for API request
        """
        return [tool.__tool_schema__ for tool in selected_tools.values()]

    def list_tools(self) -> dict:
        """List all available tools categorized by source.

        Returns:
            Dict with 'builtin' and 'user' keys containing tool info
        """
        result = {"builtin": {}, "user": {}}

        for name, tool_func in sorted(self.tools.items()):
            desc = tool_func.__tool_schema__["function"]["description"]
            source = "builtin" if getattr(tool_func, "__builtin__", False) else "user"
            result[source][name] = desc

        return result

    def get_tool_info(self, tool_name: str) -> dict:
        """Get detailed information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dict with tool details

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")

        tool_func = self.tools[tool_name]
        schema = tool_func.__tool_schema__["function"]

        return {
            "name": tool_name,
            "description": schema["description"],
            "parameters": schema["parameters"],
            "source": "builtin" if getattr(tool_func, "__builtin__", False) else "user"
        }


class ToolExecutor:
    """Handles tool execution and result formatting."""

    def __init__(self, registry: ToolRegistry):
        """Initialize executor with tool registry.

        Args:
            registry: ToolRegistry instance with loaded tools
        """
        self.registry = registry

    def execute_tool_call(self, tool_call: dict, selected_tools: Dict[str, Callable]) -> dict:
        """Execute a single tool call and return formatted result.

        Args:
            tool_call: Tool call dict from API response with structure:
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "arguments": "{\"param\": \"value\"}"
                    }
                }
            selected_tools: Dict of currently enabled tools

        Returns:
            Tool message dict for API with structure:
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "name": "tool_name",
                "content": "{\"result\": \"...\"}"
            }
        """
        tool_call_id = tool_call["id"]
        func_name = tool_call["function"]["name"]
        args_json = tool_call["function"]["arguments"]

        try:
            # Parse arguments
            args = json.loads(args_json)

            # Verify tool is in selected tools
            if func_name not in selected_tools:
                raise ValueError(f"Tool '{func_name}' not in enabled tools")

            # Execute tool
            tool_func = selected_tools[func_name]
            result = tool_func(**args)

            # Format successful result
            content = json.dumps(result) if not isinstance(result, str) else result

        except json.JSONDecodeError as e:
            content = json.dumps({"error": f"Invalid arguments JSON: {str(e)}"})
        except TypeError as e:
            content = json.dumps({"error": f"Invalid arguments for tool: {str(e)}"})
        except Exception as e:
            content = json.dumps({"error": f"Tool execution failed: {str(e)}"})

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": func_name,
            "content": content
        }

    def execute_tool_calls(self, tool_calls: list[dict], selected_tools: Dict[str, Callable]) -> list[dict]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool call dicts
            selected_tools: Dict of currently enabled tools

        Returns:
            List of tool message dicts
        """
        return [self.execute_tool_call(tc, selected_tools) for tc in tool_calls]
