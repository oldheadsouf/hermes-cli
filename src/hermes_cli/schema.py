"""JSON schema handling logic."""

import json
from pathlib import Path
from typing import Dict, Any


def load_schema(schema_input: str) -> Dict[str, Any]:
    """Load a JSON schema from a string or file path.

    Args:
        schema_input: Either a JSON string or path to a JSON file

    Returns:
        Parsed JSON schema as a dictionary

    Raises:
        ValueError: If schema is invalid JSON
        FileNotFoundError: If schema file doesn't exist
    """
    # Try to determine if it's a JSON string or file path
    # JSON strings start with { or [
    if schema_input.strip().startswith('{') or schema_input.strip().startswith('['):
        # Treat as JSON string
        try:
            return json.loads(schema_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema string: {str(e)}")
    else:
        # Treat as file path
        schema_path = Path(schema_input)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_input}")

        if not schema_path.is_file():
            raise ValueError(f"Schema path is not a file: {schema_input}")

        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file {schema_input}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading schema file {schema_input}: {str(e)}")


def build_system_prompt_with_schema(
    user_system_prompt: str | None,
    schema: Dict[str, Any]
) -> str:
    """Build a complete system prompt with schema instructions.

    Args:
        user_system_prompt: Optional user-provided system prompt
        schema: JSON schema dictionary

    Returns:
        Complete system prompt with schema instructions
    """
    schema_json = json.dumps(schema, indent=2)
    schema_instruction = f"\n\nYou must respond with valid JSON matching this schema: {schema_json}"

    if user_system_prompt:
        return user_system_prompt + schema_instruction
    else:
        return schema_instruction.strip()
