"""JSON schema handling logic."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import ValidationError


def is_json_string(input_str: str) -> bool:
    """Check if a string is a JSON string or a file path.

    Args:
        input_str: String to check

    Returns:
        True if it looks like a JSON string, False if it looks like a file path
    """
    stripped = input_str.strip()
    return stripped.startswith('{') or stripped.startswith('[')


def validate_schema_structure(schema: Dict[str, Any]) -> None:
    """Validate that a schema has a basic valid structure.

    Args:
        schema: Schema dictionary to validate

    Raises:
        ValueError: If schema structure is invalid
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a JSON object (dictionary)")

    # Basic JSON Schema validation - check for common fields
    if "type" not in schema:
        # It's okay if type is missing at the root level in some schemas
        # but we should warn about potential issues
        pass

    # Check for obviously invalid schemas
    if not schema:
        raise ValueError("Schema cannot be empty")


def load_schema(schema_input: str) -> Dict[str, Any]:
    """Load a JSON schema from a string or file path.

    This function intelligently detects whether the input is a JSON string
    (starting with { or [) or a file path, then loads and validates it.

    Args:
        schema_input: Either a JSON string or path to a JSON file

    Returns:
        Parsed JSON schema as a dictionary

    Raises:
        ValueError: If schema is invalid JSON or has invalid structure
        FileNotFoundError: If schema file doesn't exist
    """
    if not schema_input or not schema_input.strip():
        raise ValueError("Schema input cannot be empty")

    # Determine if it's a JSON string or file path
    if is_json_string(schema_input):
        # Treat as JSON string
        try:
            schema = json.loads(schema_input)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON schema string: {str(e)}\n"
                f"Make sure your schema is valid JSON. Example:\n"
                f'  --schema \'{{"type": "object", "properties": {{"key": {{"type": "string"}}}}}}\''
            )
    else:
        # Treat as file path
        schema_path = Path(schema_input).expanduser()

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {schema_input}\n"
                f"Make sure the file path is correct and the file exists."
            )

        if not schema_path.is_file():
            raise ValueError(f"Schema path is not a file: {schema_input}")

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in schema file '{schema_input}': {str(e)}\n"
                f"Make sure the file contains valid JSON."
            )
        except PermissionError:
            raise ValueError(f"Permission denied reading schema file: {schema_input}")
        except Exception as e:
            raise ValueError(f"Error reading schema file '{schema_input}': {str(e)}")

    # Validate the schema structure
    validate_schema_structure(schema)

    return schema


def build_system_prompt_with_schema(
    user_system_prompt: Optional[str],
    schema: Dict[str, Any]
) -> str:
    """Build a complete system prompt with schema instructions.

    According to CLAUDE.md, the system prompt should be:
    {user_system_prompt}

    You must respond with valid JSON matching this schema: {schema_json}

    Args:
        user_system_prompt: Optional user-provided system prompt
        schema: JSON schema dictionary

    Returns:
        Complete system prompt with schema instructions
    """
    # Format schema as compact JSON for shorter prompts, or indented for readability
    # Using indent=2 for better readability in the prompt
    schema_json = json.dumps(schema, indent=2)
    schema_instruction = f"\n\nYou must respond with valid JSON matching this schema: {schema_json}"

    if user_system_prompt:
        return user_system_prompt + schema_instruction
    else:
        # If no user system prompt, just use the schema instruction without leading newlines
        return schema_instruction.strip()


def should_disable_streaming(schema: Optional[Dict[str, Any]]) -> bool:
    """Determine if streaming should be automatically disabled.

    According to CLAUDE.md, when --schema is provided, streaming should be
    automatically disabled for clean JSON parsing (unless user explicitly
    sets --stream or --no-stream).

    Args:
        schema: Schema dictionary if provided, None otherwise

    Returns:
        True if streaming should be disabled, False otherwise
    """
    # Disable streaming when schema is provided for clean JSON output
    return schema is not None
