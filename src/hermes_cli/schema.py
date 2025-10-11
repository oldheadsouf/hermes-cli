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
    # TODO: Implement schema loading logic
    pass


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
    # TODO: Implement system prompt building logic
    pass
