"""Helper functions for Hermes CLI."""

import sys
from typing import Optional


def read_stdin() -> Optional[str]:
    """Read input from stdin if available.

    Returns:
        Content from stdin if piped, None otherwise
    """
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


def get_user_prompt(cli_prompt: Optional[str]) -> str:
    """Get the user prompt from CLI argument or stdin.

    Args:
        cli_prompt: Prompt provided as CLI argument

    Returns:
        The user prompt to use

    Raises:
        ValueError: If no prompt is provided via CLI or stdin
    """
    # TODO: Implement prompt detection logic
    pass
