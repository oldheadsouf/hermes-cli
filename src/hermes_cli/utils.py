"""Helper functions for Hermes CLI."""

import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.style import Style


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
    # Check for piped input first (takes priority)
    stdin_input = read_stdin()
    if stdin_input:
        return stdin_input

    # Fall back to CLI argument
    if cli_prompt:
        return cli_prompt

    # No input provided
    raise ValueError(
        "No prompt provided. Please provide a prompt as an argument or pipe input.\n"
        "Examples:\n"
        "  hermes \"Your prompt here\"\n"
        "  echo \"Your prompt\" | hermes"
    )


def format_with_border(content: str, model: str) -> str:
    """Format content with a decorative ASCII border.

    Args:
        content: The text content to wrap in a border

    Returns:
        The content formatted with a handsome ASCII border
    """
    console = Console()

    # Create a beautiful panel with rounded corners and a gradient-style border
    panel = Panel(
        content,
        border_style=Style(color="cyan", bold=True),
        padding=(1, 2),
        expand=False,
        title=f"[bold magenta]✨ {model} ✨[/bold magenta]",
        title_align="center",
    )

    # Capture the output to a string
    with console.capture() as capture:
        console.print(panel)

    return capture.get().rstrip()
