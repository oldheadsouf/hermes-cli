"""Built-in tools for Hermes CLI."""

from hermes_cli.builtin_tools.math import calculate
from hermes_cli.builtin_tools.shell import execute_shell_command
from hermes_cli.builtin_tools.file import read_file, write_file
from hermes_cli.builtin_tools.search import web_search

__all__ = [
    "calculate",
    "execute_shell_command",
    "read_file",
    "write_file",
    "web_search"
]
