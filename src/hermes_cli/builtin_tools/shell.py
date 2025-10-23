"""Shell command execution tools."""

import subprocess
from hermes_cli.tools import tool


@tool(
    name="execute_shell_command",
    description="Execute a shell command and return its output. Use with caution. Has a 30 second timeout.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            }
        },
        "required": ["command"]
    },
    builtin=True
)
def execute_shell_command(command: str) -> dict:
    """Execute a shell command.

    Args:
        command: Shell command string

    Returns:
        Dict with stdout, stderr, and returncode
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out (30s limit)"}
    except Exception as e:
        return {"error": f"Command execution failed: {str(e)}"}
