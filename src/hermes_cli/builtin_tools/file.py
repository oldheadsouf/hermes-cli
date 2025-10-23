"""File operation tools."""

from pathlib import Path
from hermes_cli.tools import tool


@tool(
    name="read_file",
    description="Read the contents of a file. Maximum file size is 1MB.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read (supports ~ for home directory)"
            }
        },
        "required": ["file_path"]
    },
    builtin=True
)
def read_file(file_path: str) -> dict:
    """Read file contents.

    Args:
        file_path: Path to file

    Returns:
        Dict with 'content' key or 'error' key
    """
    try:
        path = Path(file_path).expanduser()
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        if not path.is_file():
            return {"error": f"Not a file: {file_path}"}

        # Limit file size to 1MB
        if path.stat().st_size > 1_000_000:
            return {"error": "File too large (max 1MB)"}

        content = path.read_text(encoding='utf-8')
        return {"content": content}

    except UnicodeDecodeError:
        return {"error": "File is not UTF-8 text"}
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}


@tool(
    name="write_file",
    description="Write content to a file. Creates parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to write (supports ~ for home directory)"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        "required": ["file_path", "content"]
    },
    builtin=True
)
def write_file(file_path: str, content: str) -> dict:
    """Write content to file.

    Args:
        file_path: Path to file
        content: Content to write

    Returns:
        Dict with 'success' key or 'error' key
    """
    try:
        path = Path(file_path).expanduser()

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding='utf-8')
        return {"success": True, "message": f"Wrote {len(content)} characters to {file_path}"}

    except Exception as e:
        return {"error": f"Failed to write file: {str(e)}"}
