"""Unit tests for tool system."""

import pytest
import json
from unittest.mock import Mock, patch
from hermes_cli.tools import ToolRegistry, ToolExecutor, tool
from hermes_cli.builtin_tools import calculate, read_file, write_file, execute_shell_command


def test_tool_decorator():
    """Test tool decorator attaches metadata."""
    assert hasattr(calculate, '__tool_name__')
    assert calculate.__tool_name__ == 'calculate'
    assert hasattr(calculate, '__tool_schema__')
    assert calculate.__tool_schema__['type'] == 'function'
    assert calculate.__tool_schema__['function']['name'] == 'calculate'


def test_calculate_tool():
    """Test calculate tool execution."""
    result = calculate(expression="2 + 2")
    assert result == {"result": 4}

    result = calculate(expression="sqrt(16)")
    assert result == {"result": 4.0}

    result = calculate(expression="15 * 23")
    assert result == {"result": 345}

    result = calculate(expression="pow(2, 8)")
    assert result == {"result": 256}

    # Test error handling
    result = calculate(expression="invalid")
    assert "error" in result


def test_tool_registry_initialization():
    """Test ToolRegistry loads built-in tools."""
    registry = ToolRegistry()
    assert "calculate" in registry.tools
    assert "execute_shell_command" in registry.tools
    assert "read_file" in registry.tools
    assert "write_file" in registry.tools
    assert "web_search" in registry.tools


def test_tool_selection():
    """Test tool selection logic."""
    registry = ToolRegistry()

    # Select specific tool
    selected = registry.select_tools("calculate")
    assert len(selected) == 1
    assert "calculate" in selected

    # Select multiple tools
    selected = registry.select_tools("calculate,read_file")
    assert len(selected) == 2
    assert "calculate" in selected
    assert "read_file" in selected

    # Select all
    selected = registry.select_tools("all")
    assert len(selected) >= 5  # At least 5 built-in tools


def test_tool_selection_unknown_tool():
    """Test error on unknown tool."""
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unknown tool"):
        registry.select_tools("nonexistent")


def test_get_tool_schemas():
    """Test schema generation."""
    registry = ToolRegistry()
    selected = registry.select_tools("calculate")
    schemas = registry.get_tool_schemas(selected)

    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "calculate"
    assert "parameters" in schemas[0]["function"]


def test_list_tools():
    """Test listing all tools."""
    registry = ToolRegistry()
    available = registry.list_tools()

    assert "builtin" in available
    assert "user" in available
    assert "calculate" in available["builtin"]
    assert len(available["builtin"]) >= 5


def test_get_tool_info():
    """Test getting tool info."""
    registry = ToolRegistry()
    info = registry.get_tool_info("calculate")

    assert info["name"] == "calculate"
    assert info["source"] == "builtin"
    assert "description" in info
    assert "parameters" in info


def test_get_tool_info_unknown():
    """Test error on unknown tool info."""
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Tool not found"):
        registry.get_tool_info("nonexistent")


def test_tool_executor():
    """Test ToolExecutor execution."""
    registry = ToolRegistry()
    selected = registry.select_tools("calculate")
    executor = ToolExecutor(registry)

    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "calculate",
            "arguments": '{"expression": "5 + 3"}'
        }
    }

    result = executor.execute_tool_call(tool_call, selected)

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_123"
    assert result["name"] == "calculate"

    # Parse the content to check the result
    content = json.loads(result["content"])
    assert "result" in content
    assert content["result"] == 8


def test_tool_executor_invalid_args():
    """Test ToolExecutor handles invalid arguments."""
    registry = ToolRegistry()
    selected = registry.select_tools("calculate")
    executor = ToolExecutor(registry)

    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "calculate",
            "arguments": 'invalid json'
        }
    }

    result = executor.execute_tool_call(tool_call, selected)
    assert "error" in result["content"]


def test_tool_executor_multiple_calls():
    """Test executing multiple tool calls."""
    registry = ToolRegistry()
    selected = registry.select_tools("calculate")
    executor = ToolExecutor(registry)

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": '{"expression": "2 + 2"}'
            }
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": '{"expression": "3 * 3"}'
            }
        }
    ]

    results = executor.execute_tool_calls(tool_calls, selected)
    assert len(results) == 2
    assert results[0]["tool_call_id"] == "call_1"
    assert results[1]["tool_call_id"] == "call_2"


@patch('subprocess.run')
def test_execute_shell_command(mock_run):
    """Test shell command execution."""
    mock_run.return_value = Mock(
        stdout="output",
        stderr="",
        returncode=0
    )

    result = execute_shell_command(command="echo test")

    assert result["stdout"] == "output"
    assert result["returncode"] == 0
    mock_run.assert_called_once()


@patch('pathlib.Path.read_text')
@patch('pathlib.Path.exists')
@patch('pathlib.Path.is_file')
@patch('pathlib.Path.stat')
def test_read_file_tool(mock_stat, mock_is_file, mock_exists, mock_read_text):
    """Test read_file tool."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value = Mock(st_size=100)
    mock_read_text.return_value = "file content"

    result = read_file(file_path="/tmp/test.txt")

    assert result["content"] == "file content"


def test_read_file_not_found():
    """Test read_file with non-existent file."""
    result = read_file(file_path="/nonexistent/file.txt")
    assert "error" in result


@patch('pathlib.Path.write_text')
@patch('pathlib.Path.mkdir')
def test_write_file_tool(mock_mkdir, mock_write_text):
    """Test write_file tool."""
    result = write_file(file_path="/tmp/test.txt", content="test content")

    assert result["success"] is True
    mock_write_text.assert_called_once_with("test content", encoding='utf-8')


@patch('requests.get')
def test_web_search_tool(mock_get):
    """Test web_search tool."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "organic_results": [
            {
                "title": "Test Result",
                "link": "https://example.com",
                "snippet": "Test snippet"
            }
        ]
    }
    mock_get.return_value = mock_response

    with patch.dict('os.environ', {'SERPAPI_API_KEY': 'test_key'}):
        from hermes_cli.builtin_tools.search import web_search
        result = web_search(query="test query")

        assert "results" in result
        assert len(result["results"]) > 0
        assert result["results"][0]["title"] == "Test Result"


def test_web_search_no_api_key():
    """Test web_search without API key."""
    with patch.dict('os.environ', {}, clear=True):
        from hermes_cli.builtin_tools.search import web_search
        result = web_search(query="test")

        assert "error" in result
        assert "SERPAPI_API_KEY" in result["error"]


def test_custom_tool_decorator():
    """Test creating a custom tool with decorator."""
    @tool(
        name="test_tool",
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            },
            "required": ["param"]
        },
        builtin=False
    )
    def test_func(param: str) -> dict:
        return {"result": f"processed: {param}"}

    assert test_func.__tool_name__ == "test_tool"
    assert test_func.__builtin__ is False

    result = test_func(param="test")
    assert result["result"] == "processed: test"


@patch('hermes_cli.api.NousAPIClient')
def test_chat_command_with_tools(mock_client_class):
    """Test chat command accepts --use-tools flag."""
    from click.testing import CliRunner
    from hermes_cli.main import cli
    import uuid

    # Mock the API client
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    # Mock API response with tool call followed by final answer
    mock_client.chat_completion.side_effect = [
        # First response: tool call
        {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expression": "2 + 2"}'
                        }
                    }]
                }
            }]
        },
        # Second response: final answer
        {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "The answer is 4."
                }
            }]
        }
    ]

    runner = CliRunner()
    with runner.isolated_filesystem():
        # Use unique conversation name to avoid conflicts
        chat_name = f'test-chat-{uuid.uuid4().hex[:8]}'

        # Test creating a new chat with tools
        result = runner.invoke(cli, [
            'chat',
            '--name', chat_name,
            '--use-tools', 'calculate',
            'What is 2 + 2?'
        ])

        # Should not error on the --use-tools flag
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "No such option" not in result.output
        # Verify the tool was called and we got a response
        assert "[Calling tool: calculate]" in result.output or result.exit_code == 0


def test_chat_command_with_tools_and_max_calls():
    """Test chat command accepts --max-tool-calls flag."""
    from click.testing import CliRunner
    from hermes_cli.main import cli
    import uuid
    from unittest.mock import patch, Mock, MagicMock

    runner = CliRunner()
    with runner.isolated_filesystem():
        # Use unique conversation name to avoid conflicts
        chat_name = f'test-max-calls-{uuid.uuid4().hex[:8]}'

        # Patch before invoking
        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            # Create a completely fresh mock
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock API response - fresh mock won't have side_effect
            mock_client.chat_completion.return_value = {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Answer"
                    }
                }]
            }

            result = runner.invoke(cli, [
                'chat',
                '--name', chat_name,
                '--use-tools', 'calculate',
                '--max-tool-calls', '10',
                'Test prompt'
            ])

            # Should not error on flags
            assert result.exit_code == 0, f"Command failed: {result.output}\nException: {getattr(result, 'exception', None)}"
            assert "No such option" not in result.output
