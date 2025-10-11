"""Tests for CLI main module."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
from hermes_cli.main import cli
from hermes_cli.api import APIError


class TestCLIBasicFunctionality:
    """Tests for basic CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cli_requires_prompt(self, runner):
        """Test that CLI requires a prompt from either argument or stdin."""
        with patch('hermes_cli.main.NousAPIClient'):
            with patch('sys.stdin.isatty', return_value=True):
                result = runner.invoke(cli, [])
                assert result.exit_code != 0

    def test_cli_with_positional_prompt(self, runner):
        """Test CLI with positional prompt argument."""
        mock_response = {
            "choices": [{"message": {"content": "Test response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, ['--no-stream', 'Test prompt'])

            assert result.exit_code == 0
            assert "Test response" in result.output
            mock_client.chat_completion.assert_called_once()

    def test_cli_with_system_prompt(self, runner):
        """Test CLI with system prompt option."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--no-stream',
                '-s', 'You are helpful',
                'Hello'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            messages = call_args['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert messages[0]['content'] == 'You are helpful'


class TestCLITemperatureAndMaxTokens:
    """Tests for temperature and max_tokens parameters."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cli_default_temperature(self, runner):
        """Test that default temperature is 0.7."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, ['--no-stream', 'Test'])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['temperature'] == 0.7

    def test_cli_default_max_tokens(self, runner):
        """Test that default max_tokens is 2048."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, ['--no-stream', 'Test'])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['max_tokens'] == 2048

    def test_cli_custom_temperature(self, runner):
        """Test CLI with custom temperature."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--no-stream',
                '-t', '0.5',
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['temperature'] == 0.5

    def test_cli_custom_temperature_long_flag(self, runner):
        """Test CLI with custom temperature using long flag."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--no-stream',
                '--temperature', '1.2',
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['temperature'] == 1.2

    def test_cli_custom_max_tokens(self, runner):
        """Test CLI with custom max_tokens."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--no-stream',
                '-mt', '1024',
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['max_tokens'] == 1024

    def test_cli_custom_max_tokens_long_flag(self, runner):
        """Test CLI with custom max_tokens using long flag."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--no-stream',
                '--max-tokens', '512',
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['max_tokens'] == 512

    def test_cli_both_temperature_and_max_tokens(self, runner):
        """Test CLI with both temperature and max_tokens."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--no-stream',
                '-t', '0.3',
                '-mt', '256',
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['temperature'] == 0.3
            assert call_args['max_tokens'] == 256

    def test_cli_temperature_with_streaming(self, runner):
        """Test temperature parameter works with streaming."""
        mock_chunks = ["Hello", " world"]

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = iter(mock_chunks)
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--stream',
                '-t', '0.9',
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['temperature'] == 0.9
            assert call_args['stream'] is True

    def test_cli_max_tokens_with_streaming(self, runner):
        """Test max_tokens parameter works with streaming."""
        mock_chunks = ["Test"]

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = iter(mock_chunks)
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--stream',
                '-mt', '100',
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['max_tokens'] == 100
            assert call_args['stream'] is True


class TestCLIModelSelection:
    """Tests for model selection."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cli_default_model(self, runner):
        """Test that default model is Hermes-4-405B."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, ['--no-stream', 'Test'])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['model'] == 'Hermes-4-405B'

    def test_cli_with_hermes_70b_model(self, runner):
        """Test CLI with Hermes-4-70B model."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, [
                '--no-stream',
                '-m', 'Hermes-4-70B',
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['model'] == 'Hermes-4-70B'


class TestCLIStreaming:
    """Tests for streaming functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cli_streaming_output(self, runner):
        """Test CLI with streaming enabled."""
        mock_chunks = ["Hello", " ", "world"]

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = iter(mock_chunks)
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, ['--stream', 'Test prompt'])

            assert result.exit_code == 0
            assert "Hello world" in result.output

    def test_cli_non_streaming_output(self, runner):
        """Test CLI with streaming disabled."""
        mock_response = {
            "choices": [{"message": {"content": "Complete response"}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, ['--no-stream', 'Test prompt'])

            assert result.exit_code == 0
            assert "Complete response" in result.output


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cli_missing_api_key(self, runner):
        """Test CLI handles missing API key error."""
        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client_class.side_effect = ValueError("API key not found")

            result = runner.invoke(cli, ['--no-stream', 'Test'])

            assert result.exit_code == 1
            assert "API key not found" in result.output

    def test_cli_api_error(self, runner):
        """Test CLI handles API errors."""
        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.side_effect = APIError("API failed")
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, ['--no-stream', 'Test'])

            assert result.exit_code == 1
            assert "API Error" in result.output

    def test_cli_keyboard_interrupt(self, runner):
        """Test CLI handles keyboard interrupt gracefully."""
        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.side_effect = KeyboardInterrupt()
            mock_client_class.return_value = mock_client

            result = runner.invoke(cli, ['--no-stream', 'Test'])

            assert result.exit_code == 130
            assert "Interrupted" in result.output


class TestCLISchema:
    """Tests for schema functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cli_with_schema_json_output(self, runner):
        """Test CLI with schema produces JSON output."""
        json_response = '{"answer": "42"}'
        mock_response = {
            "choices": [{"message": {"content": json_response}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            schema = '{"type": "object", "properties": {"answer": {"type": "string"}}}'
            result = runner.invoke(cli, [
                '--schema', schema,
                'Test prompt'
            ])

            assert result.exit_code == 0
            # Should pretty-print JSON
            assert '"answer"' in result.output
            assert '"42"' in result.output

    def test_cli_schema_disables_streaming_by_default(self, runner):
        """Test that schema disables streaming by default."""
        mock_response = {
            "choices": [{"message": {"content": '{"result": "test"}'}}]
        }

        with patch('hermes_cli.main.NousAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client.chat_completion.return_value = mock_response
            mock_client_class.return_value = mock_client

            schema = '{"type": "object"}'
            result = runner.invoke(cli, [
                '--schema', schema,
                'Test prompt'
            ])

            assert result.exit_code == 0
            call_args = mock_client.chat_completion.call_args[1]
            assert call_args['stream'] is False
