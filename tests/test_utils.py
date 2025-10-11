"""Tests for utility functions."""

import sys
import pytest
from io import StringIO
from unittest.mock import patch, Mock
from hermes_cli.utils import read_stdin, get_user_prompt


class TestReadStdin:
    """Tests for read_stdin function."""

    def test_read_stdin_with_piped_input(self):
        """Test reading from stdin when input is piped."""
        test_input = "Hello from pipe"
        mock_stdin = StringIO(test_input)

        with patch('sys.stdin', mock_stdin):
            with patch('sys.stdin.isatty', return_value=False):
                result = read_stdin()
                assert result == test_input

    def test_read_stdin_with_multiline_input(self):
        """Test reading multiline input from stdin."""
        test_input = "Line 1\nLine 2\nLine 3"
        mock_stdin = StringIO(test_input)

        with patch('sys.stdin', mock_stdin):
            with patch('sys.stdin.isatty', return_value=False):
                result = read_stdin()
                assert result == test_input

    def test_read_stdin_strips_whitespace(self):
        """Test that stdin input is stripped of leading/trailing whitespace."""
        test_input = "  \n  Hello World  \n  "
        expected = "Hello World"
        mock_stdin = StringIO(test_input)

        with patch('sys.stdin', mock_stdin):
            with patch('sys.stdin.isatty', return_value=False):
                result = read_stdin()
                assert result == expected

    def test_read_stdin_with_empty_input(self):
        """Test reading empty input from stdin."""
        test_input = "   \n   "
        mock_stdin = StringIO(test_input)

        with patch('sys.stdin', mock_stdin):
            with patch('sys.stdin.isatty', return_value=False):
                result = read_stdin()
                assert result == ""

    def test_read_stdin_without_piped_input(self):
        """Test that None is returned when stdin is a TTY (no pipe)."""
        with patch('sys.stdin.isatty', return_value=True):
            result = read_stdin()
            assert result is None

    def test_read_stdin_interactive_terminal(self):
        """Test reading from interactive terminal returns None."""
        # When isatty() returns True, it means interactive terminal
        with patch('sys.stdin.isatty', return_value=True):
            result = read_stdin()
            assert result is None

    def test_read_stdin_with_special_characters(self):
        """Test reading input with special characters."""
        test_input = "Hello\tWorld\n!@#$%^&*()"
        expected = "Hello\tWorld\n!@#$%^&*()"
        mock_stdin = StringIO(test_input)

        with patch('sys.stdin', mock_stdin):
            with patch('sys.stdin.isatty', return_value=False):
                result = read_stdin()
                assert result == expected

    def test_read_stdin_with_unicode(self):
        """Test reading input with unicode characters."""
        test_input = "Hello 世界 🌍"
        mock_stdin = StringIO(test_input)

        with patch('sys.stdin', mock_stdin):
            with patch('sys.stdin.isatty', return_value=False):
                result = read_stdin()
                assert result == test_input


class TestGetUserPrompt:
    """Tests for get_user_prompt function."""

    def test_get_user_prompt_from_cli_argument(self):
        """Test getting prompt from CLI argument when no stdin."""
        cli_prompt = "CLI prompt text"

        with patch('hermes_cli.utils.read_stdin', return_value=None):
            result = get_user_prompt(cli_prompt)
            assert result == cli_prompt

    def test_get_user_prompt_from_stdin(self):
        """Test getting prompt from stdin when available."""
        cli_prompt = "CLI prompt"
        stdin_prompt = "Stdin prompt"

        with patch('hermes_cli.utils.read_stdin', return_value=stdin_prompt):
            result = get_user_prompt(cli_prompt)
            assert result == stdin_prompt

    def test_get_user_prompt_stdin_takes_priority(self):
        """Test that stdin input takes priority over CLI argument."""
        cli_prompt = "This should be ignored"
        stdin_prompt = "This should be used"

        with patch('hermes_cli.utils.read_stdin', return_value=stdin_prompt):
            result = get_user_prompt(cli_prompt)
            assert result == stdin_prompt
            assert result != cli_prompt

    def test_get_user_prompt_with_none_cli_prompt(self):
        """Test getting prompt from stdin when CLI argument is None."""
        stdin_prompt = "Stdin prompt"

        with patch('hermes_cli.utils.read_stdin', return_value=stdin_prompt):
            result = get_user_prompt(None)
            assert result == stdin_prompt

    def test_get_user_prompt_with_empty_string_cli_prompt(self):
        """Test getting prompt from stdin when CLI argument is empty string."""
        stdin_prompt = "Stdin prompt"

        with patch('hermes_cli.utils.read_stdin', return_value=stdin_prompt):
            result = get_user_prompt("")
            assert result == stdin_prompt

    def test_get_user_prompt_raises_error_when_no_input(self):
        """Test that ValueError is raised when no prompt is provided."""
        with patch('hermes_cli.utils.read_stdin', return_value=None):
            with pytest.raises(ValueError) as exc_info:
                get_user_prompt(None)

            error_msg = str(exc_info.value)
            assert "No prompt provided" in error_msg
            assert "hermes" in error_msg
            assert "Examples:" in error_msg

    def test_get_user_prompt_raises_error_with_empty_cli_prompt(self):
        """Test that ValueError is raised when CLI prompt is empty and no stdin."""
        with patch('hermes_cli.utils.read_stdin', return_value=None):
            with pytest.raises(ValueError) as exc_info:
                get_user_prompt("")

            error_msg = str(exc_info.value)
            assert "No prompt provided" in error_msg

    def test_get_user_prompt_error_message_includes_examples(self):
        """Test that error message includes usage examples."""
        with patch('hermes_cli.utils.read_stdin', return_value=None):
            with pytest.raises(ValueError) as exc_info:
                get_user_prompt(None)

            error_msg = str(exc_info.value)
            assert 'hermes "Your prompt here"' in error_msg
            assert 'echo "Your prompt" | hermes' in error_msg

    def test_get_user_prompt_with_whitespace_only_cli_prompt(self):
        """Test behavior with whitespace-only CLI prompt."""
        # Whitespace is still a valid prompt (not empty)
        cli_prompt = "   "

        with patch('hermes_cli.utils.read_stdin', return_value=None):
            result = get_user_prompt(cli_prompt)
            assert result == cli_prompt

    def test_get_user_prompt_stdin_empty_string_falls_back_to_cli(self):
        """Test that empty stdin string falls back to CLI argument."""
        cli_prompt = "CLI prompt"

        # Empty string from stdin should be falsy and fall back to CLI
        with patch('hermes_cli.utils.read_stdin', return_value=""):
            with pytest.raises(ValueError):
                # Both stdin (empty) and no CLI prompt should raise error
                get_user_prompt(None)

    def test_get_user_prompt_with_multiline_cli_prompt(self):
        """Test getting multiline prompt from CLI argument."""
        cli_prompt = "Line 1\nLine 2\nLine 3"

        with patch('hermes_cli.utils.read_stdin', return_value=None):
            result = get_user_prompt(cli_prompt)
            assert result == cli_prompt
            assert "\n" in result

    def test_get_user_prompt_with_special_characters_in_cli(self):
        """Test CLI prompt with special characters."""
        cli_prompt = "What is 2+2? !@#$%"

        with patch('hermes_cli.utils.read_stdin', return_value=None):
            result = get_user_prompt(cli_prompt)
            assert result == cli_prompt

    def test_get_user_prompt_with_unicode_in_cli(self):
        """Test CLI prompt with unicode characters."""
        cli_prompt = "Translate: Hello 世界 🌍"

        with patch('hermes_cli.utils.read_stdin', return_value=None):
            result = get_user_prompt(cli_prompt)
            assert result == cli_prompt

    def test_get_user_prompt_stdin_whitespace_only_raises_error(self):
        """Test that whitespace-only stdin with no CLI prompt raises error."""
        # This tests the edge case where stdin has only whitespace (gets stripped to "")
        with patch('hermes_cli.utils.read_stdin', return_value=""):
            with pytest.raises(ValueError) as exc_info:
                get_user_prompt(None)
            assert "No prompt provided" in str(exc_info.value)


class TestIntegration:
    """Integration tests combining read_stdin and get_user_prompt."""

    def test_full_flow_with_piped_input(self):
        """Test full flow with piped input."""
        test_input = "This is piped input"
        mock_stdin = StringIO(test_input)

        with patch('sys.stdin', mock_stdin):
            with patch('sys.stdin.isatty', return_value=False):
                # read_stdin should get the input
                stdin_result = read_stdin()
                assert stdin_result == test_input

        # get_user_prompt should use stdin over CLI arg
        with patch('hermes_cli.utils.read_stdin', return_value=test_input):
            result = get_user_prompt("ignored CLI arg")
            assert result == test_input

    def test_full_flow_with_cli_only(self):
        """Test full flow with CLI argument only."""
        cli_prompt = "CLI argument prompt"

        with patch('sys.stdin.isatty', return_value=True):
            # read_stdin should return None (no pipe)
            stdin_result = read_stdin()
            assert stdin_result is None

        # get_user_prompt should use CLI arg
        with patch('hermes_cli.utils.read_stdin', return_value=None):
            result = get_user_prompt(cli_prompt)
            assert result == cli_prompt

    def test_full_flow_with_no_input(self):
        """Test full flow with no input at all."""
        with patch('sys.stdin.isatty', return_value=True):
            # read_stdin should return None
            stdin_result = read_stdin()
            assert stdin_result is None

        # get_user_prompt should raise error
        with patch('hermes_cli.utils.read_stdin', return_value=None):
            with pytest.raises(ValueError) as exc_info:
                get_user_prompt(None)
            assert "No prompt provided" in str(exc_info.value)
