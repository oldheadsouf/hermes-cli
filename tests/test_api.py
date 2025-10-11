"""Tests for API client."""

import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from hermes_cli.api import NousAPIClient, APIError


class TestNousAPIClientInitialization:
    """Tests for NousAPIClient initialization."""

    def test_init_with_api_key_parameter(self):
        """Test initialization with explicit API key."""
        client = NousAPIClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == "Bearer test-key-123"
        assert client.session.headers["Content-Type"] == "application/json"

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with NOUS_API_KEY environment variable."""
        monkeypatch.setenv("NOUS_API_KEY", "env-key-456")
        client = NousAPIClient()
        assert client.api_key == "env-key-456"
        assert client.session.headers["Authorization"] == "Bearer env-key-456"

    def test_init_without_api_key_raises_error(self, monkeypatch):
        """Test that initialization fails without API key."""
        monkeypatch.delenv("NOUS_API_KEY", raising=False)
        with pytest.raises(ValueError) as exc_info:
            NousAPIClient()
        assert "API key not found" in str(exc_info.value)
        assert "NOUS_API_KEY" in str(exc_info.value)

    def test_explicit_api_key_overrides_env_var(self, monkeypatch):
        """Test that explicit API key takes precedence over env var."""
        monkeypatch.setenv("NOUS_API_KEY", "env-key")
        client = NousAPIClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_base_url_is_set_correctly(self):
        """Test that BASE_URL is set to the correct endpoint."""
        assert NousAPIClient.BASE_URL == "https://inference-api.nousresearch.com/v1"


class TestChatCompletion:
    """Tests for chat_completion method."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return NousAPIClient(api_key="test-key")

    def test_chat_completion_non_streaming_success(self, client):
        """Test successful non-streaming chat completion."""
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "Hermes-4-405B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        with patch.object(client.session, 'post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            result = client.chat_completion(messages, stream=False)

            assert result == mock_response
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['json']['model'] == "Hermes-4-405B"
            assert call_args[1]['json']['messages'] == messages
            assert call_args[1]['json']['stream'] is False

    def test_chat_completion_with_custom_parameters(self, client):
        """Test chat completion with custom model, temperature, and max_tokens."""
        with patch.object(client.session, 'post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"choices": []}

            messages = [{"role": "user", "content": "Test"}]
            client.chat_completion(
                messages,
                model="Hermes-4-70B",
                temperature=0.5,
                max_tokens=1024,
                stream=False
            )

            call_args = mock_post.call_args[1]['json']
            assert call_args['model'] == "Hermes-4-70B"
            assert call_args['temperature'] == 0.5
            assert call_args['max_tokens'] == 1024

    def test_chat_completion_default_temperature(self, client):
        """Test that default temperature is 0.7."""
        with patch.object(client.session, 'post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"choices": []}

            messages = [{"role": "user", "content": "Test"}]
            client.chat_completion(messages, stream=False)

            call_args = mock_post.call_args[1]['json']
            assert call_args['temperature'] == 0.7

    def test_chat_completion_default_max_tokens(self, client):
        """Test that default max_tokens is 2048."""
        with patch.object(client.session, 'post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"choices": []}

            messages = [{"role": "user", "content": "Test"}]
            client.chat_completion(messages, stream=False)

            call_args = mock_post.call_args[1]['json']
            assert call_args['max_tokens'] == 2048

    def test_chat_completion_extreme_temperature_values(self, client):
        """Test chat completion with extreme temperature values."""
        with patch.object(client.session, 'post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"choices": []}

            messages = [{"role": "user", "content": "Test"}]

            # Test minimum temperature (0.0)
            client.chat_completion(messages, temperature=0.0, stream=False)
            call_args = mock_post.call_args[1]['json']
            assert call_args['temperature'] == 0.0

            # Test maximum temperature (2.0)
            client.chat_completion(messages, temperature=2.0, stream=False)
            call_args = mock_post.call_args[1]['json']
            assert call_args['temperature'] == 2.0

    def test_chat_completion_custom_max_tokens_values(self, client):
        """Test chat completion with various max_tokens values."""
        with patch.object(client.session, 'post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"choices": []}

            messages = [{"role": "user", "content": "Test"}]

            # Test small value
            client.chat_completion(messages, max_tokens=100, stream=False)
            call_args = mock_post.call_args[1]['json']
            assert call_args['max_tokens'] == 100

            # Test large value
            client.chat_completion(messages, max_tokens=4096, stream=False)
            call_args = mock_post.call_args[1]['json']
            assert call_args['max_tokens'] == 4096

    def test_chat_completion_streaming_success(self, client):
        """Test successful streaming chat completion."""
        mock_chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n',
            b'data: [DONE]\n'
        ]

        with patch.object(client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = iter(mock_chunks)
            mock_post.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            result = client.chat_completion(messages, stream=True)

            # Collect streamed content
            content = list(result)
            assert content == ["Hello", " world"]

    def test_chat_completion_streaming_with_empty_lines(self, client):
        """Test streaming handles empty lines correctly."""
        mock_chunks = [
            b'',
            b'data: {"choices": [{"delta": {"content": "Test"}}]}\n',
            b'',
            b'data: [DONE]\n'
        ]

        with patch.object(client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = iter(mock_chunks)
            mock_post.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            result = client.chat_completion(messages, stream=True)
            content = list(result)
            assert content == ["Test"]

    def test_chat_completion_streaming_skips_empty_content(self, client):
        """Test streaming skips chunks with empty content."""
        mock_chunks = [
            b'data: {"choices": [{"delta": {"content": ""}}]}\n',
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'data: {"choices": [{"delta": {}}]}\n',
            b'data: [DONE]\n'
        ]

        with patch.object(client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = iter(mock_chunks)
            mock_post.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            result = client.chat_completion(messages, stream=True)
            content = list(result)
            assert content == ["Hello"]

    def test_chat_completion_http_error(self, client):
        """Test handling of HTTP errors."""
        with patch.object(client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {
                "error": {"message": "Invalid API key"}
            }
            mock_post.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            with pytest.raises(APIError) as exc_info:
                client.chat_completion(messages, stream=False)

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value)

    def test_chat_completion_timeout_error(self, client):
        """Test handling of timeout errors."""
        with patch.object(client.session, 'post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()

            messages = [{"role": "user", "content": "Hello"}]
            with pytest.raises(APIError) as exc_info:
                client.chat_completion(messages, stream=False)

            assert "timed out" in str(exc_info.value).lower()

    def test_chat_completion_connection_error(self, client):
        """Test handling of connection errors."""
        with patch.object(client.session, 'post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()

            messages = [{"role": "user", "content": "Hello"}]
            with pytest.raises(APIError) as exc_info:
                client.chat_completion(messages, stream=False)

            assert "connect" in str(exc_info.value).lower()

    def test_chat_completion_generic_request_exception(self, client):
        """Test handling of generic request exceptions."""
        with patch.object(client.session, 'post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("Something went wrong")

            messages = [{"role": "user", "content": "Hello"}]
            with pytest.raises(APIError) as exc_info:
                client.chat_completion(messages, stream=False)

            assert "Request failed" in str(exc_info.value)


class TestParseErrorResponse:
    """Tests for _parse_error_response method."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return NousAPIClient(api_key="test-key")

    def test_parse_error_with_dict_error(self, client):
        """Test parsing error response with dict error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Bad request error"}
        }

        error_msg = client._parse_error_response(mock_response)
        assert error_msg == "Bad request error"

    def test_parse_error_with_string_error(self, client):
        """Test parsing error response with string error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": "Internal server error"
        }

        error_msg = client._parse_error_response(mock_response)
        assert error_msg == "Internal server error"

    def test_parse_error_fallback_400(self, client):
        """Test fallback error message for 400."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)

        error_msg = client._parse_error_response(mock_response)
        assert "Bad request" in error_msg

    def test_parse_error_fallback_401(self, client):
        """Test fallback error message for 401."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {}

        error_msg = client._parse_error_response(mock_response)
        assert "Invalid API key" in error_msg

    def test_parse_error_fallback_403(self, client):
        """Test fallback error message for 403."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {}

        error_msg = client._parse_error_response(mock_response)
        assert "forbidden" in error_msg.lower()

    def test_parse_error_fallback_404(self, client):
        """Test fallback error message for 404."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {}

        error_msg = client._parse_error_response(mock_response)
        assert "not found" in error_msg.lower()

    def test_parse_error_fallback_429(self, client):
        """Test fallback error message for 429."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {}

        error_msg = client._parse_error_response(mock_response)
        assert "Rate limit" in error_msg

    def test_parse_error_fallback_500(self, client):
        """Test fallback error message for 500."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}

        error_msg = client._parse_error_response(mock_response)
        assert "server error" in error_msg.lower()

    def test_parse_error_fallback_503(self, client):
        """Test fallback error message for 503."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.json.return_value = {}

        error_msg = client._parse_error_response(mock_response)
        assert "unavailable" in error_msg.lower()

    def test_parse_error_fallback_unknown_status(self, client):
        """Test fallback error message for unknown status code."""
        mock_response = Mock()
        mock_response.status_code = 418  # I'm a teapot
        mock_response.json.return_value = {}

        error_msg = client._parse_error_response(mock_response)
        assert "418" in error_msg


class TestStreamResponse:
    """Tests for _stream_response method."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return NousAPIClient(api_key="test-key")

    def test_stream_response_normal_flow(self, client):
        """Test streaming with normal flow."""
        mock_chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'data: {"choices": [{"delta": {"content": " there"}}]}\n',
            b'data: [DONE]\n'
        ]

        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_chunks)

        result = list(client._stream_response(mock_response))
        assert result == ["Hello", " there"]

    def test_stream_response_with_malformed_json(self, client, capsys):
        """Test streaming handles malformed JSON gracefully."""
        mock_chunks = [
            b'data: {"choices": [{"delta": {"content": "Good"}}]}\n',
            b'data: {invalid json}\n',
            b'data: {"choices": [{"delta": {"content": " data"}}]}\n',
            b'data: [DONE]\n'
        ]

        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_chunks)

        result = list(client._stream_response(mock_response))
        assert result == ["Good", " data"]

        # Check that warning was printed to stderr
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "Failed to parse chunk" in captured.err

    def test_stream_response_with_missing_choices(self, client):
        """Test streaming handles missing choices gracefully."""
        mock_chunks = [
            b'data: {"choices": [{"delta": {"content": "Start"}}]}\n',
            b'data: {"no_choices": true}\n',
            b'data: {"choices": [{"delta": {"content": " End"}}]}\n',
            b'data: [DONE]\n'
        ]

        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_chunks)

        result = list(client._stream_response(mock_response))
        assert result == ["Start", " End"]

    def test_stream_response_exception_during_iteration(self, client):
        """Test streaming handles exceptions during iteration."""
        def raise_exception():
            yield b'data: {"choices": [{"delta": {"content": "Start"}}]}\n'
            raise RuntimeError("Stream interrupted")

        mock_response = Mock()
        mock_response.iter_lines.return_value = raise_exception()

        with pytest.raises(APIError) as exc_info:
            list(client._stream_response(mock_response))

        assert "Error while streaming" in str(exc_info.value)
        assert "Stream interrupted" in str(exc_info.value)

    def test_stream_response_non_data_lines(self, client):
        """Test streaming ignores lines that don't start with 'data: '."""
        mock_chunks = [
            b'event: message\n',
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'id: 123\n',
            b'data: [DONE]\n'
        ]

        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_chunks)

        result = list(client._stream_response(mock_response))
        assert result == ["Hello"]


class TestAPIError:
    """Tests for APIError exception class."""

    def test_api_error_with_status_code(self):
        """Test APIError with status code."""
        error = APIError("Test error", status_code=404)
        assert error.message == "Test error"
        assert error.status_code == 404
        assert str(error) == "Test error"

    def test_api_error_without_status_code(self):
        """Test APIError without status code."""
        error = APIError("Another error")
        assert error.message == "Another error"
        assert error.status_code is None
        assert str(error) == "Another error"


class TestIntegration:
    """Integration tests with actual API calls."""

    @pytest.mark.skipif(
        not os.getenv("NOUS_API_KEY"),
        reason="NOUS_API_KEY not set - skipping integration test"
    )
    def test_real_api_call_non_streaming(self):
        """Test actual API call with Hermes-4-70B (non-streaming)."""
        client = NousAPIClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "Say 'test successful' and nothing else."}
        ]

        result = client.chat_completion(
            messages,
            model="Hermes-4-70B",
            temperature=0.1,
            max_tokens=50,
            stream=False
        )

        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]
        assert "content" in result["choices"][0]["message"]
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert len(result["choices"][0]["message"]["content"]) > 0

    @pytest.mark.skipif(
        not os.getenv("NOUS_API_KEY"),
        reason="NOUS_API_KEY not set - skipping integration test"
    )
    def test_real_api_call_streaming(self):
        """Test actual API call with Hermes-4-70B (streaming)."""
        client = NousAPIClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be very brief."},
            {"role": "user", "content": "Count to 3."}
        ]

        result = client.chat_completion(
            messages,
            model="Hermes-4-70B",
            temperature=0.1,
            max_tokens=50,
            stream=True
        )

        # Collect streamed chunks
        chunks = list(result)

        # Verify we got some content
        assert len(chunks) > 0

        # Verify all chunks are strings
        assert all(isinstance(chunk, str) for chunk in chunks)

        # Verify combined content is non-empty
        full_content = "".join(chunks)
        assert len(full_content) > 0
