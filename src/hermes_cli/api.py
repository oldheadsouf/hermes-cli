"""Nous Research API client."""

import os
import json
import sys
from typing import Optional, Dict, Any, Iterator
import requests


class APIError(Exception):
    """Exception raised for API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class NousAPIClient:
    """Client for interacting with Nous Research API."""

    BASE_URL = "https://inference-api.nousresearch.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the API client.

        Args:
            api_key: Nous Research API key. If not provided, reads from NOUS_API_KEY env var.

        Raises:
            ValueError: If API key is not provided and NOUS_API_KEY env var is not set
        """
        self.api_key = api_key or os.getenv("NOUS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set the NOUS_API_KEY environment variable.\n"
                "Example: export NOUS_API_KEY='your-api-key-here'"
            )

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def chat_completion(
        self,
        messages: list[Dict[str, str]],
        model: str = "Hermes-4-405B",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = True
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Send a chat completion request to the API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use (Hermes-4-405B or Hermes-4-70B)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response

        Returns:
            Response dict or iterator of response chunks if streaming

        Raises:
            APIError: If the API request fails
        """
        url = f"{self.BASE_URL}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            response = self.session.post(url, json=payload, stream=stream, timeout=30)

            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = self._parse_error_response(response)
                raise APIError(error_msg, response.status_code)

            if stream:
                return self._stream_response(response)
            else:
                return response.json()

        except requests.exceptions.Timeout:
            raise APIError("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError as e:
            raise APIError("Failed to connect to Nous Research API. Please check your internet connection.")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}")

    def _parse_error_response(self, response: requests.Response) -> str:
        """Parse error response from API.

        Args:
            response: The failed response object

        Returns:
            Human-readable error message
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                if isinstance(error_data["error"], dict):
                    return error_data["error"].get("message", str(error_data["error"]))
                return str(error_data["error"])
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback to status code message
        status_messages = {
            400: "Bad request. Please check your input.",
            401: "Invalid API key. Please check your NOUS_API_KEY.",
            403: "Access forbidden. Please check your API key permissions.",
            404: "API endpoint not found.",
            429: "Rate limit exceeded. Please try again later.",
            500: "Internal server error. Please try again later.",
            503: "Service unavailable. Please try again later."
        }

        return status_messages.get(
            response.status_code,
            f"API request failed with status code {response.status_code}"
        )

    def _stream_response(self, response: requests.Response) -> Iterator[str]:
        """Stream SSE response from API.

        Args:
            response: The streaming response object

        Yields:
            Content chunks from the assistant's response
        """
        try:
            for line in response.iter_lines():
                if not line:
                    continue

                # Decode bytes to string
                line_str = line.decode('utf-8')

                # SSE format: "data: {json}"
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove "data: " prefix

                    # Check for end of stream
                    if data_str.strip() == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data_str)

                        # Extract content from chunk
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                yield content

                    except json.JSONDecodeError:
                        # Skip malformed JSON chunks
                        print(f"Warning: Failed to parse chunk: {data_str}", file=sys.stderr)
                        continue

        except Exception as e:
            raise APIError(f"Error while streaming response: {str(e)}")
