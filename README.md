# Hermes CLI

A Python CLI tool for interfacing with Nous Research's Hermes-4 models (405b and 70b) via their API.

## Features

- Simple command-line interface for interacting with Hermes-4 models
- Support for both streaming and non-streaming responses
- JSON schema-based structured output support
- System prompt configuration
- Piped input support
- Choice between Hermes-4-405b and Hermes-4-70b models

## Installation

Install using `uv tool`:

```bash
uv tool install .
```

Or install from the repository:

```bash
uv tool install git+https://github.com/yourusername/hermes-cli
```

## Configuration

Set your Nous Research API key as an environment variable:

```bash
export NOUS_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
# Simple user prompt (streams by default)
hermes "What is the capital of France?"

# With system prompt
hermes -s "You are a helpful assistant" "Explain quantum computing"

# Disable streaming for complete response
hermes --no-stream "Give me a short answer"
```

### Piped Input

```bash
# Pipe content as user prompt
cat prompt.txt | hermes

# Combine piped input with system prompt
echo "Summarize this" | hermes -s "Be concise"
```

### Structured JSON Output

```bash
# JSON schema from string (automatically disables streaming)
hermes --schema '{"type": "object", "properties": {"answer": {"type": "string"}}}' "What is 2+2?"

# JSON schema from file
hermes --schema ./schemas/response.json "Analyze this data"

# Explicitly disable streaming with schema
hermes --no-stream --schema ./schema.json "Respond with JSON"
```

### Model Selection

```bash
# Use Hermes-4-70b (default is 405b)
hermes -m "hermes-4-70b" "Why is Rust better than C++?"

# Or use the full flag
hermes --model "hermes-4-70b" "Explain neural networks"
```

## Command-Line Options

- **Positional argument**: User prompt text (required unless piped)
- `-s, --system`: System prompt (optional)
- `--schema`: JSON schema for structured output (JSON string or file path)
- `--stream / --no-stream`: Enable/disable streaming output (default: `--stream`)
- `-m, --model`: Model to use - `hermes-4-405b` or `hermes-4-70b` (default: `hermes-4-405b`)

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/hermes-cli
cd hermes-cli

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
