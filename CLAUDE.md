# Hermes CLI Tool

A Python CLI tool for interfacing with Nous Research's Hermes-4 models (405b and 70b) via their API.

## Project Overview

**Tool Name**: `hermes`  
**Installation Method**: uv tool  
**Primary Models**: Hermes-4-405b, Hermes-4-70b  
**Key Feature**: Native JSON schema-based structured output support

## Core Functionality

### Basic Usage Patterns

```bash
# Simple user prompt (streams by default)
hermes "What is the capital of France?"

# With system prompt
hermes -s "You are a helpful assistant" "Explain quantum computing"

# Piped input
cat prompt.txt | hermes
echo "Summarize this" | hermes -s "Be concise"

# Disable streaming for complete response
hermes --no-stream "Give me a short answer"

# Structured JSON output with schema (auto-disables streaming)
hermes --schema '{"type": "object", "properties": {"answer": {"type": "string"}}}' "What is 2+2?"

# Schema from file
hermes --schema ./schemas/response.json "Analyze this data"

# Explicitly disable streaming with schema
hermes --no-stream --schema ./schema.json "Respond with JSON"

# Use hermes-4-70b
hermes --m "hermes-4-70b" "Why is Rust better than C++?"
```

### CLI Arguments

- **Positional argument** (required): User prompt text
- `-s, --system`: System prompt (optional)
- `--schema`: JSON schema for structured output (accepts JSON string or file path)
- `--stream / --no-stream`: Enable/disable streaming output (default: `--stream`)
- `--m / --model`: Model to be used for completion (default: hermres-4-405b,optional: hermes-4-70b)
- Standard input: Accepts piped input for user prompt

### Schema Behavior

When `--schema` is provided:
1. Load the JSON schema (from string or file)
2. Append JSON output instructions to system prompt
3. The complete system prompt should be: `{user_system_prompt}\n\nYou must respond with valid JSON matching this schema: {schema_json}`
4. If no user system prompt exists, just use the schema instructions

## Technical Requirements

### Project Structure

```
hermes-cli/
├── pyproject.toml          # uv-compatible project config
├── README.md               # User documentation
├── CLAUDE.md              # This file
├── src/
│   └── hermes_cli/
│       ├── __init__.py
│       ├── main.py        # CLI entry point
│       ├── api.py         # Nous Research API client
│       ├── schema.py      # Schema handling logic
│       └── utils.py       # Helper functions
└── tests/
    └── test_*.py          # Pytest tests
```

### Dependencies

**Core**:
- `requests` - HTTP client for API calls
- `click` or `typer` - CLI framework (recommend `click` for simplicity)
- `pydantic` - JSON schema validation 

**Dev**:
- `pytest` - Testing
- `pytest-cov` - Coverage

### Installation as uv Tool

The `pyproject.toml` must:
1. Define a `[project.scripts]` entry pointing to the CLI entry point
2. Be compatible with `uv tool install`
3. Include all required dependencies

Example entry point:
```toml
[project.scripts]
hermes = "hermes_cli.main:cli"
```

## API Integration Details

### Nous Research API Documentation

**Base URL**: `https://api.nousresearch.com/v1`

**Authentication**: API key via `Authorization: Bearer <token>` header

**Chat Completions Endpoint**: `POST /chat/completions`

**Request Format**:
```json
{
  "model": "hermes-4-405b" | "hermes-4-70b",
  "messages": [
    {"role": "system", "content": "system prompt here"},
    {"role": "user", "content": "user prompt here"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": true
}
```

**Response Format**:
```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "hermes-4-405b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "response text here"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

**Available Models**:
- `hermes-4-405b` - Larger, more capable model
- `hermes-4-70b` - Faster, efficient model

**Error Handling**: API returns standard HTTP error codes with JSON error messages

### API Configuration

- API key should be read from environment variable `NOUS_API_KEY`
- Default model: `hermes-4-405b` (can switch to hermes-4-70b with --m/--model flag)
- Handle rate limits and network errors gracefully
- Provide clear error messages for missing API key

## Implementation Guidelines

### Input Handling

1. **Piped Input Detection**: Check if stdin is piped using `sys.stdin.isatty()`
2. **Priority**: Piped input should override positional argument if both exist
3. **Validation**: Ensure either piped input or positional argument is provided

### Schema Processing

1. **Detection**: Check if argument starts with `{` or `[` for JSON string vs file path
2. **File Loading**: Use `pathlib.Path` for file operations
3. **Validation**: Validate JSON before sending (parse with `json.loads()`)
4. **Error Messages**: Clear errors for invalid JSON or missing files

### Output

- Print only the assistant's response content to stdout
- Errors and warnings should go to stderr
- For JSON responses, consider pretty-printing with `json.dumps(indent=2)`

### Error Handling

**Must handle**:
- Missing API key
- Invalid API responses
- Network errors
- Invalid JSON schema
- File not found (for schema files)
- Empty/missing prompts

**Error message format**: Clear, actionable messages on stderr

## Testing Strategy

### Unit Tests
- Schema loading (string vs file)
- System prompt construction with schemas
- Input detection (piped vs argument)
- API request building

### Integration Tests
- Mock API responses
- End-to-end CLI invocation
- Error scenarios

### Manual Testing Commands
```bash
# After installation with: uv tool install .

# Basic test
hermes "Hello"

# With system prompt
hermes -s "Be brief" "What is AI?"

# Piped input
echo "Explain Python" | hermes

# Schema from string
hermes --schema '{"type":"object","properties":{"answer":{"type":"string"}}}' "What is 2+2?"

# Schema from file (create test file first)
echo '{"type":"object","properties":{"result":{"type":"string"}}}' > test_schema.json
hermes --schema test_schema.json "Respond with JSON"
```

## Environment Setup

Users should set `NOUS_API_KEY` environment variable:
```bash
export NOUS_API_KEY="your-api-key-here"
```

Consider adding a check on first run to guide users if key is missing.

## Nice-to-Have Features (Future)

- `--temperature` flag for controlling randomness
- `--max-tokens` flag for response length
- Configuration file support (~/.hermes/config.toml)
- Conversation history/context

## Success Criteria

✅ Installs cleanly via `uv tool install`  
✅ Handles all specified usage patterns  
✅ Properly integrates with Nous Research API  
✅ Schema functionality works with both strings and files  
✅ Clear error messages for common issues  
✅ Comprehensive test coverage  
✅ Works with piped input from other commands