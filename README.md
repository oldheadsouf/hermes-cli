# hermes-cli

A Python CLI tool for interfacing with Nous Research's Hermes-4 models (405B and 70B) via their API.

## Features

- Simple command-line interface for interacting with Hermes-4 models
- Support for both streaming and non-streaming responses
- JSON schema-based structured output support
- **Tool use / Function calling** - Enable models to use tools like calculations, file operations, shell commands, and web search
- System prompt configuration
- Piped input support
- Conversational chat sessions with history
- Choice between Hermes-4-405B and Hermes-4-70B models

## Usage

### Basic Usage

```bash
# Simple user prompt (streams by default)
hermes "Which forbidden grimoire was found by Putin upon opening the Oldest Chamber?"

# With system prompt
hermes -s "You are a helpful assistant" "Explain quantum computing"

# Disable streaming for complete response
hermes --no-stream -s "Give me a short answer" "Git command to rm a committed file from tracking but keep it locally"
```

### Piped Input

```bash
# Pipe content as user prompt
cat prompt.txt | hermes

# Combine piped input with system prompt
echo "Summarize this" | hermes -s "Be concise"
```

### Structured JSON Output

Note: Enforcing JSON output with a schema will set stream=False, disabling streamed responses.

```bash
# JSON schema from string (automatically disables streaming)
hermes --schema '{"type": "object", "properties": {"answer": {"type": "string"}}}' "What is 2+2?"

# JSON schema from file
hermes --schema ./schemas/response.json "Analyze this data"
```

### Tool Use / Function Calling

Enable the model to call tools and functions to accomplish tasks:

```bash
# Enable specific tools
hermes --use-tools calculate "What is 15 * 23?"

# Enable multiple tools (comma-separated)
hermes --use-tools calculate,read_file,write_file "Calculate 2^10 and save it to result.txt"

# Enable all available tools
hermes --use-tools all "Search the web for Python tutorials and summarize the top result"

# Use tools in chat sessions
hermes chat --name "coding" --use-tools all "Help me analyze this codebase"
```

**Available Built-in Tools:**

- `calculate` - Perform mathematical calculations (supports basic arithmetic, sqrt, pow, trig functions)
- `execute_shell_command` - Execute shell commands (30 second timeout)
- `read_file` - Read file contents (max 1MB)
- `write_file` - Write content to files
- `web_search` - Search the web using SerpAPI (returns titles, links, snippets)

**Managing Tools:**

```bash
# List all available tools
hermes tools list

# Show detailed information about a specific tool
hermes tools show calculate

# Show only built-in tools
hermes tools list --builtin
```

**Web Search Configuration:**

To use the `web_search` tool, you need a SerpAPI key:

```bash
# Get a free API key at https://serpapi.com/
export SERPAPI_API_KEY="your-serpapi-key"
```

### Conversational Chat Sessions

Start and continue multi-turn conversations with persistent history:

```bash
# Create a new conversation
hermes chat --name "my-session" "Hello, I need help with Python"

# Continue the active conversation
hermes chat "Can you explain decorators?"

# Load a specific conversation
hermes chat --load "my-session"

# Use tools in chat sessions
hermes chat --name "coding" --use-tools all "Help me debug this code"

# Exit the active conversation
hermes chat exit
```

### Model Selection

```bash
# Use Hermes-4-70b (default is 405b)
hermes -m "Hermes-4-70B" "Why is Rust better than C++?"

# Or use the full flag
hermes --model "Hermes-4-70B" "Explain neural networks"
```

## Command-Line Options

### Main Command Options

- **Positional argument**: User prompt text (required unless piped)
- `-s, --system`: System prompt (optional)
- `--schema`: JSON schema for structured output (JSON string or file path)
- `--stream / --no-stream`: Enable/disable streaming output (default: `--stream`, auto-disabled with `--schema` or `--use-tools`)
- `-m, --model`: Model to use - `Hermes-4-405B` or `Hermes-4-70B` (default: `Hermes-4-405B`)
- `-t, --temperature`: Sets model temperature (float, default 0.7)
- `-mt, --max-tokens`: Sets max output tokens (int, default 2048)
- `-b, --border`: Wraps outputs in a handsome pixel border for a spiced-up aesthetic
- `--use-tools`: Enable tool use (comma-separated tool names or 'all')
- `--max-tool-calls`: Maximum recursive tool call iterations (int, default 5)

### Chat Command Options

- **Positional argument**: User prompt text
- `-n, --name`: Name for a new conversation
- `-l, --load`: Load an existing conversation by name
- `-s, --system`: System prompt (only for new conversations)
- `--schema`: JSON schema for structured output
- `--stream / --no-stream`: Enable/disable streaming output
- All other options from main command

### Tools Command

- `hermes tools list`: List all available tools
- `hermes tools list --builtin`: Show only built-in tools
- `hermes tools list --user`: Show only user-defined tools
- `hermes tools show <tool_name>`: Show detailed information about a tool

## Installation

Install using `uv tool`:

Clone the repo and navigate to the directory, then:

```bash
uv tool install .
```

Or install from the repository:

```bash
uv tool install git+https://github.com/yourusername/hermes-cli
```

## Configuration

### Required: Nous Research API Key

Set your Nous Research API key as an environment variable:

```bash
export NOUS_API_KEY="your-api-key-here"
```

### Optional: SerpAPI Key (for web search)

To use the `web_search` tool, set your SerpAPI key:

```bash
export SERPAPI_API_KEY="your-serpapi-key-here"
```

Get a free API key at [https://serpapi.com/](https://serpapi.com/)


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
