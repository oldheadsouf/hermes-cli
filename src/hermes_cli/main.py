"""Main CLI entry point for Hermes CLI."""

import sys
import json
import click
from io import StringIO

from hermes_cli.api import NousAPIClient, APIError
from hermes_cli.utils import get_user_prompt, format_with_border
from hermes_cli.schema import load_schema, build_system_prompt_with_schema
from hermes_cli.chat import ConversationManager
from hermes_cli.tools import ToolRegistry, ToolExecutor


def _execute_with_tools(
    client: NousAPIClient,
    messages: list[dict],
    selected_tools: dict,
    tool_schemas: list[dict],
    executor: ToolExecutor,
    model: str,
    temperature: float,
    max_tokens: int,
    max_calls: int,
    border: bool,
    schema_dict: dict = None
):
    """Execute request with tool calling loop.

    Args:
        client: API client instance
        messages: Initial messages list
        selected_tools: Dict of enabled tool functions
        tool_schemas: List of tool schemas for API
        executor: ToolExecutor instance
        model: Model name
        temperature: Temperature setting
        max_tokens: Max tokens setting
        max_calls: Maximum tool call iterations
        border: Whether to format with border
        schema_dict: Optional JSON schema
    """
    current_messages = messages.copy()

    for iteration in range(max_calls):
        try:
            # Make API request with tools
            response = client.chat_completion(
                messages=current_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,  # Required for tool use
                tools=tool_schemas
            )

            # Check if we have a response
            if "choices" not in response or len(response["choices"]) == 0:
                click.echo("Error: No response from API", err=True)
                sys.exit(1)

            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            message = choice.get("message", {})

            if finish_reason == "tool_calls":
                # Model wants to call tools
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    click.echo("Error: finish_reason is tool_calls but no tool_calls in response", err=True)
                    break

                # Display which tools are being called
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    click.echo(f"[Calling tool: {func_name}]", err=True)

                # Add assistant message with tool calls to history
                current_messages.append(message)

                # Execute tools and add results
                tool_results = executor.execute_tool_calls(tool_calls, selected_tools)
                current_messages.extend(tool_results)

                # Continue loop to get final answer

            else:
                # Final answer reached
                content = message.get("content", "")

                # Format JSON if schema provided
                if schema_dict and content:
                    try:
                        parsed_json = json.loads(content)
                        content = json.dumps(parsed_json, indent=2)
                    except json.JSONDecodeError:
                        pass

                # Display result
                if border:
                    click.echo(format_with_border(content, model))
                else:
                    click.echo(content)

                return  # Success

        except APIError as e:
            click.echo(f"API Error: {e.message}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)

    # Reached max iterations
    click.echo(f"Warning: Reached maximum tool call limit ({max_calls})", err=True)


def _execute_chat_with_tools(
    client: NousAPIClient,
    conversation_data: dict,
    conversation_name: str,
    conv_manager: ConversationManager,
    selected_tools: dict,
    tool_schemas: list[dict],
    executor: ToolExecutor,
    tools_config: dict,
    border: bool
):
    """Execute chat message with tool calling loop.

    Args:
        client: API client
        conversation_data: Full conversation data dict
        conversation_name: Name of conversation
        conv_manager: ConversationManager instance
        selected_tools: Dict of enabled tools
        tool_schemas: Tool schemas for API
        executor: ToolExecutor instance
        tools_config: Tool configuration dict
        border: Whether to format with border
    """
    model = conversation_data.get("model")
    temperature = conversation_data.get("temperature")
    max_tokens = conversation_data.get("max_tokens")
    schema_dict = conversation_data.get("schema")
    max_calls = tools_config.get('max_calls', 5)

    messages = conversation_data["messages"].copy()

    for iteration in range(max_calls):
        try:
            response = client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                tools=tool_schemas
            )

            if "choices" not in response or len(response["choices"]) == 0:
                click.echo("Error: No response from API", err=True)
                sys.exit(1)

            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            message = choice.get("message", {})

            if finish_reason == "tool_calls":
                tool_calls = message.get("tool_calls", [])

                # Display tool calls
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    click.echo(f"[Calling tool: {func_name}]", err=True)

                # Add assistant message with tool calls
                messages.append(message)
                conversation_data["messages"].append(message)

                # Execute tools
                tool_results = executor.execute_tool_calls(tool_calls, selected_tools)

                # Add tool results to messages and save
                for result in tool_results:
                    messages.append(result)
                    conversation_data["messages"].append(result)

                conv_manager.save_conversation(conversation_name, conversation_data)

            else:
                # Final answer
                content = message.get("content", "")

                if schema_dict and content:
                    try:
                        parsed_json = json.loads(content)
                        display_content = json.dumps(parsed_json, indent=2)
                    except json.JSONDecodeError:
                        display_content = content
                else:
                    display_content = content

                # Display
                if border:
                    click.echo(format_with_border(display_content, model))
                else:
                    click.echo(display_content)

                # Save final assistant message
                conversation_data["messages"].append({"role": "assistant", "content": content})
                conv_manager.save_conversation(conversation_name, conversation_data)

                return

        except APIError as e:
            click.echo(f"API Error: {e.message}", err=True)
            sys.exit(1)

    click.echo(f"Warning: Reached maximum tool call limit ({max_calls})", err=True)


class HermesGroup(click.Group):
    """Custom Group that treats unknown commands as prompt arguments."""

    def resolve_command(self, ctx, args):
        """Override to treat unknown commands as prompt arguments."""
        # If no args, use parent behavior
        if not args:
            return super().resolve_command(ctx, args)

        # Check if first arg is a known subcommand
        cmd_name = args[0]
        if cmd_name in self.commands:
            # Valid subcommand
            return super().resolve_command(ctx, args)

        # Not a subcommand - just treat all args as prompt
        # Don't try to resolve as a command, just leave args for group callback
        # We signal no subcommand by raising a special exception that invoke() catches
        raise click.exceptions.UsageError(f"__PROMPT_ARGS__{args}", ctx=ctx)

    def invoke(self, ctx):
        """Override invoke to handle prompt arguments."""
        try:
            return super().invoke(ctx)
        except click.exceptions.UsageError as e:
            # Check if this is our special marker for prompt args
            if str(e).startswith("__PROMPT_ARGS__"):
                # Extract the args from the error message
                import ast
                args_str = str(e).replace("__PROMPT_ARGS__", "")
                try:
                    args = ast.literal_eval(args_str)
                    ctx.args = list(args)
                    ctx.invoked_subcommand = None
                    # Invoke just the group callback
                    with ctx.scope(cleanup=False):
                        return click.Group.invoke(self, ctx)
                except (SystemExit, KeyboardInterrupt):
                    # Let system exits and keyboard interrupts propagate
                    raise
                except Exception:
                    # Only catch parsing errors, not system control flow
                    pass
            # Re-raise if not our special case
            raise


# Main group - handles both direct prompts and subcommands
@click.group(cls=HermesGroup, invoke_without_command=True, context_settings={'allow_interspersed_args': False})
@click.pass_context
@click.option("-s", "--system", help="System prompt")
@click.option("--schema", help="JSON schema for structured output (JSON string or file path)")
@click.option("--stream/--no-stream", "stream", default=None, help="Enable/disable streaming output (default: stream enabled, auto-disabled with --schema)")
@click.option("-m", "--model", default="Hermes-4-405B", help="Model to use (Hermes-4-405B or Hermes-4-70B)")
@click.option("-t", "--temperature", type=float, default=0.7, help="Sampling temperature (0.0-2.0, default: 0.7)")
@click.option("-mt", "--max-tokens", type=int, default=2048, help="Maximum tokens in response (default: 2048)")
@click.option("-b", "--border", is_flag=True, default=False, help="Format output with a decorative ASCII border")
@click.option("--use-tools", help="Enable tool use. Comma-separated tool names or 'all' for all available tools")
@click.option("--max-tool-calls", type=int, default=5, help="Maximum recursive tool call iterations (default: 5)")
def cli(ctx, system, schema, stream, model, temperature, max_tokens, border, use_tools, max_tool_calls):
    """Hermes CLI - Interface with Nous Research's Hermes-4 models.

    Examples:

      hermes "What is the capital of France?"

      hermes -s "You are a helpful assistant" "Explain quantum computing"

      echo "Summarize this" | hermes -s "Be concise"

      hermes --schema '{"type": "object", "properties": {"answer": {"type": "string"}}}' "What is 2+2?"

      hermes --use-tools calculate "What is 15 * 23?"

      hermes --use-tools all "Search the web for Python tutorials"

      hermes chat --name "my-chat" "Start a conversation"

      hermes chat --load "my-chat"

      hermes chat exit
    """
    # Store tool config in context for subcommands
    ctx.ensure_object(dict)
    if use_tools:
        ctx.obj['tools_config'] = {
            'use_tools': use_tools,
            'max_calls': max_tool_calls
        }
    else:
        ctx.obj['tools_config'] = None

    # If a subcommand is being invoked, don't run main logic
    if ctx.invoked_subcommand is not None:
        return

    # Otherwise, treat remaining args as prompt for main command
    try:
        # Get prompt from remaining args or stdin
        # ctx.args contains unparsed arguments after options
        prompt = " ".join(ctx.args) if ctx.args else None
        user_prompt = get_user_prompt(prompt)

        # Handle schema if provided
        schema_dict = None
        if schema:
            schema_dict = load_schema(schema)
            # When schema is provided, build system prompt with schema instructions
            system = build_system_prompt_with_schema(system, schema_dict)

        # Determine streaming behavior
        if stream is None:
            use_streaming = not bool(schema_dict)
        else:
            use_streaming = stream

        # Build messages array
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prompt})

        # Initialize API client
        try:
            client = NousAPIClient()
        except ValueError as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)

        # Check if tools are enabled
        tools_config = ctx.obj.get('tools_config')

        if tools_config:
            # Tool use mode - requires non-streaming
            use_streaming = False

            # Initialize tool registry and select tools
            try:
                registry = ToolRegistry()
                selected_tools = registry.select_tools(tools_config['use_tools'])
                tool_schemas = registry.get_tool_schemas(selected_tools)
                executor = ToolExecutor(registry)
            except ValueError as e:
                click.echo(f"Error: {str(e)}", err=True)
                sys.exit(1)

            # Execute with tool loop
            _execute_with_tools(
                client=client,
                messages=messages,
                selected_tools=selected_tools,
                tool_schemas=tool_schemas,
                executor=executor,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_calls=tools_config['max_calls'],
                border=border,
                schema_dict=schema_dict
            )
            return

        # Make API request
        try:
            response = client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=use_streaming
            )

            if use_streaming:
                # Handle streaming response
                if border:
                    collected_output = StringIO()
                    for chunk in response:
                        collected_output.write(chunk)
                    output_text = collected_output.getvalue()
                    click.echo(format_with_border(output_text, model))
                else:
                    for chunk in response:
                        click.echo(chunk, nl=False)
                    click.echo()
            else:
                # Handle non-streaming response
                if "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]

                    if schema_dict:
                        try:
                            parsed_json = json.loads(content)
                            content = json.dumps(parsed_json, indent=2)
                        except json.JSONDecodeError:
                            pass

                    if border:
                        click.echo(format_with_border(content, model))
                    else:
                        click.echo(content)
                else:
                    click.echo("Error: No response content received", err=True)
                    sys.exit(1)

        except APIError as e:
            click.echo(f"API Error: {e.message}", err=True)
            sys.exit(1)

    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
@click.argument("prompt", required=False)
@click.option("-n", "--name", help="Name for the conversation (required for new chats)")
@click.option("-l", "--load", "load_name", help="Load an existing conversation by name")
@click.option("-s", "--system", help="System prompt (only for new conversations)")
@click.option("--schema", help="JSON schema for structured output (JSON string or file path)")
@click.option("--stream/--no-stream", "stream", default=None, help="Enable/disable streaming output")
@click.option("-m", "--model", default="Hermes-4-405B", help="Model to use (Hermes-4-405B or Hermes-4-70B)")
@click.option("-t", "--temperature", type=float, default=0.7, help="Sampling temperature (0.0-2.0, default: 0.7)")
@click.option("-mt", "--max-tokens", type=int, default=2048, help="Maximum tokens in response (default: 2048)")
@click.option("--use-tools", help="Enable tool use. Comma-separated tool names or 'all' for all available tools")
@click.option("-b", "--border", is_flag=True, default=False, help="Format output with a decorative ASCII border")
def chat(ctx, prompt, name, load_name, system, schema, stream, model, temperature, max_tokens, border):
    """Start or continue a conversational chat session.

    Examples:

      hermes chat --name "my-chat" "Hello, how are you?"

      hermes chat --load "my-chat"

      hermes chat exit
    """
    try:
        conv_manager = ConversationManager()

        # Handle "exit" command
        if prompt and prompt.lower() == "exit":
            active = conv_manager.get_active_session()
            if active:
                conv_manager.clear_active_session()
                click.echo(f"Exited conversation '{active}'")
            else:
                click.echo("No active conversation to exit")
            return

        # Check if there's an active session
        active_session = conv_manager.get_active_session()

        # Determine which conversation to use
        if load_name:
            # Load an existing conversation
            conversation_name = load_name
            try:
                conversation_data = conv_manager.load_conversation(conversation_name)
                conv_manager.set_active_session(conversation_name)
                click.echo(f"Loaded conversation '{conversation_name}'", err=True)

                # If no prompt provided, just activate the session
                if not prompt:
                    click.echo("Session activated. Use 'hermes chat <prompt>' to continue the conversation.", err=True)
                    return

            except FileNotFoundError as e:
                click.echo(f"Error: {str(e)}", err=True)
                sys.exit(1)

        elif name:
            # Create a new conversation
            if not prompt:
                click.echo("Error: Prompt is required when creating a new conversation", err=True)
                sys.exit(1)

            # Get user prompt
            user_prompt = get_user_prompt(prompt)

            # Handle schema if provided
            schema_dict = None
            if schema:
                schema_dict = load_schema(schema)
                system = build_system_prompt_with_schema(system, schema_dict)

            # Get tools config from context
            tools_config = ctx.obj.get('tools_config')

            # Create the conversation with the first message
            actual_name, conv_path = conv_manager.create_conversation(
                name=name,
                initial_message={"role": "user", "content": user_prompt},
                system_prompt=system,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                schema=schema_dict,
                tools_config=tools_config
            )

            if actual_name != name:
                click.echo(f"Note: Conversation name '{name}' already exists. Created as '{actual_name}' instead.", err=True)

            # Set as active session
            conv_manager.set_active_session(actual_name)
            conversation_name = actual_name
            conversation_data = conv_manager.load_conversation(conversation_name)

            click.echo(f"Created new conversation '{actual_name}'", err=True)

        elif active_session:
            # Continue the active session
            if not prompt:
                click.echo(f"Active session: '{active_session}'. Provide a prompt to continue.", err=True)
                return

            conversation_name = active_session
            conversation_data = conv_manager.load_conversation(conversation_name)

            # Get user prompt and add to conversation
            user_prompt = get_user_prompt(prompt)
            conversation_data["messages"].append({"role": "user", "content": user_prompt})
            # Save the user message immediately
            conv_manager.save_conversation(conversation_name, conversation_data)

        else:
            # No active session and no --name or --load provided
            click.echo(
                "Error: No active conversation session.\n"
                "Use --name to create a new conversation or --load to resume an existing one.\n"
                "Examples:\n"
                "  hermes chat --name 'my-chat' 'Hello'\n"
                "  hermes chat --load 'my-chat'",
                err=True
            )
            sys.exit(1)

        # At this point, we have a conversation_data with messages to send
        # Get settings from conversation data
        model = conversation_data.get("model", model)
        temperature = conversation_data.get("temperature", temperature)
        max_tokens = conversation_data.get("max_tokens", max_tokens)
        schema_dict = conversation_data.get("schema")

        # Determine tool configuration for this message
        # Priority: CLI flag > conversation default > None
        tools_config = None

        if ctx.obj.get('tools_config'):
            # CLI flag takes precedence
            tools_config = ctx.obj['tools_config']
        elif 'conversation_data' in locals():
            # Use conversation default
            tools_config = conversation_data.get('tools_config')

        # Determine streaming behavior
        if stream is None:
            use_streaming = not bool(schema_dict) and not bool(tools_config)
        else:
            use_streaming = stream

        # Initialize API client
        try:
            client = NousAPIClient()
        except ValueError as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)

        # Determine if we should use tools for this request
        if tools_config:
            # Initialize tool system
            try:
                registry = ToolRegistry()
                selected_tools = registry.select_tools(tools_config['use_tools'])
                tool_schemas = registry.get_tool_schemas(selected_tools)
                executor = ToolExecutor(registry)
            except ValueError as e:
                click.echo(f"Error: {str(e)}", err=True)
                sys.exit(1)

            # Execute with tools
            _execute_chat_with_tools(
                client=client,
                conversation_data=conversation_data,
                conversation_name=conversation_name,
                conv_manager=conv_manager,
                selected_tools=selected_tools,
                tool_schemas=tool_schemas,
                executor=executor,
                tools_config=tools_config,
                border=border
            )
            return

        # Make API request with full conversation history
        try:
            response = client.chat_completion(
                messages=conversation_data["messages"],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=use_streaming
            )

            # Collect the assistant's response
            if use_streaming:
                # Handle streaming response
                if border:
                    collected_output = StringIO()
                    for chunk in response:
                        collected_output.write(chunk)
                    assistant_content = collected_output.getvalue()
                    click.echo(format_with_border(assistant_content, model))
                else:
                    collected_chunks = []
                    for chunk in response:
                        click.echo(chunk, nl=False)
                        collected_chunks.append(chunk)
                    click.echo()
                    assistant_content = "".join(collected_chunks)
            else:
                # Handle non-streaming response
                if "choices" in response and len(response["choices"]) > 0:
                    assistant_content = response["choices"][0]["message"]["content"]

                    if schema_dict:
                        try:
                            parsed_json = json.loads(assistant_content)
                            display_content = json.dumps(parsed_json, indent=2)
                        except json.JSONDecodeError:
                            display_content = assistant_content
                    else:
                        display_content = assistant_content

                    if border:
                        click.echo(format_with_border(display_content, model))
                    else:
                        click.echo(display_content)
                else:
                    click.echo("Error: No response content received", err=True)
                    sys.exit(1)

            # Save the assistant's response to the conversation
            conv_manager.add_message(conversation_name, "assistant", assistant_content)

        except APIError as e:
            click.echo(f"API Error: {e.message}", err=True)
            sys.exit(1)

    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def tools():
    """Manage available tools."""
    pass


@tools.command(name="list")
@click.option("--builtin", is_flag=True, help="Show only built-in tools")
@click.option("--user", is_flag=True, help="Show only user-defined tools")
def list_tools(builtin, user):
    """List all available tools."""
    try:
        registry = ToolRegistry()
        available = registry.list_tools()

        if not user and available["builtin"]:
            click.echo("Built-in tools:")
            for name, desc in sorted(available["builtin"].items()):
                click.echo(f"  {name}")
                click.echo(f"    {desc}")

        if not builtin and available["user"]:
            if available["builtin"] and not user:
                click.echo()
            click.echo("User-defined tools:")
            for name, desc in sorted(available["user"].items()):
                click.echo(f"  {name}")
                click.echo(f"    {desc}")

        if not available["builtin"] and not available["user"]:
            click.echo("No tools available")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@tools.command()
@click.argument("tool_name")
def show(tool_name):
    """Show detailed information about a specific tool."""
    try:
        registry = ToolRegistry()
        info = registry.get_tool_info(tool_name)

        click.echo(f"Tool: {info['name']}")
        click.echo(f"Type: {info['source']}")
        click.echo(f"Description: {info['description']}")
        click.echo()
        click.echo("Parameters:")

        params = info['parameters']
        if params.get('properties'):
            for param_name, param_def in params['properties'].items():
                required = " (required)" if param_name in params.get('required', []) else ""
                param_type = param_def.get('type', 'unknown')
                param_desc = param_def.get('description', 'No description')
                click.echo(f"  {param_name}: {param_type}{required}")
                click.echo(f"    {param_desc}")
        else:
            click.echo("  None")

    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
