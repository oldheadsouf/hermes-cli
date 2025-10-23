"""Main CLI entry point for Hermes CLI."""

import sys
import json
import click
from io import StringIO

from hermes_cli.api import NousAPIClient, APIError
from hermes_cli.utils import get_user_prompt, format_with_border
from hermes_cli.schema import load_schema, build_system_prompt_with_schema
from hermes_cli.chat import ConversationManager


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
def cli(ctx, system, schema, stream, model, temperature, max_tokens, border):
    """Hermes CLI - Interface with Nous Research's Hermes-4 models.

    Examples:

      hermes "What is the capital of France?"

      hermes -s "You are a helpful assistant" "Explain quantum computing"

      echo "Summarize this" | hermes -s "Be concise"

      hermes --schema '{"type": "object", "properties": {"answer": {"type": "string"}}}' "What is 2+2?"

      hermes chat --name "my-chat" "Start a conversation"

      hermes chat --load "my-chat"

      hermes chat exit
    """
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
@click.argument("prompt", required=False)
@click.option("-n", "--name", help="Name for the conversation (required for new chats)")
@click.option("-l", "--load", "load_name", help="Load an existing conversation by name")
@click.option("-s", "--system", help="System prompt (only for new conversations)")
@click.option("--schema", help="JSON schema for structured output (JSON string or file path)")
@click.option("--stream/--no-stream", "stream", default=None, help="Enable/disable streaming output")
@click.option("-m", "--model", default="Hermes-4-405B", help="Model to use (Hermes-4-405B or Hermes-4-70B)")
@click.option("-t", "--temperature", type=float, default=0.7, help="Sampling temperature (0.0-2.0, default: 0.7)")
@click.option("-mt", "--max-tokens", type=int, default=2048, help="Maximum tokens in response (default: 2048)")
@click.option("-b", "--border", is_flag=True, default=False, help="Format output with a decorative ASCII border")
def chat(prompt, name, load_name, system, schema, stream, model, temperature, max_tokens, border):
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

            # Create the conversation with the first message
            actual_name, conv_path = conv_manager.create_conversation(
                name=name,
                initial_message={"role": "user", "content": user_prompt},
                system_prompt=system,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                schema=schema_dict
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

        # Determine streaming behavior
        if stream is None:
            use_streaming = not bool(schema_dict)
        else:
            use_streaming = stream

        # Initialize API client
        try:
            client = NousAPIClient()
        except ValueError as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)

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


if __name__ == "__main__":
    cli()
