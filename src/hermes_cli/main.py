"""Main CLI entry point for Hermes CLI."""

import sys
import json
import click
from io import StringIO

from hermes_cli.api import NousAPIClient, APIError
from hermes_cli.utils import get_user_prompt, format_with_border
from hermes_cli.schema import load_schema, build_system_prompt_with_schema


@click.command()
@click.argument("prompt", required=False)
@click.option("-s", "--system", help="System prompt")
@click.option("--schema", help="JSON schema for structured output (JSON string or file path)")
@click.option("--stream/--no-stream", "stream", default=None, help="Enable/disable streaming output (default: stream enabled, auto-disabled with --schema)")
@click.option("-m", "--model", default="Hermes-4-405B", help="Model to use (Hermes-4-405B or Hermes-4-70B)")
@click.option("-t", "--temperature", type=float, default=0.7, help="Sampling temperature (0.0-2.0, default: 0.7)")
@click.option("-mt", "--max-tokens", type=int, default=2048, help="Maximum tokens in response (default: 2048)")
@click.option("-b", "--border", is_flag=True, default=False, help="Format output with a decorative ASCII border")
def cli(prompt, system, schema, stream, model, temperature, max_tokens, border):
    """Hermes CLI - Interface with Nous Research's Hermes-4 models.

    Examples:

      hermes "What is the capital of France?"

      hermes -s "You are a helpful assistant" "Explain quantum computing"

      echo "Summarize this" | hermes -s "Be concise"

      hermes --schema '{"type": "object", "properties": {"answer": {"type": "string"}}}' "What is 2+2?"
    """
    try:
        # Get user prompt from CLI argument or stdin
        user_prompt = get_user_prompt(prompt)

        # Handle schema if provided
        schema_dict = None
        if schema:
            schema_dict = load_schema(schema)
            # When schema is provided, build system prompt with schema instructions
            system = build_system_prompt_with_schema(system, schema_dict)

        # Determine streaming behavior
        # If stream is None, user didn't explicitly set it
        # Auto-disable streaming when schema is provided (unless explicitly overridden)
        if stream is None:
            # Default behavior: stream enabled, but disabled when schema is used
            use_streaming = not bool(schema_dict)
        else:
            # User explicitly set --stream or --no-stream, respect their choice
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
                    # Collect all chunks first for border formatting
                    collected_output = StringIO()
                    for chunk in response:
                        collected_output.write(chunk)
                    output_text = collected_output.getvalue()
                    click.echo(format_with_border(output_text, model))
                else:
                    for chunk in response:
                        click.echo(chunk, nl=False)
                    # Add newline at end of streaming output
                    click.echo()
            else:
                # Handle non-streaming response
                if "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]

                    # If schema was used, try to pretty-print JSON
                    if schema_dict:
                        try:
                            parsed_json = json.loads(content)
                            content = json.dumps(parsed_json, indent=2)
                        except json.JSONDecodeError:
                            # If it's not valid JSON, just use as-is
                            pass

                    # Apply border if requested
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


if __name__ == "__main__":
    cli()
