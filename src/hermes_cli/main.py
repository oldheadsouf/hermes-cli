"""Main CLI entry point for Hermes CLI."""

import click


@click.command()
@click.argument("prompt", required=False)
@click.option("-s", "--system", help="System prompt")
@click.option("--schema", help="JSON schema for structured output (JSON string or file path)")
@click.option("--stream/--no-stream", default=True, help="Enable/disable streaming output")
@click.option("-m", "--model", default="hermes-4-405b", help="Model to use (hermes-4-405b or hermes-4-70b)")
def cli(prompt, system, schema, stream, model):
    """Hermes CLI - Interface with Nous Research's Hermes-4 models."""
    # TODO: Implement CLI logic
    click.echo("Hermes CLI - Not yet implemented")


if __name__ == "__main__":
    cli()
