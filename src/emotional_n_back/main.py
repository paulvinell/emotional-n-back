import typer

app = typer.Typer()


@app.command()
def main(name: str):
    print(f"Hello {name}")


@app.command()
def cli():
    typer.echo("This is the CLI command.")


if __name__ == "__main__":
    app()
