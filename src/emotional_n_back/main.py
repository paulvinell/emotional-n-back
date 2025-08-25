import typer

from emotional_n_back.game import DualNBackPygame
from emotional_n_back.nback import DualNBackTerminal

app = typer.Typer()


@app.command()
def main(name: str):
    print(f"Hello {name}")


@app.command()
def cli():
    DualNBackTerminal(
        length=20,
        n=2,
        repeat_probability=0.3,
        distinct_items=4,
        seed=42,
        show_truth=True,
    ).play()


@app.command()
def gui():
    DualNBackPygame(
        length=20,
        n=1,
        visual_items=9,
        auditory_items=8,
        repeat_probability=0.3,
        seed=42,
        stim_ms=1000,
        isi_ms=500,
        show_feedback_ms=300,
    ).run()


if __name__ == "__main__":
    app()
