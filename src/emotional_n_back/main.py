import os
import shutil

import typer

from emotional_n_back.constants import DATA_DIR
from emotional_n_back.game import DualNBackPygame, EmotionalDualNBack
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


@app.command()
def emo():
    game = EmotionalDualNBack(
        length=20,
        n=2,
        repeat_probability=0.3,
        seed=42,
        stim_ms=900,
        isi_ms=500,
        feedback_ms=220,
        show_debug_labels=False,  # set True to display current emotions + GT
    )
    game.run()


@app.command()
def kdef():
    kdef_dst = DATA_DIR / "kdef"
    if kdef_dst.exists():
        print(f"{kdef_dst} already exists, deleting...")
        shutil.rmtree(kdef_dst)

    import kagglehub

    os.environ["DISABLE_COLAB_CACHE"] = "true"
    os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)
    kagglehub.dataset_download("chenrich/kdef-database", force_download=True)

    kaggle_datasets_dir = DATA_DIR / "datasets"
    src_dir = kaggle_datasets_dir / "chenrich" / "kdef-database"
    versions = list(src_dir.glob("versions/*"))
    assert len(versions) == 1, "Expected exactly one version of the KDEF dataset"

    kdef_src = versions[0]
    if kdef_dst.exists():
        if any(kdef_dst.iterdir()):
            raise FileExistsError(f"{kdef_dst} already exists and is not empty")
    else:
        kdef_dst.mkdir(parents=True)

    for item in kdef_src.iterdir():
        shutil.move(str(item), kdef_dst / item.name)

    # Clean up Kaggle datasets directory
    if kaggle_datasets_dir.exists():
        shutil.rmtree(kaggle_datasets_dir)


if __name__ == "__main__":
    app()
