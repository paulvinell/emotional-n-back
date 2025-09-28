import os
import shutil

import typer

from emotional_n_back.constants import DATA_DIR
from emotional_n_back.game import (
    AudioNBackGame,
    AudioSentimentNBackGame,
    AudioSentimentVisualPositionDualNBack,
    EmotionalDualNBack,
    SentimentDualNBack,
    VisualNBackGame,
    VisualSentimentNBackGame,
)
from emotional_n_back.nback import DualNBackTerminal
from emotional_n_back.stroop import (
    AlternatingStroopGame,
    SentimentStroopGame,
)
from emotional_n_back.speed_reading import SpeedReadingGame

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
def dual(
    length: int = 20,
    n: int = 2,
    stim_ms: int = 2000,
    seed: int | None = None,
    help_labels: bool = False,
):
    game = EmotionalDualNBack(
        length=length,
        n=n,
        repeat_probability=0.35,
        seed=seed,
        stim_ms=stim_ms,
        isi_ms=500,
        show_debug_labels=False,  # set True to display current emotions + GT
        show_help_labels=help_labels,
    )
    game.run()


@app.command()
def visual(
    length: int = 20,
    n: int = 2,
    stim_ms: int = 2000,
    seed: int | None = None,
    help_labels: bool = False,
):
    game = VisualNBackGame(
        length=length,
        n=n,
        repeat_probability=0.5,
        seed=seed,
        stim_ms=stim_ms,
        isi_ms=500,
        show_debug_labels=False,  # set True to display current emotions + GT
        show_help_labels=help_labels,
    )
    game.run()


@app.command()
def audio(
    length: int = 20,
    n: int = 2,
    stim_ms: int = 2000,
    seed: int | None = None,
    help_labels: bool = False,
):
    game = AudioNBackGame(
        length=length,
        n=n,
        repeat_probability=0.5,
        seed=seed,
        stim_ms=stim_ms,
        isi_ms=500,
        show_debug_labels=False,  # set True to display current emotions + GT
        show_help_labels=help_labels,
    )
    game.run()


@app.command()
def visual_sentiment(
    length: int = 20,
    n: int = 2,
    stim_ms: int = 2000,
    seed: int | None = None,
    help_labels: bool = False,
    binary: bool = False,
):
    game = VisualSentimentNBackGame(
        length=length,
        n=n,
        repeat_probability=0.5,
        seed=seed,
        stim_ms=stim_ms,
        isi_ms=500,
        show_debug_labels=False,
        show_help_labels=help_labels,
        binary=binary,
    )
    game.run()


@app.command()
def audio_sentiment(
    length: int = 20,
    n: int = 2,
    stim_ms: int = 2000,
    seed: int | None = None,
    help_labels: bool = False,
    binary: bool = False,
):
    game = AudioSentimentNBackGame(
        length=length,
        n=n,
        repeat_probability=0.5,
        seed=seed,
        stim_ms=stim_ms,
        isi_ms=500,
        show_debug_labels=False,
        show_help_labels=help_labels,
        binary=binary,
    )
    game.run()


@app.command()
def dual_sentiment(
    length: int = 20,
    n: int = 2,
    stim_ms: int = 2000,
    seed: int | None = None,
    help_labels: bool = False,
    binary: bool = False,
):
    game = SentimentDualNBack(
        length=length,
        n=n,
        repeat_probability=0.35,
        seed=seed,
        stim_ms=stim_ms,
        isi_ms=500,
        show_debug_labels=False,
        show_help_labels=help_labels,
        binary=binary,
    )
    game.run()


@app.command()
def dual_position_sentiment(
    length: int = 20,
    n: int = 2,
    stim_ms: int = 2000,
    seed: int | None = None,
    help_labels: bool = False,
    binary: bool = False,
):
    game = AudioSentimentVisualPositionDualNBack(
        length=length,
        n=n,
        repeat_probability=0.35,
        seed=seed,
        stim_ms=stim_ms,
        isi_ms=500,
        show_debug_labels=False,
        show_help_labels=help_labels,
        binary=binary,
    )
    game.run()


@app.command()
def stroop(
    length: int = 30,
    seed: int | None = None,
    visual_intro_ms: int = 500,
    response_window_ms: int = 2000,
):
    game = SentimentStroopGame(
        length=length,
        seed=seed,
        visual_intro_ms=visual_intro_ms,
        response_window_ms=response_window_ms,
    )
    game.run()


@app.command()
def alternating_stroop(
    length: int = 30,
    seed: int | None = None,
    intro_delay_ms: int = 500,
    response_window_ms: int = 2000,
):
    game = AlternatingStroopGame(
        length=length,
        seed=seed,
        intro_delay_ms=intro_delay_ms,
        response_window_ms=response_window_ms,
    )
    game.run()


@app.command()
def speed_reading(
    lang: str = "en",
    speed: int = 200,
    audio_distraction_freq: float = 0.1,
    visual_distraction_freq: float = 0.3,
    visual_distraction_duration: int = 700,
    seed: int | None = None,
):
    """
    Run the speed reading game with distractions.
    """
    game = SpeedReadingGame(
        language=lang,
        scroll_speed=speed,
        audio_distraction_freq=audio_distraction_freq,
        visual_distraction_freq=visual_distraction_freq,
        visual_distraction_duration_ms=visual_distraction_duration,
        seed=seed,
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

    # Remove black images
    black_images = {
        "neutral": ["48_13", "42_30"],
        "disgust": ["63_33", "116_32"],
    }
    for emotion, img_basenames in black_images.items():
        for img_basename in img_basenames:
            img_path = kdef_dst / emotion / f"{img_basename}.jpg"
            if img_path.exists():
                print(f"Removing {img_path}...")
                img_path.unlink()
    # Clean up Kaggle datasets directory
    if kaggle_datasets_dir.exists():
        shutil.rmtree(kaggle_datasets_dir)


@app.command()
def tess():
    tess_dst = DATA_DIR / "tess"
    if tess_dst.exists():
        print(f"{tess_dst} already exists, deleting...")
        shutil.rmtree(tess_dst)

    import kagglehub

    os.environ["DISABLE_COLAB_CACHE"] = "true"
    os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)
    kagglehub.dataset_download(
        "ejlok1/toronto-emotional-speech-set-tess", force_download=True
    )

    kaggle_datasets_dir = DATA_DIR / "datasets"
    src_dir = kaggle_datasets_dir / "ejlok1" / "toronto-emotional-speech-set-tess"
    versions = list(src_dir.glob("versions/*"))
    assert len(versions) == 1, "Expected exactly one version of the TESS dataset"

    tess_src = versions[0] / "TESS Toronto emotional speech set data"
    if tess_dst.exists():
        if any(tess_dst.iterdir()):
            raise FileExistsError(f"{tess_dst} already exists and is not empty")
    else:
        tess_dst.mkdir(parents=True)

    tess_src_subfolder = tess_src / "TESS Toronto emotional speech set data"
    for item in tess_src_subfolder.iterdir():
        shutil.move(str(item), tess_dst / item.name)

    # Rename dir OAF_Pleasant_surprise to OAF_pleasant_surprise
    # Rename dir YAF_pleasant_surprised to YAF_pleasant_surprise
    for item in tess_dst.iterdir():
        if item.name == "OAF_Pleasant_surprise":
            item.rename(tess_dst / "OAF_pleasant_surprise")
        elif item.name == "YAF_pleasant_surprised":
            item.rename(tess_dst / "YAF_pleasant_surprise")

    # Clean up Kaggle datasets directory
    if kaggle_datasets_dir.exists():
        shutil.rmtree(kaggle_datasets_dir)


if __name__ == "__main__":
    app()
