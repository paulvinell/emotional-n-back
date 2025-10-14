import os
import shutil

import typer

from emotional_n_back.constants import DATA_DIR
from emotional_n_back.games.eeg.eeg_stroop.game import EEGStroopGame
from emotional_n_back.games.game import (
    AudioNBackGame,
    AudioSentimentNBackGame,
    AudioSentimentVisualPositionDualNBack,
    EmotionalDualNBack,
    SentimentDualNBack,
    VisualNBackGame,
    VisualSentimentNBackGame,
)
from emotional_n_back.games.regular.speed_reading import SpeedReadingGame
from emotional_n_back.games.regular.stroop import (
    AlternatingStroopGame,
    SentimentStroopGame,
)

app = typer.Typer()
osc_app = typer.Typer()
app.add_typer(osc_app, name="osc")


@osc_app.command()
def reader(
    ip: str = "127.0.0.1",
    port: int = 5005,
    visualize: bool = typer.Option(
        False, "--visualize", help="Visualize incoming EEG signal."
    ),
    buffer_size: int = 256 * 5,  # 5 seconds of data at 256Hz
):
    """OSC reader."""
    from emotional_n_back.utils.osc_reader import OSCReader

    osc_reader = OSCReader(ip, port, visualize, buffer_size)
    osc_reader.start()


@osc_app.command()
def writer(
    mode: str = typer.Argument(
        "dummy", help="The streaming mode to use. Can be 'dummy' or 'eeg'."
    ),
    ip: str = "127.0.0.1",
    port: int = 5005,
    address: str = None,
    message: str = "Hello OSC",
    fs: int = 256,
    duration: float = 20.0,
    seed: int = 7,
    rate_per_sec: float = 0.5,
    dur_range: str = "0.2,0.8",
    amp_range: str = "0.15,0.7",
    eeg_path: str = typer.Option(
        None, "--eeg-path", help="Path to the EEG recording file."
    ),
):
    """OSC writer."""
    from emotional_n_back.utils.streaming.factory import create_streamer

    try:
        streamer = create_streamer(
            mode=mode,
            ip=ip,
            port=port,
            address=address,
            message=message,
            fs=fs,
            duration=duration,
            seed=seed,
            rate_per_sec=rate_per_sec,
            dur_range=dur_range,
            amp_range=amp_range,
            eeg_path=eeg_path,
        )
        streamer.stream()
    except ValueError as e:
        print(e)
        raise typer.Exit(code=1)


@app.command()
def dual(
    length: int = 20,
    n: int = 2,
    stim_ms: int = 2000,
    seed: int | None = None,
    help_labels: bool = False,
):
    """
    Run an emotional dual n-back game with visual and audio stimuli.
    """
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
    """
    Run a visual n-back game.
    """
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
    """
    Run an audio n-back game.
    """
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
    """
    Run a visual n-back game with sentiment classification.
    """
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
    """
    Run an audio n-back game with sentiment classification.
    """
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
    """
    Run a dual n-back game with sentiment classification on both audio and visual streams.
    """
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
    """
    Run a dual n-back game with audio sentiment and visual position.
    """
    game = AudioSentimentVisualPositionDualNBack(
        length=length,
        n=n,
        repeat_probability=[0.25, 0.35],
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
    """
    Run a sentiment Stroop test.
    """
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
    """
    Run an alternating Stroop test.
    """
    game = AlternatingStroopGame(
        length=length,
        seed=seed,
        intro_delay_ms=intro_delay_ms,
        response_window_ms=response_window_ms,
    )
    game.run()


@app.command()
def eeg_stroop(
    seed: int | None = None,
    stimulus_intro_ms: int = 500,
    initial_calibration_trials: int = typer.Option(
        10, help="Number of trials for initial calibration."
    ),
    recalibration_interval: int = typer.Option(
        10, help="How often to recalibrate the threshold."
    ),
    window_size: str = "900,650",
):
    """Run an EEG-integrated sentiment Stroop test."""
    import pygame
    from emotional_n_back.games.eeg.eeg_stroop.game import EEGStroopGame
    from emotional_n_back.games.eeg.eeg_stroop.render import GameRenderer
    from emotional_n_back.games.eeg.eeg_stroop.thread import GameThread

    renderer = GameRenderer(window_size=tuple(map(int, window_size.split(','))))
    game = EEGStroopGame(
        seed=seed,
        stimulus_intro_ms=stimulus_intro_ms,
        initial_calibration_trials=initial_calibration_trials,
        recalibration_interval=recalibration_interval,
        window_size=tuple(map(int, window_size.split(','))),
    )
    game.start()

    game_thread = GameThread(game)
    game_thread.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
        
        renderer.render_game(game)
        renderer.clock.tick(60)
    
    game_thread.stop()
    game_thread.join()
    game.shutdown()
    pygame.quit()


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
    """
    Download and prepare the KDEF dataset.
    """
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
    """
    Download and prepare the TESS dataset.
    """
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
