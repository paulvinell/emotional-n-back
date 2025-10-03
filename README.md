# Emotional N-Back

This project contains a collection of N-back and other cognitive tasks with an emotional component.

For a review of the cognitive benefits of emotional n-back training, see [this article](https://www.iqmindware.com/increase-cognitive-resilience/emotional-dual-n-back-science/).

### Classes of games
There's three main classes of games included:

* Emotional N-Back games
  * Challenges working memory and emotional regulation
* Emotional Stroop games
  * Challenges cognitive control (inhibition) and emotional regulation
* Speed-reading with distraction
  * Meant to be read out loud
  * Speed-reading out loud is thought to improve working memory, verbal fluency, and processing speed

## Prerequisites

This project uses `uv` for package management. Before you begin, please install `uv`.

Please refer to the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/paulvinell/emotional-n-back.git
    cd emotional-n-back
    ```

2.  Install the dependencies using `uv`:
    ```bash
    uv sync
    ```

## Usage

The project uses `typer` to create a command-line interface. You can see all available commands by running:

```bash
uv run nback --help
```

### Downloading Datasets

The project requires the KDEF and TESS datasets. You can download and prepare them using the following commands:

```bash
uv run nback kdef
uv run nback tess
```

Alternatively, you can use the `Makefile`:

```bash
make data
```

This will download both datasets. You can also download them individually:

```bash
make kdef
make tess
```

For now, the MAV and text datasets are included in the repository.

## Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for details.

## License

This project is licensed under CC BY-NC 4.0. See the `LICENSE` file for details.
