# GPT Chess Arena

Any two OpenAI models play chess against each other. Pick the models in a web app, or run a round-robin tournament from the command line.

[![Watch the video](https://img.youtube.com/vi/jeHYvkHVPsA/maxresdefault.jpg)](https://youtu.be/jeHYvkHVPsA)

**Demo Video:** [GPT-6 Luna vs. GPT-6 Luna on YouTube](https://youtu.be/jeHYvkHVPsA). Two copies of `gpt-6-luna` play each other, and the game ends in a draw after 40 moves (80 half-moves).

**Demo Video (Earlier Version):** [Video of the original code on LinkedIn](https://lnkd.in/p/g4R-76pg).

**Replay the Game:** Paste the PGN into [chess.com/analysis](https://www.chess.com/analysis).

## How It Works

1. The program shows the current models, the board, the FEN, the move history, and the list of legal moves.
2. The model replies with a single line: `MOVE: <uci>`.
3. The move is checked against the legal moves. If the reply is missing or illegal, the program retries up to 5 times, then falls back to a random legal move.
4. When the game ends, it saves a PGN file in `output/`.

## Models

Any two models can play each other, including the same model against itself. Prices are in USD per 1M tokens, checked against [developers.openai.com](https://developers.openai.com/api/docs/pricing) on 08-Nov-2026. Check that page for current prices before long runs.

| Model | Input | Output | Status |
|---|---|---|---|
| `gpt-6-luna` | $0.10 | $0.50 | current |
| `gpt-6.1-terra` | $1.00 | $5.00 | current |
| `gpt-6.1-sol` | $2.00 | $10.00 | current |
| `gpt-6-astra` | $10.00 | $50.00 | current |
| `gpt-5-mini` | $0.25 | $2.00 | legacy |
| `gpt-5` | $1.25 | $10.00 | legacy |
| `o3` | $2.00 | $8.00 | legacy |
| `gpt-4o-mini` | $0.15 | $0.60 | legacy |

Legacy models are deprecated by OpenAI and may stop working. Reasoning tokens bill as output tokens, so reasoning models cost more per move than the table suggests. The program prints an estimated cost based on token counts, and it ignores cached-input discounts.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
git clone https://github.com/dsadrianzadeh/gpt_chess_arena.git
cd gpt_chess_arena
uv sync
```

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=sk-your-key-here
```

## Usage

Start the web app, then open http://127.0.0.1:7860:

```bash
uv run python -m gpt_chess_arena.main
```

Choose White and Black, set the reasoning effort and the half-move limit, then click **Start Game**. For a cheap first test, use `gpt-6-luna` with a limit of 6 half-moves.

Run a round robin, where every pair plays once:

```bash
uv run python -m gpt_chess_arena.main --tournament --models gpt-6-luna gpt-5-mini --max-plies 40
```

| Flag | Meaning |
|---|---|
| `--tournament` | Skip the web app and play every pair of models once |
| `--models` | Models to include (default: all) |
| `--max-plies` | Half-moves before a game is declared a draw (default: 80) |
| `--effort` | `none`, `minimal`, `low`, `medium` or `high` (default: `medium`) |
| `--both-colors` | Play each pair twice, swapping colors |

Each model uses the closest effort level it supports. `gpt-4o-mini` has no reasoning effort.

## Output

Games are saved as PGN files in `output/` (e.g., `20261108_004512_gpt-6-luna_vs_gpt-6-luna.pgn`). To replay one, open [chess.com/analysis](https://www.chess.com/analysis) and import the PGN file.

A game that reaches the half-move limit is recorded as a draw with termination `MOVE_LIMIT`. That is a rule of this program, not a draw under the rules of chess.

## Project Layout

```text
gpt_chess_arena/
├── docs/example/
│   └── 20261108_012543_gpt-6-luna_vs_gpt-6-luna.pgn  # the game from the video
├── src/gpt_chess_arena/
│   ├── __init__.py
│   └── main.py  # models, game loop, web app, tournament
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
└── uv.lock
```

## Notes

- Models can play illegal or weak moves. Results show how each model behaves under this prompt, not its true chess strength.
- API keys stay in `.env`, which is ignored by Git.

Built with Python, [python-chess](https://python-chess.readthedocs.io/), [Gradio](https://www.gradio.app/), and the [OpenAI API](https://platform.openai.com/).
