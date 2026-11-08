"""main.py
GPT Chess Arena: Any two OpenAI models play chess against each other.

Setup:
    uv add chess openai gradio python-dotenv
    Put `OPENAI_API_KEY=sk-...` in `.env`.

Run the web app (then open http://127.0.0.1:7860):
    uv run python main.py

Run a round robin (every pair of models plays once):
    uv run python main.py --tournament --models gpt-6-luna gpt-5-mini
"""


import argparse, itertools, os, random, re, textwrap, time

import chess
import chess.pgn
import gradio as gr

from collections.abc import Generator
from datetime import date, datetime
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from pathlib import Path


# Build paths relative to the file, so the program works from any
# working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Read `.env` into environment variables. `override=True` makes `.env`
# win over a stale variable that is already set in the shell.
load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=True,
)

API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Model roster. Any two entries can play each other.
# - kind: selects the request settings below.
# - input: price in USD per 1M input tokens.
# - output: price in USD per 1M output tokens (reasoning tokens count).
# - status: "current", or "legacy" (deprecated, may stop working).
# Prices checked against developers.openai.com on 08-Nov-2026.
# Removed because OpenAI has shut them down: gpt-4.1-nano, o4-mini.
MODELS: dict[str, dict[str, str | float]] = {
    "gpt-6-luna": {
        "kind": "gpt6",
        "input": 0.10,
        "output": 0.50,
        "status": "current",
    },
    "gpt-6.1-terra": {
        "kind": "gpt6",
        "input": 1.00,
        "output": 5.00,
        "status": "current",
    },
    "gpt-6.1-sol": {
        "kind": "gpt6",
        "input": 2.00,
        "output": 10.00,
        "status": "current",
    },
    "gpt-6-astra": {
        "kind": "gpt6",
        "input": 10.00,
        "output": 50.00,
        "status": "current",
    },
    "gpt-5-mini": {
        "kind": "gpt5",
        "input": 0.25,
        "output": 2.00,
        "status": "legacy",
    },
    "gpt-5": {
        "kind": "gpt5",
        "input": 1.25,
        "output": 10.00,
        "status": "legacy",
    },
    "o3": {
        "kind": "thinking",
        "input": 2.00,
        "output": 8.00,
        "status": "legacy",
    },
    "gpt-4o-mini": {
        "kind": "standard",
        "input": 0.15,
        "output": 0.60,
        "status": "legacy",
    },
}

# Output-token budget per kind. Reasoning models need room to think.
MAX_TOKENS: dict[str, int] = {
    "standard": 20,
    "thinking": 4096,
    "gpt5": 8192,
    "gpt6": 8192,
}

# Reasoning-effort levels per kind. Standard models have none.
EFFORTS: dict[str, tuple[str, ...]] = {
    "standard": (),
    "thinking": ("low", "medium", "high"),
    "gpt5": ("minimal", "low", "medium", "high"),
    "gpt6": ("none", "low", "medium", "high"),
}

EFFORT_ORDER: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
)

DEFAULT_EFFORT: str = "medium"

# Standard models only.
TEMPERATURE: float = 0.2

# Retry behavior.
MAX_RETRIES: int = 5
API_CALL_DELAY: float = 1.0
CEILING_MARGIN: int = 32
CEILING_HITS_BEFORE_DOWNGRADE: int = 2

# Defaults.
DEFAULT_MAX_PLIES: int = 80
STREAM_CHAR_DELAY: float = 0.006
UCI_PATTERN: str = r"[a-h][1-8][a-h][1-8][qrbn]?"
SCORES: dict[str, tuple[float, float]] = {
    "1-0": (1.0, 0.0),
    "0-1": (0.0, 1.0),
    "1/2-1/2": (0.5, 0.5),
}

OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Track the costs of each model (in USD).
costs: dict[str, float] = {}

_client: OpenAI | None = None


def require_key() -> None:
    """Stop the program if no real API key is configured.

    Raises
    ------
    SystemExit
        If `OPENAI_API_KEY` is empty or still a placeholder.
    """
    if not API_KEY or API_KEY.startswith("REPLACE_ME"):
        raise SystemExit(
            "[SystemExit - main.py] Set a real `OPENAI_API_KEY` "
            "in `.env` first.\n"
        )


def get_client() -> OpenAI:
    """Return the shared OpenAI client, creating it on first use.

    Returns
    -------
    OpenAI
        The cached client.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY)
    return _client


def add_cost(
    label: str,
    amount: float | None,
) -> None:
    """Add `amount` USD to the running cost of `label`.

    Parameters
    ----------
    label : str
        Cost bucket, normally a model name.
    amount : float or None
        Cost in USD. `None` counts as zero.
    """
    costs[label] = costs.get(label, 0.0) + float(amount or 0)


def token_counts(resp: ChatCompletion) -> tuple[int, int, int]:
    """Read the token usage of a response.

    Parameters
    ----------
    resp : ChatCompletion
        Chat Completions response.

    Returns
    -------
    tuple of int
        `(prompt, completion, reasoning)` token counts. Reasoning
        tokens are already included in the completion count.
    """
    used = getattr(resp, "usage", None)
    if used is None:
        return 0, 0, 0
    details = getattr(used, "completion_tokens_details", None)
    return (
        getattr(used, "prompt_tokens", 0) or 0,
        getattr(used, "completion_tokens", 0) or 0,
        getattr(details, "reasoning_tokens", 0) or 0,
    )


def response_cost(
    model: str,
    resp: ChatCompletion,
) -> float:
    """Estimate the USD cost of one response from its token counts.

    Parameters
    ----------
    model : str
        Key in `MODELS`.
    resp : ChatCompletion
        Chat Completions response.

    Returns
    -------
    float
        Estimated cost in USD (ignores cached-input discounts).
    """
    prompt: int
    completion: int
    prompt, completion, _ = token_counts(resp)
    info: dict[str, str | float] = MODELS[model]
    return (
        prompt * float(info["input"]) +
        completion * float(info["output"])
    ) / 1_000_000


def print_costs() -> None:
    """Print the estimated cost of every model and the total."""
    print("\n[INFO - main.py] Cost summary (USD, estimated)\n")
    for label, amount in costs.items():
        print(f"    - {label:<14} ${amount:.4f}\n")
    print(f"    - {'TOTAL':<14} ${sum(costs.values()):.4f}\n")


def price_label(model: str) -> str:
    """Format a model's price for display.

    Parameters
    ----------
    model : str
        Key in `MODELS`.

    Returns
    -------
    str
        Text such as `$0.10 in / $0.50 out per 1M tokens`.
    """
    info: dict[str, str | float] = MODELS[model]
    return (
        f"${float(info['input']):.2f} in / "
        f"${float(info['output']):.2f} out per 1M tokens"
    )


def normalize_effort(
    model: str,
    effort: str,
) -> str:
    """Pick the effort the model supports that is closest to `effort`.

    Parameters
    ----------
    model : str
        Key in `MODELS`.
    effort : str
        Requested level from `EFFORT_ORDER`.

    Returns
    -------
    str
        A supported level, or `""` for models without reasoning effort.
    """
    allowed: tuple[str, ...] = EFFORTS[str(MODELS[model]["kind"])]
    if not allowed:
        return ""
    target: int = EFFORT_ORDER.index(effort)
    return min(
        allowed,
        key=lambda e: abs(EFFORT_ORDER.index(e) - target),
    )


def request_kwargs(
    kind: str,
    effort: str,
) -> dict[str, str | float]:
    """Build the extra API arguments for a model kind.

    Parameters
    ----------
    kind : str
        One of `standard`, `thinking`, `gpt5` or `gpt6`.
    effort : str
        Reasoning effort already accepted by the model.

    Returns
    -------
    dict
        Keyword arguments for `chat.completions.create`.
    """
    if kind == "thinking":
        return {"reasoning_effort": effort}
    if kind == "gpt5":
        return {"reasoning_effort": effort, "verbosity": "low"}
    if kind == "gpt6":
        return {"reasoning_effort": effort}
    return {"temperature": TEMPERATURE}


def format_history(board: chess.Board) -> str:
    """Format the move list in numbered SAN.

    Parameters
    ----------
    board : chess.Board
        Board with its move stack.

    Returns
    -------
    str
        Text such as `1. e4 e5 2. Nf3`, or a placeholder at game start.
    """
    replay: chess.Board = chess.Board()
    san_moves: list[str] = []
    for move in board.move_stack:
        san_moves.append(replay.san(move))
        replay.push(move=move)
    pairs: list[str] = [
        f"{i // 2 + 1}. {' '.join(san_moves[i:i + 2])}"
        for i in range(0, len(san_moves), 2)
    ]
    return " ".join(pairs) or "(game start | no moves yet)"


def build_prompt(
    board: chess.Board,
    color_name: str,
) -> str:
    """Build the move prompt shared by every model.

    Parameters
    ----------
    board : chess.Board
        Current position.
    color_name : str
        `"White"` or `"Black"`.

    Returns
    -------
    str
        Prompt with the board, FEN, history and legal moves.
    """
    legal: str = ", ".join(m.uci() for m in board.legal_moves)
    return textwrap.dedent(f"""\
        You are a chess grandmaster playing as {color_name}.

        OUTPUT FORMAT: STRICT

        Your ENTIRE response must be ONE line only:

        MOVE: <uci_move>

        Example: "MOVE: e2e4"

        DO NOT write any explanation, reasoning, sentences, punctuation,
        or extra text. ONLY the MOVE: line.

        CHESS STRATEGY:

        - Pick the BEST move available.
        - Choose ONLY from the Legal Moves list below.
        - Do NOT invent a move outside that list.
        - If checkmate is available, play it immediately.
        - If losing material, defend or counter-attack.
        - Control the center, develop pieces, protect your king.

        === Board (ASCII) ===

        {board}

        === FEN ===

        {board.fen()}

        === Move History (SAN) ===

        {format_history(board)}

        === Legal Moves (UCI, pick ONLY from this list) ===

        {legal}

        Reason silently. Output only:

        MOVE: <uci_move>
        """).strip()


def extract_uci_move(
    text: str,
    board: chess.Board,
) -> str | None:
    """Find a move in the model's reply.

    Parameters
    ----------
    text : str
        Raw reply text.
    board : chess.Board
        Current position, used to accept only legal fallback matches.

    Returns
    -------
    str or None
        Lowercase UCI move, or `None` if none was found.
    """
    match: re.Match[str] | None = re.search(
        pattern=rf"MOVE:\s*({UCI_PATTERN})",
        string=text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    legal: set[str] = {m.uci() for m in board.legal_moves}
    for token in reversed(re.findall(UCI_PATTERN, text, re.IGNORECASE)):
        if token.lower() in legal:
            print(
                f"[INFO - main.py] Recovered move from text: "
                f"{token.lower()}\n"
            )
            return token.lower()
    return None


def get_move(
    board: chess.Board,
    model: str,
    effort: str,
) -> chess.Move:
    """Ask a model for a legal move, retrying on failure.

    In `gpt5` and `gpt6` modes, if the reasoning budget is exhausted
    repeatedly, the effort drops to the model's lowest level.

    Parameters
    ----------
    board : chess.Board
        Current position.
    model : str
        Key in `MODELS`.
    effort : str
        Requested reasoning effort.

    Returns
    -------
    chess.Move
        A legal move. A random legal move is used after `MAX_RETRIES`
        failed attempts.
    """
    kind: str = str(MODELS[model]["kind"])
    max_tokens: int = MAX_TOKENS[kind]
    tag: str = model.upper()
    legal_moves: set[chess.Move] = set(board.legal_moves)
    color: str = "White" if board.turn == chess.WHITE else "Black"
    prompt: str = build_prompt(
        board=board,
        color_name=color,
    )

    active_effort: str = normalize_effort(
        model=model,
        effort=effort,
    )
    ceiling_hits: int = 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(API_CALL_DELAY)
            resp: ChatCompletion = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                stream=False,
                **request_kwargs(
                    kind=kind,
                    effort=active_effort
                ),
            )
            add_cost(
                label=model,
                amount=response_cost(
                    model=model,
                    resp=resp
                ),
            )
            raw: str = (resp.choices[0].message.content or "").strip()
            uci: str | None = extract_uci_move(
                text=raw,
                board=board,
            )
            completion: int
            reasoning: int
            _, completion, reasoning = token_counts(resp=resp)

            print(
                f"[INFO - main.py] {tag} "
                f"effort={active_effort or 'n/a'} "
                f"reasoning_tokens={reasoning} -> "
                f"{uci.upper() if uci else 'No move found!'}\n"
            )

            if uci is None:
                print(
                    f"[INFO - main.py] {tag} No move in response "
                    f"(raw={raw[:120]!r}). Retrying (attempt "
                    f"{attempt})...\n"
                )
                limit: int = max_tokens - CEILING_MARGIN
                lowest: str = EFFORTS[kind][0] if EFFORTS[kind] else ""
                if (
                    kind in ("gpt5", "gpt6")
                    and (reasoning >= limit or completion >= limit)
                    and active_effort != lowest
                ):
                    ceiling_hits += 1
                    if ceiling_hits >= CEILING_HITS_BEFORE_DOWNGRADE:
                        active_effort = lowest
                        print(
                            f"[INFO - main.py] {tag} Downgrading to "
                            f"reasoning_effort={lowest}.\n"
                        )
                continue

            move: chess.Move = chess.Move.from_uci(uci)
            if move in legal_moves:
                return move
            print(
                f"[INFO - main.py] {tag} Illegal move {uci}. "
                f"Retrying...\n"
            )

        except OpenAIError as exc:
            print(
                f"[OpenAIError - main.py] {tag} attempt {attempt}: "
                f"{exc}\n"
            )
        except Exception as exc:
            print(
                f"[Exception - main.py] {tag} attempt {attempt}: "
                f"{exc}\n"
            )

    fallback: chess.Move = random.choice(seq=list(legal_moves))
    print(
        f"[INFO - main.py] {tag} Fallback: random move "
        f"{fallback.uci()}\n"
    )
    return fallback


def iter_game(
    board: chess.Board,
    white: str,
    black: str,
    max_plies: int,
    effort: str,
) -> Generator[tuple[str, str, int, str, str], None, None]:
    """Play a game on `board`, one ply at a time.

    Parameters
    ----------
    board : chess.Board
        Starting position. It is updated in place.
    white, black : str
        Model names playing each color.
    max_plies : int
        Maximum number of half-moves.
    effort : str
        Requested reasoning effort.

    Yields
    ------
    tuple
        `(model, color, move_number, san, uci)` for each ply.
    """
    plies: int = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        plies += 1
        model: str
        color: str
        if board.turn == chess.WHITE:
            model, color = white, "White"
        else:
            model, color = black, "Black"
        number: int = board.fullmove_number
        move: chess.Move = get_move(
            board=board,
            model=model,
            effort=effort,
        )
        san: str = board.san(move=move)
        board.push(move=move)
        yield model, color, number, san, move.uci()


def result_summary(board: chess.Board) -> tuple[str, str]:
    """Summarize how a game ended.

    Parameters
    ----------
    board : chess.Board
        Final position.

    Returns
    -------
    tuple of str
        `(result, termination)`. Unfinished games count as draws with
        termination `MOVE_LIMIT`.
    """
    if board.is_game_over(claim_draw=True):
        outcome: chess.Outcome | None = board.outcome(claim_draw=True)
        termination: str = (
            outcome.termination.name
            if outcome
            else "DRAW"
        )
        return board.result(claim_draw=True), termination
    return "1/2-1/2", "MOVE_LIMIT"


def save_pgn(
    board: chess.Board,
    white: str,
    black: str,
    tag: str,
    path: Path,
) -> Path:
    """Write a finished game to a PGN file.

    Parameters
    ----------
    board : chess.Board
        Final position with its move stack.
    white, black : str
        Model names playing each color.
    tag : str
        Short label added to the PGN event header.
    path : Path
        Output file. Parent folders are created.

    Returns
    -------
    Path
        The file that was written.
    """
    result: str
    result, _ = result_summary(board=board)
    game: chess.pgn.Game = chess.pgn.Game()
    game.headers.update({
        "Event": f"{white} vs. {black} | GPT Chess Arena [{tag}]",
        "Site": "Local Computer",
        "Date": date.today().isoformat(),
        "Round": "1",
        "White": white,
        "Black": black,
        "Result": result,
    })
    game.add_line(moves=board.move_stack)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        data=f"{game}\n\n",
        encoding="utf-8",
    )
    return path


def effort_label(
    white: str,
    black: str,
    effort: str,
) -> str:
    """Describe the effort each player actually uses.

    Parameters
    ----------
    white, black : str
        Model names playing each color.
    effort : str
        Requested reasoning effort.

    Returns
    -------
    str
        Text such as `gpt-5=low, o3=low`, or `n/a` if none apply.
    """
    parts: list[str] = [
        f"{m}={normalize_effort(model=m, effort=effort)}"
        for m in dict.fromkeys((white, black))
        if normalize_effort(model=m, effort=effort)
    ]
    return ", ".join(parts) or "n/a"

def warn_legacy(*models: str) -> None:
    """Print a warning for every deprecated model in `models`.

    Parameters
    ----------
    *models : str
        Model names to check.
    """
    for model in dict.fromkeys(models):
        if MODELS[model]["status"] == "legacy":
            print(
                f"[WARNING - main.py] {model} is deprecated by OpenAI "
                f"and may stop working.\n"
            )


def typewrite(
    buffer: list[str],
    fragment: str,
) -> Generator[str, None, None]:
    """Append `fragment` to `buffer[0]`, yielding after every character.

    Parameters
    ----------
    buffer : list of str
        One-item list holding the accumulated text.
    fragment : str
        Text to append.

    Yields
    ------
    str
        The accumulated text so far.
    """
    for ch in fragment:
        buffer[0] += ch
        yield buffer[0]
        time.sleep(STREAM_CHAR_DELAY)


def run_game(
    white: str,
    black: str,
    randomize: bool,
    max_plies: float,
    effort: str,
) -> Generator[str, None, None]:
    """Play one game and stream it to the web app as Markdown.

    Parameters
    ----------
    white, black : str
        Model names chosen in the UI.
    randomize : bool
        If True, swap colors with probability 0.5.
    max_plies : float
        Half-move limit (a `gr.Number` value).
    effort : str
        Requested reasoning effort.

    Yields
    ------
    str
        The growing Markdown transcript.
    """
    plies: int
    try:
        plies = int(max_plies)
        if plies < 1:
            raise ValueError
    except (TypeError, ValueError):
        yield "**Error:** Max Half-Moves must be a positive integer."
        return

    if randomize and random.random() < 0.5:
        white, black = black, white

    warn_legacy(white, black)
    board: chess.Board = chess.Board()
    buffer: list[str] = [""]
    cost_before: float = sum(costs.values())

    yield from typewrite(
        buffer=buffer,
        fragment=f"## {white} (White) vs. {black} (Black)\n\n"
        f"*Reasoning effort: {effort_label(white=white, black=black, effort=effort)}.*\n\n"
        "| | Model | Price |\n|---|---|---|\n"
        f"| White | **{white}** | {price_label(model=white)} |\n"
        f"| Black | **{black}** | {price_label(model=black)} |\n\n---\n\n",
    )

    for model, color, number, san, uci in iter_game(
        board,
        white,
        black,
        plies,
        effort,
    ):
        yield from typewrite(
            buffer=buffer,
            fragment=f"**Move {number} -- {model} ({color})**\n\n"
            f"Played: `{san}` (UCI: `{uci}`)\n\n```\n{board}\n```\n\n",
        )

    result: str
    termination: str
    result, termination = result_summary(board=board)
    path: Path = save_pgn(
        board=board,
        white=white,
        black=black,
        tag=effort_label(white=white, black=black, effort=effort),
        path=OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{white}_vs_{black}.pgn",
    )
    yield from typewrite(
        buffer=buffer,
        fragment=f"---\n\n**GAME OVER** | Result: `{result}` | "
        f"Termination: `{termination}`\n\n"
        f"**Estimated cost:** ${sum(costs.values()) - cost_before:.4f}\n\n"
        f"**PGN saved:** `{path}`\n\n**Replay on Chess.com:**\n"
        "1. Open https://www.chess.com/analysis\n"
        "2. Click the PGN import icon (book).\n"
        "3. Paste the contents of the PGN file.\n",
    )


def build_ui() -> gr.Blocks:
    """Build the Gradio interface.

    Returns
    -------
    gr.Blocks
        The app, ready for `.launch()`.
    """
    models: list[str] = list(MODELS)
    rows: str = "\n".join(
        f"| `{name}` | {price_label(name)} | {info['status']} |"
        for name, info in MODELS.items()
    )
    with gr.Blocks(title="GPT Chess Arena") as demo:
        gr.Markdown("# GPT Chess Arena")
        gr.Markdown(
            "Pick any two models. They may even be the same model.\n\n"
            "| Model | Price | Status |\n|---|---|---|\n" + rows
        )
        with gr.Row():
            white_dd: gr.Dropdown = gr.Dropdown(
                choices=models,
                value=models[0],
                label="White",
            )
            black_dd: gr.Dropdown = gr.Dropdown(
                choices=models,
                value=models[1],
                label="Black",
            )
        randomize: gr.Checkbox = gr.Checkbox(
            label="Randomly swap colors",
            value=False,
        )
        effort_dd: gr.Dropdown = gr.Dropdown(
            choices=list(EFFORT_ORDER),
            value=DEFAULT_EFFORT,
            label="Reasoning Effort",
            info=(
                "Each model uses the closest level it supports. "
                "gpt-4o-mini has no reasoning effort."
            ),
        )
        max_plies: gr.Number = gr.Number(
            label="Max Half-Moves (plies) before draw",
            value=DEFAULT_MAX_PLIES,
            minimum=1,
            maximum=500,
            step=1,
        )
        run_btn: gr.Button = gr.Button("Start Game", variant="primary")
        output: gr.Markdown = gr.Markdown(label="Live Game Stream")
        run_btn.click(
            fn=run_game,
            inputs=[white_dd, black_dd, randomize, max_plies, effort_dd],
            outputs=[output],
        )
    return demo


def run_tournament(
    models: list[str],
    max_plies: int,
    effort: str,
    both_colors: bool,
) -> None:
    """Play every pair of models once and print the standings.

    Parameters
    ----------
    models : list of str
        Model names in the tournament.
    max_plies : int
        Half-move limit per game.
    effort : str
        Requested reasoning effort.
    both_colors : bool
        If True, each pair also plays with colors swapped.
    """
    run_dir: Path = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs: list[tuple[str, str]] = list(itertools.combinations(models, 2))
    if both_colors:
        pairs += [(b, a) for a, b in pairs]

    warn_legacy(*models)
    points: dict[str, float] = {model: 0.0 for model in models}
    for n, (white, black) in enumerate(iterable=pairs, start=1):
        print(
            f"[INFO - main.py] Game {n}/{len(pairs)}: "
            f"{white} (White) vs. {black} (Black)\n"
        )
        board: chess.Board = chess.Board()
        for _ in iter_game(
            board=board,
            white=white,
            black=black,
            max_plies=max_plies,
            effort=effort,
        ):
            pass
        result: str
        termination: str
        result, termination = result_summary(board=board)
        points[white] += SCORES[result][0]
        points[black] += SCORES[result][1]
        save_pgn(
            board=board,
            white=white,
            black=black,
            tag=effort,
            path=run_dir / f"{n:02d}_{white}_vs_{black}.pgn",
        )
        print(f"    - Result: {result} ({termination})\n")

    print("\n[INFO - main.py] Standings\n")
    for model, score in sorted(points.items(), key=lambda kv: -kv[1]):
        print(f"    - {model:<14} {score:.1f}\n")
    print(f"\nSaved to {run_dir}\n")
    print_costs()


def main() -> None:
    """Parse the command line, then run the tournament or the web app.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Skip the web app and play every pair of models once.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        choices=list(MODELS),
        help="Models in the tournament (default: all).",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=DEFAULT_MAX_PLIES,
        help="Half-moves before a game is declared a draw.",
    )
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=list(EFFORT_ORDER),
    )
    parser.add_argument(
        "--both-colors",
        action="store_true",
        help="Play each pair twice, swapping colors.",
    )
    args: argparse.Namespace = parser.parse_args()

    require_key()
    if args.tournament:
        run_tournament(
            models=args.models,
            max_plies=args.max_plies,
            effort=args.effort,
            both_colors=args.both_colors,
        )
    else:
        build_ui().launch()

if __name__ == "__main__":
    main()
