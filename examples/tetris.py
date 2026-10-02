import argparse
import time
from random import Random

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from PIL import Image, ImageDraw, ImageFont

from intelif import Intelif

WIDTH, HEIGHT = 10, 18
PIECES = {
    "I": ([(0, 0), (1, 0), (2, 0), (3, 0)], "cyan"),
    "O": ([(0, 0), (1, 0), (0, 1), (1, 1)], "yellow"),
    "T": ([(0, 0), (1, 0), (2, 0), (1, 1)], "magenta"),
    "S": ([(1, 0), (2, 0), (0, 1), (1, 1)], "green"),
    "Z": ([(0, 0), (1, 0), (1, 1), (2, 1)], "red"),
    "J": ([(0, 0), (0, 1), (1, 1), (2, 1)], "blue"),
    "L": ([(2, 0), (0, 1), (1, 1), (2, 1)], "orange1"),
}


def rotations(cells):
    shapes = []
    for _ in range(4):
        cells = [(y, -x) for x, y in cells]
        left, top = min(x for x, _ in cells), min(y for _, y in cells)
        shape = sorted((x - left, y - top) for x, y in cells)
        if shape not in shapes:
            shapes.append(shape)

    return shapes


def fits(board, cells, x, y):
    return all(
        0 <= x + dx < WIDTH and y + dy < HEIGHT and (y + dy < 0 or not board[y + dy][x + dx])
        for dx, dy in cells
    )


def drop(board, cells, x):
    y = -max(dy for _, dy in cells) - 1
    if not fits(board, cells, x, y):
        return None

    while fits(board, cells, x, y + 1):
        y += 1

    return y


def place(board, cells, x, y, color):
    board = [row[:] for row in board]
    for dx, dy in cells:
        if y + dy < 0:
            return None, 0
        board[y + dy][x + dx] = color

    kept = [row for row in board if not all(row)]
    cleared = HEIGHT - len(kept)

    return [[None] * WIDTH for _ in range(cleared)] + kept, cleared


def stats(board):
    heights = [next((HEIGHT - y for y in range(HEIGHT) if board[y][x]), 0) for x in range(WIDTH)]
    holes = sum(
        1 for x in range(WIDTH) for y in range(HEIGHT - heights[x], HEIGHT) if not board[y][x]
    )
    bumpiness = sum(abs(a - b) for a, b in zip(heights, heights[1:]))

    return max(heights), holes, bumpiness


def describe(before, after, cleared):
    height, holes, bumpiness = (a - b for a, b in zip(stats(after), stats(before)))

    return ", ".join(
        [
            f"clears {cleared} line{'s' * (cleared > 1)}" if cleared else "clears no lines",
            f"creates {holes} new hole{'s' * (holes > 1)}" if holes > 0 else "no new holes",
            f"raises the stack by {height}" if height > 0 else "keeps the stack low",
            "surface gets bumpier" if bumpiness > 0 else "surface stays flat",
        ]
    )


def moves(board, piece):
    cells, color = PIECES[piece]
    options = {}

    for r, shape in enumerate(rotations(cells)):
        for x in range(WIDTH - max(dx for dx, _ in shape)):
            y = drop(board, shape, x)
            if y is None:
                continue

            after, cleared = place(board, shape, x, y, color)
            if after is None:
                continue

            options[f"rotation {r}, column {x}"] = {
                "move": (shape, x, y, after, cleared),
                "description": describe(board, after, cleared),
            }

    return options


RGB = {
    "cyan": (137, 220, 235), "yellow": (249, 226, 175), "magenta": (203, 166, 247),
    "green": (166, 227, 161), "red": (243, 139, 168), "blue": (137, 180, 250),
    "orange1": (250, 179, 135),
}
BACKGROUND, DIM, TEXT, ACCENT = (30, 30, 46), (69, 71, 90), (205, 214, 244), (137, 220, 235)
CELL = 22


def picture(board, falling, piece, placed, lines, answer, seconds, status):
    image = Image.new("RGB", (820, 2 * CELL + HEIGHT * CELL), BACKGROUND)
    pen = ImageDraw.Draw(image)
    font, big = ImageFont.load_default(16), ImageFont.load_default(20)

    cells = {}
    if falling:
        shape, x, y, color = falling
        cells = {(x + dx, y + dy): color for dx, dy in shape}

    left, top = CELL, CELL
    pen.rectangle([left - 3, top - 3, left + WIDTH * CELL + 2, top + HEIGHT * CELL + 2], outline=DIM, width=2)
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            color = cells.get((x, y)) or cell
            box = [left + x * CELL + 1, top + y * CELL + 1, left + (x + 1) * CELL - 2, top + (y + 1) * CELL - 2]
            if color:
                pen.rectangle(box, fill=RGB[color])
            else:
                pen.point((left + x * CELL + CELL // 2, top + y * CELL + CELL // 2), fill=DIM)

    x, y = left + WIDTH * CELL + 40, top
    pen.text((x, y), "intelif plays tetris", font=big, fill=ACCENT)
    y += 40
    if piece:
        pen.text((x, y), "piece", font=font, fill=TEXT)
        pen.text((x + 60, y), piece, font=font, fill=RGB[PIECES[piece][1]])
    y += 26
    pen.text((x, y), f"pieces {placed}   lines {lines}", font=font, fill=TEXT)
    y += 44

    if answer:
        ranked = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])[:5]
        for key, probability in ranked:
            pen.rectangle([x, y + 3, x + 160, y + 15], fill=DIM)
            pen.rectangle([x, y + 3, x + round(160 * probability), y + 15], fill=ACCENT)
            color = RGB["green"] if key == answer.choice else TEXT
            pen.text((x + 172, y), f"{probability:6.1%}  {key}", font=font, fill=color)
            y += 28
        y += 16
        pen.text((x, y), f"{len(answer.probabilities)} landings scored in one pass · {seconds * 1000:.0f} ms", font=font, fill=DIM)

    if status:
        pen.text((x, y + 40), status, font=big, fill=TEXT)

    return image


def frame(board):
    return "\n".join("".join("#" if cell else "." for cell in row) for row in board)


def draw(board, falling=None):
    text = Text()
    cells = {}
    if falling:
        shape, x, y, color = falling
        cells = {(x + dx, y + dy): color for dx, dy in shape}

    for y, row in enumerate(board):
        text.append("│", style="grey50")
        for x, cell in enumerate(row):
            color = cells.get((x, y)) or cell
            text.append("██" if color else " ·", style=color or "grey23")
        text.append("│\n", style="grey50")
    text.append("└" + "──" * WIDTH + "┘", style="grey50")

    return text


def side(piece, placed, lines, answer=None, seconds=0.0, status=""):
    body = [
        Text.assemble(("piece  ", "bold"), (piece, f"bold {PIECES[piece][1]}")) if piece else Text(""),
        Text(f"pieces {placed}   lines {lines}", style="white"),
        Text(""),
    ]

    if answer:
        table = Table.grid(padding=(0, 1))
        for key, probability in sorted(answer.probabilities.items(), key=lambda kv: -kv[1])[:5]:
            filled = round(probability * 16)
            table.add_row(
                Text("█" * filled + "░" * (16 - filled), style="cyan"),
                Text(f"{probability:6.1%}"),
                Text(key, style="bold green" if key == answer.choice else "white"),
            )
        body += [
            table,
            Text(""),
            Text(
                f"{len(answer.probabilities)} landings scored in one pass · "
                f"{seconds * 1000:.0f} ms",
                style="grey50",
            ),
        ]

    if status:
        body += [Text(""), Text(status, style="bold")]

    return Panel(Group(*body), title="intelif plays tetris", border_style="cyan", width=56)


def decide(model, board, piece, options):
    return model.choice(
        {"board": frame(board), "piece": piece},
        {key: option["description"] for key, option in options.items()},
        f"You are playing Tetris. Where should the {piece} piece land? "
        "Clear lines, avoid holes and keep the stack low and flat.",
    )


def play(model, seed, max_pieces, speed, console, gif=None):
    rng = Random(seed)
    board = [[None] * WIDTH for _ in range(HEIGHT)]
    lines, placed, status = 0, 0, f"{max_pieces} pieces placed"
    frames, durations = [], []

    def show(live, board, falling, piece, placed, lines, answer=None, seconds=0.0, status="", hold=0.0):
        color = falling and (falling[0], falling[1], falling[2], PIECES[piece][1])
        live.update(Columns([draw(board, color), side(piece, placed, lines, answer, seconds, status)]))
        if gif:
            frames.append(picture(board, color, piece, placed, lines, answer, seconds, status))
            durations.append(max(round((hold or speed) * 1000), 20))

    with Live(console=console, refresh_per_second=30) as live:
        for placed in range(max_pieces):
            piece = rng.choice(list(PIECES))
            options = moves(board, piece)
            if not options:
                status = f"topped out after {placed} pieces"
                break

            started = time.perf_counter()
            answer = decide(model, board, piece, options)
            seconds = time.perf_counter() - started

            shape, x, y, after, cleared = options[answer.choice]["move"]
            for fall in range(-max(dy for _, dy in shape) - 1, y + 1):
                show(live, board, (shape, x, fall), piece, placed, lines, answer, seconds)
                time.sleep(speed)

            board, lines = after, lines + cleared
        else:
            placed = max_pieces

        show(live, board, None, None, placed, lines, status=status, hold=3.0)

    if gif:
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0)
        console.print(f"wrote {gif} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="UserMoonlight/intelif-qwen3-4b")
    parser.add_argument("--revision", default="v0.1")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--pieces", type=int, default=100)
    parser.add_argument("--speed", type=float, default=0.02)
    parser.add_argument("--gif", default=None)
    args = parser.parse_args()

    console = Console(force_terminal=True, color_system="truecolor")
    with console.status("loading intelif…"):
        model = Intelif.from_pretrained(args.model, revision=args.revision, dtype=args.dtype)
        empty = [[None] * WIDTH for _ in range(HEIGHT)]
        decide(model, empty, "T", moves(empty, "T"))

    play(model, args.seed, args.pieces, args.speed, console, args.gif)


if __name__ == "__main__":
    main()
