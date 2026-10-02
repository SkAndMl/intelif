import argparse
import json
import re
import time
from html.parser import HTMLParser
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from intelif import Intelif

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "intelif-wikigame/0.1 (https://github.com/SkAndMl/intelif)"
INSTRUCTIONS = (
    "You are playing the Wikipedia game: get from the current article to the "
    "target article by clicking links, in as few clicks as possible. "
    "Which link should you click next?"
)


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.skip = 0
        self.paragraphs, self.links, self.text = [], {}, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("style", "sup"):
            self.skip += 1

        if tag == "p":
            self.depth += 1

        href = attrs.get("href") or ""
        if self.depth and tag == "a" and href.startswith("/wiki/") and ":" not in href:
            title = attrs.get("title") or unquote(href[6:]).replace("_", " ")
            self.links.setdefault(title, None)

    def handle_endtag(self, tag):
        if tag in ("style", "sup") and self.skip:
            self.skip -= 1

        if tag == "p" and self.depth:
            self.depth -= 1
            paragraph = re.sub(r"\s+", " ", "".join(self.text)).strip()
            if paragraph:
                self.paragraphs.append(paragraph)
            self.text = []

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.text.append(data)


def fetch(title: str) -> dict:
    query = urlencode(
        {
            "action": "parse",
            "page": title,
            "prop": "text",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
        }
    )
    request = Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    page = json.load(urlopen(request, timeout=30))["parse"]

    parser = ArticleParser()
    parser.feed(page["text"])

    return {
        "title": page["title"],
        "summary": parser.paragraphs[0][:600] if parser.paragraphs else "",
        "links": [link for link in parser.links if link != page["title"]],
    }


def bar(probability: float, width: int = 28) -> Text:
    filled = round(probability * width)
    return Text("█" * filled, style="cyan") + Text("░" * (width - filled), style="grey30")


def render(target, path, article, answer, step, seconds, chosen=None, status=""):
    header = Text.assemble(
        ("Wikipedia game  ", "bold"),
        (path[0], "bold yellow"),
        ("  →  ", "grey50"),
        (target, "bold green"),
    )

    trail = Text(" → ".join(path), style="yellow")

    body = [header, Text(""), trail, Text("")]

    if article:
        body.append(Text(article["title"], style="bold white"))
        body.append(Text(article["summary"][:280] + "…", style="grey62"))
        body.append(Text(""))

    if answer:
        table = Table.grid(padding=(0, 2))
        ranked = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])[:8]

        for link, probability in ranked:
            style = "bold green" if link == chosen else "white"
            table.add_row(
                bar(probability), Text(f"{probability:6.1%}"), Text(link, style=style)
            )

        body.append(table)
        body.append(Text(""))
        body.append(
            Text(
                f"step {step} · {len(answer.probabilities)} links scored in one "
                f"forward pass · {seconds * 1000:.0f} ms",
                style="grey50",
            )
        )

    if status:
        body.append(Text(""))
        body.append(Text(status, style="bold"))

    return Panel(Group(*body), border_style="cyan", padding=(1, 2), width=100)


def play(model, start, target, max_steps, max_links, delay, console):
    target = fetch(target)["title"]
    path, visited = [], set()
    article = fetch(start)
    path.append(article["title"])

    with Live(console=console, refresh_per_second=12) as live:
        for step in range(1, max_steps + 1):
            visited.add(article["title"])

            if article["title"] == target:
                live.update(
                    render(target, path, article, None, step, 0,
                           status=f"Reached {target} in {len(path) - 1} clicks")
                )
                return path

            links = [link for link in article["links"] if link not in visited]
            links = links[:max_links]
            if target in article["links"]:
                links = list(dict.fromkeys([*links, target]))

            if not links:
                break

            live.update(render(target, path, article, None, step, 0, status="thinking…"))

            state = {
                "target article": target,
                "current article": article["title"],
                "current article summary": article["summary"],
                "path so far": path,
            }
            started = time.perf_counter()
            answer = model.choice(state, {link: None for link in links}, INSTRUCTIONS)
            seconds = time.perf_counter() - started

            live.update(render(target, path, article, answer, step, seconds, answer.choice))
            time.sleep(delay)

            visited.add(answer.choice)
            article = fetch(answer.choice)
            path.append(article["title"])

        live.update(render(target, path, None, None, max_steps, 0,
                           status=f"Gave up after {len(path) - 1} clicks"))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", default="Banana")
    parser.add_argument("target", nargs="?", default="Quantum mechanics")
    parser.add_argument("--model", default="UserMoonlight/intelif-qwen3-4b")
    parser.add_argument("--revision", default="v0.1")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-links", type=int, default=200)
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    console = Console()
    with console.status("loading intelif…"):
        model = Intelif.from_pretrained(args.model, revision=args.revision, dtype=args.dtype)

    play(model, args.start, args.target, args.max_steps, args.max_links, args.delay, console)


if __name__ == "__main__":
    main()
