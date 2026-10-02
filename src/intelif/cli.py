import argparse
import json
import os

from intelif._version import __version__
from intelif.hub import DEFAULT_MODEL, DEFAULT_REVISION


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)


def load(args: argparse.Namespace):
    from intelif.model import Intelif

    return Intelif.from_pretrained(
        args.model, revision=args.revision, device=args.device, dtype=args.dtype
    )


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    from intelif.server import create_app

    app = create_app(load(args), api_key=os.environ.get("INTELIF_API_KEY"))
    uvicorn.run(app, host=args.host, port=args.port)


def ask(args: argparse.Namespace) -> None:
    from intelif.types import Choice

    criteria = {key.strip(): None for key in args.choice.split(",")}
    question = Choice(criteria=criteria, instructions=args.instructions)
    response = load(args).system_one(args.state, {"answer": question})
    print(json.dumps(response.model_dump(mode="json"), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="intelif")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve_parser = commands.add_parser("serve")
    add_model_args(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(handler=serve)

    ask_parser = commands.add_parser("ask")
    add_model_args(ask_parser)
    ask_parser.add_argument("--state", required=True)
    ask_parser.add_argument("--choice", required=True)
    ask_parser.add_argument("--instructions", default=None)
    ask_parser.set_defaults(handler=ask)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
