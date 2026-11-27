#!/usr/bin/env python3
# file: eval.py

import argparse
import shlex
import sys
from contextlib import ExitStack

import chess
import chess.engine
import chess.pgn


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def set_elo(engine, name, elo):
    options = {"UCI_LimitStrength": True, "UCI_Elo": elo}
    # Validate both options before enabling limited strength.
    for key, value in options.items():
        if key not in engine.options:
            print(f"{name}: skipping Elo {elo}; engine lacks {key}", file=sys.stderr)
            return
        try:
            engine.options[key].parse(value)
        except chess.engine.EngineError as error:
            print(f"{name}: skipping Elo {elo}; {error}", file=sys.stderr)
            return

    engine.configure(options)
    engine.ping()


def main():
    parser = argparse.ArgumentParser(
        description="Play a match between two UCI engines."
    )

    parser.add_argument("--alpha", required=True)
    parser.add_argument("--beta", required=True)
    parser.add_argument("--alpha-time", required=True, type=positive_int)
    parser.add_argument("--beta-time", required=True, type=positive_int)
    parser.add_argument("--alpha-elo", type=int, help="Requested UCI Elo for Alpha")
    parser.add_argument("--beta-elo", type=int, help="Requested UCI Elo for Beta")
    parser.add_argument("--games", type=positive_int, default=1)
    parser.add_argument("--pgn", default="match.pgn")

    args = parser.parse_args()

    commands = {
        "Alpha": shlex.split(args.alpha),
        "Beta": shlex.split(args.beta),
    }

    times = {
        "Alpha": args.alpha_time,
        "Beta": args.beta_time,
    }

    elos = {"Alpha": args.alpha_elo, "Beta": args.beta_elo}
    scores = {"Alpha": 0.0, "Beta": 0.0}

    with ExitStack() as stack:
        engines = {}

        for name, command in commands.items():
            if not command:
                parser.error(f"{name} command must not be empty")

            engine = chess.engine.SimpleEngine.popen_uci(
                command, timeout=10.0
            )

            stack.callback(engine.close)
            if elos[name] is not None:
                set_elo(engine, name, elos[name])
            engines[name] = engine

            print(
                f"{name}: {engine.id.get('name', 'Unknown')} "
                f"({times[name]} ms/move)",
                flush=True,
            )

        output = stack.enter_context(
            open(args.pgn, "w", encoding="utf-8")
        )

        for number in range(args.games):
            # Alternate colors between games.
            white, black = (
                ("Alpha", "Beta")
                if number % 2 == 0
                else ("Beta", "Alpha")
            )

            players = {
                chess.WHITE: white,
                chess.BLACK: black,
            }

            board = chess.Board()
            game = chess.pgn.Game()

            game.headers.update({
                "Event": "UCI engine match",
                "Round": str(number + 1),
                "White": white,
                "Black": black,
            })

            node = game

            print(
                f"\nGame {number + 1}: "
                f"{white} (White) vs {black} (Black)"
            )

            while not board.is_game_over(claim_draw=True):
                name = players[board.turn]
                engine = engines[name]

                # python-chess takes seconds and converts
                # them to UCI movetime milliseconds.
                result = engine.play(
                    board,
                    chess.engine.Limit(
                        time=times[name] / 1000
                    ),
                    game=game,
                )

                move = result.move

                if move is None or move not in board.legal_moves:
                    raise RuntimeError(
                        f"{name} returned no legal move"
                    )

                print(
                    f"{board.fullmove_number}"
                    f"{'.' if board.turn else '...'} "
                    f"{name}: {board.san(move)}",
                    flush=True,
                )

                board.push(move)
                node = node.add_variation(move)

            outcome = board.outcome(claim_draw=True)

            game.headers["Result"] = outcome.result()
            game.headers["Termination"] = (
                outcome.termination.name.lower()
            )

            print(game, file=output, end="\n\n")
            output.flush()

            if outcome.winner is None:
                scores[white] += 0.5
                scores[black] += 0.5
            else:
                scores[players[outcome.winner]] += 1

            print(
                f"Result: {outcome.result()} "
                f"({outcome.termination.name})"
            )

            print(
                f"Score: Alpha {scores['Alpha']:g} - "
                f"{scores['Beta']:g} Beta",
                flush=True,
            )

        for engine in engines.values():
            engine.quit()

    print(f"PGN saved to {args.pgn}")


if __name__ == "__main__":
    main()