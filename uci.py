#!/usr/bin/env python3
"""UCI front end for lc0-from-scratch teacher and student checkpoints."""

import argparse
import os
import select
import sys
import time
from collections import Counter, deque
from pathlib import Path

import chess


def position(tokens):
    if tokens[:1] == ["startpos"]:
        board, tail = chess.Board(), tokens[1:]
    elif tokens[:1] == ["fen"] and len(tokens) >= 7:
        board, tail = chess.Board(" ".join(tokens[1:7])), tokens[7:]
        if not board.is_valid():
            raise ValueError("invalid FEN position")
    else:
        raise ValueError("expected startpos or fen followed by six FEN fields")
    if tail and tail[0] != "moves":
        raise ValueError("expected moves")
    for uci in tail[1:]:
        move = board.parse_uci(uci)
        if not move:
            raise ValueError("null move in position history")
        board.push(move)
    return board


def limits(tokens, turn):
    values = {}
    words = iter(tokens)
    for word in words:
        if word == "infinite":
            values[word] = True
        elif word in ("wtime", "btime", "movetime", "winc", "binc", "movestogo", "nodes"):
            n = int(next(words, ""))
            if n < (1 if word in ("movestogo", "nodes") else 0):
                raise ValueError(f"invalid {word}")
            values[word] = n
        else:
            raise ValueError(f"unsupported go parameter {word}")
    if "infinite" in values:
        return None, None
    side = "w" if turn else "b"
    if "movetime" in values:
        ms = values["movetime"]
    elif side + "time" in values:
        clock = values[side + "time"]
        ms = min(clock, clock / values.get("movestogo", 30) + 0.8 * values.get(side + "inc", 0))
    else:
        return None, values.get("nodes")
    reserve = min(30, max(2, 0.1 * ms))
    return max(0, ms - reserve) / 1000, values.get("nodes")


def load_predictor(args):
    import torch
    from hardware import configure_device
    from inference import Predictor
    from network import ChessNet

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device, dtype = configure_device(args.device, "auto")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != 1:
        raise ValueError("expected a format-1 trainer or distillation checkpoint")
    architecture = state.get("architecture")
    if architecture is None:
        cfg = state["config"]
        architecture = {"width": cfg["width"], "blocks": cfg["blocks"]}
    net = ChessNet(**architecture)
    net.load_state_dict(state["model"])
    predictor = Predictor(net.eval().to(device, memory_format=torch.channels_last), device, dtype)
    # Warm the root evaluation before announcing UCI readiness.
    from encoding import encode, move_ids
    board = chess.Board()
    predictor([(encode(board, 1), move_ids(board, list(board.legal_moves)))])
    return predictor


MAX_THREADS = 128
DEFAULT_BATCH_SIZE = 8
MAX_BATCH_SIZE = 128


def simulations(board, predict, batch_size=DEFAULT_BATCH_SIZE, node_limit=None, deadline=None):
    from inference import visit, visit_batch
    from search import Node, key

    # Rebuild counts from the actual game history, including a custom FEN root.
    scratch = board.root()
    counts = Counter({key(scratch): 1})
    for move in board.move_stack:
        scratch.push(move)
        counts[key(scratch)] += 1
    root = Node()
    start = time.monotonic()
    step = visit(scratch, counts, root)
    request = next(step, None)
    if request is None:
        return
    try:
        step.send(predict([request])[0])
    except StopIteration:
        pass
    seconds_per_visit = time.monotonic() - start
    unvisited = deque(sorted(range(len(root.moves)), key=root.prior.__getitem__, reverse=True))
    yield root.moves[unvisited[0]], 0

    while node_limit is None or root.total < node_limit:
        size = batch_size if node_limit is None else min(batch_size, node_limit - root.total)
        start = time.monotonic()
        if deadline is not None:
            remaining = deadline - start
            if remaining <= 0:
                return
            # Leave time for the UCI loop to publish a move. One forward pass is indivisible.
            size = min(size, max(1, int(remaining / (2 * max(seconds_per_visit, 1e-6)))))
        before = root.total
        mate = visit_batch(scratch, counts, root, predict, unvisited, size)
        if mate is not None:
            yield root.moves[mate], root.total
            return
        seconds_per_visit = (time.monotonic() - start) / (root.total - before)
        # During initial coverage many moves tie on visits. Compare their mean values
        # (equivalently their sums for equal counts), then break remaining ties by prior.
        best = max(range(len(root.moves)), key=lambda i: (root.n[i], root.w[i], root.prior[i]))
        yield root.moves[best], root.total


def protocol(config):
    board = chess.Board()
    predict = search = None
    active = False
    batch_size = DEFAULT_BATCH_SIZE
    started = None
    deadline = node_limit = None
    best, nodes = "0000", 0
    fd = sys.stdin.fileno()
    pending = bytearray()

    def initialize():
        nonlocal predict
        if predict is None:
            predict = load_predictor(config)

    def finish():
        nonlocal active, search
        if active:
            elapsed = max(time.monotonic() - started, 1e-9)
            print(f"info nodes {nodes} time {int(1000 * elapsed)} nps {int(nodes / elapsed)}"
                  f"\nbestmove {best}", flush=True)
        active, search = False, None

    while True:
        if active and deadline is not None and time.monotonic() >= deadline:
            finish()
        end = pending.find(b"\n")
        if end < 0:
            timeout = 0 if search is not None else None
            if select.select([fd], [], [], timeout)[0]:
                chunk = os.read(fd, 65536)
                if not chunk:
                    if pending:
                        pending.extend(b"\n")
                        continue
                    return 0
                pending.extend(chunk)
                continue
            if search is not None:
                update = next(search, None)
                if update is None:
                    search = None
                    if deadline is not None or node_limit is not None:
                        finish()
                else:
                    move, nodes = update
                    best = move.uci()
                    if node_limit is not None and nodes >= node_limit:
                        finish()
            continue
        words = pending[:end].decode(errors="replace").split()
        del pending[:end + 1]
        if not words:
            continue
        cmd, args = words[0].lower(), words[1:]
        try:
            if cmd == "quit":
                return 0
            if cmd == "uci":
                print("id name lc0-from-scratch", flush=True)
                print(f"option name Threads type spin default 1 min 1 max {MAX_THREADS}")
                print(f"option name BatchSize type spin default {DEFAULT_BATCH_SIZE} min 1 max {MAX_BATCH_SIZE}\nuciok", flush=True)
            elif cmd == "setoption":
                if len(args) != 4 or args[0].lower() != "name" or args[2].lower() != "value":
                    raise ValueError("expected name Threads or BatchSize followed by value N")
                option = args[1].lower()
                if option not in ("threads", "batchsize"):
                    raise ValueError(f"unknown option {args[1]}")
                n = int(args[3])
                maximum = MAX_THREADS if option == "threads" else MAX_BATCH_SIZE
                if not 1 <= n <= maximum:
                    raise ValueError(f"{args[1]} must be 1..{maximum}")
                if active:
                    raise ValueError(f"stop search before changing {args[1]}")
                if option == "batchsize":
                    batch_size = n
                else:
                    if predict is not None and n != config.threads:
                        import torch
                        torch.set_num_threads(n)
                    config.threads = n
            elif cmd == "isready":
                initialize()
                print("readyok", flush=True)
            elif cmd == "stop":
                finish()
            elif cmd in ("position", "ucinewgame"):
                new_board = position(args) if cmd == "position" else chess.Board()
                finish()
                board = new_board
            elif cmd == "go":
                received = time.monotonic()
                seconds, node_limit = limits(args, board.turn)
                finish()
                started = received
                moves = list(board.legal_moves)
                best, nodes = moves[0].uci() if moves else "0000", 0
                active = True
                deadline = None if seconds is None else started + seconds
                if len(moves) < 2 or seconds == 0:
                    if deadline is not None or node_limit is not None:
                        finish()
                else:
                    initialize()
                    search = simulations(board, predict, batch_size, node_limit, deadline)
        except ValueError as exc:
            print(f"info string invalid {cmd}: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path,
                        default=Path(__file__).resolve().with_name("student.pt"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("supply an existing checkpoint")
    args.threads = 1
    # Set native-pool defaults before NumPy and PyTorch are imported.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    return protocol(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
