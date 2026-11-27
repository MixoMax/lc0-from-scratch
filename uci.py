#!/usr/bin/env python3
"""UCI front end for lc0-from-scratch teacher and student checkpoints."""

import argparse
import multiprocessing as mp
import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
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
    return Predictor(net.eval().to(device, memory_format=torch.channels_last), device, dtype)


def run_search(job, shared, predict):
    from inference import visit
    from search import Node, key

    serial, base, history, nodes = job
    board = chess.Board(base)
    counts = Counter({key(board): 1})
    for move in history:
        board.push(move)
        counts[key(board)] += 1
    root = Node()
    while shared[0] == serial:
        simulation = visit(board, counts, root)
        try:
            request = next(simulation)
            if shared[0] != serial:
                return
            simulation.send(predict([request])[0])
        except StopIteration as done:
            if not done.value:  # A drawn root with legal moves retains its legal fallback.
                return
        scores = root.n if root.total else root.prior
        index = int(scores.argmax())
        with shared.get_lock():
            if shared[0] != serial:
                return
            shared[1:3] = (index, root.total)
        if nodes is not None and root.total >= nodes:
            return


def worker(args, jobs, shared, status):
    try:
        predict = load_predictor(args)
        status.send(("ready", None))
        while True:
            job = jobs.get()
            if job[0] == shared[0]:
                run_search(job, shared, predict)
                status.send(("done", job[0]))
    except Exception as exc:
        status.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        status.close()


def commands(events):
    with open(sys.stdin.fileno(), encoding="utf-8", closefd=False) as stream:
        for line in stream:
            events.put(("command", line))
            if line.strip() == "quit":
                return
    events.put(("command", "quit"))


def statuses(conn, events):
    try:
        while True:
            events.put(conn.recv())
    except EOFError:
        events.put(("error", "search worker exited"))
    finally:
        conn.close()


@dataclass
class Search:
    serial: int
    moves: list
    deadline: float | None
    wait_for_stop: bool


def protocol(jobs, shared, events):
    board = chess.Board()
    active = None
    serial = pending_ready = 0
    ready = False

    def finish():
        nonlocal active
        if active is None:
            return
        with shared.get_lock():
            _, index, nodes = shared[:]
            shared[0] = 0
        move = active.moves[index].uci() if active.moves else "0000"
        print(f"info nodes {nodes}\nbestmove {move}", flush=True)
        active = None

    while True:
        timeout = None
        if active and active.deadline is not None:
            timeout = active.deadline - time.monotonic()
            if timeout <= 0:
                finish()
                continue
        try:
            kind, value = events.get(timeout=timeout)
        except queue.Empty:
            finish()
            continue
        if kind == "error":
            finish()
            print(f"UCI engine: {value}", file=sys.stderr, flush=True)
            return 1
        if kind == "ready":
            ready = True
            for _ in range(pending_ready):
                print("readyok", flush=True)
            pending_ready = 0
            continue
        if kind == "done":
            if active and value == active.serial and not active.wait_for_stop:
                finish()
            continue

        words = value.split()
        if not words:
            continue
        cmd, args = words[0], words[1:]
        try:
            if cmd == "quit":
                return 0
            if cmd == "uci":
                print("id name lc0-from-scratch\nuciok", flush=True)
            elif cmd == "isready":
                if ready:
                    print("readyok", flush=True)
                else:
                    pending_ready += 1
            elif cmd == "stop":
                finish()
            elif cmd in ("position", "ucinewgame"):
                new_board = position(args) if cmd == "position" else chess.Board()
                finish()
                board = new_board
            elif cmd == "go":
                started = time.monotonic()
                seconds, nodes = limits(args, board.turn)
                finish()
                serial += 1
                moves = list(board.legal_moves)
                deadline = None if seconds is None else started + seconds
                active = Search(serial, moves, deadline, seconds is None and nodes is None)
                with shared.get_lock():
                    shared[:] = (serial, 0, 0)
                if len(moves) < 2 or seconds == 0:
                    if not active.wait_for_stop:
                        finish()
                else:
                    jobs.put((serial, board.root().fen(), board.move_stack, nodes))
        except ValueError as exc:
            print(f"info string invalid {cmd}: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path,
                        default=Path(__file__).resolve().with_name("student.pt"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads per engine")
    args = parser.parse_args()
    if args.threads < 1 or not args.checkpoint.is_file():
        parser.error("supply an existing checkpoint and a positive thread count")

    ctx = mp.get_context("spawn")
    # One atomic snapshot: search ID, index in root legal_moves, completed edge visits.
    shared = ctx.Array("q", [0, 0, 0])
    jobs = ctx.Queue()
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=worker, args=(args, jobs, shared, writer), daemon=True)
    proc.start()
    writer.close()
    events = queue.Queue()
    threading.Thread(target=commands, args=(events,), daemon=True).start()
    threading.Thread(target=statuses, args=(reader, events), daemon=True).start()
    try:
        return protocol(jobs, shared, events)
    finally:
        # The worker has no writable external state. Quit can interrupt a native forward pass.
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        jobs.cancel_join_thread()
        jobs.close()


if __name__ == "__main__":
    sys.exit(main())
