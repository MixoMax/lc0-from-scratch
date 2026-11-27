#!/usr/bin/env python3
"""Place beside uci.py and run with a trainer or student checkpoint. Reads the checkpoint only."""
import argparse
import os
from pathlib import Path
from threading import Event
from time import perf_counter


GAME = (
    "d2d4 d7d5 h2h3 c7c5 d1d3 c5d4 d3d4 b8c6 d4d1 e7e5 e2e3 g8f6 "
    "g2g3 c8f5 f1d3 f5e6 a2a4 d8c7 d3f1 d5d4 e3d4 c6d4 c2c3 e6b3 "
    "d1d2 d4c2 e1e2 c7c4 e2f3"
).split()


def history(board):
    from collections import Counter
    from search import key

    scratch = board.root()
    counts = Counter({key(scratch): 1})
    for move in board.move_stack:
        scratch.push(move)
        counts[key(scratch)] += 1
    return scratch, counts


def requests_for(boards):
    from encoding import encode, move_ids
    from search import key

    requests = []
    for board in boards:
        scratch, counts = history(board)
        moves = list(scratch.legal_moves)
        requests.append((encode(scratch, counts[key(scratch)]), move_ids(scratch, moves)))
    return requests


def check_mate_choice():
    """Use a known chess position and controlled priors to isolate final move selection."""
    import chess
    import numpy as np
    from uci import simulations

    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    stalemate = board.parse_san("Qe4")
    mate = board.parse_san("Qg7#")
    for move, expected in ((stalemate, "stalemate"), (mate, "checkmate")):
        board.push(move)
        outcome = board.outcome()
        assert outcome and outcome.termination.name.lower() == expected
        board.pop()
    root_state, _ = requests_for([board])[0]
    moves = list(board.legal_moves)

    def controlled_predict(requests):
        results = []
        for state, actions in requests:
            probs = np.full(len(actions), 1.0 / len(actions), dtype=np.float32)
            if state == root_state:
                probs.fill(0)
                probs[moves.index(stalemate)] = 0.98
                probs[moves.index(mate)] = 0.02
            results.append((probs, 0.0))
        return results

    for best, visits in simulations(board, controlled_predict):
        if visits >= 128:
            print(f"Controlled search (separate from checkpoint): 98% prior Qe4 stalemate, "
                  f"2% Qg7 mate. At {visits} visits UCI selects {board.san(best)}.")
            break


def check_serving(predict, requests):
    """Pass identical positions, move order, weights and batch through the two serving paths."""
    import numpy as np
    from encoding import MAX_LEGAL, N_ACTIONS, STATE_BYTES
    from train import Inference

    size = len(requests)
    states = np.empty((size, STATE_BYTES), np.uint8)
    ids = np.full((size, MAX_LEGAL), N_ACTIONS, np.uint16)
    answers = np.empty((size, MAX_LEGAL + 1), np.float32)
    width = max(len(actions) for _, actions in requests)
    for i, (packed, actions) in enumerate(requests):
        states[i] = np.frombuffer(packed, np.uint8)
        ids[i, :len(actions)] = actions
    training = Inference(predict.net, predict.device, predict.dtype, size,
                         [(states, ids, answers)], [Event()])
    training.serve([(0, width, 0, [])], size)
    uci_answers = predict(requests)
    p_diff = max(float(np.abs(answers[i, :len(p)] - p).max())
                 for i, (p, _) in enumerate(uci_answers))
    v_diff = max(abs(float(answers[i, -1]) - v) for i, (_, v) in enumerate(uci_answers))
    print(f"Trainer vs UCI, identical batch: max |prior difference|={p_diff:.7g}, "
          f"max |W-L difference|={v_diff:.7g}")


def time_batches(predict, requests):
    print("Predictor timings (warm, includes packing, transfers and synchronisation)")
    for size in (1, 4, 8, 16, 32):
        batch = [requests[i % len(requests)] for i in range(size)]
        for _ in range(2):
            predict(batch)
        start = perf_counter()
        repeats = 12
        for _ in range(repeats):
            predict(batch)
        duration = (perf_counter() - start) / repeats
        print(f"  {size:2d} positions   {duration * 1000:7.2f} ms/batch   "
              f"{size / duration:8.0f} positions/s")


def inspect_search(predict, board, nodes):
    """Run the UCI visit primitive and retain its root so the moves can be inspected."""
    import numpy as np
    from inference import visit
    from search import Node

    scratch, counts = history(board)
    root = Node()
    before_seconds, before_positions = predict.seconds, predict.positions
    start = perf_counter()
    while root.total < nodes:
        step = visit(scratch, counts, root)
        try:
            request = next(step)
            step.send(predict([request])[0])
        except StopIteration as done:
            if not done.value:
                raise ValueError("selected position is terminal")
    elapsed = perf_counter() - start
    forward = predict.seconds - before_seconds
    positions = predict.positions - before_positions
    chosen = int(root.n.argmax())
    print(f"  {nodes} visits in {elapsed:.3f}s ({nodes / elapsed:.0f}/s), "
          f"{positions} neural evaluations, {forward / elapsed:.1%} in timed predictor section")
    print(f"  Network W-L={root.v:+.3f}, highest prior={root.moves[int(root.prior.argmax())]}, "
          f"most visited={root.moves[chosen]}, explored={np.count_nonzero(root.n)}/{len(root.n)}")
    mates = set()
    for i, move in enumerate(root.moves):
        scratch.push(move)
        if scratch.is_checkmate():
            mates.add(i)
        scratch.pop()
    order = sorted(range(len(root.n)), key=lambda i: (-int(root.n[i]), -float(root.prior[i])))
    print("  move      prior    visits   root Q")
    for i in [i for j, i in enumerate(order) if j < 5 or i in mates]:
        q = f"{root.w[i] / root.n[i]:+.3f}" if root.n[i] else "     ."
        flag = "  mate in one" if i in mates else ""
        print(f"  {root.moves[i].uci():6s}   {root.prior[i]:5.1%}   {root.n[i]:6d}   {q}{flag}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--nodes", type=int, default=128)
    args = p.parse_args()
    if not 1 <= args.threads <= 128 or args.nodes < 1:
        p.error("threads must be 1..128 and nodes must be positive")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    import torch
    from uci import load_predictor, position

    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    print("Checkpoint:", args.checkpoint)
    print("Saved settings:", state.get("config", state.get("architecture")))
    print("Counters:", {k: state[k] for k in ("step", "games", "rows", "capped") if k in state})
    if "optimizer" in state:
        print("Saved optimizer LR:", state["optimizer"]["param_groups"][0]["lr"])
    cfg = state.get("config", {})
    if state.get("rows") and "plies" in cfg and "capped" in state:
        print(f"Lifetime capped-row share (constant saved ply cap): "
              f"{state['capped'] * cfg['plies'] / state['rows']:.1%}")
    del state
    predict = load_predictor(args)
    print(f"Runtime: {predict.device}, {predict.dtype}, PyTorch {torch.__version__}, "
          f"{torch.get_num_threads()} intra-op threads")
    check_mate_choice()
    cases = [
        ("Starting position", []),
        ("White before 13.Qd2", GAME[:24]),
        ("Black before 15...Qe4#", GAME),
        ("White with mate in one", "e2e4 f7f6 d2d4 g7g5".split()),
    ]
    boards = [position(["startpos", "moves", *moves]) for _, moves in cases]
    requests = requests_for(boards)
    check_serving(predict, requests)
    time_batches(predict, requests)
    for (label, _), board in zip(cases, boards):
        print(label)
        inspect_search(predict, board, args.nodes)


if __name__ == "__main__":
    main()
