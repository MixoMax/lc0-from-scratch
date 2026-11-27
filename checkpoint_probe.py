#!/usr/bin/env python3
"""Print checkpoint information and score every legal move in a standard chess FEN."""
import argparse
import os
from collections import Counter, deque
from pathlib import Path


def request_for(board):
    from encoding import encode, move_ids

    # FEN supplies no repetition history. The root and each one-ply child
    # occur once in the available history.
    return encode(board, 1), move_ids(board, list(board.legal_moves))


def search_position(board, predict, batch, visits):
    """Return the position score, UCI move and root statistics from one search."""
    from inference import visit, visit_batch
    from search import Node, key

    scratch = board.root()
    counts = Counter({key(scratch): 1})
    for move in board.move_stack:
        scratch.push(move)
        counts[key(scratch)] += 1

    root = Node()
    step = visit(scratch, counts, root)
    request = next(step, None)
    if request is None:
        return float(root.terminal), None, root
    try:
        step.send(predict([request])[0])
    except StopIteration:
        pass

    # UCI returns a forced move immediately. Use the root network evaluation
    # for its position, since there are no backed-up visits in this case.
    if len(root.moves) == 1:
        return float(root.v), root.moves[0], root

    edges = range(len(root.moves))
    unvisited = deque(sorted(edges, key=root.prior.__getitem__, reverse=True))
    while root.total < visits:
        size = min(batch, visits - root.total)
        mate = visit_batch(scratch, counts, root, predict, unvisited, size)
        if mate is not None:
            return 1.0, root.moves[mate], root
    best = max(edges, key=lambda i: (root.n[i], root.w[i], root.prior[i]))
    return float(root.w[best] / root.n[best]), root.moves[best], root


def evaluate_moves(predict, board, batch, visits):
    mover = board.turn
    side = "White" if mover else "Black"
    print(f"Position: {board.fen(en_passant='fen')}")
    print(f"W-L from {side}'s perspective (higher is better)")
    outcome = board.outcome()
    if outcome is not None:
        value = 0 if outcome.winner is None else (1 if outcome.winner == mover else -1)
        print(f"Game over: {outcome.termination.name.lower()} ({outcome.result()}), "
              f"W-L {value:+.4f}. No moves to play.")
        return

    best = None
    root_stats = {}
    if visits is None:
        (_, baseline), = predict([request_for(board)])
        before = float(baseline)
    else:
        print(f"Search: up to {visits} visits per position, batch size {batch}")
        before, best, root = search_position(board, predict, batch, visits)
        if root.moves is not None:
            for move, n, w in zip(root.moves, root.n, root.w):
                root_stats[move] = (int(n), f"{w / n:+.4f}" if n else "-")
    moves = list(board.legal_moves)
    scores = {}
    requests = []
    pending = []
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        outcome = child.outcome()
        if outcome is None and visits is not None:
            child_value, _, _ = search_position(child, predict, batch, visits)
            scores[move] = -child_value, ""
        elif outcome is None:
            pending.append(move)
            requests.append(request_for(child))
        else:
            score = 0.0 if outcome.winner is None else float(1 if outcome.winner == mover else -1)
            scores[move] = score, outcome.termination.name.lower().replace("_", " ")

    for start in range(0, len(requests), batch):
        answers = predict(requests[start:start + batch])
        for move, (_, value) in zip(pending[start:start + batch], answers, strict=True):
            # The child value head sees the opponent as the side to move.
            scores[move] = -float(value), ""

    choice = ""
    if visits is not None:
        choice = (f" | UCI bestmove {best.uci()} ({board.san(best)})"
                  if best is not None else " | terminal search root")
    print(f"{len(moves)} legal moves{choice}")
    extra = f" {'Root N':>7} {'Root Q':>9}" if visits is not None else ""
    print(f"{'Move':<9} {'UCI':<6} {'Before':>9} {'After':>9} {'Change':>9}{extra}")
    for move in sorted(moves, key=lambda m: (-scores[m][0], m.uci())):
        after, note = scores[move]
        if visits is not None:
            n, q = root_stats.get(move, (0, "-"))
            extra = f" {n:7d} {q:>9}"
        print(f"{board.san(move):<9} {move.uci():<6} {before:+9.4f} "
              f"{after:+9.4f} {after - before:+9.4f}{extra}  {note}".rstrip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--fen", help="quoted FEN string (prompts if omitted)")
    p.add_argument("--batch", type=int, help="UCI BatchSize (default 8, range 1..128)")
    p.add_argument("--visits", type=int,
                   help="UCI go nodes limit per position (omit for direct network scores)")
    args = p.parse_args()
    if args.visits is not None and args.visits < 1:
        p.error("visits must be positive")
    if not 1 <= args.threads <= 128:
        p.error("threads must be 1..128")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    import chess
    from uci import DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE, load_predictor
    batch = DEFAULT_BATCH_SIZE if args.batch is None else args.batch
    if not 1 <= batch <= MAX_BATCH_SIZE:
        p.error(f"batch must be 1..{MAX_BATCH_SIZE}")

    import torch

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

    try:
        fen = args.fen if args.fen is not None else input("FEN: ")
        board = chess.Board(fen.strip())
    except EOFError:
        p.error("enter a FEN at the prompt or pass --fen")
    except ValueError as exc:
        p.error(f"invalid FEN: {exc}")
    if not board.is_valid():
        p.error(f"invalid chess position (board status {board.status()})")
    evaluate_moves(predict, board, batch, args.visits)


if __name__ == "__main__":
    main()
