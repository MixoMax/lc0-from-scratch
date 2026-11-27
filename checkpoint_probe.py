#!/usr/bin/env python3
"""Print checkpoint information and score every legal move in a standard chess FEN."""
import argparse
import os
from pathlib import Path


def request_for(board):
    from encoding import encode, move_ids

    # FEN supplies no repetition history. The root and each one-ply child
    # occur once in the available history.
    return encode(board, 1), move_ids(board, list(board.legal_moves))


def evaluate_moves(predict, board):
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

    (_, baseline), = predict([request_for(board)])
    before = float(baseline)
    moves = list(board.legal_moves)
    scores = {}
    requests = []
    pending = []
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        outcome = child.outcome()
        if outcome is None:
            pending.append(move)
            requests.append(request_for(child))
        else:
            score = 0.0 if outcome.winner is None else float(1 if outcome.winner == mover else -1)
            scores[move] = score, outcome.termination.name.lower().replace("_", " ")

    for start in range(0, len(requests), 32):
        answers = predict(requests[start:start + 32])
        for move, (_, value) in zip(pending[start:start + 32], answers, strict=True):
            # The child value head sees the opponent as the side to move.
            scores[move] = -float(value), ""

    print(f"{len(moves)} legal moves")
    print(f"{'Move':<9} {'UCI':<6} {'Before':>9} {'After':>9} {'Change':>9}")
    for move in sorted(moves, key=lambda m: (-scores[m][0], m.uci())):
        after, note = scores[move]
        print(f"{board.san(move):<9} {move.uci():<6} {before:+9.4f} "
              f"{after:+9.4f} {after - before:+9.4f}  {note}".rstrip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--fen", help="quoted FEN string (prompts if omitted)")
    args = p.parse_args()
    if not 1 <= args.threads <= 128:
        p.error("threads must be 1..128")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    import chess
    import torch
    from uci import load_predictor

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
    evaluate_moves(predict, board)


if __name__ == "__main__":
    main()
