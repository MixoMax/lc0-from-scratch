#!/usr/bin/env python3
"""Evaluate lc0-from-scratch teacher checkpoints against a strength-limited Stockfish.

Place this file next to train.py. Install the training dependencies, PyTorch, tqdm
and a Stockfish executable. The default runs 10 games per checkpoint at Elo 1600.
Each pair uses the same opening with colors reversed. Use an even number of games
for exact color balance. Adaptive mode seeks a 45–55% model score in up to five
rounds. Opponent settings refer to this engine and node budget.

Examples:

    python eval.py /mnt/photos/checkpoints/run-3m --interval-hours .25 --elo 1600 --games 100
    python eval.py /mnt/photos/checkpoints/run-3m --mode adaptive --elo 1600 --games 10
    python eval.py /mnt/photos/checkpoints/run-3m --device cpu --games 2 --visits 8
"""

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

try:
    import chess.engine
    import torch
    from tqdm import tqdm

    from inference import Predictor, visit
    from hardware import configure_device, host_limits
    from network import ChessNet
    from search import Node, key
except ImportError as exc:
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = None

# Short opening lines, played from both sides. The empty line is the initial position.
OPENINGS = (
    "", "e4 e5 Nf3 Nc6 Bb5", "e4 e5 Nf3 Nc6 Bc4", "e4 e5 Nf3 Nc6 d4",
    "e4 e5 Nf3 Nf6", "e4 e5 Nc3 Nf6", "e4 c5 Nf3 d6", "e4 c5 Nf3 Nc6",
    "e4 c5 c3 Nf6", "e4 e6 d4 d5", "e4 c6 d4 d5", "e4 d6 d4 Nf6",
    "d4 d5 c4 e6", "d4 d5 c4 c6", "d4 d5 c4 dxc4", "d4 d5 Bf4 Nf6",
    "d4 Nf6 c4 g6", "d4 Nf6 c4 e6", "d4 Nf6 c4 c5", "d4 f5 g3 Nf6",
    "c4 e5 Nc3 Nf6", "c4 c5 Nf3 Nc6", "Nf3 d5 g3 Nf6", "Nf3 Nf6 c4 e6",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", type=Path, help="server run directory or its versions directory")
    parser.add_argument("--interval-hours", type=float, default=1.0,
                        help="minimum wall-clock hours since last sample; 0 selects all versions")
    parser.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--elo", type=int, default=1600, help="fixed or initial Stockfish UCI_Elo")
    parser.add_argument("--games", type=int, default=10, help="games per checkpoint per Elo round")
    parser.add_argument("--rounds", type=int, default=5, help="maximum rounds in adaptive mode")
    parser.add_argument("--stockfish", default="/usr/games/stockfish", help="Stockfish executable")
    parser.add_argument("--sf-nodes", type=int, default=10000, help="Stockfish nodes per move")
    parser.add_argument("--visits", type=int, help="model root visit target per move (default: saved recipe)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--parallel-games", type=int, help="games to batch (auto: up to 4 CPU or 8 CUDA)")
    parser.add_argument("--cpu-threads", type=int, help="PyTorch intra-op threads (auto: up to 4 on CPU)")
    parser.add_argument("--seed", type=int, default=1234, help="seed for the paired opening schedule")
    parser.add_argument("--output", type=Path, help="new JSON report path (default: eval-<UTC time>.json)")
    args = parser.parse_args()
    if not math.isfinite(args.interval_hours) or args.interval_hours < 0:
        parser.error("--interval-hours must be finite and nonnegative")
    for name in ("games", "rounds", "sf_nodes", "visits", "parallel_games", "cpu_threads"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


@dataclass(frozen=True)
class Version:
    path: Path
    mtime_ns: int


def sample_versions(run, hours):
    versions = run if run.name == "versions" else run / "versions"
    if not versions.is_dir():
        raise ValueError(f"Missing directory {versions}. Pass the run directory or versions/.")
    entries = sorted(
        (Version(p, p.stat().st_mtime_ns) for p in versions.iterdir()
         if re.fullmatch("[0-9a-f]{64}", p.name) and p.is_file()),
        key=lambda v: (v.mtime_ns, v.path.name),
    )
    if not entries:
        raise ValueError(f"No checkpoint versions at {versions}. Pass the run directory or versions/.")
    gap_ns = round(hours * 3_600_000_000_000)
    selected = []
    for entry in entries:
        if not selected or entry.mtime_ns - selected[-1].mtime_ns >= gap_ns:
            selected.append(entry)
    return selected, len(entries)


def opening_schedule(n_games, seed):
    rng = random.Random(seed)
    pairs = []
    while 2 * len(pairs) < n_games:
        batch = list(OPENINGS)
        rng.shuffle(batch)
        pairs.extend(batch)
    return [pairs[i // 2] for i in range(n_games)]


def score(games):
    counts = Counter(g["model_result"] for g in games)
    w, d, l = (counts[k] for k in ("win", "draw", "loss"))
    return {"wins": w, "draws": d, "losses": l, "score": (w + 0.5 * d) / len(games) if games else None}


def adaptive_next(rounds, low, high):
    """Return the next opponent or a reason to stop. Bracket a 50% score."""
    last = rounds[-1]
    elo, s = last["elo"], last["summary"]
    if abs(s["wins"] - s["losses"]) * 10 <= len(last["games"]):
        return None, "45–55% score"
    up = s["score"] > 0.5
    if (not up and elo == low) or (up and elo == high):
        return None, "scored above 50% at max Elo" if up else "scored below 50% at min Elo"
    beaten = [r["elo"] for r in rounds if r["summary"]["score"] > 0.5]
    lost = [r["elo"] for r in rounds if r["summary"]["score"] < 0.5]
    if beaten and lost:
        lo, hi = max(beaten), min(lost)
        if hi - lo <= 1:
            return None, "adjacent Elo settings bracket 50%"
        return (lo + hi) // 2, None
    step = 200 * 2 ** (len(rounds) - 1)
    candidate = min(high, elo + step) if up else max(low, elo - step)
    return candidate, None


def write_report(path, report):
    # An interruption during serialization leaves the previous complete JSON readable.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.close()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def utc_now():
    return datetime.now(timezone.utc)


def runtime(args):
    cores, ram = host_limits()
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    try:
        device, dtype = configure_device("cuda" if use_cuda else "cpu", "auto")
    except RuntimeError as exc:
        if args.device != "auto" or not use_cuda:
            raise
        print(f"CUDA probe failed ({exc}). Using CPU.", file=sys.stderr)
        device, dtype = configure_device("cpu", "auto")
    threads = args.cpu_threads or (min(cores, 4) if device.type == "cpu" else 1)
    torch.set_num_threads(threads)
    # Each lane owns a single-threaded Stockfish. Budget memory for its net and hash.
    limit = min(cores, max(1, ram // (512 * 1024**2)), 8 if device.type == "cuda" else 4)
    lanes = min(args.games, args.parallel_games or limit)
    info = {"device": str(device), "dtype": str(dtype), "cpu_quota_cores": cores,
            "torch_threads": threads, "parallel_games": lanes, "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    return device, dtype, info


class Match:
    def __init__(self, index, line, engine):
        self.index, self.engine = index, engine
        self.white = index % 2 == 0
        self.line = line
        self.board = chess.Board()
        self.counts = Counter({key(self.board): 1})
        self.root = Node()
        self.sf_seconds = 0.0
        self.model_moves = 0
        for san in line.split():
            self.play(self.board.parse_san(san))

    def play(self, move):
        root = self.root
        child = None if root.moves is None else root.children.get(root.moves.index(move))
        self.root = child if child is not None else Node()
        self.board.push(move)
        self.counts[key(self.board)] += 1

    def stockfish_move(self, limit):
        start = perf_counter()
        move = self.engine.play(self.board, limit, game=self).move
        self.sf_seconds += perf_counter() - start
        if move is None or move not in self.board.legal_moves:
            raise ValueError(f"Stockfish returned {move} in {self.board.fen()}")
        return move

    def result(self):
        # Automatically take draws on reaching the third occurrence or 100th quiet ply.
        board = self.board
        outcome = board.outcome()
        if outcome:
            winner, reason = outcome.winner, outcome.termination.name.lower()
        elif self.counts[key(board)] >= 3 or board.halfmove_clock >= 100:
            winner = None
            reason = "threefold_repetition" if self.counts[key(board)] >= 3 else "fifty_moves"
        else:
            return None
        board_result, model_result = "1/2-1/2", "draw"
        if winner is not None:
            board_result = "1-0" if winner else "0-1"
            model_result = "win" if winner == self.white else "loss"
        return {
            "game": self.index + 1,
            "model_color": "white" if self.white else "black",
            "opening_san": self.line,
            "result": board_result,
            "model_result": model_result,
            "termination": reason,
            "plies": board.ply(),
            "model_moves": self.model_moves,
            "stockfish_seconds": self.sf_seconds,
            "moves_uci": " ".join(m.uci() for m in board.move_stack),
        }

    def search(self, visits):
        """PUCT with the trainer's Node, encoding and value convention. Yield neural leaves."""
        root = self.root
        while root.total < visits:
            if not (yield from visit(self.board, self.counts, root)):
                raise ValueError("model search requested in a terminal position")
        return root.moves[int(root.n.argmax())]


def model_moves(matches, predict, visits):
    # One pending leaf per independent game. A batch can shrink when a search finishes.
    searches = [(g, g.search(visits), None) for g in matches]
    while searches:
        pending, requests = [], []
        for g, search, answer in searches:
            try:
                requests.append(search.send(answer))
                pending.append((g, search))
            except StopIteration as done:
                g.play(done.value)
                g.model_moves += 1
        searches = []
        if pending:
            for (g, search), answer in zip(pending, predict(requests), strict=True):
                searches.append((g, search, answer))


def play_round(row, engines, workers, lines, predict, visits, nodes):
    elo = row["elo"]
    for engine in engines:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    limit = chess.engine.Limit(nodes=nodes)
    with tqdm(total=len(lines), desc=f"  SF {elo}", unit="game", leave=False,
              dynamic_ncols=True, miniters=0) as bar:
        for start in range(0, len(lines), len(engines)):
            batch = lines[start:start + len(engines)]
            live = [Match(start + i, line, engines[i]) for i, line in enumerate(batch)]
            while live:
                active = []
                for g in live:
                    result = g.result()
                    if result is None:
                        active.append(g)
                    else:
                        row["games"].append(result)
                        row["summary"] = score(row["games"])
                        bar.update(1)
                live = active
                to_search = [g for g in live if g.board.turn == g.white]
                replies = [(g, workers.submit(g.stockfish_move, limit)) for g in live
                           if g.board.turn != g.white]
                model_moves(to_search, predict, visits)
                for g, reply in replies:
                    g.play(reply.result())
                if live:
                    s = row["summary"]
                    bar.set_postfix_str(f"ply {max(g.board.ply() for g in live)}  W/D/L "
                                        f"{s['wins']}/{s['draws']}/{s['losses']}", refresh=False)
                    bar.update(0)
    row["games"].sort(key=lambda g: g["game"])


def evaluate(version, entry, device, dtype, engines, workers, bounds, args, lines, save):
    started = perf_counter()
    state = torch.load(version.path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != 1:
        raise ValueError(f"Expected a format-1 teacher checkpoint at {version.path}")
    cfg = state["config"]
    net = ChessNet(cfg["width"], cfg["blocks"])
    net.load_state_dict(state["model"], strict=True)
    net = net.eval().to(device, memory_format=torch.channels_last)
    visits = args.visits or cfg["visits"]
    entry.update(step=state["step"], training_games=state["games"], training_rows=state["rows"],
                 config=cfg, parameters=sum(p.numel() for p in net.parameters()), visits=visits)
    del state
    predict = Predictor(net, device, dtype)
    elo, reason = args.elo, None
    tqdm.write(f"Step {entry['step']:,}  |  {version.path.name[:12]}  |  "
               f"+{entry['hours_from_first']:.2f} h  |  {visits} visits/move")
    try:
        for _ in range(args.rounds if args.mode == "adaptive" else 1):
            row = {"elo": elo, "summary": score([]), "games": []}
            entry["rounds"].append(row)
            play_round(row, engines, workers, lines, predict, visits, args.sf_nodes)
            s = row["summary"]
            tqdm.write(f"  SF {elo}  {s['wins']}W {s['draws']}D {s['losses']}L  ({s['score']:.1%})")
            entry["inference"] = predict.timing()
            if args.mode == "fixed":
                reason = "fixed"
                break
            elo, reason = adaptive_next(entry["rounds"], *bounds)
            if reason:
                break
            save()
        best = entry["rounds"][-1]
        if reason is None:
            # Round limit: closest tested score to 50%, preferring the latest on ties.
            best = min(reversed(entry["rounds"]),
                       key=lambda r: abs(r["summary"]["wins"] - r["summary"]["losses"]))
        entry.update(reported_elo=best["elo"], reported_score=best["summary"], stop=reason or "round limit")
    finally:
        entry.update(inference=predict.timing(), wall_seconds=perf_counter() - started)


def print_table(report):
    rows = [["Step", "Age h", "Version", "SF Elo", "W", "D", "L", "Score", "All games", "NN ms/pos", "Stop"]]
    for c in report["checkpoints"]:
        if "reported_score" in c:
            s, ms = c["reported_score"], c["inference"]["ms_per_position"]
            rows.append([str(c["step"]), f"{c['hours_from_first']:.2f}", c["sha"][:10],
                         str(c["reported_elo"]), str(s["wins"]), str(s["draws"]), str(s["losses"]),
                         f"{s['score']:.1%}", str(sum(len(r["games"]) for r in c["rounds"])),
                         f"{ms:.2f}" if ms is not None else "–", c["stop"]])
    widths = [max(map(len, col)) for col in zip(*rows)]
    print("\nCheckpoint scores (model perspective; W/D/L at displayed SF Elo)")
    for i, row in enumerate(rows):
        print("  ".join(cell.rjust(w) for cell, w in zip(row, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def run(args):
    selected, total = sample_versions(args.run, args.interval_hours)
    device, dtype, hw = runtime(args)
    lines = opening_schedule(args.games, args.seed)
    now = utc_now()
    path = args.output or Path(f"eval-{now:%Y%m%dT%H%M%S%fZ}.json")
    report = {"started_utc": now.isoformat(), "status": "running",
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "hardware": hw, "available_versions": total, "selected_versions": [v.path.name for v in selected],
              "time_basis": "server version file mtime; gaps include trainer downtime",
              "inference_timing": "wall time packing network batches through CPU-visible legal priors and WDL value, including first call",
              "checkpoints": []}

    def save():
        write_report(path, report)

    with ExitStack() as stack:
        engines = []
        for _ in range(hw["parallel_games"]):
            engine = stack.enter_context(chess.engine.SimpleEngine.popen_uci(args.stockfish))
            if "UCI_LimitStrength" not in engine.options or "UCI_Elo" not in engine.options:
                raise ValueError(f"{args.stockfish} must advertise UCI_LimitStrength and UCI_Elo")
            opt = engine.options["UCI_Elo"]
            bounds = (opt.min, opt.max)
            if not bounds[0] <= args.elo <= bounds[1]:
                raise ValueError(f"--elo {args.elo} outside {engine.id.get('name')} range {bounds[0]}–{bounds[1]}")
            engine.configure({"Threads": 1})
            engines.append(engine)
        workers = stack.enter_context(ThreadPoolExecutor(max_workers=len(engines)))
        report["engine"] = {"id": engines[0].id, "uci_elo_range": bounds, "threads": 1,
                            "UCI_LimitStrength": True, "nodes_per_move": args.sf_nodes}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=False)
        print(f"Selected {len(selected)}/{total} versions  |  {hw['gpu'] or device} ({dtype})  |  "
              f"{hw['parallel_games']} game lanes  |  {hw['torch_threads']} PyTorch threads\n"
              f"{engines[0].id.get('name')}  |  {args.mode}, starting Elo {args.elo}  |  "
              f"{args.games} games/round  |  {args.sf_nodes:,} Stockfish nodes/move\nReport: {path.resolve()}", flush=True)
        start = perf_counter()
        try:
            save()
            for v in tqdm(selected, desc="Checkpoints", unit="ckpt", dynamic_ncols=True):
                entry = {"sha": v.path.name, "mtime_utc": datetime.fromtimestamp(v.mtime_ns / 1e9, timezone.utc).isoformat(),
                         "hours_from_first": (v.mtime_ns - selected[0].mtime_ns) / 3.6e12, "rounds": []}
                report["checkpoints"].append(entry)
                evaluate(v, entry, device, dtype, engines, workers, bounds, args, lines, save)
                save()
            report["status"] = "complete"
        except BaseException as exc:
            report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report.update(finished_utc=utc_now().isoformat(), wall_seconds=perf_counter() - start)
            save()
    print_table(report)
    print(f"\nFull game records and timing: {path.resolve()}")


if __name__ == "__main__":
    options = arguments()
    if IMPORT_ERROR:
        sys.exit(f"Evaluation dependencies unavailable: {IMPORT_ERROR}. Place eval.py next to train.py "
                 "and install the project requirements, PyTorch and tqdm in this environment.")
    try:
        run(options)
    except KeyboardInterrupt:
        sys.exit("Evaluation interrupted.")
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        sys.exit(f"Evaluation failed: {error}")
