"""UCI regressions. Run with python test_uci.py alongside uci.py."""

import multiprocessing as mp
import queue
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

from network import ChessNet
from uci import limits, position, protocol, run_search


class UciTests(unittest.TestCase):
    def test_clocks_and_history(self):
        clocks = "wtime 30000 btime 60000".split()
        self.assertEqual(limits(clocks, chess.WHITE), (0.97, None))
        self.assertEqual(limits(clocks, chess.BLACK), (1.97, None))
        self.assertEqual(limits(clocks + ["movetime", "1"], chess.BLACK), (0, None))
        self.assertEqual(limits(["wtime", "5", "winc", "10000"], chess.WHITE), (0.003, None))
        board = position("startpos moves g1f3 g8f6 f3g1 f6g8".split())
        self.assertTrue(board.is_repetition(2))
        with self.assertRaises(ValueError):
            position("startpos moves e2e4 e2e5".split())

    def test_blocked_forward_and_late_result(self):
        # Keep an actual PUCT visit suspended inside predict while the protocol handles
        # stop, a second deadline, and a third search. Then release the obsolete visit.
        shared = mp.Array("q", [0, 0, 0])
        jobs, events = queue.Queue(), queue.Queue()
        blocked, release = threading.Event(), threading.Event()

        def slow_predict(requests):
            size = len(requests[0][1])
            blocked.set()
            if not release.wait(5):
                raise TimeoutError()
            prior = np.zeros(size, np.float32)
            prior[-1] = 1.0
            return [(prior, 0.0)]

        def command(text):
            events.put(("command", text))

        result = StringIO()
        output = queue.Queue()

        class Capture:
            def write(self, text):
                result.write(text)
                for line in text.splitlines():
                    if line:
                        output.put(line)

            def flush(self):
                pass

        with redirect_stdout(Capture()):
            handler = threading.Thread(target=protocol, args=(jobs, shared, events), daemon=True)
            handler.start()
            try:
                events.put(("ready", None))
                command("go infinite")
                old_job = jobs.get(timeout=2)
                search = threading.Thread(target=run_search, args=(old_job, shared, slow_predict))
                search.start()
                self.assertTrue(blocked.wait(2))
                command("isready")
                self.assertEqual(output.get(timeout=1), "readyok")
                command("stop")
                self.assertEqual(output.get(timeout=1), "info nodes 0")
                self.assertIn(chess.Move.from_uci(output.get(timeout=1).split()[1]), chess.Board().legal_moves)
                command("position startpos moves e2e4")
                command("go movetime 50")
                self.assertEqual(output.get(timeout=1), "info nodes 0")
                reply = chess.Move.from_uci(output.get(timeout=1).split()[1])
                self.assertIn(reply, position("startpos moves e2e4".split()).legal_moves)
                command("go infinite")
                jobs.get(timeout=2)  # Timed-out job
                new_job = jobs.get(timeout=2)
                release.set()
                search.join(timeout=1)
                self.assertFalse(search.is_alive())
                self.assertEqual(shared[:], [new_job[0], 0, 0])
                events.put(("done", old_job[0]))
                command("stop")
                command("stop")
            finally:
                release.set()
                command("quit")
                handler.join(timeout=2)
        self.assertEqual(result.getvalue().count("bestmove"), 3)

    def test_real_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("config", "architecture"):
                with self.subTest(kind=kind):
                    arch = dict(width=16, blocks=1) if kind == "config" else dict(width=88, blocks=3, key_size=16)
                    checkpoint = Path(tmp) / "model.pt"
                    torch.save({"format": 1, kind: arch, "model": ChessNet(**arch).state_dict()}, checkpoint)
                    args = [sys.executable, str(Path(__file__).with_name("uci.py")), str(checkpoint)]
                    engine = chess.engine.SimpleEngine.popen_uci(args, timeout=20)
                    try:
                        engine.ping()
                        for board in (chess.Board(), position("startpos moves e2e4".split()),
                                      position("fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 12 moves e1g1".split())):
                            result = engine.play(board, chess.engine.Limit(nodes=4), info=chess.engine.INFO_ALL)
                            self.assertIn(result.move, board.legal_moves)
                            self.assertEqual(result.info["nodes"], 4)
                        for fen in ("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"):
                            result = engine.play(chess.Board(fen), chess.engine.Limit(time=0.05))
                            self.assertFalse(result.move)
                    finally:
                        engine.quit()


if __name__ == "__main__":
    unittest.main()
