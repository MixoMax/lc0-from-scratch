"""Regression checks for the move encoding, search signs, and portable replay."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import torch

from checkpoints import sha256
from encoding import ACTIONS, N_ACTIONS, STATE_BYTES, encode, move_ids
from network import ChessNet, legal_logits
from replay import Replay, encode_shard
from search import Game, Node, key, terminal_value


class ChessTests(unittest.TestCase):
    def test_canonical_input_and_actions(self):
        board = chess.Board()
        for uci in ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4"):
            board.push_uci(uci)
        mirrored = chess.Board().mirror()
        for m in board.move_stack:
            mirrored.push(chess.Move(m.from_square ^ 56, m.to_square ^ 56))
        self.assertEqual(encode(board, 2), encode(mirrored, 2))
        self.assertEqual(len(encode(board, 2)), STATE_BYTES)
        moves = list(board.legal_moves)
        reflected = [chess.Move(m.from_square ^ 56, m.to_square ^ 56) for m in moves]
        np.testing.assert_array_equal(move_ids(board, moves), move_ids(mirrored, reflected))
        castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        special = [castle.parse_uci(m) for m in ("e8g8", "e8c8")]
        self.assertEqual([ACTIONS[i] for i in move_ids(castle, special)], [(4, 7, 0), (4, 0, 0)])
        promo = chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1")
        ids = move_ids(promo, [promo.parse_uci("a7a8" + p) for p in "qrbn"])
        self.assertEqual(len(set(ids)), 4)
        self.assertTrue(np.all(ids >= N_ACTIONS - 88))

    def test_mate_sign_and_terminal_priority(self):
        game = Game(np.random.default_rng(4), 4, 10)
        game.board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 99 1")
        game.counts = Counter({key(game.board): 1})
        game.root = Node()
        _, ids = game.request()
        moves = game.pending[1]
        prior = np.zeros(len(ids), np.float32)
        prior[moves.index(chess.Move.from_uci("f7g7"))] = 1.0
        game.accept(prior, 0.0)
        game.need_noise = False
        self.assertIsNone(game.request())
        states, _, visits, targets, capped = game.finished
        self.assertEqual(len(states), 1)
        self.assertEqual(int(visits.max()), 4)
        self.assertFalse(capped)
        np.testing.assert_array_equal(targets, [[1.0, 0.0, 0.0]])
        self.assertTrue(game.board.is_checkmate())
        self.assertEqual(terminal_value(game.board, 3, []), -1.0)

    def test_repetition_and_terminal_labels(self):
        board = chess.Board()
        counts = Counter({key(board): 1})
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            board.push_uci(uci)
            counts[key(board)] += 1
        self.assertEqual(counts[key(board)], 3)
        self.assertEqual(terminal_value(board, 3, list(board.legal_moves)), 0.0)

    def test_policy_mask_and_student_size(self):
        model = ChessNet(88, 3, 16)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 470315)
        packed = torch.frombuffer(bytearray(encode(chess.Board(), 1)), dtype=torch.uint8)[None]
        p, v = model(packed)
        self.assertEqual(tuple(p.shape), (1, 1880))
        self.assertEqual(tuple(v.shape), (1, 3))
        legal = legal_logits(p, torch.tensor([[42, 56, N_ACTIONS]])).softmax(1)
        self.assertEqual(legal[0, -1].item(), 0.0)
        self.assertAlmostEqual(legal.sum().item(), 1.0, places=6)

    def test_incremental_replay_after_wrap(self):
        replay = Replay(8)

        def add(start, n):
            idx = np.arange(start, start + n, dtype=np.uint8)
            x = np.broadcast_to(idx[:, None], (n, STATE_BYTES))
            a = np.tile(np.array([0, 1], np.uint16), (n, 1))
            v = a + 1
            replay.append(x, a, v, np.ones((n, 3), np.float32) / 3)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            refs = []

            def freeze(since):
                for lo, hi, arrays in replay.freeze(since):
                    blob = encode_shard(arrays)
                    sha = sha256(blob)
                    (directory / sha).write_bytes(blob)
                    refs.append((lo, hi, sha))

            add(0, 6)
            freeze(0)
            add(6, 7)
            freeze(6)
            restored = Replay(8)
            restored.restore(refs, directory, 13)
            for a, b in zip(replay.read(np.arange(8)), restored.read(np.arange(8)), strict=True):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
