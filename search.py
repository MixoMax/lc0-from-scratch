# file: search.py
"""Independent CPU actors. Each game can have one outstanding neural evaluation."""

import math
from collections import Counter

import chess
import numpy as np

from encoding import MAX_LEGAL, N_ACTIONS, STATE_BYTES, encode, move_ids


class Node:
    __slots__ = ("children", "ids", "moves", "n", "prior", "terminal", "total", "v", "w")

    def __init__(self):
        self.moves = None
        self.terminal = None
        self.total = 0

    def expand(self, moves, ids, prior, v):
        self.moves, self.ids = moves, ids
        self.prior = prior.copy()
        self.n = np.zeros(len(moves), np.int32)
        self.w = np.zeros(len(moves), np.float32)
        self.children = {}
        self.v = v

    def select(self, in_flight=None, blocked=None):
        n, w, total = self.n, self.w, self.total
        if in_flight is not None:
            # A pending visit temporarily counts as a loss for its selecting player.
            n, w = n + in_flight, w - in_flight
            total += int(in_flight.sum())
        q = np.divide(w, n, out=np.full_like(self.w, self.v - 0.2), where=n > 0)
        scores = q + 1.5 * math.sqrt(1 + total) * self.prior / (1 + n)
        if blocked:
            scores[blocked] = -np.inf
        return int(np.argmax(scores))


def key(board):
    # Pinned python-chess key includes castling and legally available en passant.
    return board._transposition_key()


def terminal_value(board, occurrences, moves):
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if occurrences >= 3 or board.halfmove_clock >= 100 or board.is_insufficient_material():
        return 0.0
    return None


class Game:
    def __init__(self, rng, visits, cap):
        self.rng, self.budget, self.cap = rng, visits, cap
        self.board = chess.Board()
        self.counts = Counter({key(self.board): 1})
        for _ in range(int(rng.integers(9))):
            moves = list(self.board.legal_moves)
            if terminal_value(self.board, self.counts[key(self.board)], moves) is not None:
                break
            self.push(moves[int(rng.integers(len(moves)))])
        self.root = Node()
        self.need_noise = True
        self.rows = []
        self.path = []
        self.finished = None
        self.searched = 0

    def push(self, move):
        self.board.push(move)
        self.counts[key(self.board)] += 1

    def backup(self, value):
        for parent, i in reversed(self.path):
            k = key(self.board)
            self.counts[k] -= 1
            if self.counts[k] == 0:
                del self.counts[k]
            self.board.pop()
            value = -value
            parent.n[i] += 1
            parent.w[i] += value
            parent.total += 1
        self.path.clear()

    def finish(self, value, capped):
        # End result belongs to the side to move at this position.
        value = max(-1.0, min(1.0, value))
        final_turn = self.board.turn
        game = self.rows
        if game:
            n = len(game)
            width = max(len(r[1]) for r in game)
            ids = np.full((n, width), N_ACTIONS, np.uint16)
            visits = np.zeros((n, width), np.uint16)
            states = np.frombuffer(b"".join(r[0] for r in game), np.uint8).reshape(n, STATE_BYTES)
            wdl = np.zeros((n, 3), np.float32)
            for j, (_, actions, counts, turn) in enumerate(game):
                ids[j, : len(actions)] = actions
                visits[j, : len(actions)] = counts
                v = value if turn == final_turn else -value
                uncertain = (1 - abs(v)) / 3 if capped else 0.0
                wdl[j] = (
                    max(v, 0) + uncertain,
                    uncertain if capped else 1 - abs(v),
                    max(-v, 0) + uncertain,
                )
            self.finished = (states, ids, visits, wdl, capped)
        else:
            self.finished = ()

    def request(self):
        while self.finished is None:
            root = self.root
            if root.moves is not None and self.need_noise:
                root.prior = 0.75 * root.prior + 0.25 * self.rng.dirichlet(
                    np.full(len(root.moves), 0.3)
                ).astype(np.float32)
                self.need_noise = False
            if root.moves is not None and root.total >= self.budget:
                if len(self.rows) >= self.cap:
                    self.finish(float(root.w.sum() / root.total), True)
                    break
                self.rows.append(
                    (
                        encode(self.board, self.counts[key(self.board)]),
                        root.ids,
                        root.n.astype(np.uint16),
                        self.board.turn,
                    )
                )
                if len(self.rows) <= 24:
                    i = int(self.rng.choice(len(root.n), p=root.n / root.total))
                else:
                    i = int(root.n.argmax())
                self.push(root.moves[i])
                self.root = root.children[i]
                self.need_noise = True
                continue
            node = root
            while node.moves is not None:
                i = node.select()
                self.path.append((node, i))
                self.push(node.moves[i])
                child = node.children.get(i)
                if child is None:
                    child = node.children[i] = Node()
                node = child
            if node.terminal is None:
                moves = list(self.board.legal_moves)
                occurrences = self.counts[key(self.board)]
                node.terminal = terminal_value(self.board, occurrences, moves)
                if node.terminal is None:
                    ids = move_ids(self.board, moves)
                    self.pending = (node, moves, ids)
                    return encode(self.board, occurrences), ids
            value = node.terminal
            if not self.path:
                self.finish(value, False)
            else:
                self.searched += 1
                self.backup(value)
        return None

    def accept(self, probs, v):
        node, moves, ids = self.pending
        node.expand(moves, ids, probs[: len(moves)], v)
        self.searched += bool(self.path)
        self.backup(v)


def shared_views(raw, games):
    states, actions, answers = raw
    return (
        np.frombuffer(states, np.uint8).reshape(games, STATE_BYTES),
        np.frombuffer(actions, np.uint16).reshape(games, MAX_LEGAL),
        np.frombuffer(answers, np.float32).reshape(games, MAX_LEGAL + 1),
    )


def actor(index, raw, event, queue, games, visits, cap, seed):
    rng = np.random.default_rng(seed)
    boards = [Game(rng, visits, cap) for _ in range(games)]
    states, ids, answers = shared_views(raw, games)
    while True:
        complete = []
        widest = searched = 0
        for j, game in enumerate(boards):
            req = game.request()
            while req is None:
                if game.finished:
                    complete.append(game.finished)
                searched += game.searched
                boards[j] = game = Game(rng, visits, cap)
                req = game.request()
            searched += game.searched
            game.searched = 0
            state, actions = req
            states[j] = np.frombuffer(state, np.uint8)
            ids[j].fill(N_ACTIONS)
            ids[j, : len(actions)] = actions
            widest = max(widest, len(actions))
        queue.put((index, widest, searched, complete))
        event.wait()
        event.clear()
        for j, game in enumerate(boards):
            game.accept(answers[j, :-1], float(answers[j, -1]))
