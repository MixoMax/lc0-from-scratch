# file: inference.py

from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch

from encoding import N_ACTIONS, STATE_BYTES, encode, move_ids
from hardware import amp
from network import legal_logits
from search import Node, key, terminal_value


class Predictor:
    def __init__(self, net, device, dtype):
        self.net, self.device, self.dtype = net, device, dtype
        self.seconds = 0.0
        self.batches = self.positions = 0

    def __call__(self, requests):
        start = perf_counter()
        states = bytearray().join(x for x, _ in requests)
        ids = np.full((len(requests), max(len(a) for _, a in requests)), N_ACTIONS, dtype=np.int64)
        for i, (_, actions) in enumerate(requests):
            ids[i, :len(actions)] = actions
        with torch.inference_mode(), amp(self.device, self.dtype):
            x = torch.frombuffer(states, dtype=torch.uint8).reshape(-1, STATE_BYTES).to(self.device)
            policy, value = self.net(x)
            priors = legal_logits(policy, torch.from_numpy(ids).to(self.device)).softmax(1)
            wdl = value.float().softmax(1)
            # Blocking copy includes GPU completion in the measured interval.
            out = torch.cat((priors, wdl[:, :1] - wdl[:, 2:3]), dim=1).cpu().numpy()
        self.seconds += perf_counter() - start
        self.batches += 1
        self.positions += len(requests)
        if not np.isfinite(out).all():
            raise ValueError("network produced non-finite policy or value")
        return [(row[:len(actions)], float(row[-1])) for row, (_, actions) in zip(out, requests)]

    def timing(self):
        return {"seconds": self.seconds, "positions": self.positions, "batches": self.batches,
                "ms_per_position": 1000 * self.seconds / self.positions if self.positions else None,
                "mean_batch_size": self.positions / self.batches if self.batches else None}


@contextmanager
def _leaf(board, counts, root, select, root_edge=None):
    """Expose a leaf and its path, then restore the shared board and repetition counts."""
    node, path = root, []
    try:
        while node.moves is not None:
            edge = root_edge if node is root and root_edge is not None else select(node)
            board.push(node.moves[edge])
            pos = key(board)
            counts[pos] += 1
            path.append((node, edge, pos))
            child = node.children.get(edge)
            if child is None:
                child = node.children[edge] = Node()
            node = child
        moves = request = None
        if node.terminal is None:
            moves = list(board.legal_moves)
            occurrences = counts[key(board)]
            node.terminal = terminal_value(board, occurrences, moves)
            if node.terminal is None:
                request = encode(board, occurrences), move_ids(board, moves)
        yield node, path, moves, request
    finally:
        for _, _, pos in reversed(path):
            board.pop()
            counts[pos] -= 1
            if counts[pos] == 0:
                del counts[pos]


def _backup(path, value):
    for parent, edge, _ in reversed(path):
        value = -value
        parent.n[edge] += 1
        parent.w[edge] += value
        parent.total += 1


def visit(board, counts, root):
    """Run one simulation, yielding at most one neural request. False means terminal root."""
    with _leaf(board, counts, root, Node.select) as (node, path, moves, request):
        if request is None:
            if not path:
                return False
            value = node.terminal
        else:
            prior, value = yield request
            node.expand(moves, request[1], prior, value)
        _backup(path, value)
    return True


def visit_batch(board, counts, root, predict, unvisited, size):
    """Search up to size root edges (counting repeats). Return an immediate winning edge if found."""
    in_flight, blocked = {}, {}
    leaves, requests = [], []

    def select(node):
        return node.select(in_flight.get(node), blocked.get(node))

    for slot in range(size):
        if len(blocked.get(root, ())) == len(root.moves):
            break  # Every reachable leaf is waiting for this batch.
        first = unvisited.popleft() if unvisited else None
        with _leaf(board, counts, root, select, first) as (node, path, moves, request):
            if request is None:
                _backup(path, node.terminal)
                if len(path) == 1 and node.terminal == -1.0:
                    return path[0][1]  # Queued requests have not consumed any visits.
                continue
            leaves.append((node, path, moves))
            requests.append(request)
            if slot + 1 == size:
                break
            for parent, edge, _ in path:
                if parent not in in_flight:
                    in_flight[parent] = np.zeros_like(parent.n)
                in_flight[parent][edge] += 1
            # Exclude the queued leaf. A fully booked subtree blocks its edge one level up.
            for parent, edge, _ in reversed(path):
                edges = blocked.setdefault(parent, [])
                edges.append(edge)
                if len(edges) < len(parent.moves):
                    break
    if requests:
        answers = predict(requests)
        if len(answers) != len(requests):
            raise ValueError("predictor returned a different number of answers than requests")
        for (node, path, moves), (_, actions), (prior, value) in zip(leaves, requests, answers):
            node.expand(moves, actions, prior, value)
            _backup(path, value)
    return None
