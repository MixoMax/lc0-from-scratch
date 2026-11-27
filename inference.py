"""Neural inference and deterministic PUCT shared by match evaluation and UCI."""

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


def visit(board, counts, root):
    """Run one simulation, yielding at most one neural request. False means terminal root."""
    node, path = root, []
    while node.moves is not None:
        edge = node.select()
        path.append((node, edge))
        board.push(node.moves[edge])
        counts[key(board)] += 1
        if edge not in node.children:
            node.children[edge] = Node()
        node = node.children[edge]
    if node.terminal is None:
        moves = list(board.legal_moves)
        occurrences = counts[key(board)]
        node.terminal = terminal_value(board, occurrences, moves)
    if node.terminal is None:
        actions = move_ids(board, moves)
        prior, value = yield encode(board, occurrences), actions
        node.expand(moves, actions, prior, value)
    else:
        if not path:
            return False
        value = node.terminal
    for parent, edge in reversed(path):
        pos = key(board)
        counts[pos] -= 1
        if counts[pos] == 0:
            del counts[pos]
        board.pop()
        value = -value
        parent.n[edge] += 1
        parent.w[edge] += value
        parent.total += 1
    return True
