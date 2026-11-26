"""Uniform ring buffer, with immutable incremental shards for persistence."""

import io

import numpy as np

from encoding import MAX_LEGAL, N_ACTIONS, STATE_BYTES

SHARD_ROWS = 4096


class Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.total = 0
        self.states = np.empty((capacity, STATE_BYTES), np.uint8)
        self.actions = np.empty((capacity, MAX_LEGAL), np.uint16)
        self.visits = np.empty_like(self.actions)
        self.wdl = np.empty((capacity, 3), np.float32)
        self.width = np.empty(capacity, np.uint16)

    def __len__(self):
        return min(self.total, self.capacity)

    def append(self, states, actions, visits, wdl):
        n = len(states)
        end = self.total + n
        take = min(n, self.capacity)
        idx = np.arange(end - take, end) % self.capacity
        width = actions.shape[1]
        self.states[idx] = states[-take:]
        self.actions[idx, :width] = actions[-take:]
        self.actions[idx, width:] = N_ACTIONS
        self.visits[idx, :width] = visits[-take:]
        self.visits[idx, width:] = 0
        self.wdl[idx] = wdl[-take:]
        self.width[idx] = width
        self.total = end

    def read(self, idx):
        width = int(self.width[idx].max())
        return self.states[idx], self.actions[idx, :width], self.visits[idx, :width], self.wdl[idx]

    def sample(self, rng, count):
        return self.read(rng.integers(len(self), size=count))

    def freeze(self, since):
        start = max(since, self.total - len(self))
        chunks = []
        for lo in range(start, self.total, SHARD_ROWS):
            hi = min(lo + SHARD_ROWS, self.total)
            chunks.append((lo, hi, self.read(np.arange(lo, hi) % self.capacity)))
        return chunks

    def restore(self, refs, directory, total):
        start = max(0, total - self.capacity)
        self.total = start
        for lo, hi, sha in refs:
            if hi <= start:
                continue
            if max(lo, start) != self.total:
                raise ValueError("noncontiguous replay checkpoint")
            with np.load(directory / sha, allow_pickle=False) as data:
                if len(data["states"]) != hi - lo:
                    raise ValueError("replay length mismatch")
                skip = max(start - lo, 0)
                self.append(*(data[k][skip:] for k in ("states", "actions", "visits", "wdl")))
        if self.total != total:
            raise ValueError("incomplete replay checkpoint")


def encode_shard(arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **dict(zip(("states", "actions", "visits", "wdl"), arrays)))
    return buf.getvalue()
