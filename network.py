import torch
from torch import nn

from encoding import ACTIONS, N_ACTIONS


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
        )

    def forward(self, x):
        return (x + self.path(x)).relu()


class ChessNet(nn.Module):
    def __init__(self, width, blocks, key_size=None):
        super().__init__()
        key_size = key_size or max(16, width // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(52, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        self.tower = nn.Sequential(*(Block(width) for _ in range(blocks)))
        self.qk = nn.Conv2d(width, 2 * key_size, 1)
        self.promotion = nn.Conv2d(width, 4, 1)
        self.value = nn.Sequential(
            nn.Conv2d(width, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 3),
        )
        self.scale = key_size**-0.5
        self.register_buffer(
            "pairs", torch.tensor([64 * a + b for a, b, p in ACTIONS]), persistent=False
        )
        self.register_buffer(
            "promos", torch.tensor([(p - 2) * 64 + a for a, b, p in ACTIONS if p]), persistent=False
        )
        self.register_buffer("bits", torch.arange(8, dtype=torch.uint8), persistent=False)

    def forward(self, packed):
        n = packed.shape[0]
        masks = ((packed[:, :320, None] >> self.bits) & 1).reshape(n, 40, 8, 8)
        scalars = packed[:, 320:].contiguous().view(torch.int8).reshape(n, 12, 1, 1)
        # Scale counts and clocks to a useful range for a small network.
        scalars = scalars.float()
        scalars[:, 6] /= 100
        scalars[:, :6] /= 8
        scalars[:, 7:] /= 8
        x = torch.cat((masks, scalars.expand(-1, -1, 8, 8)), 1).contiguous(
            memory_format=torch.channels_last
        )
        x = self.tower(self.stem(x))
        q, k = self.qk(x).flatten(2).chunk(2, 1)
        pairs = (q.transpose(1, 2) @ k).flatten(1).index_select(1, self.pairs) * self.scale
        offsets = self.promotion(x).flatten(1).index_select(1, self.promos)
        policy = torch.cat((pairs[:, :-88], pairs[:, -88:] + offsets), 1)
        return policy, self.value(x)


def legal_logits(policy, ids):
    # N_ACTIONS is the padding sentinel. Every example has at least one legal move.
    return (
        policy.float().gather(1, ids.clamp_max(N_ACTIONS - 1)).masked_fill(ids == N_ACTIONS, -1e9)
    )
