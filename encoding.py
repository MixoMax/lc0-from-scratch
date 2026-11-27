# file: encoding.py
"""The paper's side-to-move 52 planes, packed as 40 masks and 12 integers."""

import struct

import chess
import numpy as np

PACK = struct.Struct("<40Q12b")
STATE_BYTES = PACK.size
MAX_LEGAL = 256
FULL = (1 << 64) - 1
CHECKERBOARD = chess.BB_DARK_SQUARES


def action_table():
    pairs = []
    for a in chess.SQUARES:
        for b in chess.SQUARES:
            dx = abs((a % 8) - (b % 8))
            dy = abs((a // 8) - (b // 8))
            if a != b and (dx == 0 or dy == 0 or dx == dy or dx * dy == 2):
                pairs.append((a, b, 0))
    for a in range(48, 56):
        for b in range(56, 64):
            if abs(a % 8 - b % 8) <= 1:
                pairs.extend((a, b, p) for p in (5, 4, 3, 2))
    assert len(pairs) == 1880
    return pairs


ACTIONS = action_table()
INDEX = {key: i for i, key in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)


def move_ids(board, moves):
    flip = 0 if board.turn else 56
    king = board.king(board.turn)

    def index(m):
        a, b = m.from_square, m.to_square
        if a == king and abs(b - a) == 2:
            b = a + 3 if b > a else a - 4
        return INDEX[a ^ flip, b ^ flip, m.promotion or 0]

    return np.fromiter(map(index, moves), dtype=np.uint16, count=len(moves))


def encode(board, occurrences):
    own = board.turn
    flip = 0 if own else 56
    masks = [board.pieces_mask(p, c) for c in (own, not own) for p in chess.PIECE_TYPES]
    counts = [v.bit_count() for v in masks]
    flags = (
        board.has_kingside_castling_rights(own),
        board.has_queenside_castling_rights(own),
        board.has_kingside_castling_rights(not own),
        board.has_queenside_castling_rights(not own),
    )
    masks.extend(FULL * f for f in flags)
    masks.extend(
        (
            board.occupied_co[own],
            board.occupied_co[not own],
            board.checkers_mask(),
            0 if board.ep_square is None else 1 << board.ep_square,
        )
    )
    if flip:
        masks = [int.from_bytes(v.to_bytes(8, "little"), "big") for v in masks]
    masks.extend((FULL * (occurrences >= 2), FULL * (occurrences >= 3)))
    for m in reversed(board.move_stack[-8:]):
        masks.extend((1 << (m.from_square ^ flip), 1 << (m.to_square ^ flip)))
    masks.extend([0] * (38 - len(masks)))
    masks.extend(
        (
            CHECKERBOARD,
            FULL
            * (counts[2] == counts[8] == 1 and (board.bishops & CHECKERBOARD).bit_count() == 1),
        )
    )
    balance = [a - b for a, b in zip(counts[:6], counts[6:])]
    return PACK.pack(*masks, *balance, min(100, board.halfmove_clock), *counts[:5])
