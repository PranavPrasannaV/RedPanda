
"""
Monte-Carlo Graph Search support (ChessMamba v4).

Chess is transposition-heavy: the same position is reached by many move orders.
A tree re-evaluates each occurrence; a graph shares one evaluation. This module
provides a lightweight transposition table keyed by Zobrist hash that MARS uses to
**share position value estimates across the whole search**, so a position seen on
one rollout line is not re-evaluated on another.

A position's value is path-independent (it's a property of the board), so caching
it by board key is sound. The recurrent SSM rollout *state* is path-dependent and
is therefore NOT stored here — it is carried along the live rollout and corrected
by MARS's periodic re-anchoring.
"""

import chess
import chess.polyglot


def zobrist_key(board: chess.Board) -> int:
    """Transposition key (incremental Zobrist hash from python-chess)."""
    return chess.polyglot.zobrist_hash(board)


class TranspositionTable:
    """key -> [visits, value_sum] in that position's side-to-move perspective."""

    def __init__(self):
        self.t = {}

    def get_value(self, key):
        e = self.t.get(key)
        if e is None or e[0] == 0:
            return None
        return e[1] / e[0]

    def visits(self, key) -> int:
        e = self.t.get(key)
        return e[0] if e else 0

    def update(self, key, value: float):
        e = self.t.get(key)
        if e is None:
            self.t[key] = [1, float(value)]
        else:
            e[0] += 1
            e[1] += float(value)

    def clear(self):
        self.t.clear()

    def __len__(self):
        return len(self.t)
