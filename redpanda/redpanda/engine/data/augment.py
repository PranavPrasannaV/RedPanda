
"""
Board-symmetry data augmentation (Phase 1.1).

Two label-preserving symmetries multiply the dataset up to 4×:

  1. Horizontal mirror (files a↔h).  Side-to-move and evaluation are unchanged.
     Skipped when either side still has castling rights (a file-flip turns
     O-O into O-O-O and breaks the castling encoding).
  2. Colour flip (python-chess board.mirror(): vertical flip + swap colours).
     Always valid. Because our targets are stored **side-to-move relative**,
     the WDL / centipawn / value labels are IDENTICAL after the flip — only the
     moves are transformed.

All move objects (best move + PV) are transformed with the same square map so
the policy / PV targets stay correct.
"""

import chess
from typing import List, Optional, Tuple


def _transform_square(sq: int, flip_h: bool, flip_v: bool) -> int:
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    if flip_h:
        f = 7 - f
    if flip_v:
        r = 7 - r
    return chess.square(f, r)


def transform_move(move: chess.Move, flip_h: bool, flip_v: bool) -> chess.Move:
    return chess.Move(
        _transform_square(move.from_square, flip_h, flip_v),
        _transform_square(move.to_square, flip_h, flip_v),
        promotion=move.promotion,
    )


def _has_castling(board: chess.Board) -> bool:
    return bool(board.castling_rights)


def mirror_horizontal(board: chess.Board) -> Optional[chess.Board]:
    """File-flip; None if castling rights make it invalid."""
    if _has_castling(board):
        return None
    return board.transform(chess.flip_horizontal)


def flip_colors(board: chess.Board) -> chess.Board:
    """Vertical flip + colour swap (side-to-move-relative eval unchanged)."""
    return board.mirror()


def _moves_to_ucis(moves: List[chess.Move]) -> List[str]:
    return [m.uci() for m in moves]


def augment_position(
    board: chess.Board,
    best_move_uci: str,
    pv_ucis: List[str],
    include_hflip: bool = True,
    include_color: bool = True,
) -> List[Tuple[chess.Board, str, List[str]]]:
    """
    Return [(board, best_move_uci, pv_ucis), ...] — the original plus every
    requested symmetry. Labels (WDL/cp) are unchanged for all variants, so the
    caller reuses them verbatim.
    """
    variants = [(board, best_move_uci, list(pv_ucis))]

    def _apply(flip_h: bool, flip_v: bool, new_board: chess.Board):
        try:
            bm = transform_move(chess.Move.from_uci(best_move_uci), flip_h, flip_v).uci()
            pv = [transform_move(chess.Move.from_uci(u), flip_h, flip_v).uci()
                  for u in pv_ucis]
        except Exception:
            return
        variants.append((new_board, bm, pv))

    hflip_board = mirror_horizontal(board) if include_hflip else None
    if hflip_board is not None:
        _apply(True, False, hflip_board)

    if include_color:
        color_board = flip_colors(board)
        _apply(False, True, color_board)
        if hflip_board is not None:
            # h-flip + colour flip
            both = flip_colors(hflip_board)
            _apply(True, True, both)

    return variants
