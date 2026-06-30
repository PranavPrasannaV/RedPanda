"""
fastchess — a Numba-JIT bitboard chess core for the MARS search hot loop.

WHY: profiling showed python-chess legal-move generation (~33 us/call) is the
dominant per-node cost that caps MARS's throughput below MCTS. This module does
the same operations (legal moves, make-move, terminal detection, zobrist) at
~C speed via Numba, so the per-node cost becomes NEURAL-bound — the regime where
MARS's cheap O(1) Search-Mamba step beats MCTS's full eval, decisively.

Design choices for CORRECTNESS (paper-grade — a single illegal move is fatal):
  - Bitboard board state in a fixed numpy uint64 array (Numba-friendly).
  - Leaper attacks (knight/king/pawn) from precomputed tables.
  - Sliding attacks (bishop/rook/queen) via Kogge-Stone fills (no magic tables,
    branch-free, Numba-fast).
  - Legality by MAKE-then-king-safety-check (simplest provably-correct method).
  - Validated by PERFT against known node counts AND move-for-move vs
    python-chess across millions of positions (see test_fastchess.py).

Everything here is pure Python + Numba — it imports back into the engine with no
build step, so the validated core rewires straight into the Mamba-3 engine.

Board array layout (np.uint64[17]):
   0..11  piece bitboards: WP WN WB WR WQ WK  BP BN BB BR BQ BK   (sq bit = a1..h8)
   12     side to move (0 = white, 1 = black)
   13     castling rights bitmask: 1=WK 2=WQ 4=BK 8=BQ
   14     en-passant target square + 1 (0 = none)
   15     halfmove clock
   16     fullmove number
Moves are packed into a uint32: from(0..5) | to(6..11) | promo(12..14) | flag(15..17)
   promo: 0 none, 1 N, 2 B, 3 R, 4 Q
   flag : 0 normal, 1 double-pawn-push, 2 en-passant, 3 castle, 4 promotion
"""

import numpy as np
from numba import njit, uint64, uint32, int64

# ── Piece indices ────────────────────────────────────────────────────────────
WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK = range(12)
SIDE, CASTLE, EP, HALF, FULL = 12, 13, 14, 15, 16
BOARD_LEN = 17

# ── Precomputed leaper attack tables (computed once at import) ───────────────
def _precompute_leapers():
    knight = np.zeros(64, dtype=np.uint64)
    king = np.zeros(64, dtype=np.uint64)
    wpawn = np.zeros(64, dtype=np.uint64)   # white pawn capture targets
    bpawn = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        r, f = sq // 8, sq % 8
        for dr, df in ((2, 1), (2, -1), (-2, 1), (-2, -1),
                       (1, 2), (1, -2), (-1, 2), (-1, -2)):
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                knight[sq] |= np.uint64(1) << np.uint64(rr * 8 + ff)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                rr, ff = r + dr, f + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    king[sq] |= np.uint64(1) << np.uint64(rr * 8 + ff)
        for df in (-1, 1):                  # white pawn captures (up the board)
            rr, ff = r + 1, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                wpawn[sq] |= np.uint64(1) << np.uint64(rr * 8 + ff)
        for df in (-1, 1):                  # black pawn captures (down)
            rr, ff = r - 1, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                bpawn[sq] |= np.uint64(1) << np.uint64(rr * 8 + ff)
    return knight, king, wpawn, bpawn

KNIGHT_ATK, KING_ATK, WPAWN_ATK, BPAWN_ATK = _precompute_leapers()

# File masks for Kogge-Stone (avoid wrap-around).
NOT_A = np.uint64(0xfefefefefefefefe)
NOT_H = np.uint64(0x7f7f7f7f7f7f7f7f)
RANK2 = np.uint64(0x000000000000ff00)
RANK7 = np.uint64(0x00ff000000000000)
FULL64 = np.uint64(0xffffffffffffffff)
U1 = np.uint64(1)


# ── Bit helpers ──────────────────────────────────────────────────────────────
@njit(uint64(uint64), cache=True)
def _lsb(b):
    return b & ((~b) + U1)               # b & (-b): isolate least significant set bit

@njit(int64(uint64), cache=True)
def _bsf(b):
    # index of least significant set bit (b != 0). De Bruijn-free: count trailing.
    n = 0
    while (b & U1) == 0:
        b >>= U1
        n += 1
    return n

@njit(int64(uint64), cache=True)
def _popcount(b):
    n = 0
    while b:
        b &= b - U1
        n += 1
    return n


# ── Kogge-Stone sliding fills (occluded) ─────────────────────────────────────
# Each returns the set of squares a slider on `gen` attacks given `empty`.
@njit(uint64(uint64, uint64), cache=True)
def _north(gen, empty):
    gen |= empty & (gen << np.uint64(8))
    empty &= empty << np.uint64(8)
    gen |= empty & (gen << np.uint64(16))
    empty &= empty << np.uint64(16)
    gen |= empty & (gen << np.uint64(32))
    return gen << np.uint64(8)

@njit(uint64(uint64, uint64), cache=True)
def _south(gen, empty):
    gen |= empty & (gen >> np.uint64(8))
    empty &= empty >> np.uint64(8)
    gen |= empty & (gen >> np.uint64(16))
    empty &= empty >> np.uint64(16)
    gen |= empty & (gen >> np.uint64(32))
    return gen >> np.uint64(8)

@njit(uint64(uint64, uint64), cache=True)
def _east(gen, empty):
    empty &= NOT_A
    gen |= empty & (gen << U1)
    empty &= empty << U1
    gen |= empty & (gen << np.uint64(2))
    empty &= empty << np.uint64(2)
    gen |= empty & (gen << np.uint64(4))
    return (gen << U1) & NOT_A

@njit(uint64(uint64, uint64), cache=True)
def _west(gen, empty):
    empty &= NOT_H
    gen |= empty & (gen >> U1)
    empty &= empty >> U1
    gen |= empty & (gen >> np.uint64(2))
    empty &= empty >> np.uint64(2)
    gen |= empty & (gen >> np.uint64(4))
    return (gen >> U1) & NOT_H

@njit(uint64(uint64, uint64), cache=True)
def _ne(gen, empty):
    empty &= NOT_A
    gen |= empty & (gen << np.uint64(9))
    empty &= empty << np.uint64(9)
    gen |= empty & (gen << np.uint64(18))
    empty &= empty << np.uint64(18)
    gen |= empty & (gen << np.uint64(36))
    return (gen << np.uint64(9)) & NOT_A

@njit(uint64(uint64, uint64), cache=True)
def _nw(gen, empty):
    empty &= NOT_H
    gen |= empty & (gen << np.uint64(7))
    empty &= empty << np.uint64(7)
    gen |= empty & (gen << np.uint64(14))
    empty &= empty << np.uint64(14)
    gen |= empty & (gen << np.uint64(28))
    return (gen << np.uint64(7)) & NOT_H

@njit(uint64(uint64, uint64), cache=True)
def _se(gen, empty):
    empty &= NOT_A
    gen |= empty & (gen >> np.uint64(7))
    empty &= empty >> np.uint64(7)
    gen |= empty & (gen >> np.uint64(14))
    empty &= empty >> np.uint64(14)
    gen |= empty & (gen >> np.uint64(28))
    return (gen >> np.uint64(7)) & NOT_A

@njit(uint64(uint64, uint64), cache=True)
def _sw(gen, empty):
    empty &= NOT_H
    gen |= empty & (gen >> np.uint64(9))
    empty &= empty >> np.uint64(9)
    gen |= empty & (gen >> np.uint64(18))
    empty &= empty >> np.uint64(18)
    gen |= empty & (gen >> np.uint64(36))
    return (gen >> np.uint64(9)) & NOT_H

@njit(uint64(uint64, uint64), cache=True)
def _rook_atk(sq_bb, empty):
    return _north(sq_bb, empty) | _south(sq_bb, empty) | _east(sq_bb, empty) | _west(sq_bb, empty)

@njit(uint64(uint64, uint64), cache=True)
def _bishop_atk(sq_bb, empty):
    return _ne(sq_bb, empty) | _nw(sq_bb, empty) | _se(sq_bb, empty) | _sw(sq_bb, empty)


# ── python-chess bridge (for integration + validation) ───────────────────────
def from_pychess(board):
    """Build a fastchess board array from a python-chess Board."""
    import chess
    a = np.zeros(BOARD_LEN, dtype=np.uint64)
    pmap = {(chess.PAWN, True): WP, (chess.KNIGHT, True): WN, (chess.BISHOP, True): WB,
            (chess.ROOK, True): WR, (chess.QUEEN, True): WQ, (chess.KING, True): WK,
            (chess.PAWN, False): BP, (chess.KNIGHT, False): BN, (chess.BISHOP, False): BB,
            (chess.ROOK, False): BR, (chess.QUEEN, False): BQ, (chess.KING, False): BK}
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc is not None:
            a[pmap[(pc.piece_type, pc.color)]] |= np.uint64(1) << np.uint64(sq)
    a[SIDE] = np.uint64(0 if board.turn == chess.WHITE else 1)
    c = 0
    if board.has_kingside_castling_rights(chess.WHITE):  c |= 1
    if board.has_queenside_castling_rights(chess.WHITE): c |= 2
    if board.has_kingside_castling_rights(chess.BLACK):  c |= 4
    if board.has_queenside_castling_rights(chess.BLACK): c |= 8
    a[CASTLE] = np.uint64(c)
    a[EP] = np.uint64((board.ep_square + 1) if board.ep_square is not None
                      and board.has_legal_en_passant() else 0)
    a[HALF] = np.uint64(board.halfmove_clock)
    a[FULL] = np.uint64(board.fullmove_number)
    return a


# ── Occupancy / attack detection ─────────────────────────────────────────────
@njit(cache=True)
def _occ(a):
    w = a[WP] | a[WN] | a[WB] | a[WR] | a[WQ] | a[WK]
    b = a[BP] | a[BN] | a[BB] | a[BR] | a[BQ] | a[BK]
    return w, b, w | b

@njit(cache=True)
def _attacked(a, sq, by_white, occ):
    """Is square `sq` attacked by the given side? Super-piece method."""
    sqb = U1 << np.uint64(sq)
    empty = ~occ
    if by_white:
        if BPAWN_ATK[sq] & a[WP]: return True
        if KNIGHT_ATK[sq] & a[WN]: return True
        if KING_ATK[sq] & a[WK]: return True
        if _bishop_atk(sqb, empty) & (a[WB] | a[WQ]): return True
        if _rook_atk(sqb, empty) & (a[WR] | a[WQ]): return True
    else:
        if WPAWN_ATK[sq] & a[BP]: return True
        if KNIGHT_ATK[sq] & a[BN]: return True
        if KING_ATK[sq] & a[BK]: return True
        if _bishop_atk(sqb, empty) & (a[BB] | a[BQ]): return True
        if _rook_atk(sqb, empty) & (a[BR] | a[BQ]): return True
    return False

@njit(cache=True)
def _king_sq(a, white):
    return _bsf(a[WK] if white else a[BK])

@njit(cache=True)
def _in_check(a, white):
    return _attacked(a, _king_sq(a, white), not white, _occ(a)[2])

_LIGHT = np.uint64(0x55AA55AA55AA55AA)
_DARK = np.uint64(0xAA55AA55AA55AA55)

@njit(cache=True)
def is_insufficient(a):
    """KvK, K+single-minor vs K, or same-colour-bishops-only — matches
    python-chess is_insufficient_material on the cases that arise in play."""
    if a[WP] | a[BP] | a[WR] | a[BR] | a[WQ] | a[BQ]:
        return False
    knights = _popcount(a[WN]) + _popcount(a[BN])
    bishops_bb = a[WB] | a[BB]
    minors = knights + _popcount(bishops_bb)
    if minors <= 1:
        return True
    if knights == 0:
        if (bishops_bb & _LIGHT) == 0 or (bishops_bb & _DARK) == 0:
            return True
    return False


# ── Move packing ─────────────────────────────────────────────────────────────
@njit(uint32(int64, int64, int64, int64), cache=True)
def _mk(frm, to, promo, flag):
    return uint32(frm | (to << 6) | (promo << 12) | (flag << 15))

@njit(cache=True)
def m_from(m): return int(m & np.uint32(63))
@njit(cache=True)
def m_to(m):   return int((m >> np.uint32(6)) & np.uint32(63))
@njit(cache=True)
def m_promo(m):return int((m >> np.uint32(12)) & np.uint32(7))
@njit(cache=True)
def m_flag(m): return int((m >> np.uint32(15)) & np.uint32(7))


# ── Pseudo-legal move generation ─────────────────────────────────────────────
@njit(cache=True)
def _gen_pseudo(a, out):
    n = 0
    white = a[SIDE] == 0
    w, b, occ = _occ(a)
    own = w if white else b
    enemy = b if white else w
    empty = ~occ
    if white:
        pawns = a[WP]
        one = (pawns << np.uint64(8)) & empty
        two = ((one & (RANK2 << np.uint64(8))) << np.uint64(8)) & empty
        promo_rank = np.uint64(0xff00000000000000)
        while one:
            t = _lsb(one); to = _bsf(t); frm = to - 8
            if t & promo_rank:
                out[n] = _mk(frm, to, 4, 4); n += 1
                out[n] = _mk(frm, to, 3, 4); n += 1
                out[n] = _mk(frm, to, 2, 4); n += 1
                out[n] = _mk(frm, to, 1, 4); n += 1
            else:
                out[n] = _mk(frm, to, 0, 0); n += 1
            one &= one - U1
        while two:
            t = _lsb(two); to = _bsf(t); frm = to - 16
            out[n] = _mk(frm, to, 0, 1); n += 1
            two &= two - U1
        pp = pawns
        while pp:
            t = _lsb(pp); frm = _bsf(t)
            caps = WPAWN_ATK[frm] & enemy
            while caps:
                ct = _lsb(caps); to = _bsf(ct)
                if ct & promo_rank:
                    out[n] = _mk(frm, to, 4, 4); n += 1
                    out[n] = _mk(frm, to, 3, 4); n += 1
                    out[n] = _mk(frm, to, 2, 4); n += 1
                    out[n] = _mk(frm, to, 1, 4); n += 1
                else:
                    out[n] = _mk(frm, to, 0, 0); n += 1
                caps &= caps - U1
            pp &= pp - U1
        if a[EP] != 0:
            ep = np.int64(a[EP]) - 1
            srcs = BPAWN_ATK[ep] & a[WP]
            while srcs:
                t = _lsb(srcs); frm = _bsf(t)
                out[n] = _mk(frm, ep, 0, 2); n += 1
                srcs &= srcs - U1
    else:
        pawns = a[BP]
        one = (pawns >> np.uint64(8)) & empty
        two = ((one & (RANK7 >> np.uint64(8))) >> np.uint64(8)) & empty
        promo_rank = np.uint64(0x00000000000000ff)
        while one:
            t = _lsb(one); to = _bsf(t); frm = to + 8
            if t & promo_rank:
                out[n] = _mk(frm, to, 4, 4); n += 1
                out[n] = _mk(frm, to, 3, 4); n += 1
                out[n] = _mk(frm, to, 2, 4); n += 1
                out[n] = _mk(frm, to, 1, 4); n += 1
            else:
                out[n] = _mk(frm, to, 0, 0); n += 1
            one &= one - U1
        while two:
            t = _lsb(two); to = _bsf(t); frm = to + 16
            out[n] = _mk(frm, to, 0, 1); n += 1
            two &= two - U1
        pp = pawns
        while pp:
            t = _lsb(pp); frm = _bsf(t)
            caps = BPAWN_ATK[frm] & enemy
            while caps:
                ct = _lsb(caps); to = _bsf(ct)
                if ct & promo_rank:
                    out[n] = _mk(frm, to, 4, 4); n += 1
                    out[n] = _mk(frm, to, 3, 4); n += 1
                    out[n] = _mk(frm, to, 2, 4); n += 1
                    out[n] = _mk(frm, to, 1, 4); n += 1
                else:
                    out[n] = _mk(frm, to, 0, 0); n += 1
                caps &= caps - U1
            pp &= pp - U1
        if a[EP] != 0:
            ep = np.int64(a[EP]) - 1
            srcs = WPAWN_ATK[ep] & a[BP]
            while srcs:
                t = _lsb(srcs); frm = _bsf(t)
                out[n] = _mk(frm, ep, 0, 2); n += 1
                srcs &= srcs - U1
    kn = a[WN] if white else a[BN]
    while kn:
        t = _lsb(kn); frm = _bsf(t)
        tg = KNIGHT_ATK[frm] & ~own
        while tg:
            tt = _lsb(tg); out[n] = _mk(frm, _bsf(tt), 0, 0); n += 1
            tg &= tg - U1
        kn &= kn - U1
    bq = (a[WB] | a[WQ]) if white else (a[BB] | a[BQ])
    while bq:
        t = _lsb(bq); frm = _bsf(t)
        tg = _bishop_atk(t, empty) & ~own
        while tg:
            tt = _lsb(tg); out[n] = _mk(frm, _bsf(tt), 0, 0); n += 1
            tg &= tg - U1
        bq &= bq - U1
    rq = (a[WR] | a[WQ]) if white else (a[BR] | a[BQ])
    while rq:
        t = _lsb(rq); frm = _bsf(t)
        tg = _rook_atk(t, empty) & ~own
        while tg:
            tt = _lsb(tg); out[n] = _mk(frm, _bsf(tt), 0, 0); n += 1
            tg &= tg - U1
        rq &= rq - U1
    kb = a[WK] if white else a[BK]
    frm = _bsf(kb)
    tg = KING_ATK[frm] & ~own
    while tg:
        tt = _lsb(tg); out[n] = _mk(frm, _bsf(tt), 0, 0); n += 1
        tg &= tg - U1
    cr = np.int64(a[CASTLE])
    if white and not _attacked(a, 4, False, occ):
        if (cr & 1) and (occ & np.uint64(0x60)) == 0:
            if not _attacked(a, 5, False, occ) and not _attacked(a, 6, False, occ):
                out[n] = _mk(4, 6, 0, 3); n += 1
        if (cr & 2) and (occ & np.uint64(0x0e)) == 0:
            if not _attacked(a, 3, False, occ) and not _attacked(a, 2, False, occ):
                out[n] = _mk(4, 2, 0, 3); n += 1
    if (not white) and not _attacked(a, 60, True, occ):
        if (cr & 4) and (occ & np.uint64(0x6000000000000000)) == 0:
            if not _attacked(a, 61, True, occ) and not _attacked(a, 62, True, occ):
                out[n] = _mk(60, 62, 0, 3); n += 1
        if (cr & 8) and (occ & np.uint64(0x0e00000000000000)) == 0:
            if not _attacked(a, 59, True, occ) and not _attacked(a, 58, True, occ):
                out[n] = _mk(60, 58, 0, 3); n += 1
    return n


# ── Make move (returns a new board array) ────────────────────────────────────
@njit(cache=True)
def make(a, m):
    b = a.copy()
    white = a[SIDE] == 0
    frm = m_from(m); to = m_to(m); promo = m_promo(m); flag = m_flag(m)
    fb = U1 << np.uint64(frm); tb = U1 << np.uint64(to)
    lo, hi = (0, 6) if white else (6, 12)
    elo, ehi = (6, 12) if white else (0, 6)
    pidx = lo
    for i in range(lo, hi):
        if b[i] & fb:
            pidx = i; break
    # capture flag (from ORIGINAL board) for the halfmove clock
    cap_bb = a[elo] | a[elo+1] | a[elo+2] | a[elo+3] | a[elo+4] | a[elo+5]
    is_cap = ((cap_bb & tb) != 0) or (flag == 2)
    # remove captured enemy on `to`
    for i in range(elo, ehi):
        b[i] &= ~tb
    if flag == 2:                              # en-passant: remove pawn behind
        capsq = to - 8 if white else to + 8
        b[BP if white else WP] &= ~(U1 << np.uint64(capsq))
    b[pidx] &= ~fb
    if flag == 4:                              # promotion
        if white:
            pp = WQ if promo == 4 else WR if promo == 3 else WB if promo == 2 else WN
        else:
            pp = BQ if promo == 4 else BR if promo == 3 else BB if promo == 2 else BN
        b[pp] |= tb
    else:
        b[pidx] |= tb
    if flag == 3:                              # castle: move the rook
        if to == 6:    b[WR] &= ~(U1 << np.uint64(7));  b[WR] |= (U1 << np.uint64(5))
        elif to == 2:  b[WR] &= ~(U1 << np.uint64(0));  b[WR] |= (U1 << np.uint64(3))
        elif to == 62: b[BR] &= ~(U1 << np.uint64(63)); b[BR] |= (U1 << np.uint64(61))
        elif to == 58: b[BR] &= ~(U1 << np.uint64(56)); b[BR] |= (U1 << np.uint64(59))
    cr = np.int64(a[CASTLE])
    if pidx == WK: cr &= ~3
    if pidx == BK: cr &= ~12
    if frm == 0 or to == 0:   cr &= ~2
    if frm == 7 or to == 7:   cr &= ~1
    if frm == 56 or to == 56: cr &= ~8
    if frm == 63 or to == 63: cr &= ~4
    b[CASTLE] = np.uint64(cr)
    if flag == 1:
        b[EP] = np.uint64((to - 8 if white else to + 8) + 1)
    else:
        b[EP] = np.uint64(0)
    is_pawn = (pidx == WP or pidx == BP)
    b[HALF] = np.uint64(0) if (is_pawn or is_cap) else a[HALF] + U1
    if not white:
        b[FULL] = a[FULL] + U1
    b[SIDE] = np.uint64(1) if white else np.uint64(0)
    return b


# ── Legal move generation ────────────────────────────────────────────────────
@njit(cache=True)
def gen_legal(a, out):
    pseudo = np.empty(256, dtype=np.uint32)
    npc = _gen_pseudo(a, pseudo)
    white = a[SIDE] == 0
    n = 0
    for i in range(npc):
        bb = make(a, pseudo[i])
        if not _in_check(bb, white):
            out[n] = pseudo[i]; n += 1
    return n


# ── Move <-> token bridge (fastchess move  ->  encoder action id) ────────────
# Precomputed once: key = from | to<<6 | promo<<12  ->  encoder token id (or -1).
# O(1) array lookup in the hot loop, no per-call uci() string build.
_MOVE_TOKEN = None

def build_move_token_map(encoder):
    global _MOVE_TOKEN
    import chess
    arr = np.full(1 << 15, -1, dtype=np.int32)
    pt = {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}
    for frm in range(64):
        for to in range(64):
            for promo in range(5):
                mv = chess.Move(frm, to) if promo == 0 else chess.Move(frm, to, promotion=pt[promo])
                tok = encoder.encode_move(mv)
                arr[frm | (to << 6) | (promo << 12)] = -1 if tok is None else tok
    _MOVE_TOKEN = arr
    return arr

def move_token(m):
    """encoder action id for a fastchess move (flag bits masked off)."""
    return int(_MOVE_TOKEN[int(m) & 0x7fff])


# ── Zobrist hashing (Numba; for repetition detection / TT) ───────────────────
_zr = np.random.default_rng(0x5eed)
_TOPBIT = np.uint64(1) << np.uint64(63)
ZOB_PIECE = _zr.integers(0, 1 << 63, size=(12, 64), dtype=np.uint64) | _TOPBIT
ZOB_SIDE = np.uint64(_zr.integers(0, 1 << 63, dtype=np.uint64)) | _TOPBIT
ZOB_CASTLE = _zr.integers(0, 1 << 63, size=16, dtype=np.uint64) | _TOPBIT
ZOB_EP = _zr.integers(0, 1 << 63, size=8, dtype=np.uint64) | _TOPBIT

@njit(uint64(uint64[:]), cache=True)
def zobrist(a):
    h = np.uint64(0)
    for p in range(12):
        bb = a[p]
        while bb:
            sq = _bsf(bb)
            h ^= ZOB_PIECE[p, sq]
            bb &= bb - U1
    if a[SIDE] == 1:
        h ^= ZOB_SIDE
    h ^= ZOB_CASTLE[int(a[CASTLE]) & 15]
    if a[EP] != 0:
        h ^= ZOB_EP[(int(a[EP]) - 1) & 7]
    return h


# ── Board encoding from a fastchess board (matches encoder.encode_board) ─────
_PIECE_SYM = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
_MAT_VAL = [100, 320, 330, 500, 900, 0]          # P N B R Q K (by piece % 6)
_FILES = "abcdefgh"
# minor home squares (color, is_knight) for phase 'developed' count
_WN_HOME, _WB_HOME = {1, 6}, {2, 5}
_BN_HOME, _BB_HOME = {57, 62}, {58, 61}

def fc_encode_board(a, history_tokens, rep_count, encoder):
    """Token sequence identical to encoder.encode_board, built from a fastchess
    board `a`, the move-token history, and a precomputed repetition count
    (0/1/2). Called only at anchors/leaves, so a Python implementation is fine —
    the per-ply hot loop stays in Numba."""
    mid = encoder.move_to_id
    pid = encoder.piece_to_id
    enc = [mid["[START]"]]
    h = [t for t in history_tokens if t >= 0]
    if len(h) > 32:
        h = h[-32:]
    enc.extend(h)
    enc.append(mid["[BOARD_START]"])
    # 64 piece tokens
    sym = ['.'] * 64
    for p in range(12):
        bb = int(a[p])
        while bb:
            sq = (bb & -bb).bit_length() - 1
            sym[sq] = _PIECE_SYM[p]
            bb &= bb - 1
    for sq in range(64):
        enc.append(pid[sym[sq]])
    enc.append(mid["W_TURN"] if a[SIDE] == 0 else mid["B_TURN"])
    cr = int(a[CASTLE])
    if cr & 1: enc.append(mid["W_OO"])
    if cr & 2: enc.append(mid["W_OOO"])
    if cr & 4: enc.append(mid["B_OO"])
    if cr & 8: enc.append(mid["B_OOO"])
    # en passant (legal iff a pawn attacks the ep square — matches has_legal_en_passant)
    ep_tok = mid["NO_EP"]
    if a[EP] != 0:
        ep = int(a[EP]) - 1
        legal_ep = (BPAWN_ATK[ep] & a[WP]) if a[SIDE] == 0 else (WPAWN_ATK[ep] & a[BP])
        if legal_ep:
            key = "EP_" + _FILES[ep % 8] + str(ep // 8 + 1)
            ep_tok = mid.get(key, mid["NO_EP"])
    enc.append(ep_tok)
    enc.append(mid["HMC_%d" % min(int(a[HALF]) // 10, 5)])
    enc.append(mid["REP_%d" % rep_count])
    # material balance
    bal = 0
    for p in range(12):
        c = _popcount(a[p])
        bal += (_MAT_VAL[p % 6] * c) if p < 6 else (-_MAT_VAL[p % 6] * c)
    enc.append(mid["MAT_%d" % max(-5, min(5, bal // 100))])
    # phase
    pieces = 0
    for p in range(12):
        pieces += _popcount(a[p])
    mm = (_popcount(a[WN]) + _popcount(a[WB]) + _popcount(a[WR]) + _popcount(a[WQ])
          + _popcount(a[BN]) + _popcount(a[BB]) + _popcount(a[BR]) + _popcount(a[BQ]))
    if pieces <= 12 or mm <= 4:
        enc.append(mid["[ENDGAME]"])
    else:
        dev = 0
        for sq in _bits(int(a[WN])):
            if sq not in _WN_HOME: dev += 1
        for sq in _bits(int(a[WB])):
            if sq not in _WB_HOME: dev += 1
        for sq in _bits(int(a[BN])):
            if sq not in _BN_HOME: dev += 1
        for sq in _bits(int(a[BB])):
            if sq not in _BB_HOME: dev += 1
        enc.append(mid["[OPENING]"] if dev <= 2 else mid["[MIDDLEGAME]"])
    return enc

def _bits(bb):
    while bb:
        yield (bb & -bb).bit_length() - 1
        bb &= bb - 1


# ── Perft (correctness oracle) ───────────────────────────────────────────────
@njit(cache=True)
def perft(a, depth):
    if depth == 0:
        return 1
    out = np.empty(256, dtype=np.uint32)
    n = gen_legal(a, out)
    if depth == 1:
        return n
    tot = 0
    for i in range(n):
        tot += perft(make(a, out[i]), depth - 1)
    return tot
