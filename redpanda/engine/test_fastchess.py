"""Perft validation for fastchess — the gold-standard move-generator correctness
test. Compares fastchess.perft against KNOWN node counts for the standard test
positions (which exercise en passant, castling, promotions, pins, checks), and
cross-checks the legal move SET against python-chess on random positions."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
import fastchess as fc

# (fen, [perft(1..n)]) — canonical values from the Chess Programming Wiki.
CASES = [
    (chess.STARTING_FEN, [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [48, 2039, 97862]),                                              # Kiwipete
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [44, 1486, 62379]),
]

print("=== PERFT vs known node counts ===")
allok = True
for fen, expect in CASES:
    a = fc.from_pychess(chess.Board(fen))
    for d, exp in enumerate(expect, start=1):
        got = fc.perft(a, d)
        ok = got == exp
        allok &= ok
        print(f"  perft({d}) {got:>9} exp {exp:>9} {'OK' if ok else 'FAIL <<<'}  "
              f"{fen.split()[0][:24]}")
        if not ok:
            break

# Cross-check the legal-move SET vs python-chess on random walks.
print("\n=== legal-move set vs python-chess (random positions) ===")
def fc_moves_uci(a):
    out = __import__('numpy').empty(256, dtype='uint32')
    n = fc.gen_legal(a, out)
    res = set()
    for i in range(n):
        m = out[i]
        frm, to, promo = fc.m_from(m), fc.m_to(m), fc.m_promo(m)
        uci = chess.square_name(frm) + chess.square_name(to)
        if promo:
            uci += "nbrq"[promo - 1]
        res.add(uci)
    return res

random.seed(0)
mismatch = 0; checked = 0
for g in range(3000):
    b = chess.Board()
    for _ in range(random.randint(0, 40)):
        lm = list(b.legal_moves)
        if not lm:
            break
        b.push(random.choice(lm))
        py = set(m.uci() for m in b.legal_moves)
        fcm = fc_moves_uci(fc.from_pychess(b))
        checked += 1
        if py != fcm:
            mismatch += 1
            if mismatch <= 3:
                print(f"  MISMATCH {b.fen()}")
                print(f"    py - fc : {py - fcm}")
                print(f"    fc - py : {fcm - py}")
        if b.is_game_over():
            break
print(f"  checked {checked} positions, {mismatch} mismatches")
print("\n" + ("ALL FASTCHESS CHECKS PASSED" if allok and mismatch == 0
              else "FAILURES PRESENT <<<"))
sys.exit(0 if allok and mismatch == 0 else 1)
