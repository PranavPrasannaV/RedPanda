
"""
Tournament / Elo testing harness (Phase 6.3).

Plays ChessMamba (MCTS-primary MoveSelector) against an opponent and reports the
score + an Elo-difference estimate. Opponents:
    - stockfish  - a UCI Stockfish binary capped by depth / movetime / Skill Level
    - self       - another ChessMamba checkpoint (regression vs a previous net)
    - random     - sanity baseline

It uses python-chess directly (no CuteChess dependency) so it runs anywhere,
including inside the Kaggle notebook for periodic strength checks.

Usage:
    python tournament.py --model chess_mamba.pt --opponent stockfish \
        --stockfish /usr/games/stockfish --sf-depth 6 --games 20 --sims 400
"""

import os
import math
import argparse
import random
import chess
import chess.pgn
import torch

from self_play import load_model
from move_selector import MoveSelector
from search_mamba import SearchMamba
from geometric import GeometricNavigator, load_advantage_vectors


def elo_diff(score, n):
    """Elo difference from a fractional score (0..1) over n games."""
    s = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)


def _san_line(moves, opening_len=0):
    """SAN movetext for a move list; marks where the random opening ends with '|'."""
    b = chess.Board(); out = []
    for i, m in enumerate(moves):
        if b.turn == chess.WHITE:
            out.append(f"{b.fullmove_number}.")
        out.append(b.san(m)); b.push(m)
        if i + 1 == opening_len and opening_len:
            out.append("|")          # everything before '|' is the forced opening
    return " ".join(out)


def _to_pgn(moves, white_name, black_name, result_str, opening_len=0):
    g = chess.pgn.Game()
    g.headers["White"] = white_name
    g.headers["Black"] = black_name
    g.headers["Result"] = result_str
    if opening_len:
        g.headers["FEN_after_opening"] = "(first %d plies = forced random opening)" % opening_len
    node = g
    for m in moves:
        node = node.add_variation(m)
    return str(g)


def random_opening(plies, rng):
    """A short random legal opening (move list). Played twice with colors
    swapped so neither side gets the opening's bias — gives game diversity so
    the match score is statistically meaningful (B8), not one game repeated."""
    b = chess.Board()
    moves = []
    for _ in range(plies):
        lm = list(b.legal_moves)
        if not lm:
            break
        m = lm[rng.randrange(len(lm))]
        b.push(m); moves.append(m)
        if b.is_game_over():
            break
    return moves


def build_selector(model_path, device, sims, search_path=None, adv_path=None,
                   backend="mcts"):
    model, _ = load_model(model_path, device=device)
    search_model = None
    if search_path and os.path.exists(search_path):
        search_model = SearchMamba.from_pretrained(search_path, device=device)
        search_model.to(device).eval()
    navigator = None
    if adv_path and os.path.exists(adv_path):
        navigator = GeometricNavigator(load_advantage_vectors(adv_path, device)).to(device)
    return MoveSelector(model, search_model, navigator, backend=backend,
                        num_simulations=sims, adaptive_sims=False, add_root_noise=False,
                        mars_kwargs={"sim_budget": sims})


class StockfishOpponent:
    def __init__(self, path, depth=6, movetime=None, skill=None, uci_elo=None):
        import chess.engine
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        # uci_elo: Stockfish's OWN calibrated limited-strength mode — the proper
        # absolute anchor (~1320-3190). When set, it overrides depth/skill so the
        # implied Mamba Elo = uci_elo + elo_diff(score) is a real number.
        if uci_elo is not None:
            try:
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(uci_elo)})
                self.limit = chess.engine.Limit(movetime=0.1)   # let it use its strength cap
                return
            except Exception:
                pass
        if skill is not None:
            try:
                self.engine.configure({"Skill Level": skill})
            except Exception:
                pass
        self.limit = chess.engine.Limit(
            depth=depth if movetime is None else None,
            time=(movetime / 1000.0) if movetime else None)

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


def play_match(selector, opponent, games=20, opening_plies=4, max_plies=400, seed=0,
               print_moves=True, pgn_path=None):
    """Play `games` games with DIVERSE openings, each opening played twice with
    colors swapped (so opening/color bias cancels). Returns (score_fraction,
    W/D/L). Mamba plays at temperature 0 (deterministic, match-strength).
    print_moves -> SAN movetext per game; pgn_path -> append loadable PGN."""
    score = 0.0
    results = {"win": 0, "draw": 0, "loss": 0}
    rng = random.Random(seed)
    n_pairs = (games + 1) // 2
    g = 0
    pgn_f = open(pgn_path, "w") if pgn_path else None
    for p in range(n_pairs):
        opening = random_opening(opening_plies, rng)
        for mamba_is_white in (True, False):     # same opening, both colors
            if g >= games:
                break
            board = chess.Board()
            for m in opening:
                board.push(m)
            while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
                mamba_turn = (board.turn == chess.WHITE) == mamba_is_white
                mv = selector.select_move(board) if mamba_turn else opponent.move(board)
                if mv is None or mv not in board.legal_moves:
                    mv = rng.choice(list(board.legal_moves))
                board.push(mv)
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                score += 0.5; results["draw"] += 1; res_str = "1/2-1/2"
            elif (outcome.winner == chess.WHITE) == mamba_is_white:
                score += 1.0; results["win"] += 1
                res_str = "1-0" if mamba_is_white else "0-1"
            else:
                results["loss"] += 1
                res_str = "0-1" if mamba_is_white else "1-0"
            g += 1
            moves = list(board.move_stack)
            # outcome from Mamba's POV for the human-readable tag
            tag = ("WIN " if (outcome and outcome.winner is not None and
                              (outcome.winner == chess.WHITE) == mamba_is_white)
                   else "DRAW" if (outcome is None or outcome.winner is None) else "LOSS")
            term = (outcome.termination.name if outcome else "MAXPLIES")
            print(f"  game {g}/{games} | mamba={'W' if mamba_is_white else 'B'} | "
                  f"{tag} ({term}) | score {score:.1f} | "
                  f"W/D/L {results['win']}/{results['draw']}/{results['loss']}")
            if print_moves:
                print("      " + _san_line(moves, len(opening)))
            if pgn_f:
                wn = "Mamba" if mamba_is_white else "Stockfish"
                bn = "Stockfish" if mamba_is_white else "Mamba"
                pgn_f.write(_to_pgn(moves, wn, bn, res_str, len(opening)) + "\n\n")
                pgn_f.flush()
    if pgn_f:
        pgn_f.close()
        print(f"\n  [pgn] all games written to {pgn_path}")
    frac = score / games
    print(f"\nScore {score}/{games} ({frac:.1%}) | Elo diff vs opponent ~ "
          f"{elo_diff(frac, games):+.0f}")
    return frac, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chess_mamba.pt")
    ap.add_argument("--search", default="search_mamba.pt")
    ap.add_argument("--adv", default="advantage_vectors.pt")
    ap.add_argument("--opponent", choices=["stockfish", "self", "random"], default="stockfish")
    ap.add_argument("--opponent-model", default=None, help="for --opponent self")
    ap.add_argument("--backend", choices=["mars", "mcts"], default="mcts")
    ap.add_argument("--opponent-backend", choices=["mars", "mcts"], default="mcts",
                    help="for --opponent self (A/B: MARS vs MCTS, same net)")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--sf-depth", type=int, default=6)
    ap.add_argument("--sf-movetime", type=int, default=None)
    ap.add_argument("--sf-skill", type=int, default=None)
    ap.add_argument("--opp-elo", type=int, default=None,
                    help="anchor: Stockfish UCI_LimitStrength at this Elo (~1320-3190). "
                         "Implied Mamba Elo = opp-elo + Elo-diff. The clean way to get "
                         "an absolute number; overrides --sf-depth/--sf-skill.")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="random opening length for game diversity (each played both colors)")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--no-moves", action="store_true",
                    help="suppress the per-game SAN movetext (printed by default)")
    ap.add_argument("--save-pgn", default="tournament_games.pgn",
                    help="write all games to this PGN file (load into Lichess to analyze)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    selector = build_selector(args.model, device, args.sims, args.search, args.adv,
                              backend=args.backend)

    if args.opponent == "stockfish":
        opp = StockfishOpponent(args.stockfish, args.sf_depth, args.sf_movetime,
                                args.sf_skill, uci_elo=args.opp_elo)
    elif args.opponent == "self":
        opp_sel = build_selector(args.opponent_model or args.model, device, args.sims,
                                 args.search, args.adv, backend=args.opponent_backend)
        opp = type("S", (), {"move": lambda self, b: opp_sel.select_move(b),
                             "close": lambda self: None})()
    else:
        opp = type("R", (), {"move": lambda self, b: random.choice(list(b.legal_moves)),
                             "close": lambda self: None})()

    try:
        frac, _ = play_match(selector, opp, games=args.games,
                             opening_plies=args.opening_plies,
                             print_moves=not args.no_moves, pgn_path=args.save_pgn)
        if args.opponent == "stockfish" and args.opp_elo is not None:
            est = args.opp_elo + elo_diff(frac, args.games)
            print(f"\n>>> Estimated Mamba Elo ~ {est:.0f}  "
                  f"(anchor: Stockfish UCI_Elo {args.opp_elo}, {args.games} games)")
            print("    (rough first read — widen with more games / multiple anchors)")
    finally:
        opp.close()


if __name__ == "__main__":
    main()
