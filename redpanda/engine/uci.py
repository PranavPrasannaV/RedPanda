
"""
UCI (Universal Chess Interface) protocol implementation for ChessMamba.

Enables ChessMamba to communicate with:
    - CuteChess (tournament testing)
    - Lichess bot accounts
    - Arena / Fritz / any UCI-compatible GUI
    - Command-line play

Usage:
    python uci.py

    Then type UCI commands:
        uci
        isready
        position startpos moves e2e4 e7e5
        go movetime 5000
        quit
"""

import sys
import os
import time
import chess
import torch

# Add engine dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoding import encoder, ACTION_SPACE
from mamba import MambaConfig
from model import ChessMamba
from search_mamba import SearchMamba, SearchMambaConfig
from geometric import GeometricNavigator, load_advantage_vectors
from move_selector import MoveSelector


ENGINE_NAME = "ChessMamba v3"
ENGINE_AUTHOR = "ChessMamba Team"


class UCIEngine:
    """
    Full UCI protocol implementation.
    
    Handles all standard UCI commands and routes them to the
    ChessMamba move selection pipeline.
    """
    
    def __init__(self):
        self.board = chess.Board()
        self.eval_model = None
        self.search_model = None
        self.navigator = None
        self.selector = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # UCI options
        self.options = {
            "Hash": 256,
            "Threads": 1,
            "Simulations": 800,     # MCTS simulations per move
            "Contempt": 0.0,        # draw aversion [0..1]
            "AdaptiveSims": True,   # let the uncertainty head set the sim budget
        }
    
    def load_models(self):
        """Load all model components."""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load Evaluation Mamba
        eval_path = os.path.join(model_dir, "chess_mamba.pt")
        config_path = os.path.join(model_dir, "chess_mamba_config.json")
        
        if os.path.exists(eval_path) and os.path.exists(config_path):
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            config = MambaConfig(**{
                k: v for k, v in config_dict.items()
                if k in MambaConfig.__dataclass_fields__
            })
            self.eval_model = ChessMamba(config, action_space=ACTION_SPACE)
            state_dict = torch.load(eval_path, map_location=self.device, weights_only=True)  # nosec
            self.eval_model.load_state_dict(state_dict)
            self.eval_model.to(self.device)
            self.eval_model.eval()
            self._log(f"Eval model loaded from {eval_path}")
        else:
            # Random model for testing
            config = MambaConfig(d_model=128, n_layer=4, vocab_size=encoder.vocab_size() + 100)
            self.eval_model = ChessMamba(config, action_space=ACTION_SPACE).to(self.device)
            self.eval_model.eval()
            self._log("No trained model found, using random initialization")
        
        # Load Search Mamba (optional)
        search_path = os.path.join(model_dir, "search_mamba.pt")
        if os.path.exists(search_path):
            self.search_model = SearchMamba.from_pretrained(
                search_path, device=self.device
            )
            self.search_model.to(self.device)
            self.search_model.eval()
            self._log("Search Mamba loaded")
        else:
            self.search_model = None
            self._log("No Search Mamba found, using policy-only mode")
        
        # Load advantage vectors (optional)
        adv_path = os.path.join(model_dir, "advantage_vectors.pt")
        if os.path.exists(adv_path):
            vectors = load_advantage_vectors(adv_path, device=self.device)
            self.navigator = GeometricNavigator(vectors).to(self.device)
            self._log("Geometric navigator loaded")
        else:
            self.navigator = GeometricNavigator()
            self._log("No advantage vectors found, geometric scoring disabled")
        
        # Build move selector: MARS when a Search Mamba is present, else MCTS.
        backend = "mars" if self.search_model is not None else "mcts"
        self.selector = MoveSelector(
            eval_model=self.eval_model,
            search_model=self.search_model,
            navigator=self.navigator,
            backend=backend,
            num_simulations=self.options["Simulations"],
            contempt=self.options["Contempt"],
            adaptive_sims=self.options["AdaptiveSims"],
            mars_kwargs={"sim_budget": self.options["Simulations"]},
        )
        self._log(f"search backend: {backend}")
    
    def run(self):
        """Main UCI loop - reads stdin, writes stdout."""
        self._log(f"{ENGINE_NAME} starting...")
        
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            
            if not line:
                continue
            
            self._log(f">> {line}")
            
            tokens = line.split()
            command = tokens[0]
            
            if command == "uci":
                self._cmd_uci()
            elif command == "isready":
                self._cmd_isready()
            elif command == "ucinewgame":
                self._cmd_ucinewgame()
            elif command == "position":
                self._cmd_position(tokens)
            elif command == "go":
                self._cmd_go(tokens)
            elif command == "setoption":
                self._cmd_setoption(tokens)
            elif command == "quit":
                break
            elif command == "d":
                # Debug: print board
                print(self.board)
                print(f"FEN: {self.board.fen()}")
    
    # ── UCI Command Handlers ──
    
    def _cmd_uci(self):
        print(f"id name {ENGINE_NAME}")
        print(f"id author {ENGINE_AUTHOR}")
        print()
        # Expose tunable options
        print(f"option name Hash type spin default 256 min 1 max 4096")
        print(f"option name Simulations type spin default 800 min 16 max 20000")
        print(f"option name Contempt type string default 0.0")
        print(f"option name AdaptiveSims type check default true")
        print("uciok")
        sys.stdout.flush()
    
    def _cmd_isready(self):
        if self.eval_model is None:
            self.load_models()
        print("readyok")
        sys.stdout.flush()
    
    def _cmd_ucinewgame(self):
        self.board = chess.Board()
    
    def _cmd_position(self, tokens):
        """Parse position command: 'position startpos [moves ...]' or 'position fen ... [moves ...]'"""
        idx = 1
        
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx = 2
        elif tokens[idx] == "fen":
            # Collect FEN string (6 parts)
            fen_parts = tokens[2:8]
            fen = " ".join(fen_parts)
            self.board = chess.Board(fen)
            idx = 8
        
        # Apply moves if present
        if idx < len(tokens) and tokens[idx] == "moves":
            for move_uci in tokens[idx + 1:]:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        # Try to find legal move matching UCI
                        for legal in self.board.legal_moves:
                            if legal.uci() == move_uci:
                                self.board.push(legal)
                                break
                except Exception as e:
                    self._log(f"Invalid move {move_uci}: {e}")
    
    def _cmd_go(self, tokens):
        """Handle go command with time management."""
        # Parse time control
        movetime = None
        wtime = None
        btime = None
        winc = 0
        binc = 0
        depth = None
        
        i = 1
        while i < len(tokens):
            if tokens[i] == "movetime" and i + 1 < len(tokens):
                movetime = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "wtime" and i + 1 < len(tokens):
                wtime = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "btime" and i + 1 < len(tokens):
                btime = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "winc" and i + 1 < len(tokens):
                winc = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "binc" and i + 1 < len(tokens):
                binc = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "depth" and i + 1 < len(tokens):
                depth = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "infinite":
                movetime = 30000
                i += 1
            else:
                i += 1
        
        # Simple time management
        if movetime is None:
            my_time = wtime if self.board.turn == chess.WHITE else btime
            my_inc = winc if self.board.turn == chess.WHITE else binc
            
            if my_time is not None:
                # Use ~5% of remaining time + increment
                movetime = max(100, my_time // 20 + (my_inc or 0) // 2)
            else:
                movetime = 5000  # Default: 5 seconds
        
        # Search for best move
        start_time = time.time()
        
        if self.selector is not None:
            best_move = self.selector.select_move(self.board)
        else:
            # Fallback: random legal move
            import random
            legal = list(self.board.legal_moves)
            best_move = random.choice(legal) if legal else None
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        if best_move:
            # Send info string (optional, for GUI display)
            if self.eval_model is not None:
                with torch.no_grad():
                    enc = encoder.encode_board(self.board)
                    x = torch.tensor([enc], dtype=torch.long, device=self.device)
                    outputs = self.eval_model.forward_full(x)
                    wdl = outputs["wdl"][0]
                    # Convert WDL to centipawn estimate
                    value = (wdl[0] - wdl[2]).item()
                    cp = int(value * 300)  # Rough conversion
                    
                    print(f"info depth {self.options['Simulations']} "
                          f"score cp {cp} "
                          f"time {elapsed_ms} "
                          f"pv {best_move.uci()}")
            
            print(f"bestmove {best_move.uci()}")
        else:
            print("bestmove 0000")  # No legal moves
        
        sys.stdout.flush()
    
    def _cmd_setoption(self, tokens):
        """Parse: setoption name <name> value <value>"""
        try:
            name_idx = tokens.index("name") + 1
            value_idx = tokens.index("value") + 1
            
            name = tokens[name_idx]
            value = tokens[value_idx]
            
            if name in self.options:
                # Type-appropriate conversion (bool before int - bool is an int).
                current = self.options[name]
                if isinstance(current, bool):
                    self.options[name] = str(value).lower() in ("true", "1", "yes")
                elif isinstance(current, int):
                    self.options[name] = int(value)
                elif isinstance(current, float):
                    self.options[name] = float(value)
                else:
                    self.options[name] = value

                self._log(f"Set {name} = {self.options[name]}")
        except (ValueError, IndexError):
            pass
    
    def _log(self, msg):
        """Log to stderr (not captured by UCI protocol on stdout)."""
        print(f"info string {msg}", file=sys.stderr)
        sys.stderr.flush()


if __name__ == "__main__":
    engine = UCIEngine()
    engine.run()
