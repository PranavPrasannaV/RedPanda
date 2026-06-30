
"""
ChessMamba v2 - FastAPI Server.

Serves the dual-Mamba chess engine:
    1. Evaluation Mamba -> policy + WDL + uncertainty + embedding
    2. Geometric Navigator -> strategic direction scoring
    3. Search Mamba -> learned look-ahead evaluation
    4. Move Selector -> weighted combination -> best move

Endpoints:
    POST /move     - Get best move for a FEN
    POST /analyze  - Get detailed analysis
    POST /train/*  - Training control
    GET  /health   - Server health check
"""

import os
import json
import torch
import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from mamba import MambaConfig
from model import ChessMamba
from search_mamba import SearchMamba, SearchMambaConfig
from geometric import GeometricNavigator, load_advantage_vectors
from move_selector import MoveSelector
from encoding import encoder, ACTION_SPACE


# ── Request / Response Models ──

class MoveRequest(BaseModel):
    fen: str
    depth: int = 6            # Search depth for Search Mamba
    contempt: float = 0.0     # Contempt factor [0, 1]
    verbose: bool = False     # Return detailed analysis

class MoveResponse(BaseModel):
    best_move: str
    evaluation: float = 0.0
    wdl: list = None          # [P(Win), P(Draw), P(Loss)]
    uncertainty: float = None
    search_depth: int = None

class AnalysisResponse(BaseModel):
    moves: list               # Top-N moves with scores
    wdl: list
    uncertainty: float
    phase: str
    strategy: list = None


# ── Globals ──

eval_model = None
search_model = None
navigator = None
selector = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Loading ──

def load_eval_model():
    """Load Evaluation Mamba safely."""
    global eval_model
    
    vocab_size = encoder.vocab_size() + 100
    checkpoint_path = "chess_mamba.pt"
    config_path = "chess_mamba_config.json"
    
    if os.path.exists(checkpoint_path):
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
            config = MambaConfig(**{
                k: v for k, v in config_dict.items()
                if k in MambaConfig.__dataclass_fields__
            })
        else:
            config = MambaConfig(d_model=128, n_layer=4, vocab_size=vocab_size)
        # NB: do NOT override config.vocab_size here - the saved config is already
        # vocab-padded; re-setting it would un-pad and break the checkpoint load.
        eval_model = ChessMamba(config, action_space=ACTION_SPACE)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)  # nosec
        eval_model.load_state_dict(state_dict)
        print(f"[OK] Eval Mamba loaded from {checkpoint_path}")
    else:
        print("[!] No Eval Mamba checkpoint - using random initialization")
        config = MambaConfig(
            d_model=128, n_layer=4,
            vocab_size=vocab_size, expand=2
        )
        eval_model = ChessMamba(config, action_space=ACTION_SPACE)
    
    eval_model.to(device)
    eval_model.eval()
    
    param_count = sum(p.numel() for p in eval_model.parameters())
    print(f"  Parameters: {param_count:,}")


def load_search_model():
    """Load Search Mamba safely."""
    global search_model
    
    search_path = "search_mamba.pt"
    if os.path.exists(search_path):
        search_model = SearchMamba.from_pretrained(search_path, device=str(device))
        search_model.to(device)
        search_model.eval()
        print(f"[OK] Search Mamba loaded from {search_path}")
    else:
        search_model = None
        print("[!] No Search Mamba - using policy-only mode")


def load_navigator():
    """Load geometric navigator."""
    global navigator
    
    adv_path = "advantage_vectors.pt"
    if os.path.exists(adv_path):
        vectors = load_advantage_vectors(adv_path, device=str(device))
        navigator = GeometricNavigator(vectors).to(device)
        print(f"[OK] Geometric navigator loaded ({len(vectors)} phase vectors)")
    else:
        navigator = GeometricNavigator()
        print("[!] No advantage vectors - geometric scoring disabled")


def build_selector():
    """Build the move selector from loaded components."""
    global selector
    
    selector = MoveSelector(
        eval_model=eval_model,
        search_model=search_model,
        navigator=navigator,
    )
    print("[OK] Move selector ready")


# ── App Lifecycle ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 50)
    print("ChessMamba v2 - Starting")
    print("=" * 50)
    
    try:
        # Initialize Syzygy tablebase
        from tablebase import init_tablebase
        init_tablebase()
        
        # Load all models
        load_eval_model()
        load_search_model()
        load_navigator()
        build_selector()
        
        print("\n" + "=" * 50)
        print("ChessMamba v2 - Ready")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"[ERR] Error during startup: {e}")
        import traceback
        traceback.print_exc()
        # Ensure server still starts with minimal config
        load_eval_model()
        build_selector()
    
    yield
    print("Shutting down ChessMamba v2...")


app = FastAPI(
    title="ChessMamba v2",
    description="Dual-Mamba chess engine with geometric embedding navigation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "ChessMamba v2",
        "eval_model": eval_model is not None,
        "search_model": search_model is not None,
        "navigator": navigator is not None and bool(navigator.advantage_vectors),
        "device": str(device),
    }


@app.post("/move", response_model=MoveResponse)
def get_move(request: MoveRequest):
    if not eval_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    board = chess.Board(request.fen)
    
    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is over")
    
    # Syzygy tablebase probe for endgames
    from tablebase import can_probe, get_tablebase_move, get_tablebase_wdl_probs
    
    if can_probe(board):
        tb_move = get_tablebase_move(board)
        tb_wdl = get_tablebase_wdl_probs(board)
        
        if tb_move is not None:
            wdl_list = tb_wdl if tb_wdl else [0.33, 0.34, 0.33]
            return MoveResponse(
                best_move=tb_move.uci(),
                evaluation=wdl_list[0] - wdl_list[2],
                wdl=wdl_list,
                uncertainty=0.0,
                search_depth=999,
            )
    
    # Use the full dual-Mamba pipeline
    best_move = selector.select_move(board, verbose=request.verbose)
    
    if best_move is None:
        raise HTTPException(status_code=500, detail="No move found")
    
    # Get evaluation info
    with torch.no_grad():
        enc = encoder.encode_board(board)
        x = torch.tensor([enc], dtype=torch.long, device=device)
        outputs = eval_model.forward_full(x)
        wdl = outputs["wdl"][0].cpu().tolist()
        uncertainty = outputs["uncertainty"][0, 0].item()
    
    evaluation = wdl[0] - wdl[2]
    
    return MoveResponse(
        best_move=best_move.uci(),
        evaluation=evaluation,
        wdl=wdl,
        uncertainty=uncertainty,
        search_depth=request.depth,
    )


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_position(request: MoveRequest):
    if not eval_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    board = chess.Board(request.fen)
    
    # Get full analysis
    analysis = selector.get_analysis(board, top_n=10) if selector else []
    
    # Get WDL and uncertainty
    with torch.no_grad():
        enc = encoder.encode_board(board)
        x = torch.tensor([enc], dtype=torch.long, device=device)
        outputs = eval_model.forward_full(x)
    
    wdl = outputs["wdl"][0].cpu().tolist()
    uncertainty = outputs["uncertainty"][0, 0].item()
    strategy = outputs["strategy"][0].cpu().tolist()
    
    # Detect game phase
    phase = encoder._detect_game_phase(board)
    
    return AnalysisResponse(
        moves=analysis,
        wdl=wdl,
        uncertainty=uncertainty,
        phase=phase,
        strategy=strategy,
    )


# ── Training Control ──

training_process = None

class TrainRequest(BaseModel):
    phase: str = "all"       # eval, vectors, search, all
    epochs: int = 20
    batch_size: int = 128
    data_dir: str = "data/training"

@app.post("/train/start")
async def start_training(request: TrainRequest = TrainRequest()):
    global training_process
    import subprocess
    import sys
    
    if training_process is not None and training_process.poll() is None:
        return {"status": "already_running", "message": "Training in progress"}
    
    cmd = [
        sys.executable, "train.py",
        "--phase", request.phase,
        "--epochs", str(request.epochs),
        "--batch-size", str(request.batch_size),
        "--data", request.data_dir,
    ]
    
    training_process = subprocess.Popen(
        cmd, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    return {"status": "started", "message": f"Training started (PID: {training_process.pid})"}

@app.post("/train/stop")
async def stop_training():
    global training_process
    
    if training_process is None:
        return {"status": "not_running"}
    
    if training_process.poll() is None:
        training_process.terminate()
        training_process.wait()
        training_process = None
        return {"status": "stopped"}
    else:
        training_process = None
        return {"status": "already_finished"}

@app.get("/train/status")
async def training_status():
    global training_process
    
    if training_process is None:
        return {"running": False, "message": "No training started"}
    
    if training_process.poll() is None:
        return {"running": True, "pid": training_process.pid}
    else:
        return {"running": False, "exit_code": training_process.returncode}

@app.get("/train/progress")
async def get_training_progress():
    progress_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "training_progress.json"
    )
    
    if not os.path.exists(progress_file):
        return {"status": "not_started", "epoch": 0}
    
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {"status": "error", "message": "Could not read progress file"}


# ── Entry Point ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
