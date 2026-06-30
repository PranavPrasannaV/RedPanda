
"""
Geometric Embedding Navigator - SOLIS-inspired advantage-vector navigation (v3).

Maps positions into the Eval Mamba's embedding space where an "advantage vector"
points from losing -> winning regions. Move selection's geometric tie-break picks
the move whose resulting position projects furthest along that vector.

v3 fixes & upgrades:
    - v2 computed advantage vectors by reconstructing a board from the token
      encoding - a stub that always returned None, so NO vectors were ever built.
      We now carry phase + strategy context ALONGSIDE each position (computed at
      data-prep time) instead of trying to invert the encoding.
    - 15-vector system: 3 phases x 5 strategic contexts
      (attacking / defending / maneuvering / converting / holding), selected at
      inference using the Eval Mamba's strategy head. Falls back phase -> global.

Reference: SOLIS (KDD 2025).
"""

import torch
import torch.nn.functional as F
import chess
import numpy as np
from typing import Dict, List, Optional

from encoding import encoder, PHASE_NAMES
from strategy import StrategicTheme, NUM_THEMES


# ── Strategic contexts: groups of strategy themes ─────────────────────────────

CONTEXTS = ["attacking", "defending", "maneuvering", "converting", "holding"]

# Map each strategy theme to a context (holding = drawish fallback, no themes).
THEME_CONTEXT = {
    StrategicTheme.KINGSIDE_ATTACK: "attacking",
    StrategicTheme.QUEENSIDE_ATTACK: "attacking",
    StrategicTheme.PIECE_ACTIVITY: "attacking",
    StrategicTheme.SACRIFICE_PREP: "attacking",
    StrategicTheme.BATTERY: "attacking",
    StrategicTheme.DISCOVERED_ATTACK_PREP: "attacking",
    StrategicTheme.EXCHANGE_SACRIFICE: "attacking",
    StrategicTheme.PIN_EXPLOITATION: "attacking",
    StrategicTheme.PROPHYLAXIS: "defending",
    StrategicTheme.FORTRESS: "defending",
    StrategicTheme.CONSOLIDATION: "defending",
    StrategicTheme.WEAK_SQUARES: "defending",
    StrategicTheme.CENTER_CONTROL: "maneuvering",
    StrategicTheme.OPEN_FILE: "maneuvering",
    StrategicTheme.OUTPOST: "maneuvering",
    StrategicTheme.MINORITY_ATTACK: "maneuvering",
    StrategicTheme.SPACE_ADVANTAGE: "maneuvering",
    StrategicTheme.BISHOP_PAIR: "maneuvering",
    StrategicTheme.PAWN_BREAK: "maneuvering",
    StrategicTheme.KNIGHT_MANEUVER: "maneuvering",
    StrategicTheme.ROOK_LIFT: "maneuvering",
    StrategicTheme.PASSED_PAWN: "converting",
    StrategicTheme.ROOK_ACTIVITY: "converting",
    StrategicTheme.KING_ACTIVITY: "converting",
}

# (NUM_THEMES,) context index per theme, for fast vectorised bucketing.
_THEME_TO_CTX = np.full(NUM_THEMES, CONTEXTS.index("maneuvering"), dtype=np.int64)
for _theme, _ctx in THEME_CONTEXT.items():
    _THEME_TO_CTX[int(_theme)] = CONTEXTS.index(_ctx)


def strategy_to_context(strategy_probs) -> str:
    """Pick the dominant context from a (NUM_THEMES,) strategy activation vector."""
    probs = np.asarray(strategy_probs, dtype=np.float64)
    if probs.max() < 0.15:
        return "holding"                      # nothing active -> drawish/holding
    scores = np.zeros(len(CONTEXTS))
    for t in range(min(NUM_THEMES, len(probs))):
        scores[_THEME_TO_CTX[t]] += probs[t]
    return CONTEXTS[int(scores.argmax())]


class GeometricNavigator:
    def __init__(self, advantage_vectors: Dict[str, torch.Tensor] = None):
        self.advantage_vectors = advantage_vectors or {}
        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.advantage_vectors = {
            k: v.to(device) for k, v in self.advantage_vectors.items()
        }
        return self

    # ── Vector lookup ──

    def _select_vector(self, phase: str, context: str) -> Optional[torch.Tensor]:
        for key in (f"{phase}:{context}", phase, "global"):
            if key in self.advantage_vectors:
                return self.advantage_vectors[key]
        if self.advantage_vectors:
            return torch.stack(list(self.advantage_vectors.values())).mean(0)
        return None

    def score_moves(self, board: chess.Board, model,
                    candidate_moves: List[chess.Move]) -> torch.Tensor:
        """Project each candidate's resulting-position embedding onto the
        phase/context-appropriate advantage vector. Higher = more 'winning'."""
        if not self.advantage_vectors or not candidate_moves:
            return torch.zeros(len(candidate_moves), device=self.device)

        phase = PHASE_NAMES[encoder.phase_index(board)]

        # Strategy context from the eval model's strategy head.
        with torch.no_grad():
            enc = encoder.encode_board(board)
            x = torch.tensor([enc], dtype=torch.long, device=self.device)
            strat = model.forward_full(x)["strategy"][0].cpu().numpy()
        context = strategy_to_context(strat)

        adv = self._select_vector(phase, context)
        if adv is None:
            return torch.zeros(len(candidate_moves), device=self.device)
        adv = F.normalize(adv.unsqueeze(0), dim=-1)

        encodings = []
        for mv in candidate_moves:
            board.push(mv)
            encodings.append(encoder.encode_board(board))
            board.pop()
        max_len = max(len(e) for e in encodings)
        padded = torch.zeros(len(encodings), max_len, dtype=torch.long, device=self.device)
        for i, e in enumerate(encodings):
            padded[i, : len(e)] = torch.tensor(e, dtype=torch.long)

        with torch.no_grad():
            emb = F.normalize(model.get_embedding(padded), dim=-1)
        return torch.mm(emb, adv.t()).squeeze(-1)

    def score_embedding(self, embedding: torch.Tensor,
                        phase: str = "middlegame", context: str = "maneuvering") -> float:
        adv = self._select_vector(phase, context)
        if adv is None:
            return 0.0
        adv = F.normalize(adv.unsqueeze(0), dim=-1)
        emb = F.normalize(embedding.unsqueeze(0), dim=-1)
        return torch.dot(emb.squeeze(), adv.squeeze()).item()

    def interpolate_toward_winning(self, embedding, phase="middlegame",
                                   context="maneuvering", alpha=0.1):
        adv = self._select_vector(phase, context)
        if adv is None:
            return embedding
        return F.normalize(embedding + alpha * adv, dim=0)

    def _get_phase(self, board: chess.Board) -> str:
        return PHASE_NAMES[encoder.phase_index(board)]


# ─── Advantage-vector computation (Phase 2 of training) ──────────────────────

@torch.no_grad()
def compute_advantage_vectors(model, inputs, wdl, phases, strategy_labels=None,
                              device="cpu", sample: int = 60000,
                              batch_size: int = 256, win_thresh: float = 0.6,
                              loss_thresh: float = 0.6) -> Dict[str, torch.Tensor]:
    """
    Build phasexcontext advantage vectors from trained embeddings.

    Args:
        inputs:          (N, L) int board encodings (np array / memmap)
        wdl:             (N, 3) float, side-to-move [W, D, L]
        phases:          (N,) int phase ids (0/1/2)
        strategy_labels: (N, NUM_THEMES) float multi-hot, or None
    Returns:
        dict { "phase:context": vec, "phase": vec, "global": vec }, all unit-norm
    """
    model.eval()
    n = len(inputs)
    idx = np.arange(n)
    if n > sample:
        idx = np.random.choice(n, sample, replace=False)

    d = model.config.d_model
    # accumulators: sum + count of winning/losing embeddings per bucket
    keys = (["global"]
            + list(PHASE_NAMES)
            + [f"{p}:{c}" for p in PHASE_NAMES for c in CONTEXTS])
    acc = {k: {"win": torch.zeros(d), "wn": 0, "loss": torch.zeros(d), "ln": 0}
           for k in keys}

    for start in range(0, len(idx), batch_size):
        bidx = idx[start:start + batch_size]
        x = torch.as_tensor(np.asarray(inputs[bidx]), dtype=torch.long, device=device)
        emb = model.get_embedding(x).cpu()
        for j, gi in enumerate(bidx):
            w, l = float(wdl[gi][0]), float(wdl[gi][2])
            if w < win_thresh and l < loss_thresh:
                continue
            phase = PHASE_NAMES[int(phases[gi])]
            if strategy_labels is not None:
                context = strategy_to_context(strategy_labels[gi])
            else:
                context = "maneuvering"
            e = emb[j]
            for key in ("global", phase, f"{phase}:{context}"):
                bucket = acc[key]
                if w >= win_thresh:
                    bucket["win"] += e
                    bucket["wn"] += 1
                if l >= loss_thresh:
                    bucket["loss"] += e
                    bucket["ln"] += 1

    vectors = {}
    for key, b in acc.items():
        if b["wn"] >= 20 and b["ln"] >= 20:
            adv = b["win"] / b["wn"] - b["loss"] / b["ln"]
            vectors[key] = F.normalize(adv, dim=0)
    # Report
    print(f"  Built {len(vectors)} advantage vectors "
          f"({sum(':' in k for k in vectors)} phasexcontext, "
          f"{sum(k in PHASE_NAMES for k in vectors)} phase, "
          f"{'global' in vectors} global)")
    return vectors


def save_advantage_vectors(vectors: Dict[str, torch.Tensor], path: str):
    torch.save(vectors, path)
    print(f"Advantage vectors saved to {path} ({len(vectors)} vectors)")


def load_advantage_vectors(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    vectors = torch.load(path, map_location=device, weights_only=True)
    print(f"Loaded {len(vectors)} advantage vectors")
    return vectors
