
"""
Memory-mapped datasets for ChessMamba v3 supervised training.

ChessDataset returns a dict per position:
    input_ids     (L,)  long   - board encoding
    policy_moves  (K,)  long   - soft-policy move ids (-1 = pad)
    policy_probs  (K,)  float  - soft-policy probabilities
    best_move     ()    long   - top move id (action-value target index)
    wdl           (3,)  float  - side-to-move [W,D,L]
    value         ()    float  - wdl[0]-wdl[2] in [-1,1]
    strategy      (T,)  float  - multi-hot strategy themes
    phase         ()    long   - 0/1/2

Everything is mmap-backed so 10M+ positions never load into RAM at once.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from encoding import ACTION_SPACE


class ChessDataset(Dataset):
    def __init__(self, data_dir: str, augment: bool = False):
        self.data_dir = data_dir
        self.augment = augment  # reserved; symmetry aug lives in self-play

        self.inputs = np.load(os.path.join(data_dir, "inputs.npy"), mmap_mode="r")
        self.policy_moves = np.load(os.path.join(data_dir, "policy_moves.npy"), mmap_mode="r")
        self.policy_probs = np.load(os.path.join(data_dir, "policy_probs.npy"), mmap_mode="r")
        self.wdl = np.load(os.path.join(data_dir, "wdl.npy"), mmap_mode="r")
        self.best_moves = np.load(os.path.join(data_dir, "best_moves.npy"), mmap_mode="r")
        self.phases = np.load(os.path.join(data_dir, "phases.npy"), mmap_mode="r")
        strat_path = os.path.join(data_dir, "strategy_labels.npy")
        self.strategy = np.load(strat_path, mmap_mode="r") if os.path.exists(strat_path) else None

        meta_path = os.path.join(data_dir, "metadata.json")
        self.metadata = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.num_positions = int(self.metadata.get("num_positions", len(self.inputs)))
        self.action_space = int(self.metadata.get("action_space", ACTION_SPACE))
        print(f"ChessDataset: {self.num_positions:,} positions from {data_dir}")

    def __len__(self):
        return self.num_positions

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.as_tensor(np.asarray(self.inputs[idx]).copy(), dtype=torch.long),
            "policy_moves": torch.as_tensor(self.policy_moves[idx].astype(np.int64)),
            "policy_probs": torch.as_tensor(self.policy_probs[idx].copy(), dtype=torch.float32),
            "best_move": torch.tensor(int(self.best_moves[idx]), dtype=torch.long),
            "wdl": torch.as_tensor(self.wdl[idx].copy(), dtype=torch.float32),
            "phase": torch.tensor(int(self.phases[idx]), dtype=torch.long),
        }
        item["value"] = (item["wdl"][0] - item["wdl"][2]).clamp(-1, 1)
        if self.strategy is not None:
            item["strategy"] = torch.as_tensor(self.strategy[idx].copy(), dtype=torch.float32)
        return item


def build_dense_policy(policy_moves, policy_probs, action_space):
    """(B,K) ids + (B,K) probs -> dense (B, action_space) soft-policy target."""
    B = policy_moves.shape[0]
    dense = torch.zeros(B, action_space, device=policy_moves.device)
    valid = policy_moves >= 0
    safe_ids = policy_moves.clamp(min=0)
    dense.scatter_add_(1, safe_ids, policy_probs * valid.float())
    return dense


class PVLineDataset(Dataset):
    """Search-Mamba training: (board_encoding, move_sequence, quality) from PVs."""

    def __init__(self, data_dir: str, max_pv_length: int = 8):
        from encoding import encoder
        self.encoder = encoder
        self.max_pv_length = max_pv_length
        self.inputs = np.load(os.path.join(data_dir, "inputs.npy"), mmap_mode="r")
        self.centipawns = np.load(os.path.join(data_dir, "centipawns.npy"), mmap_mode="r")
        with open(os.path.join(data_dir, "pv_lines.json")) as f:
            self.pv_lines = json.load(f)
        self.valid = [i for i, pv in enumerate(self.pv_lines) if len(pv) >= 2]
        print(f"PVLineDataset: {len(self.valid):,} valid PV lines")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        i = self.valid[idx]
        board_enc = torch.as_tensor(self.inputs[i], dtype=torch.long)
        pv = self.pv_lines[i][: self.max_pv_length]
        ids = []
        for uci in pv:
            mid = self.encoder.encode_move(uci)
            ids.append(mid if mid is not None else 0)
        while len(ids) < self.max_pv_length:
            ids.append(0)
        move_seq = torch.tensor(ids, dtype=torch.long)
        cp = float(self.centipawns[i])
        quality = torch.tensor(1.0 / (1.0 + np.exp(-cp / 200)), dtype=torch.float32)
        return board_enc, move_seq, quality


class DynamicsDataset(Dataset):
    """
    v4: trains the Search Mamba as a value-equivalent recurrent dynamics model.

    Per sample (a Stockfish PV line m1..mK from FEN B0):
        board_enc   (SEQ_LEN,)        encoding of B0 (primed once)
        move_tokens (T,)              [tok(m1)..tok(mK)] (0 = PAD)
        value_tgt   (T,)              (-1)^(i+1) * value(B0)  -> side-to-move value of B_{i+1}
        next_move   (T,)              continuation target m_{i+2} at step i
        inter_enc   (T, SEQ_LEN)      encodings of B1..BK (for the consistency loss)
        vmask       (T,)              1 where a real move exists
        cmask       (T,)              1 where a continuation target exists
    """

    def __init__(self, data_dir: str, max_pv: int = 8, seq_len: int = 160):
        import chess
        from encoding import encoder
        self.chess = chess
        self.encoder = encoder
        self.max_pv = max_pv
        self.seq_len = seq_len
        self.cps = np.load(os.path.join(data_dir, "centipawns.npy"), mmap_mode="r")
        with open(os.path.join(data_dir, "pv_lines.json")) as f:
            self.pv_lines = json.load(f)
        with open(os.path.join(data_dir, "start_fens.json")) as f:
            self.fens = json.load(f)
        self.valid = [i for i, pv in enumerate(self.pv_lines)
                      if len(pv) >= 2 and i < len(self.fens)]
        print(f"DynamicsDataset: {len(self.valid):,} PV lines")

    def __len__(self):
        return len(self.valid)

    @staticmethod
    def _cp_to_value(cp):
        import math
        p = 1.0 / (1.0 + math.exp(-cp / 350.0))
        return max(-1.0, min(1.0, 2.0 * p - 1.0))

    def _enc(self, board):
        return self.encoder.encode_board_padded(board, self.seq_len)

    def __getitem__(self, k):
        idx = self.valid[k]
        board = self.chess.Board(self.fens[idx])
        pv = self.pv_lines[idx][: self.max_pv]
        v0 = self._cp_to_value(float(self.cps[idx]))
        T = self.max_pv

        move_tokens = np.zeros(T, dtype=np.int64)
        value_tgt = np.zeros(T, dtype=np.float32)
        next_move = np.zeros(T, dtype=np.int64)
        inter_enc = np.zeros((T, self.seq_len), dtype=np.int64)
        vmask = np.zeros(T, dtype=np.float32)
        cmask = np.zeros(T, dtype=np.float32)

        board_enc = self._enc(board)
        real = 0
        for i, uci in enumerate(pv):
            try:
                mv = self.chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    break
            except Exception:
                break
            mid = self.encoder.encode_move(mv)
            board.push(mv)
            move_tokens[i] = mid if mid is not None else 0
            value_tgt[i] = ((-1.0) ** (i + 1)) * v0
            inter_enc[i] = self._enc(board)
            vmask[i] = 1.0
            real = i + 1
        for i in range(real - 1):
            nm = self.encoder.encode_move(pv[i + 1])
            next_move[i] = nm if nm is not None else 0
            cmask[i] = 1.0 if nm is not None else 0.0

        return {
            "board_enc": torch.from_numpy(board_enc),
            "move_tokens": torch.from_numpy(move_tokens),
            "value_tgt": torch.from_numpy(value_tgt),
            "next_move": torch.from_numpy(next_move),
            "inter_enc": torch.from_numpy(inter_enc),
            "vmask": torch.from_numpy(vmask),
            "cmask": torch.from_numpy(cmask),
        }


def create_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 4,
                       val_split: float = 0.02, augment: bool = False):
    full = ChessDataset(data_dir, augment=augment)
    total = len(full)
    val_size = max(1, int(total * val_split))
    train_size = total - val_size
    train_set, val_set = torch.utils.data.random_split(full, [train_size, val_size])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=num_workers > 0)
    print(f"Train: {train_size:,} | Val: {val_size:,} | "
          f"batches {len(train_loader):,}/{len(val_loader):,}")
    return train_loader, val_loader
