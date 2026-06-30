
"""
Search Mamba - Learned search via sequence evaluation.

Instead of tree search (alpha-beta, MCTS), this model evaluates entire
MOVE SEQUENCES in parallel. Feed it a position + a candidate line of
3-8 moves, and it predicts how good that continuation is.

Key advantages over tree search:
    1. Massive GPU parallelism: evaluate 100+ lines in ONE forward pass
    2. Mamba's recurrent state "mentally plays through" each line
    3. Linear scaling with sequence length (unlike transformer's quadratic)
    4. Trained on Stockfish principal variations for expert-level judgment

Architecture: Smaller Mamba-3 (d=384, 12 layers) - needs to be fast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig, RMSNorm


class SearchMambaConfig(MambaConfig):
    """Config for the Search Mamba - smaller and faster than Eval Mamba.

    Phase 6 target is d=512/16; defaults stay at d=384/12 so the pair fits a
    single 8 GB GPU. Override via the trainer's --search-d-model / --search-n-layer.
    """
    d_model: int = 384
    n_layer: int = 12
    d_state: int = 32
    expand: int = 2
    use_complex: bool = True
    mimo_p: int = 2         # Fewer MIMO channels (speed)
    use_bcnorm: bool = True


class SearchMamba(nn.Module):
    """
    Evaluates move sequences to find the best continuation.
    
    Input: board encoding + candidate move sequence
    Output: quality score for that line
    
    Usage:
        search = SearchMamba(config)
        
        # Evaluate 50 candidate 6-move lines simultaneously
        board_enc = encoder.encode_board(board)        # (1, L_board)
        lines = generate_continuations(board, top_50)  # (50, 6)
        
        scores = search.evaluate_lines(board_enc, lines)  # (50,)
        best_idx = scores.argmax()
    """
    
    def __init__(self, config: SearchMambaConfig = None, vocab_size: int = 8192,
                 eval_dim: int = 512):
        super().__init__()
        if config is None:
            config = SearchMambaConfig(vocab_size=vocab_size)
        # The Search Mamba is the in-tree rollout model: it MUST be causal.
        config.bidirectional = False
        self.config = config
        self.eval_dim = eval_dim
        self.backbone = Mamba(config)
        d = config.d_model

        # ── Line Quality Head (legacy; whole-line score) ──
        self.line_value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d),
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1),
        )

        # ── Per-step VALUE head (v4: value-equivalent dynamics) ──
        # tanh-bounded side-to-move value of the position after each stepped move.
        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, 1),
        )

        # ── Continuation / policy head (learned move ordering) ──
        self.continuation_head = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, config.vocab_size),
        )

        # ── Consistency projection (v4: EfficientZero temporal consistency) ──
        # Projects the recurrent hidden state into the Eval Mamba's embedding
        # space so a consistency loss can pull it toward the TRUE next position's
        # embedding -- the ingredient that keeps deep rollouts accurate.
        self.consistency_proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, eval_dim),
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, board_encoding, move_sequences):
        """
        Batched forward pass for training.
        Args:
            board_encoding: (B, L_board)
            move_sequences: (B, L_moves)
        Returns:
            scores: (B,)
        """
        full_sequences = torch.cat([board_encoding, move_sequences], dim=-1)
        hidden = self.backbone(full_sequences)
        final_states = hidden[:, -1, :]
        scores = self.line_value_head(final_states).squeeze(-1)
        return scores

    def evaluate_lines(self, board_encoding, move_sequences):
        """
        Evaluate multiple candidate continuations for a SINGLE board.
        
        Args:
            board_encoding: (L_board,) or (1, L_board) - current position tokens
            move_sequences: (K, L_moves) - K candidate lines, each L_moves tokens
        
        Returns:
            (K,) quality scores for each line (higher = better)
        """
        if board_encoding.dim() == 1:
            board_encoding = board_encoding.unsqueeze(0)
        
        K = move_sequences.shape[0]
        board_repeated = board_encoding.expand(K, -1)  # (K, L_board)
        
        return self.forward(board_repeated, move_sequences)
    
    def evaluate_with_continuation(self, board_encoding, move_sequences):
        """
        Evaluate lines AND predict the next best move for each.
        
        Returns:
            scores: (K,) quality scores
            next_move_logits: (K, vocab_size) predicted next moves
        """
        if board_encoding.dim() == 1:
            board_encoding = board_encoding.unsqueeze(0)
        
        K = move_sequences.shape[0]
        board_repeated = board_encoding.expand(K, -1)
        full_sequences = torch.cat([board_repeated, move_sequences], dim=-1)
        
        hidden = self.backbone(full_sequences)
        final_states = hidden[:, -1, :]
        
        scores = self.line_value_head(final_states).squeeze(-1)
        next_move_logits = self.continuation_head(final_states)
        
        return scores, next_move_logits
    
    # ── O(1) cached line evaluation (Phase 3.2) ───────────────────────────

    def prime_board(self, board_encoding):
        """
        Encode a board ONCE and return (final_hidden, cache) so candidate lines
        can be extended with O(1) step() instead of reprocessing the board.

        Accepts a list / array / tensor of token ids, shape (L,) or (1, L).
        """
        if not torch.is_tensor(board_encoding):
            board_encoding = torch.as_tensor(
                board_encoding, dtype=torch.long,
                device=next(self.parameters()).device)
        if board_encoding.dim() == 1:
            board_encoding = board_encoding.unsqueeze(0)
        return self.backbone.prime_cache(board_encoding)

    @staticmethod
    def _expand_cache(cache, k):
        """Broadcast a B=1 SSM cache to B=k (cheap: expand, no copy)."""
        expanded = []
        for state in cache:
            h, dbu = state
            expanded.append((
                h.expand(k, *h.shape[1:]).contiguous(),
                dbu.expand(k, *dbu.shape[1:]).contiguous(),
            ))
        return expanded

    @torch.no_grad()
    def evaluate_lines_fast(self, board_encoding, move_sequences, lengths=None):
        """
        Evaluate K candidate lines from a SINGLE board using cached board state.

        Starts from the board's primed SSM cache and steps through each line's
        move tokens - O(K * L_moves) instead of O(K * (L_board + L_moves)).
        ~10x faster for the typical board_len~76, move_len~8 case.

        Args:
            board_encoding: (L_board,) current position tokens
            move_sequences: (K, L_moves) candidate lines (0 = PAD)
            lengths:        optional (K,) true line lengths; if given, the score
                            is read at each line's last real move (pad-safe).
        Returns:
            (K,) quality scores
        """
        device = next(self.parameters()).device
        _, cache = self.prime_board(board_encoding.to(device))
        K, L = move_sequences.shape
        move_sequences = move_sequences.to(device)

        cache = self._expand_cache(cache, K)
        hiddens = []
        for t in range(L):
            hidden, cache = self.backbone.step(
                token_id=move_sequences[:, t], cache=cache
            )
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens, dim=1)          # (K, L, d)

        if lengths is not None:
            idx = (lengths.to(device).clamp(min=1) - 1).view(K, 1, 1).expand(-1, 1, hiddens.shape[-1])
            final_states = hiddens.gather(1, idx).squeeze(1)
        else:
            final_states = hiddens[:, -1, :]
        return self.line_value_head(final_states).squeeze(-1)

    @torch.no_grad()
    def score_branch(self, board_cache, move_sequence):
        """
        Phase 5.1 - score a single promising MCTS branch and predict its best
        continuation, reusing a pre-primed board cache.

        Args:
            board_cache: cache from prime_board() (B=1)
            move_sequence: (L_moves,) move token ids from root to this node
        Returns:
            (quality_score: float, next_move_logits: (vocab,))
        """
        device = next(self.parameters()).device
        cache = self._expand_cache(board_cache, 1)
        hidden = None
        for tok in move_sequence:
            hidden, cache = self.backbone.step(
                token_id=torch.tensor([int(tok)], device=device), cache=cache
            )
        quality = self.line_value_head(hidden).squeeze(-1)
        next_logits = self.continuation_head(hidden).squeeze(0)
        return quality.item(), next_logits

    # ── v4: value-equivalent dynamics (training + MARS) ───────────────────

    def dynamics_rollout(self, board_encoding, move_tokens):
        """
        Differentiable rollout for TRAINING the value-equivalent model.

        Prime on the board, step through `move_tokens`, and return per-step:
            values (B, T)             tanh side-to-move value after each move
            cont   (B, T, vocab)      continuation-policy logits
            feats  (B, T, eval_dim)   consistency features (-> Eval embedding)

        Args:
            board_encoding: (B, L_board) long
            move_tokens:    (B, T) long
        """
        _, cache = self.backbone.prime_cache(board_encoding)
        T = move_tokens.shape[1]
        vs, cs, fs = [], [], []
        for t in range(T):
            h, cache = self.backbone.step(token_id=move_tokens[:, t], cache=cache)
            vs.append(torch.tanh(self.value_head(h)).squeeze(-1))
            cs.append(self.continuation_head(h))
            fs.append(self.consistency_proj(h))
        return torch.stack(vs, 1), torch.stack(cs, 1), torch.stack(fs, 1)

    @torch.no_grad()
    def eval_step(self, cache, move_token):
        """
        ONE O(1) node evaluation for MARS.

        Args:
            cache: per-layer SSM cache (from prime_board or a previous eval_step)
            move_token: int move id just played
        Returns:
            (value: float, cont_logits: (vocab,), new_cache)
        """
        device = next(self.parameters()).device
        tok = torch.tensor([int(move_token)], device=device)
        h, cache = self.backbone.step(token_id=tok, cache=cache)
        value = torch.tanh(self.value_head(h)).squeeze(-1).item()
        cont = self.continuation_head(h).squeeze(0)
        return value, cont, cache

    def iterative_deepening(self, board_encoding, initial_moves, depth=8,
                            top_k_extend=3, encoder=None):
        """
        Iterative deepening via the continuation head.
        
        Start with single moves, extend the best lines deeper using
        the continuation head's predictions.
        
        Args:
            board_encoding: (L_board,) - current position
            initial_moves: (K, 1) - K candidate first moves
            depth: how many moves deep to extend
            top_k_extend: how many top lines to extend at each step
            encoder: ChessEncoder instance for decoding
        
        Returns:
            best_scores: (K,) final scores for each initial move
        """
        device = board_encoding.device
        current_lines = initial_moves  # (K, 1)
        
        for d in range(depth - 1):
            # Evaluate current lines
            scores, next_logits = self.evaluate_with_continuation(
                board_encoding, current_lines
            )
            
            # For top-K lines, extend with predicted best continuation
            top_indices = scores.topk(min(top_k_extend, len(scores))).indices
            
            for idx in top_indices:
                # Get predicted next move
                next_move = next_logits[idx].argmax().unsqueeze(0).unsqueeze(0)  # (1, 1)
                
                # Extend the line
                extended = torch.cat([
                    current_lines[idx:idx+1],
                    next_move
                ], dim=-1)  # (1, d+2)
                
                current_lines = torch.cat([
                    current_lines[:idx],
                    extended,
                    current_lines[idx+1:]
                ], dim=0)
        
        # Final evaluation of all extended lines
        # Pad shorter lines to max length
        max_len = max(line.shape[-1] for line in [current_lines])
        padded = F.pad(current_lines, (0, max_len - current_lines.shape[-1]))
        
        final_scores = self.evaluate_lines(board_encoding, padded)
        return final_scores
    
    def save_pretrained(self, path):
        """Save model weights and config safely (no pickle for config)."""
        import json
        from dataclasses import asdict
        # Save weights with torch (only tensors, safe to load with weights_only=True)
        torch.save(self.state_dict(), path)
        # Save config as JSON alongside (safe, no pickle)
        config_path = path.replace('.pt', '_config.json').replace('.pth', '_config.json')
        if config_path == path:
            config_path = path + '.config.json'
        cfg_dict = asdict(self.config)
        cfg_dict["eval_dim"] = self.eval_dim          # persist for consistency_proj
        with open(config_path, 'w') as f:
            json.dump(cfg_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, path, device='cpu', config=None):
        """Load model safely -- weights_only=True, config from JSON."""
        import json
        from dataclasses import fields as dc_fields
        eval_dim = 512
        if config is None:
            config_path = path.replace('.pt', '_config.json').replace('.pth', '_config.json')
            if config_path == path:
                config_path = path + '.config.json'
            with open(config_path) as f:
                config_dict = json.load(f)
            eval_dim = int(config_dict.get("eval_dim", 512))
            valid_fields = {f.name for f in dc_fields(SearchMambaConfig)}
            config = SearchMambaConfig(**{
                k: v for k, v in config_dict.items()
                if k in valid_fields
            })
        model = cls(config, eval_dim=eval_dim)
        state_dict = torch.load(path, map_location=device, weights_only=True)  # nosec
        model.load_state_dict(state_dict)
        return model

