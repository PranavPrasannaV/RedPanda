
"""
ChessMamba v2 - Multi-head model with geometric embedding support.

Architecture:
    Mamba-3 Backbone -> Multi-resolution Feature Fusion -> 
    {Policy Head, WDL Head, Uncertainty Head, Strategy Head}

Key changes from v1:
    - Multi-resolution pooling (last state + global average)
    - Deeper heads (3-layer policy, 3-layer WDL)
    - Uncertainty head (guides search depth)
    - get_embedding() for geometric operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
from encoding import ACTION_SPACE

# Number of strategic themes (from strategy.py)
NUM_STRATEGIC_THEMES = 24


class ChessMamba(nn.Module):
    def __init__(self, config: MambaConfig, action_space: int = ACTION_SPACE):
        super().__init__()
        self.config = config
        self.action_space = action_space
        d = config.d_model

        # ── Mamba-3 Backbone ──
        self.backbone = Mamba(config)

        # ── Multi-resolution Feature Fusion ──
        # Combines last hidden state (recurrent summary) with
        # global average pool (captures features that might be
        # "forgotten" by the recurrence).
        self.fusion = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # ── Policy Head (3-layer, wider) ──
        # Predicts move distribution over action space
        self.policy_head = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.LayerNorm(d * 4),
            nn.Linear(d * 4, d * 2),
            nn.GELU(),
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, action_space),
        )

        # ── WDL Head (3-layer) ──
        # Predicts Win/Draw/Loss probabilities
        self.wdl_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, 3),
        )

        # ── Uncertainty Head (NEW) ──
        # Predicts model confidence: 0=certain, 1=uncertain
        # Used to decide search depth adaptively
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )

        # ── Strategy Head (multi-label) ──
        # Predicts active strategic themes
        self.strategy_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, NUM_STRATEGIC_THEMES),
        )

        # ── Action-Value Head (NEW, Phase 1.4) ──
        # Per-move Q-value prediction, à la KataGo. Outputs a tanh-bounded
        # value in [-1, 1] for every move id. During MCTS this seeds child
        # Q-values BEFORE they are visited, eliminating the cold-start problem.
        self.action_value_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, action_space),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ── Core embedding extraction ──

    def get_embedding(self, input_ids):
        """
        Extract the fused embedding vector for geometric operations.
        
        This is the primary representation used for:
            - Advantage vector projection
            - Contrastive learning
            - Search Mamba input conditioning
        
        Args:
            input_ids: (B, L) token indices
        Returns:
            (B, d_model) fused embedding
        """
        hidden_states = self.backbone(input_ids)  # (B, L, D)

        # Multi-resolution pooling
        last_state = hidden_states[:, -1, :]       # (B, D) - recurrent summary
        global_pool = hidden_states.mean(dim=1)    # (B, D) - global average

        # Fuse
        fused = self.fusion(torch.cat([last_state, global_pool], dim=-1))  # (B, D)
        return fused

    # ── Forward passes ──

    def forward(self, input_ids, return_dict=False):
        """Standard forward: returns policy logits and WDL probabilities."""
        fused = self.get_embedding(input_ids)

        if return_dict:
            wdl_raw = self.wdl_head(fused)
            return {
                "embedding": fused,
                "policy": self.policy_head(fused),
                "wdl_raw": wdl_raw,
                "wdl": F.softmax(wdl_raw, dim=-1),
                "uncertainty": self.uncertainty_head(fused),
                "strategy": torch.sigmoid(self.strategy_head(fused)),
                "action_value": torch.tanh(self.action_value_head(fused)),
            }

        policy_logits = self.policy_head(fused)
        wdl_logits = self.wdl_head(fused)
        wdl = F.softmax(wdl_logits, dim=-1)

        return policy_logits, wdl

    def forward_full(self, input_ids):
        """
        Full forward: returns all head outputs + embedding.
        
        Used during inference for the complete move selection pipeline.
        
        Returns dict with:
            embedding:   (B, D) - fused embedding for geometric ops
            policy:      (B, action_space) - raw policy logits
            wdl:         (B, 3) - [P(Win), P(Draw), P(Loss)]
            uncertainty: (B, 1) - confidence level [0=certain, 1=uncertain]
            strategy:    (B, 24) - strategic theme probabilities
            action_value:(B, action_space) - per-move Q-values in [-1, 1]
        """
        fused = self.get_embedding(input_ids)

        return {
            "embedding": fused,
            "policy": self.policy_head(fused),
            "wdl": F.softmax(self.wdl_head(fused), dim=-1),
            "uncertainty": self.uncertainty_head(fused),
            "strategy": torch.sigmoid(self.strategy_head(fused)),
            "action_value": torch.tanh(self.action_value_head(fused)),
        }

    def forward_with_strategy(self, input_ids):
        """Extended forward: policy, WDL, and strategy. (Legacy compat.)"""
        fused = self.get_embedding(input_ids)

        policy_logits = self.policy_head(fused)
        wdl = F.softmax(self.wdl_head(fused), dim=-1)
        strategy_probs = torch.sigmoid(self.strategy_head(fused))

        return policy_logits, wdl, strategy_probs

    # ── Value conversion ──

    def get_value(self, wdl, contempt=0.0):
        """Convert WDL probabilities to a scalar value in [-1, 1]."""
        base_value = wdl[:, 0] - wdl[:, 2]
        draw_penalty = contempt * wdl[:, 1]
        adjusted_value = base_value + draw_penalty
        return adjusted_value.clamp(-1, 1)

    # ── Serialization ──

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path, config, action_space: int = ACTION_SPACE,
                        map_location="cpu"):
        model = cls(config, action_space=action_space)
        model.load_state_dict(torch.load(path, weights_only=True, map_location=map_location))
        return model
