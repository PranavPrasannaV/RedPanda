
"""
torch.compile integration (Phase 6.2).

Wrapping the Mamba models with torch.compile fuses the elementwise scan kernels
and cuts Python overhead - typically a 1.5–3x speedup for both training and
inference on recent PyTorch + CUDA. Safe no-ops on CPU / old PyTorch.

Usage:
    from torch_compile_wrapper import maybe_compile
    model = maybe_compile(model, mode="max-autotune")   # training
    model = maybe_compile(model, mode="reduce-overhead") # inference / MCTS
"""

import torch


def maybe_compile(model, mode="reduce-overhead", enabled=True):
    """Return a torch.compile'd model when supported, else the model unchanged."""
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        return model
    if not torch.cuda.is_available():
        # Inductor on CPU is hit-or-miss for this SSM; skip to stay robust.
        return model
    try:
        compiled = torch.compile(model, mode=mode, dynamic=True)
        print(f"[torch.compile] enabled (mode={mode})")
        return compiled
    except Exception as e:  # pragma: no cover
        print(f"[torch.compile] disabled ({e})")
        return model
