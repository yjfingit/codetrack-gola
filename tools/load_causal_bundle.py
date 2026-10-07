"""Apply a causal diagnosis/template/H bundle to a GOLA CodeTrack model."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import torch


def load_causal_bundle(model: Any, path: str | Path) -> None:
    from safetensors.torch import load_file
    state = load_file(str(path), device="cpu")
    diag = {k[len("diagnosis."):]: v for k, v in state.items() if k.startswith("diagnosis.")}
    pool = {k[len("template_pool."):]: v for k, v in state.items() if k.startswith("template_pool.")}
    if diag:
        model.codetrack.diagnosis.load_state_dict(diag, strict=True)
    if pool:
        model.codetrack.template_pool.load_state_dict(pool, strict=True)
    if "H.H" in state:
        model.codetrack.H.H.data.copy_(state["H.H"].to(model.codetrack.H.H))
