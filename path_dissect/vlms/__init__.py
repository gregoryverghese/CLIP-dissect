"""
path_dissect.vlms — VLM wrappers with a unified encode_image / encode_text interface.

Public API
----------
load_vlm(clip_name, device, **kwargs) -> VLMWrapper
    Factory that returns the appropriate wrapper for the given model name.

VLMWrapper   — base class
CLIPWrapper  — OpenAI CLIP
PLIPWrapper  — vinid/plip (HuggingFace)
CONCHWrapper — Mahmood Lab CONCH (pathology, 448x448)
"""

from .base import VLMWrapper
from .clip import CLIPWrapper
from .plip import PLIPWrapper
from .conch import CONCHWrapper


def load_vlm(clip_name: str, device: str, **kwargs) -> VLMWrapper:
    """
    clip_name: "plip", "conch", or any OpenAI CLIP model name (e.g. "ViT-B/16")
    kwargs: passed to wrapper constructors (e.g. checkpoint_path for CONCH)
    """
    if clip_name == "plip":
        return PLIPWrapper(device)
    elif clip_name == "conch":
        checkpoint_path = kwargs.get("conch_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError("Pass conch_checkpoint=<path> when using CONCH")
        return CONCHWrapper(checkpoint_path, device)
    else:
        return CLIPWrapper(clip_name, device)


__all__ = [
    "load_vlm",
    "VLMWrapper",
    "CLIPWrapper",
    "PLIPWrapper",
    "CONCHWrapper",
]
