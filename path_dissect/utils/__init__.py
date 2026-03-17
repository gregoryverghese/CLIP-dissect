"""
path_dissect.utils — utility functions for activation extraction, VLM embedding, and pipeline orchestration.

Re-exports everything from the three submodules so callers can use:
    from path_dissect.utils import save_activations, get_similarity_from_activations, ...
"""

from .activations import (
    get_activation,
    save_target_activations,
    save_cem_activations,
    get_save_names,
    _all_saved,
    _make_save_dir,
)
from .embeddings import (
    save_clip_image_features,
    save_clip_text_features,
    get_clip_text_features,
    save_plip_slide_features,
)
from .pipeline import (
    save_activations,
    get_similarity_from_activations,
    get_cos_similarity,
)

__all__ = [
    # activations
    "get_activation",
    "save_target_activations",
    "save_cem_activations",
    "get_save_names",
    "_all_saved",
    "_make_save_dir",
    # embeddings
    "save_clip_image_features",
    "save_clip_text_features",
    "get_clip_text_features",
    "save_plip_slide_features",
    # pipeline
    "save_activations",
    "get_similarity_from_activations",
    "get_cos_similarity",
]
