"""
path_dissect.probe_models.ccem — Concept Embedding Model for MIL pathology.

Public API
----------
ConceptEmbeddingModel — the main Lightning module
"""

from .cem_mil import ConceptEmbeddingModel

__all__ = ["ConceptEmbeddingModel"]
