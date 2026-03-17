"""
path_dissect.probe_models — probe (target) model definitions.

Subpackages
-----------
ccem  — Concept Embedding Model for MIL pathology (ConceptEmbeddingModel)
"""

from .ccem import ConceptEmbeddingModel

__all__ = ["ConceptEmbeddingModel"]
