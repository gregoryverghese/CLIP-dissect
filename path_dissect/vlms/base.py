"""
VLMWrapper base class defining the unified encode_image / encode_text interface.
"""
import torch


class VLMWrapper:
    """Base class. Subclasses must implement encode_image and encode_text."""

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, C, H, W] tensor, already preprocessed
        returns: [B, D] L2-normalized embeddings
        """
        raise NotImplementedError

    def encode_text(self, tokens) -> torch.Tensor:
        """
        tokens: whatever tokenize() returns for this VLM
        returns: [B, D] L2-normalized embeddings
        """
        raise NotImplementedError

    def tokenize(self, texts: list[str], device: str = "cpu"):
        """Tokenize a list of strings. Returns input for encode_text."""
        raise NotImplementedError

    @property
    def preprocess(self):
        """torchvision-compatible transform for probe images."""
        raise NotImplementedError
