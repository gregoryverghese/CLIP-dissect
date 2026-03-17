"""
CONCHWrapper — wraps CONCH (Mahmood Lab contrastive pathology VLM, 448x448).
Install conch package from https://github.com/mahmoodlab/CONCH
"""
from .base import VLMWrapper


class CONCHWrapper(VLMWrapper):
    def __init__(self, checkpoint_path: str, device: str):
        """
        checkpoint_path: path to CONCH model weights
        Install conch package from https://github.com/mahmoodlab/CONCH
        """
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        self.model, self._preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path
        )
        self.tokenizer = get_tokenizer()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def encode_image(self, images):
        feats = self.model.encode_image(images, proj_contrast=True, normalize=True)
        return feats

    def encode_text(self, tokens):
        feats = self.model.encode_text(tokens, normalize=True)
        return feats

    def tokenize(self, texts, device="cpu"):
        from conch.open_clip_custom import tokenize as conch_tokenize
        return conch_tokenize(self.tokenizer, texts).to(device)

    @property
    def preprocess(self):
        return self._preprocess
