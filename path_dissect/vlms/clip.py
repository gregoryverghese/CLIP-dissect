"""
CLIPWrapper — wraps OpenAI CLIP models.
"""
from .base import VLMWrapper


class CLIPWrapper(VLMWrapper):
    def __init__(self, model_name: str, device: str):
        import clip as openai_clip
        self.model, self._preprocess = openai_clip.load(model_name, device=device)
        self.model.eval()
        self.device = device
        self._clip = openai_clip

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    def tokenize(self, texts, device="cpu"):
        return self._clip.tokenize(texts).to(device)

    @property
    def preprocess(self):
        return self._preprocess
