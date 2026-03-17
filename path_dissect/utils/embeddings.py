"""
path_dissect.utils.embeddings — VLM image and text feature extraction.

Public API
----------
save_clip_image_features(model, dataset, save_name, batch_size, device)
save_clip_text_features(model, text, save_name, batch_size)
get_clip_text_features(model, text, batch_size)
save_plip_slide_features(plip_emb_dir, save_name, slide_ids)
"""

import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from .activations import _make_save_dir


def save_clip_image_features(model, dataset, save_name, batch_size=1000, device="cuda"):
    _make_save_dir(save_name)
    if os.path.exists(save_name):
        return

    all_features = []
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    del all_features
    torch.cuda.empty_cache()


def save_clip_text_features(model, text, save_name, batch_size=1000):
    """
    model: VLMWrapper — encode_text accepts batches of whatever tokenize() returned.
    text: output of model.tokenize() — either a tensor (CLIP) or a dict of tensors (PLIP/CONCH).
    """
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    n = text.shape[0] if isinstance(text, torch.Tensor) else len(next(iter(text.values())))
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(n / batch_size))):
            batch = (
                text[batch_size*i:batch_size*(i+1)]
                if isinstance(text, torch.Tensor)
                else {k: v[batch_size*i:batch_size*(i+1)] for k, v in text.items()}
            )
            text_features.append(model.encode_text(batch))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()


def get_clip_text_features(model, text, batch_size=1000):
    """Gets text features without saving — useful with dynamic concept sets."""
    n = text.shape[0] if isinstance(text, torch.Tensor) else len(next(iter(text.values())))
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(n / batch_size))):
            batch = (
                text[batch_size*i:batch_size*(i+1)]
                if isinstance(text, torch.Tensor)
                else {k: v[batch_size*i:batch_size*(i+1)] for k, v in text.items()}
            )
            text_features.append(model.encode_text(batch))
    text_features = torch.cat(text_features, dim=0)
    return text_features


def save_plip_slide_features(plip_emb_dir, save_name, slide_ids):
    """
    Stacks pre-computed PLIP per-slide embeddings in slide_ids order.
    slide_ids: list of slide ID strings (stems of .pt files in plip_emb_dir).
    Saves [n_slides, 512] tensor.
    """
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    embeddings = []
    for sid in tqdm(slide_ids, desc="Loading PLIP embeddings"):
        emb = torch.load(os.path.join(plip_emb_dir, f"{sid}.pt"), map_location="cpu")
        embeddings.append(emb)
    torch.save(torch.cat(embeddings), save_name)  # [n_slides, 512]
