"""
path_dissect.utils.pipeline — orchestration of the full dissect pipeline.

Public API
----------
save_activations(...)              — orchestrator: loads VLM + target model, saves all caches
get_similarity_from_activations(...)  — load cached tensors, run similarity function
get_cos_similarity(...)            — cosine similarity evaluation (CLIP + MPNet)
"""

import math
import numpy as np
import torch
from tqdm import tqdm

from ..vlms import load_vlm
from ..datasets.standard import get_target_model, get_data
from .activations import (
    get_save_names,
    save_target_activations,
    _make_save_dir,
)
from .embeddings import (
    save_clip_image_features,
    save_clip_text_features,
)


def save_activations(clip_name, target_name, target_layers, d_probe,
                     concept_set, batch_size, device, pool_mode, save_dir, **vlm_kwargs):
    """
    Orchestrates the full caching pipeline:
      1. Load VLM and target model.
      2. Extract and save CLIP image features, CLIP text features, and target activations.
    """
    vlm = load_vlm(clip_name, device, **vlm_kwargs)
    target_model, target_preprocess = get_target_model(target_name, device)

    data_c = get_data(d_probe, vlm.preprocess)
    data_t = get_data(d_probe, target_preprocess)

    with open(concept_set, 'r') as f:
        words = (f.read()).split('\n')
    words = [w for w in words if w != ""]

    text = vlm.tokenize(words, device=device)

    save_names = get_save_names(
        clip_name=clip_name, target_name=target_name,
        target_layer='{}', d_probe=d_probe, concept_set=concept_set,
        pool_mode=pool_mode, save_dir=save_dir,
    )
    target_save_name, clip_save_name, text_save_name = save_names

    save_clip_text_features(vlm, text, text_save_name, batch_size)
    save_clip_image_features(vlm, data_c, clip_save_name, batch_size, device)
    save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode)


def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name,
                                    similarity_fn, return_target_feats=True, device="cuda"):
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()

    target_feats = torch.load(target_save_name, map_location='cpu')
    similarity = similarity_fn(clip_feats, target_feats, device=device)

    del clip_feats
    torch.cuda.empty_cache()

    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity


def get_cos_similarity(preds, gt, vlm, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    vlm: VLMWrapper instance
    mpnet_model: sentence-transformers MPNet model instance
    """
    pred_tokens = vlm.tokenize(preds, device=device)
    gt_tokens = vlm.tokenize(gt, device=device)
    pred_embeds = []
    gt_embeds = []

    n = (pred_tokens.shape[0] if isinstance(pred_tokens, torch.Tensor)
         else len(next(iter(pred_tokens.values()))))
    with torch.no_grad():
        for i in range(math.ceil(n / batch_size)):
            def _batch(t, i):
                if isinstance(t, torch.Tensor):
                    return t[batch_size*i:batch_size*(i+1)]
                return {k: v[batch_size*i:batch_size*(i+1)] for k, v in t.items()}
            pred_embeds.append(vlm.encode_text(_batch(pred_tokens, i)))
            gt_embeds.append(vlm.encode_text(_batch(gt_tokens, i)))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    cos_sim_clip = torch.sum(pred_embeds * gt_embeds, dim=1)

    gt_embeds_np = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds_np = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds_np * gt_embeds_np, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))
