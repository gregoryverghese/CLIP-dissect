"""
path_dissect.utils.activations — forward-hook activation extraction for target models.

Public API
----------
get_activation(outputs, mode)        — factory for register_forward_hook callbacks
save_target_activations(...)         — run target model on dataset, save per-layer tensors
save_cem_activations(...)            — run CEM on slide embeddings, save c_sem tensors
get_save_names(...)                  — derive canonical cache file paths
_all_saved(save_names)               — check if all cache files exist
_make_save_dir(save_name)            — create parent directory if missing
"""

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..utils._constants import PM_SUFFIX


def get_activation(outputs, mode):
    """
    mode: how to pool activations — one of 'avg', 'max'.
    For FC or ViT neurons does no pooling (uses [CLS] token for ViT).
    """
    if mode == 'avg':
        def hook(model, input, output):
            if len(output.shape) == 4:       # CNN layers
                outputs.append(output.mean(dim=[2, 3]).detach())
            elif len(output.shape) == 3:     # ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape) == 2:     # FC layers
                outputs.append(output.detach())
    elif mode == 'max':
        def hook(model, input, output):
            if len(output.shape) == 4:       # CNN layers
                outputs.append(output.amax(dim=[2, 3]).detach())
            elif len(output.shape) == 3:     # ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape) == 2:     # FC layers
                outputs.append(output.detach())
    return hook


def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    target_save_name = "{}/{}_{}_{}{}.pt".format(
        save_dir, d_probe, target_name, target_layer, PM_SUFFIX[pool_mode]
    )
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))

    return target_save_name, clip_save_name, text_save_name


def save_target_activations(target_model, dataset, save_name, target_layers=("layer4",),
                             batch_size=1000, device="cuda", pool_mode='avg'):
    """
    save_name: save file path template containing '{}' which will be formatted by layer names.
    """
    _make_save_dir(save_name)
    save_names = {}
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)

    if _all_saved(save_names):
        return

    all_features = {target_layer: [] for target_layer in target_layers}

    hooks = {}
    for target_layer in target_layers:
        command = (
            "target_model.{}.register_forward_hook("
            "get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        )
        hooks[target_layer] = eval(command)

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            target_model(images.to(device))

    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    del all_features
    torch.cuda.empty_cache()


def save_cem_activations(model, dataset, save_name, device="cuda"):
    """
    Runs CEM on all slides and saves c_sem (concept state logits).
    dataset: SlideEmbeddingDataset — yields (embeddings [n_tiles, h_dim], slide_id)
    Saves [n_slides, n_concept_states] tensor.
    """
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    all_c_sem = []
    with torch.no_grad():
        for emb, _ in tqdm(dataset, desc="CEM activations"):
            x = emb.unsqueeze(0).to(device)   # [1, n_tiles, h_dim]
            outputs = model(x)
            c_sem = outputs[0]                 # [1, n_concept_states]
            all_c_sem.append(c_sem.cpu())
    torch.save(torch.cat(all_c_sem), save_name)  # [n_slides, n_concept_states]
    torch.cuda.empty_cache()


def _all_saved(save_names):
    """
    save_names: {layer_name: save_path} dict
    Returns True if there is a file for each value in save_names, else False.
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True


def _make_save_dir(save_name):
    """Creates the save directory if it does not exist."""
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
