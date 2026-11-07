"""Weighted k-NN evaluation (DINO protocol) and flip consistency.

CLS features are L2-normalised; similarity is cosine; the k = 20 nearest bank images vote for their
class with weight exp(sim / T), T = 0.07. Flip consistency is the share of queries whose prediction
equals the prediction for the mirrored query.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import mirror, normalize


@torch.no_grad()
def extract_features(adapter, loader, device, model=None, flip: bool = False):
    feats, labels = [], []
    for imgs, y in loader:
        x = normalize(imgs.to(device, non_blocking=True), adapter.dtype)
        if flip:
            x = mirror(x)
        f = adapter.cls_feature(x, model)
        feats.append(F.normalize(f.float(), dim=-1))
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels).to(device)


@torch.no_grad()
def knn_predict(bank_feats: torch.Tensor, bank_labels: torch.Tensor, query_feats: torch.Tensor,
                num_classes: int, k: int = 20, temperature: float = 0.07, chunk: int = 1024) -> torch.Tensor:
    k = min(int(k), bank_feats.shape[0])
    preds = []
    for s in range(0, query_feats.shape[0], chunk):
        sim = query_feats[s:s + chunk] @ bank_feats.T
        top_sim, top_idx = sim.topk(k, dim=1)
        weights = (top_sim / temperature).exp()
        votes = torch.zeros(sim.shape[0], num_classes, device=sim.device, dtype=weights.dtype)
        votes.scatter_add_(1, bank_labels[top_idx], weights)
        preds.append(votes.argmax(dim=1))
    return torch.cat(preds)


@torch.no_grad()
def evaluate_knn(adapter, bank_loader, query_loader, device, num_classes: int, k: int = 20,
                 temperature: float = 0.07, model=None) -> dict:
    """Top-1 (%) and flip consistency (%) for `model` (default: the adapter's own model)."""
    bf, by = extract_features(adapter, bank_loader, device, model)
    qf, qy = extract_features(adapter, query_loader, device, model)
    qf_flip, _ = extract_features(adapter, query_loader, device, model, flip=True)
    pred = knn_predict(bf, by, qf, num_classes, k, temperature)
    pred_flip = knn_predict(bf, by, qf_flip, num_classes, k, temperature)
    return {
        "top1": 100.0 * float((pred == qy).float().mean()),
        "flip_consistency": 100.0 * float((pred == pred_flip).float().mean()),
        "n_bank": int(bf.shape[0]),
        "n_query": int(qf.shape[0]),
        "pred": pred.cpu(),
        "pred_flip": pred_flip.cpu(),
    }
