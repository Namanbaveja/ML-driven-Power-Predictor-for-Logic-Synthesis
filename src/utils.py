# aig_qor/utils.py

import math
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch
from .dataset import denormalize_log_qors


class OnePercentLoss(nn.Module):
    def __init__(self, margin=0.01, eps=1e-8):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, preds, targets):
        rel_err = torch.abs(preds - targets) / (targets + self.eps)
        penalty = torch.where(
            rel_err > self.margin,
            (rel_err - self.margin) ** 2,
            torch.zeros_like(rel_err)
        )
        return penalty.mean()


def estimate_design_size(batch: Batch) -> float:
    if batch.num_graphs > 0:
        return batch.num_nodes / batch.num_graphs
    return batch.num_nodes


def get_beta(design_size: float, base_beta: float = 0.3) -> float:
    return base_beta * (1 / math.log1p(design_size))


def predict_qor(model, aig_graph, recipe_str, global_std, global_mean,
                vocab, max_recipe_len=5, device=torch.device('cuda')):
    tokens = [t.strip() for t in recipe_str.split(';') if t.strip()]
    tokens = tokens[:max_recipe_len]
    while len(tokens) < max_recipe_len:
        tokens.append('')
    unk_idx = len(vocab)
    token_indices = [vocab.get(tok, unk_idx) for tok in tokens]
    recipe_tokens = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)

    batch_graph = Batch.from_data_list([aig_graph])
    model.to(device)
    batch_graph = batch_graph.to(device)
    recipe_tokens = recipe_tokens.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(batch_graph, recipe_tokens)
    preds = preds.squeeze(0).squeeze(-1)

    denorm_preds = denormalize_log_qors(preds.cpu(), global_mean, global_std).tolist()
    return denorm_preds


def predict_design_qor(model, design_name, recipe_str,
                       global_std, global_mean, vocab,
                       aigs_dir="aigs/", max_recipe_len=20,
                       device=torch.device('cuda')):
    import os
    graph_path = os.path.join('graph_cache', design_name + '.pt')
    aig_graph = torch.load(graph_path, weights_only=False)
    preds = predict_qor(model, aig_graph, recipe_str, global_std, global_mean, vocab,
                        max_recipe_len, device)
    return preds


def relative_error_loss(preds, targets, eps=1e-8):
    return torch.mean(torch.abs((preds - targets) / (targets + eps)))


def soft_accuracy_loss(pred, target, margin=0.05, alpha=20.0):
    rel_error = torch.abs(pred - target) / (target + 1e-8)
    soft_acc = torch.sigmoid(-alpha * (rel_error - margin))
    return 1.0 - soft_acc.mean()
