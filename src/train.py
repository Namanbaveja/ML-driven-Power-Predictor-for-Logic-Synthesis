# aig_qor/train.py
import os
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from torch_geometric.data import Batch

from .utils import OnePercentLoss, estimate_design_size
from torch_geometric.data import Data

def _default_collate_train(batch):
    Gs, recs, tgts = zip(*[(b[0], b[1], b[2]) for b in batch])
    return Batch.from_data_list(Gs), torch.stack(recs), torch.stack(tgts)


def train_model(
    train_dataset,
    model: nn.Module,
    num_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    save_path: str = "models/best_model.pth",
):
    """
    Train the CombinedModel on train_dataset.

    Args:
        train_dataset: torch Dataset yielding (graph, recipe_tokens, norm_qor_seq, name)
        model: nn.Module (CombinedModel)
        num_epochs: number of epochs
        batch_size: batch size
        lr: learning rate
        device: torch.device; if None, uses cpu
        save_path: where to save the best model
    """
    device = device or torch.device("cpu")

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_default_collate_train)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_1pct = OnePercentLoss()
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        total = 0.0
        model.train()
        current_beta = None
        for G, rec, tgt in loader:
            G, rec, tgt = G.to(device), rec.to(device), tgt.to(device)
            optimizer.zero_grad()
            preds = model(G, rec).squeeze(-1)  # shape: [B, L]
            B, L = preds.shape

            design_size = estimate_design_size(G)
            current_beta = 0.3 * (1.0 / math.log1p(design_size))
            t = torch.arange(1, L + 1, dtype=torch.float, device=device)
            weights = torch.exp(current_beta * t)
            weights = weights / weights.sum()

            loss_1pct = criterion_1pct(preds, tgt)
            per_step_mse = ((preds - tgt) ** 2).mean(dim=0)
            loss = (weights * per_step_mse).sum() + 5.0 * loss_1pct

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total += loss.item() * B

        avg_loss = total / len(train_dataset) if len(train_dataset) > 0 else float("nan")
        print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Avg Beta: {current_beta:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to '{save_path}' at epoch {epoch} (Loss: {best_loss:.4f})")

    return model


def _collate_for_active(batch):
    Gs, recs, tgts, _ = zip(*batch)
    return Batch.from_data_list(Gs), torch.stack(recs), torch.stack(tgts)


def fine_tune_active(
    model: nn.Module,
    dataset,
    device: Optional[torch.device],
    n_shots: int = 50,
    num_epochs: int = 10,
    lr_fc: float = 1e-3,
    batch_size: int = 8,
    freeze_backbone: bool = True,
    use_mse_head: bool = False,
    save_path: str = "models/best_finetuned_active.pth",
):
    """
    Perform active-sampling fine-tuning:
      - select representative shots with KMeans over token embeddings
      - fine-tune classifier/regression head on subset

    Args:
        model: CombinedModel instance
        dataset: AIGDataset (or similar) which supports __getitem__ returning (G, rec_tokens, tgt_seq, name)
        device: torch.device
        n_shots: number of clusters / shots to select
        num_epochs: epochs to run on the selected subset
        lr_fc: learning rate for optimizer
        batch_size: fine-tune batch size
        freeze_backbone: whether to freeze all parameters except qor_head
        use_mse_head: if True, use MSE on final step; otherwise sequence-weighted loss
        save_path: file path to save best fine-tuned model
    """
    device = device or torch.device("cpu")
    model.to(device)
    model.eval()

    # Build embeddings from token stream using model token_embedding + pos_encoder
    embeddings = []
    for idx in range(len(dataset)):
        G, rec_tokens, _, _ = dataset[idx]
        with torch.no_grad():
            rec = rec_tokens.unsqueeze(0).to(device)  # [1, L]
            tok = model.token_embedding(rec)          # [1, L, D]
            tok = model.pos_encoder(tok)
            emb = tok.mean(dim=1).squeeze(0)          # [D]
        embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)
    kmeans = KMeans(n_clusters=min(n_shots, len(dataset)), n_init="auto", random_state=0)
    centers = kmeans.fit(embeddings).cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centers, embeddings)
    shot_idxs = sorted(set(closest.tolist()))
    print(f"[Active FT] Selected {len(shot_idxs)} shots out of {len(dataset)}")

    subset = Subset(dataset, shot_idxs)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=_collate_for_active)

    # Freeze backbone parameters except qor_head if requested
    if freeze_backbone:
        for name, p in model.named_parameters():
            if "qor_head" not in name:
                p.requires_grad = False

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_fc)
    mse_loss = nn.MSELoss()
    onepct = OnePercentLoss()
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total = 0.0
        for G, rec, tgt in loader:
            G, rec, tgt = G.to(device), rec.to(device), tgt.to(device)
            optimizer.zero_grad()
            preds = model(G, rec).squeeze(-1)
            B, L = preds.shape

            if use_mse_head:
                loss = mse_loss(preds[:, -1], tgt[:, -1])
            else:
                beta = 0.3 / math.log1p(estimate_design_size(G))
                t = torch.arange(1, L + 1, device=device, dtype=torch.float)
                w = torch.exp(beta * t)
                w = w / w.sum()
                seq_mse = ((preds - tgt) ** 2).mean(dim=0)
                loss_seq = (w * seq_mse).sum()
                loss1 = onepct(preds, tgt)
                loss = loss_seq + 3.0 * loss1

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * B

        avg = total / len(subset) if len(subset) > 0 else float("nan")
        print(f"[Active FT] Epoch {epoch}/{num_epochs}  Loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ Saved best model @ epoch {epoch}")

    return model
