#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
train_pathway266_split_sce_nocluster_nanfix.py

MorphGAT pathway266 training script with paper-oriented slide-level split.

Key changes compared with the previous train.py:
1. Uses slide-level train / val / test split instead of simple random_split.
2. Supports explicit split_csv, or automatic priority split:
   train: most pan-cancer samples
   val:   part of DLPFC + breast cancer + a small part of auxiliary samples
   test:  held-out DLPFC + breast cancer gold/priority samples
3. Defaults to Graph_pt_pathway266 and auto-detects input dimension, e.g. 266.
4. Uses SCE reconstruction loss for pathway activity vectors.
5. Adds image-edge-weighted smoothness loss based on data.edge_attr.
6. Removes DEC / clustering loss entirely. The training objective is:
   L = L_contrastive + lambda_rec * L_SCE + lambda_smooth * L_img_smooth.
7. Saves dataset_split.csv for reproducibility.
================================================================================
"""

import os
from datetime import datetime
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"

import argparse
import glob
import random
import re
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, to_dense_adj, to_dense_batch
from torch_geometric.nn.inits import glorot, zeros
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from corrupted_graph import MorphologicalDropEdge

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ==========================================
# 0. Utilities
# ==========================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("MorphGAT_pathway266")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sanitize_tensor(x, nan=0.0, posinf=0.0, neginf=0.0):
    if x is None:
        return None
    x = x.float()
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def sanitize_graph(data):
    """Remove NaN/Inf from graph fields before augmentation/training.
    This is intentionally conservative: invalid pathway scores become 0,
    invalid morphology weights become small positive values.
    """
    data.x = sanitize_tensor(data.x, nan=0.0, posinf=0.0, neginf=0.0)
    if data.edge_attr is None:
        data.edge_attr = torch.ones((data.edge_index.shape[1], 1), dtype=torch.float32)
    else:
        ea = sanitize_tensor(data.edge_attr, nan=1e-4, posinf=1.0, neginf=1e-4)
        data.edge_attr = torch.clamp(ea, min=1e-4, max=1.0)
    return data


def slide_id_from_path(path):
    return os.path.basename(path).replace(".pt", "")


def infer_slide_group(slide_id):
    """Heuristic group labels for automatic split."""
    s = str(slide_id)
    low = s.lower()

    # LIBD DLPFC commonly uses 1515xx slices.
    if re.match(r"^1515\d+", s) or "dlpfc" in low:
        return "DLPFC"

    # Breast cancer datasets / common naming.
    breast_keys = ["breast", "brca", "idc", "dcis", "her2", "bcdc", "bas1", "bas_", "st-cnts"]
    if any(k in low for k in breast_keys):
        return "BREAST"

    return "AUX"


def sample_indices(indices, n, seed):
    indices = list(indices)
    if n <= 0 or len(indices) == 0:
        return []
    n = min(n, len(indices))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:n]


def build_dataset_splits(dataset, args, logger):
    """
    Build slide-level train/val/test split.

    Option 1: Use split_csv with columns: slide_id, split.
              split should be one of train/val/test.
    Option 2: Auto priority split:
              test and val are sampled from DLPFC+BREAST priority slides,
              with a small AUX validation set. Other slides remain training.
    """
    records = []
    for idx, path in enumerate(dataset.file_list):
        sid = slide_id_from_path(path)
        records.append({
            "idx": idx,
            "slide_id": sid,
            "file_path": path,
            "group": infer_slide_group(sid),
            "split": "train",
        })

    df = pd.DataFrame(records)

    if args.split_csv and os.path.exists(args.split_csv):
        split_df = pd.read_csv(args.split_csv)
        if "slide_id" not in split_df.columns or "split" not in split_df.columns:
            raise ValueError("split_csv must contain columns: slide_id, split")

        # Normalize IDs so both "151507" and "151507.pt" can be matched.
        split_df["slide_id"] = (
            split_df["slide_id"]
            .astype(str)
            .map(lambda x: os.path.basename(x).replace(".pt", ""))
        )
        split_df["split"] = (
            split_df["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"valid": "val", "validation": "val"})
        )

        if split_df["slide_id"].duplicated().any():
            duplicated = split_df.loc[
                split_df["slide_id"].duplicated(keep=False), "slide_id"
            ].tolist()
            raise ValueError(f"Duplicate slide_id values in split_csv: {duplicated}")

        allowed = {"train", "val", "test"}
        bad = sorted(set(split_df["split"]) - allowed)
        if bad:
            raise ValueError(f"Unsupported split labels in split_csv: {bad}")

        split_map = dict(zip(split_df["slide_id"], split_df["split"]))
        df["split"] = df["slide_id"].astype(str).map(split_map).fillna("train")

        # IMPORTANT: use the group labels explicitly supplied in split_csv.
        # This prevents DLPFC 151669-151676 from being mislabeled as AUX,
        # and separates BREAST_GOLD from BREAST_UNLAB in the log.
        if "group" in split_df.columns:
            split_df["group"] = split_df["group"].astype(str).str.strip()
            group_map = dict(zip(split_df["slide_id"], split_df["group"]))
            csv_group = df["slide_id"].astype(str).map(group_map)
            matched = csv_group.notna()
            df.loc[matched, "group"] = csv_group.loc[matched]

        missing_in_csv = sorted(set(df["slide_id"].astype(str)) - set(split_df["slide_id"]))
        extra_in_csv = sorted(set(split_df["slide_id"]) - set(df["slide_id"].astype(str)))
        if missing_in_csv:
            logger.warning(
                f"⚠️ {len(missing_in_csv)} graph(s) are absent from split_csv and default to train: "
                f"{missing_in_csv[:10]}"
            )
        if extra_in_csv:
            logger.warning(
                f"⚠️ {len(extra_in_csv)} split_csv row(s) have no matching .pt graph: "
                f"{extra_in_csv[:10]}"
            )

        logger.info(f"📌 Using user-provided split_csv: {args.split_csv}")
    else:
        # Auto split.
        rng = random.Random(args.seed)
        df["split"] = "train"
        priority_idx = df.index[df["group"].isin(["DLPFC", "BREAST"])].tolist()
        aux_idx = df.index[df["group"].eq("AUX")].tolist()
        rng.shuffle(priority_idx)
        rng.shuffle(aux_idx)

        if len(priority_idx) > 0:
            n_test = max(1, int(round(len(priority_idx) * args.test_gold_frac)))
            test_idx = priority_idx[:n_test]
            remain_priority = priority_idx[n_test:]

            n_val_gold = max(1, int(round(len(remain_priority) * args.val_gold_frac))) if len(remain_priority) > 0 else 0
            val_gold_idx = remain_priority[:n_val_gold]
        else:
            # If no priority slides are detected, fallback to ordinary 70/15/15 slide split.
            all_idx = df.index.tolist()
            rng.shuffle(all_idx)
            n_test = max(1, int(round(len(all_idx) * 0.15)))
            n_val = max(1, int(round(len(all_idx) * 0.15)))
            test_idx = all_idx[:n_test]
            val_gold_idx = all_idx[n_test:n_test + n_val]

        n_val_aux = int(round(len(aux_idx) * args.val_aux_frac))
        val_aux_idx = aux_idx[:n_val_aux]

        df.loc[test_idx, "split"] = "test"
        df.loc[val_gold_idx + val_aux_idx, "split"] = "val"
        logger.info("📌 Using automatic priority split: train=pan-cancer majority, val/test=DLPFC+BREAST priority.")

    # Safety: make sure every split exists. If val/test missing, create from train.
    if (df["split"] == "val").sum() == 0:
        train_candidates = df.index[df["split"].eq("train")].tolist()
        val_take = sample_indices(train_candidates, max(1, int(0.1 * len(df))), args.seed + 11)
        df.loc[val_take, "split"] = "val"
    if (df["split"] == "test").sum() == 0:
        train_candidates = df.index[df["split"].eq("train")].tolist()
        test_take = sample_indices(train_candidates, max(1, int(0.1 * len(df))), args.seed + 22)
        df.loc[test_take, "split"] = "test"

    split_path = os.path.join(args.save_dir, "dataset_split.csv")
    df.to_csv(split_path, index=False)

    logger.info("📊 Dataset split summary:")
    logger.info("\n" + pd.crosstab(df["group"], df["split"]).to_string())
    logger.info(f"🧾 Split file saved to: {split_path}")

    train_idx = df.loc[df["split"].eq("train"), "idx"].tolist()
    val_idx = df.loc[df["split"].eq("val"), "idx"].tolist()
    test_idx = df.loc[df["split"].eq("test"), "idx"].tolist()

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx), df


# ==========================================
# 1. Augmentation and Dataset
# ==========================================
def apply_feature_masking(x, drop_prob=0.2):
    # Mask node-wise pathway vectors. This matches the previous implementation.
    x = sanitize_tensor(x, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.rand(x.size(0), device=x.device) > drop_prob
    x_masked = x.clone()
    x_masked[~mask] = 0.0
    return sanitize_tensor(x_masked, nan=0.0, posinf=0.0, neginf=0.0)


class ContrastiveGraphDataset(Dataset):
    def __init__(self, root_dir, p_overall=0.4, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.augmentor = MorphologicalDropEdge(p_overall=p_overall)
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .pt graph files found in {root_dir}")
        super().__init__(root_dir, transform, pre_transform)

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        data_orig = safe_torch_load(self.file_list[idx], map_location="cpu")
        data_orig = sanitize_graph(data_orig)
        if data_orig.x.shape[1] < 2:
            raise ValueError(f"Dirty graph: {self.file_list[idx]}, feature_dim={data_orig.x.shape[1]}")

        data_corr = self.augmentor(data_orig)
        data_corr = sanitize_graph(data_corr)

        # Preserve clean target for reconstruction and similarity calculation.
        clean_x = sanitize_tensor(data_orig.x.clone(), nan=0.0, posinf=0.0, neginf=0.0)
        data_orig.clean_x = clean_x.clone()
        data_corr.clean_x = clean_x.clone()

        data_orig.x = apply_feature_masking(data_orig.x, drop_prob=0.1)
        data_corr.x = apply_feature_masking(data_corr.x, drop_prob=0.2)
        data_orig = sanitize_graph(data_orig)
        data_corr = sanitize_graph(data_corr)
        return data_orig, data_corr


# ==========================================
# 2. Model
# ==========================================
class PathwayFeatureAttention(nn.Module):
    """SENet-style pathway feature attention for pathway activity vectors."""
    def __init__(self, in_features=266, reduction_ratio=2):
        super().__init__()
        mid_features = max(in_features // reduction_ratio, 16)
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = sanitize_tensor(x, nan=0.0, posinf=0.0, neginf=0.0)
        pathway_weights = self.attention_mlp(x)
        pathway_weights = sanitize_tensor(pathway_weights, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return x * pathway_weights, pathway_weights


class MorphGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        super().__init__(node_dim=0, aggr='add')
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        self.bias_lambda = nn.Parameter(torch.tensor(5.0))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        out = self.propagate(edge_index, x=x, alpha_src=alpha_src, alpha_dst=alpha_dst, edge_attr=edge_attr)
        out = out.view(-1, H * C) if self.concat else out.mean(dim=1)
        return out + self.bias

    def message(self, x_j, alpha_src_i, alpha_dst_j, edge_attr, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_src_i + alpha_dst_j, 0.2)
        if edge_attr is not None:
            safe_edge_attr = sanitize_tensor(edge_attr.view(-1, 1), nan=1e-4, posinf=1.0, neginf=1e-4)
            safe_edge_attr = torch.clamp(safe_edge_attr, min=1e-4, max=1.0)
            alpha = alpha + self.bias_lambda * torch.log(safe_edge_attr)
        alpha = sanitize_tensor(alpha, nan=0.0, posinf=20.0, neginf=-20.0)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = sanitize_tensor(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        return x_j * F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)


class GCLModel_Morph(nn.Module):
    def __init__(self, in_channels=266, hidden_channels=128, out_channels=64):
        super().__init__()
        self.heads1 = 8
        self.feature_attn = PathwayFeatureAttention(in_features=in_channels)

        self.gate1 = nn.Sequential(nn.Linear(hidden_channels * self.heads1, 1), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())

        self.feature_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU(),
        )

        self.conv1 = MorphGATConv(hidden_channels, hidden_channels, heads=self.heads1, concat=True)
        self.skip_proj = nn.Linear(hidden_channels, hidden_channels * self.heads1)

        self.raw_proj1 = nn.Linear(in_channels, hidden_channels * self.heads1)
        self.raw_proj2 = nn.Linear(in_channels, hidden_channels)

        self.conv2 = MorphGATConv(hidden_channels * self.heads1, hidden_channels, heads=1, concat=False)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, in_channels)
        )

        self.saved_pathway_weights = None

    def forward(self, x, edge_index, edge_attr):
        x = sanitize_tensor(x, nan=0.0, posinf=0.0, neginf=0.0)
        if edge_attr is not None:
            edge_attr = sanitize_tensor(edge_attr, nan=1e-4, posinf=1.0, neginf=1e-4).clamp(1e-4, 1.0)
        x_weighted, self.saved_pathway_weights = self.feature_attn(x)
        x_raw = x_weighted.clone()

        x_proj = self.feature_proj(x_weighted)
        x_in = x_proj

        conv1_out = F.elu(self.conv1(x_proj, edge_index, edge_attr)) + self.skip_proj(x_in)
        raw1 = self.raw_proj1(x_raw)
        g1 = self.gate1(conv1_out)
        x_inter = conv1_out * g1 + raw1 * (1.0 - g1)

        x_inter = F.dropout(x_inter, p=0.1, training=self.training)
        conv2_out = self.conv2(x_inter, edge_index, edge_attr)
        raw2 = self.raw_proj2(x_raw)
        g2 = self.gate2(conv2_out)
        node_emb = conv2_out * g2 + raw2 * (1.0 - g2)

        node_emb = sanitize_tensor(node_emb, nan=0.0, posinf=0.0, neginf=0.0)
        z = self.proj_head(node_emb)
        z = sanitize_tensor(z, nan=0.0, posinf=0.0, neginf=0.0)
        rec_x = self.decoder(z)
        rec_x = sanitize_tensor(rec_x, nan=0.0, posinf=0.0, neginf=0.0)

        return z, node_emb, rec_x


# ==========================================
# 3. Losses
# ==========================================
def spatial_contrastive_loss(z1, z2, edge_index, x_raw, batch, temperature=0.2, hard_weight=3.0):
    z1 = sanitize_tensor(z1, nan=0.0, posinf=0.0, neginf=0.0)
    z2 = sanitize_tensor(z2, nan=0.0, posinf=0.0, neginf=0.0)
    x_raw = sanitize_tensor(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
    temperature = max(float(temperature), 0.05)
    z1 = F.normalize(z1, dim=1, eps=1e-8)
    z2 = F.normalize(z2, dim=1, eps=1e-8)

    z1_b, mask = to_dense_batch(z1, batch)
    z2_b, _ = to_dense_batch(z2, batch)
    x_b, _ = to_dense_batch(x_raw, batch)
    adj_b = to_dense_adj(edge_index, batch=batch)

    batch_size, _, _ = z1_b.size()
    total_loss = 0.0
    total_nodes = 0

    for i in range(batch_size):
        valid = mask[i].bool()
        ni = valid.sum().item()
        if ni == 0:
            continue

        zi1 = z1_b[i, valid]
        zi2 = z2_b[i, valid]
        xi = x_b[i, valid]
        adj = adj_b[i][:ni, :ni]
        adj.fill_diagonal_(1.0)

        x_norm = F.normalize(xi, dim=1, eps=1e-8)
        raw_sim = torch.matmul(x_norm, x_norm.T)
        raw_sim = sanitize_tensor(raw_sim, nan=0.0, posinf=1.0, neginf=-1.0)
        neighbors_only = adj - torch.eye(ni, device=adj.device)
        valid_neighbor_sims = raw_sim[neighbors_only.bool()]

        if valid_neighbor_sims.numel() > 0:
            adaptive_thresh = torch.quantile(valid_neighbor_sims.float(), 0.10)
            hard_neg_mask = (neighbors_only * (raw_sim < adaptive_thresh)).float()
        else:
            hard_neg_mask = torch.zeros_like(adj)

        pos_mask = torch.clamp(adj - hard_neg_mask, min=0.0, max=1.0)
        logits = torch.matmul(zi1, zi2.T) / temperature
        logits = sanitize_tensor(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        sim = torch.exp(logits)
        easy_neg_mask = (adj == 0).float()

        pos_sim_sum = (sim * pos_mask).sum(dim=-1)
        easy_neg_sim_sum = (sim * easy_neg_mask).sum(dim=-1)
        hard_neg_sim_sum = (sim * hard_neg_mask * hard_weight).sum(dim=-1)
        denom = pos_sim_sum + easy_neg_sim_sum + hard_neg_sim_sum + 1e-8

        log_prob = torch.log(sim / denom.unsqueeze(-1) + 1e-8)
        loss_i = - (pos_mask * log_prob).sum(dim=-1) / (pos_mask.sum(dim=-1) + 1e-8)
        loss_i = sanitize_tensor(loss_i, nan=0.0, posinf=0.0, neginf=0.0)

        total_loss += loss_i.sum()
        total_nodes += ni

    if total_nodes == 0:
        return torch.tensor(0.0, device=z1.device)
    return total_loss / total_nodes


def sce_reconstruction_loss(reconstructed_x, original_x, alpha=2.0):
    """Scaled cosine error. Better suited for standardized pathway activity vectors than plain MSE."""
    reconstructed_x = sanitize_tensor(reconstructed_x, nan=0.0, posinf=0.0, neginf=0.0)
    original_x = sanitize_tensor(original_x, nan=0.0, posinf=0.0, neginf=0.0)
    rec = F.normalize(reconstructed_x, p=2, dim=1, eps=1e-8)
    ori = F.normalize(original_x, p=2, dim=1, eps=1e-8)
    cos = (rec * ori).sum(dim=1).clamp(-1.0, 1.0)
    loss = (1.0 - cos).pow(alpha).mean()
    return sanitize_tensor(loss, nan=0.0, posinf=0.0, neginf=0.0)


def smooth_l1_reconstruction_loss(reconstructed_x, original_x):
    return F.smooth_l1_loss(reconstructed_x, original_x, beta=1.0)


def reconstruction_loss(reconstructed_x, original_x, loss_type="sce"):
    if loss_type == "mse":
        return F.mse_loss(reconstructed_x, original_x)
    if loss_type == "smooth_l1":
        return smooth_l1_reconstruction_loss(reconstructed_x, original_x)
    return sce_reconstruction_loss(reconstructed_x, original_x)


def edge_weighted_smoothness_loss(z, edge_index, edge_attr):
    """Encourage morphologically similar neighboring spots to have close embeddings."""
    z = sanitize_tensor(z, nan=0.0, posinf=0.0, neginf=0.0)
    if edge_index is None or edge_index.numel() == 0:
        return torch.tensor(0.0, device=z.device)
    row, col = edge_index
    z_norm = F.normalize(z, dim=1, eps=1e-8)
    diff = (z_norm[row] - z_norm[col]).pow(2).sum(dim=1)
    diff = sanitize_tensor(diff, nan=0.0, posinf=0.0, neginf=0.0)
    if edge_attr is None:
        w = torch.ones_like(diff)
    else:
        w = sanitize_tensor(edge_attr.view(-1).to(z.device), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    denom = w.sum() + 1e-8
    loss = (w * diff).sum() / denom
    return sanitize_tensor(loss, nan=0.0, posinf=0.0, neginf=0.0)


def compute_losses(model, b_orig, b_corr, args, device):
    z1, _, rec_x1 = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
    z2, _, _ = model(b_corr.x, b_corr.edge_index, b_corr.edge_attr)

    batch_info = b_orig.batch if hasattr(b_orig, 'batch') else torch.zeros(b_orig.x.size(0), dtype=torch.long, device=device)
    clean_x = b_orig.clean_x if hasattr(b_orig, "clean_x") else b_orig.x

    loss_cl = spatial_contrastive_loss(
        z1, z2, b_orig.edge_index, clean_x, batch_info,
        temperature=args.temp,
        hard_weight=args.hard_weight,
    )
    loss_rec = reconstruction_loss(rec_x1, clean_x, loss_type=args.rec_loss)
    loss_smooth = edge_weighted_smoothness_loss(z1, b_orig.edge_index, b_orig.edge_attr)

    total = loss_cl + args.lambda_rec * loss_rec + args.lambda_smooth * loss_smooth
    total = sanitize_tensor(total, nan=0.0, posinf=0.0, neginf=0.0)
    return total, {
        "cl": float(sanitize_tensor(loss_cl.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)),
        "rec": float(sanitize_tensor(loss_rec.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)),
        "smooth": float(sanitize_tensor(loss_smooth.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)),
    }


# ==========================================
# 4. Training engine
# ==========================================
def train(args):
    # Do not overwrite CUDA_VISIBLE_DEVICES inside the script.
    # If CUDA_VISIBLE_DEVICES is set before launching, --gpu is the logical
    # index among visible GPUs. Otherwise, --gpu is the system CUDA index.
    set_seed(args.seed)
    logger = setup_logger(args.save_dir)
    logger.info("🚀 Initialize MorphGAT pathway266 training: CL + SCE + image-edge smoothness, no DEC.")
    logger.info(f"Data dir: {args.data_dir}")

    full_dataset = ContrastiveGraphDataset(args.data_dir, p_overall=args.dropedge_p)
    logger.info(f"📊 Loaded {len(full_dataset)} graph samples.")

    train_ds, val_ds, test_ds, split_df = build_dataset_splits(full_dataset, args, logger)
    logger.info(f"Train graphs={len(train_ds)}, Val graphs={len(val_ds)}, Test graphs={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    # test_loader is not used in self-supervised training, but split is saved for final ARI evaluation.

    if args.in_dim is None:
        sample_pt = full_dataset.file_list[0]
        tmp = safe_torch_load(sample_pt, map_location="cpu")
        in_dim = int(tmp.x.shape[1])
        logger.info(f"✅ Auto-detected input dimension: {in_dim}")
    else:
        in_dim = args.in_dim

    if in_dim != 266:
        logger.warning(f"⚠️ Input dimension is {in_dim}, not 266. Check whether you are using Graph_pt_pathway266.")

    if torch.cuda.is_available():
        visible_count = torch.cuda.device_count()
        if args.gpu < 0 or args.gpu >= visible_count:
            raise ValueError(
                f"--gpu={args.gpu} is invalid. PyTorch currently sees {visible_count} GPU(s). "
                "When CUDA_VISIBLE_DEVICES is used, --gpu must be the logical index, usually 0."
            )
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(
            f"🖥️ Using {device} | {torch.cuda.get_device_name(args.gpu)} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}"
        )
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ CUDA is unavailable; training will run on CPU.")

    model = GCLModel_Morph(
        in_channels=in_dim,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
    ).to(device)

    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    history = {
        'pre_loss': [], 'pre_cl': [], 'pre_rec': [], 'pre_smooth': [],
        'fine_train_loss': [], 'fine_val_loss': [],
        'val_cl': [], 'val_rec': [], 'val_smooth': [],
    }
    use_amp = bool(args.amp and device.type == 'cuda')
    logger.info(f"AMP mixed precision: {use_amp}")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---------------------------------------------------------
    # Phase 1: Pretraining, no DEC
    # ---------------------------------------------------------
    logger.info(f"🔥 Phase 1: Pre-train representation ({args.pretrain_epochs} epochs).")
    optimizer_pre = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(args.warmup_epochs, max(1, args.pretrain_epochs - 1))
    sched_warmup = LinearLR(optimizer_pre, start_factor=0.1, total_iters=warmup_epochs)
    sched_cosine = CosineAnnealingLR(optimizer_pre, T_max=max(1, args.pretrain_epochs - warmup_epochs))
    scheduler_pre = SequentialLR(optimizer_pre, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])

    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0.0
        meter = {"cl": 0.0, "rec": 0.0, "smooth": 0.0}

        for b_orig, b_corr in train_loader:
            b_orig, b_corr = b_orig.to(device), b_corr.to(device)
            optimizer_pre.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, comp = compute_losses(model, b_orig, b_corr, args, device)

            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(optimizer_pre)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer_pre)
            scaler.update()

            total_loss += loss.item()
            for k in meter:
                meter[k] += comp[k]

        scheduler_pre.step()
        denom = max(1, len(train_loader))
        avg_loss = total_loss / denom
        history['pre_loss'].append(avg_loss)
        history['pre_cl'].append(meter["cl"] / denom)
        history['pre_rec'].append(meter["rec"] / denom)
        history['pre_smooth'].append(meter["smooth"] / denom)

        if epoch % args.log_every == 0:
            logger.info(
                f"Pre Epoch {epoch:03d} | Loss={avg_loss:.4f} | "
                f"CL={history['pre_cl'][-1]:.4f} REC={history['pre_rec'][-1]:.4f} "
                f"SM={history['pre_smooth'][-1]:.4f} LR={optimizer_pre.param_groups[0]['lr']:.6f}"
            )

    # ---------------------------------------------------------
    # Phase 2: Fine-tuning without DEC / clustering loss
    # ---------------------------------------------------------
    logger.info(f"🚀 Phase 2: Fine-tune without DEC ({args.epochs} epochs).")
    optimizer_fine = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
    scheduler_fine = CosineAnnealingLR(optimizer_fine, T_max=args.epochs)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for b_orig, b_corr in train_loader:
            b_orig, b_corr = b_orig.to(device), b_corr.to(device)
            optimizer_fine.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, _ = compute_losses(model, b_orig, b_corr, args, device)

            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(optimizer_fine)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer_fine)
            scaler.update()
            total_train_loss += loss.item()

        scheduler_fine.step()
        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_meter = {"cl": 0.0, "rec": 0.0, "smooth": 0.0}
        with torch.no_grad():
            for b_orig, b_corr in val_loader:
                b_orig, b_corr = b_orig.to(device), b_corr.to(device)
                loss, comp = compute_losses(model, b_orig, b_corr, args, device)
                total_val_loss += loss.item()
                for k in val_meter:
                    val_meter[k] += comp[k]

        denom_val = max(1, len(val_loader))
        avg_val_loss = total_val_loss / denom_val
        history['fine_train_loss'].append(avg_train_loss)
        history['fine_val_loss'].append(avg_val_loss)
        history['val_cl'].append(val_meter["cl"] / denom_val)
        history['val_rec'].append(val_meter["rec"] / denom_val)
        history['val_smooth'].append(val_meter["smooth"] / denom_val)

        if epoch % args.log_every == 0:
            logger.info(
                f"Fine Epoch {epoch:03d} | Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | "
                f"CL={history['val_cl'][-1]:.4f} REC={history['val_rec'][-1]:.4f} "
                f"SM={history['val_smooth'][-1]:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer_fine.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'scaler': scaler.state_dict(),
                'in_dim': in_dim,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.warning(f"✋ Early stopping at fine epoch {epoch}. Best val loss={best_val_loss:.4f}")
            break

    # Save final model too
    torch.save({
        'model': model.state_dict(),
        'in_dim': in_dim,
        'args': vars(args),
    }, os.path.join(args.save_dir, "last_model.pth"))

    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(history['pre_loss'])), history['pre_loss'], label='Phase 1: Pre-train Loss')
    offset = len(history['pre_loss'])
    plt.plot(range(offset, offset + len(history['fine_val_loss'])), history['fine_val_loss'], label='Phase 3: Fine-tune Val Loss')
    plt.title('Training Pipeline: priority split + contrastive + SCE + image-edge smoothness')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_pipeline.png'), dpi=300)
    plt.close()

    pd.DataFrame({k: pd.Series(v) for k, v in history.items()}).to_csv(os.path.join(args.save_dir, "loss_history.csv"), index=False)
    logger.info(f"✅ Training finished. Best model saved to {os.path.join(args.save_dir, 'best_model.pth')}")
    logger.info(f"🧪 Held-out test slides are recorded in {os.path.join(args.save_dir, 'dataset_split.csv')}; use them only for final ARI evaluation.")


# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MorphGAT Pathway266: CL + SCE + image-edge smoothness, no DEC")

    parser.add_argument("--data_dir", type=str,
                        default="/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt_pathway266")
    parser.add_argument("--save_dir", type=str, default="checkpoints_pathway266_priority")
    parser.add_argument("--split_csv", type=str, default=None,
                        help="Optional CSV with columns slide_id,split. split in {train,val,test}.")

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help=(
            "CUDA device index. Without CUDA_VISIBLE_DEVICES, use the system index "
            "(e.g. --gpu 2). With CUDA_VISIBLE_DEVICES=2, use the logical index --gpu 0."
        ),
    )
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision. Default is disabled to avoid NaN on unstable graphs.")
    parser.add_argument("--seed", type=int, default=42)

    # Priority split parameters used when split_csv is not provided.
    parser.add_argument("--test_gold_frac", type=float, default=0.30,
                        help="Fraction of DLPFC+BREAST priority slides held out as final test.")
    parser.add_argument("--val_gold_frac", type=float, default=0.25,
                        help="Fraction of remaining DLPFC+BREAST priority slides used as validation.")
    parser.add_argument("--val_aux_frac", type=float, default=0.08,
                        help="Fraction of AUX pan-cancer slides used as validation.")

    # Training
    parser.add_argument("--pretrain_epochs", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Default 1 is safer because contrastive loss builds dense adjacency per graph.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=5)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--in_dim", type=int, default=None)

    # Loss
    parser.add_argument("--temp", type=float, default=0.15)
    parser.add_argument("--hard_weight", type=float, default=0.5)
    parser.add_argument("--rec_loss", type=str, default="sce", choices=["sce", "mse", "smooth_l1"])
    parser.add_argument("--lambda_rec", type=float, default=0.5)
    parser.add_argument("--lambda_smooth", type=float, default=0.02)
    parser.add_argument("--dropedge_p", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=1.0)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
