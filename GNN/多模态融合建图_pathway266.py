#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================================
Multi-modal Graph Construction - selected 266 pathway version

Purpose:
    Build PyG graphs for image-paired spatial transcriptomics data.

Node feature:
    selected pathway activity scores from HALLMARK + KEGG + C8 signatures.

Edge index:
    spatial adjacency graph.

Edge attribute:
    histology patch feature cosine similarity converted to a positive morphology weight.

Important changes compared with the previous version:
    - Reads selected_pathway_names.txt and filters the pathway-gene net to selected pathways.
    - Uses a dedicated graph output directory Graph_pt_pathway266 to avoid mixing with old .pt files.
    - Reads image barcode metadata if available and checks strict alignment.
    - Vectorized edge-weight calculation for speed.
    - More robust sparse/dense matrix handling and spatial-neighbor fallback.
=============================================================================================
"""

import os
import gc
import json
import traceback

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import decoupler as dc
import squidpy as sq
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm


# ==========================================
# 1. Global configuration
# ==========================================
PROJECT_ROOT = "/data/home/wangzz_group/zhaipengyuan/BEPH-main"
KZ_ROOT = os.path.join(PROJECT_ROOT, "DATA_DIRECTORY", "kz_data")

H5AD_DIR = os.path.join(KZ_ROOT, "Raw_Data", "h5ad_files")
IMG_FEAT_DIR = os.path.join(KZ_ROOT, "Graph_pt")

# Save new 266-pathway graphs into a new directory to avoid reusing old 233-dim graphs.
SAVE_DIR = os.path.join(KZ_ROOT, "Graph_pt_pathway266")
PROCESS_CSV = os.path.join(PROJECT_ROOT, "DATA_DIRECTORY", "process_list.csv")

NET_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "DATA_DIRECTORY", "Pathway", "pancancer_microenvironment_net_266.csv"),
    os.path.join(PROJECT_ROOT, "DATA_DIRECTORY", "Pathway", "pancancer_microenvironment_net.csv"),
    os.path.join(PROJECT_ROOT, "Pathway", "pancancer_microenvironment_net_266.csv"),
    os.path.join(PROJECT_ROOT, "Pathway", "pancancer_microenvironment_net.csv"),
]

SELECTED_PATHWAY_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "DATA_DIRECTORY", "Pathway", "selected_pathway_names.txt"),
    os.path.join(PROJECT_ROOT, "Pathway", "selected_pathway_names.txt"),
    os.path.join(PROJECT_ROOT, "selected_pathway_names.txt"),
    "selected_pathway_names.txt",
]

MIN_VALID_PATHWAYS_ABS = 150
MIN_VALID_PATHWAYS_FRAC = 0.55
REBUILD_EXISTING = False

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# 2. Utilities
# ==========================================
def safe_x_max(adata):
    x = adata.X
    if sparse.issparse(x):
        if x.nnz == 0:
            return 0.0
        return float(x.max())
    return float(np.nanmax(x))


def preprocess_adata_for_pathway(adata):
    adata.var_names_make_unique()

    if "spatial" not in adata.obsm.keys():
        if "X_spatial" in adata.obsm.keys():
            adata.obsm["spatial"] = adata.obsm["X_spatial"]
        else:
            raise ValueError("No spatial coordinates found: neither obsm['spatial'] nor obsm['X_spatial'] exists.")

    # Keep this QC rule identical to the image feature extraction script.
    if safe_x_max(adata) > 50:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata


def find_existing_path(candidates, required=True, name="file"):
    for p in candidates:
        if os.path.exists(p):
            return p
    if required:
        raise FileNotFoundError(f"Cannot find {name}. Tried: {candidates}")
    return None


def read_selected_pathways(path):
    if path is None:
        return None

    selected = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                selected.append(s)

    # remove duplicates while keeping order
    seen = set()
    out = []
    for s in selected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def prepare_pathway_net(net_path, selected_pathway_path=None):
    net = pd.read_csv(net_path)
    required_cols = {"source", "target"}
    missing = required_cols - set(net.columns)
    if missing:
        raise ValueError(f"Pathway net missing required columns: {missing}; columns={list(net.columns)}")

    net["source"] = net["source"].astype(str)
    net["target"] = net["target"].astype(str)

    if "weight" not in net.columns:
        net["weight"] = 1.0

    selected = read_selected_pathways(selected_pathway_path)
    if not selected:
        all_pathways = sorted(net["source"].unique().tolist())
        print(f"[-] No selected_pathway_names.txt found. Use all pathways in net: {len(all_pathways)}")
        return net, all_pathways, "all_net"

    net_sources = set(net["source"].unique().tolist())
    selected_full = selected
    selected_tail = [s.split("__", 1)[1] if "__" in s else s for s in selected]

    full_hits = [s for s in selected_full if s in net_sources]
    tail_hits = [s for s in selected_tail if s in net_sources]

    if len(full_hits) >= len(tail_hits) and len(full_hits) > 0:
        keep_sources = full_hits
        mode = "selected_full_name"
    elif len(tail_hits) > 0:
        keep_sources = tail_hits
        mode = "selected_tail_name"
    else:
        example_sources = sorted(list(net_sources))[:20]
        raise RuntimeError(
            "None of selected pathways matched net['source'].\n"
            f"Selected examples: {selected[:10]}\n"
            f"Net source examples: {example_sources}"
        )

    filtered = net[net["source"].isin(set(keep_sources))].copy()
    filtered["source"] = pd.Categorical(filtered["source"], categories=keep_sources, ordered=True)
    filtered = filtered.sort_values("source")
    filtered["source"] = filtered["source"].astype(str)

    print(f"[-] Selected pathway file: {selected_pathway_path}")
    print(f"[-] Selected pathways requested: {len(selected)}")
    print(f"[-] Matched pathways in net: {len(keep_sources)}")
    print(f"[-] Matched mode: {mode}")

    return filtered, keep_sources, mode


def build_spatial_edge_index(adata):
    try:
        sq.gr.spatial_neighbors(adata, n_rings=1, coord_type="grid", n_neighs=6)
    except Exception:
        # Some non-Visium datasets do not fit grid topology. Fall back to generic KNN.
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)

    adj_matrix = adata.obsp["spatial_connectivities"]
    edge_index_np = np.vstack(adj_matrix.nonzero()).astype(np.int64)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    return edge_index


def compute_morphology_edge_attr(edge_index, image_features):
    image_features = np.asarray(image_features, dtype=np.float32)
    norm = np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-8
    image_features = image_features / norm

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    cos_sim = np.sum(image_features[src] * image_features[dst], axis=1)

    # Original rule: exp(cos_sim) / exp(1.0) = exp(cos_sim - 1.0), range roughly [e^-2, 1].
    weights = np.exp(cos_sim - 1.0).astype(np.float32)
    edge_attr = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    return edge_attr


def read_image_barcodes(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ==========================================
# 3. Single-slide processing
# ==========================================
def process_single_slide(slide_id, h5ad_path, img_features_path, img_barcode_path, save_path, net, all_pathways):
    try:
        adata = sc.read_h5ad(h5ad_path)
        adata = preprocess_adata_for_pathway(adata)

        # Pathway activity branch
        weight_col = "weight" if "weight" in net.columns else None
        dc.run_mlm(
            mat=adata,
            net=net,
            source="source",
            target="target",
            weight=weight_col,
            use_raw=False,
        )

        pathway_scores_df = adata.obsm["mlm_estimate"]
        if not isinstance(pathway_scores_df, pd.DataFrame):
            pathway_scores_df = pd.DataFrame(
                pathway_scores_df,
                index=adata.obs_names,
                columns=adata.uns.get("mlm_estimate", {}).get("sources", None),
            )

        original_dim = pathway_scores_df.shape[1]
        target_dim = len(all_pathways)
        min_valid = max(MIN_VALID_PATHWAYS_ABS, int(target_dim * MIN_VALID_PATHWAYS_FRAC))

        if original_dim < min_valid:
            return False, f"QC failed: only {original_dim}/{target_dim} pathways available; min required {min_valid}."

        pathway_scores_df = pathway_scores_df.reindex(columns=all_pathways, fill_value=0.0)
        pathway_scores = pathway_scores_df.values.astype(np.float32)

        scaler = StandardScaler()
        scaled_pathways = scaler.fit_transform(pathway_scores).astype(np.float32)
        x_tensor = torch.tensor(scaled_pathways, dtype=torch.float32)

        # Spatial edge branch
        edge_index = build_spatial_edge_index(adata)

        # Image branch
        if not os.path.exists(img_features_path):
            return False, f"Missing image feature file: {os.path.basename(img_features_path)}"

        image_features = np.load(img_features_path)
        if image_features.shape[0] != x_tensor.shape[0]:
            return False, f"Feature count mismatch: image={image_features.shape[0]} vs gene_spots={x_tensor.shape[0]}. Re-extract image features."

        img_barcodes = read_image_barcodes(img_barcode_path)
        if img_barcodes is not None:
            gene_barcodes = adata.obs_names.astype(str).tolist()
            if img_barcodes != gene_barcodes:
                return False, "Barcode order mismatch between image features and h5ad after QC. Re-extract image features."

        edge_attr = compute_morphology_edge_attr(edge_index, image_features)

        graph_data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        graph_data.slide_id = str(slide_id)
        graph_data.barcodes = adata.obs_names.astype(str).tolist()
        graph_data.pathway_names = list(all_pathways)
        graph_data.image_feature_path = img_features_path
        graph_data.num_pathways = int(x_tensor.shape[1])
        graph_data.num_spots = int(x_tensor.shape[0])

        torch.save(graph_data, save_path)

        msg = (
            f"spots={x_tensor.shape[0]}, pathways={x_tensor.shape[1]}, "
            f"edges={edge_index.shape[1]}, image_dim={image_features.shape[1]}"
        )

        del adata, x_tensor, edge_index, edge_attr, image_features, graph_data
        gc.collect()
        return True, msg

    except Exception as e:
        return False, f"{str(e)}\n{traceback.format_exc()}"


# ==========================================
# 4. Batch processing
# ==========================================
def batch_build_graphs():
    print("=" * 70)
    print("🚀 Start multi-modal graph construction: selected pathway266 version")
    print("=" * 70)

    net_path = find_existing_path(NET_CANDIDATES, required=True, name="pathway net csv")
    selected_path = find_existing_path(SELECTED_PATHWAY_CANDIDATES, required=False, name="selected pathway names")

    print(f"[-] Pathway net: {net_path}")
    print(f"[-] Graph save dir: {SAVE_DIR}")
    print(f"[-] Image feature dir: {IMG_FEAT_DIR}")

    net, all_pathways, match_mode = prepare_pathway_net(net_path, selected_path)

    # Save actually used pathway names for reproducibility.
    used_pathway_txt = os.path.join(SAVE_DIR, "used_pathway_names.txt")
    with open(used_pathway_txt, "w") as f:
        for p in all_pathways:
            f.write(str(p) + "\n")

    config_json = os.path.join(SAVE_DIR, "graph_build_config.json")
    with open(config_json, "w") as f:
        json.dump({
            "net_path": net_path,
            "selected_pathway_path": selected_path,
            "matched_pathway_count": len(all_pathways),
            "match_mode": match_mode,
            "h5ad_dir": H5AD_DIR,
            "img_feat_dir": IMG_FEAT_DIR,
            "save_dir": SAVE_DIR,
            "min_valid_pathways_abs": MIN_VALID_PATHWAYS_ABS,
            "min_valid_pathways_frac": MIN_VALID_PATHWAYS_FRAC,
        }, f, indent=2)

    if os.path.exists(PROCESS_CSV):
        df = pd.read_csv(PROCESS_CSV)
        slide_ids = df["slide_id"].tolist() if "slide_id" in df.columns else df.iloc[:, 0].tolist()
        print(f"[-] Loaded {len(slide_ids)} slides from process_list.csv")
    else:
        slide_ids = [f.replace(".h5ad", "") for f in os.listdir(H5AD_DIR) if f.endswith(".h5ad")]
        print(f"[-] process_list.csv not found; loaded {len(slide_ids)} slides from h5ad dir")

    results = {"success": 0, "skip": 0, "fail": 0}
    pbar = tqdm(slide_ids, desc="Graph construction")

    for slide_id in pbar:
        slide_id = str(slide_id)
        h5ad_path = os.path.join(H5AD_DIR, f"{slide_id}.h5ad")
        img_features_path = os.path.join(IMG_FEAT_DIR, f"{slide_id}_image_features.npy")
        img_barcode_path = os.path.join(IMG_FEAT_DIR, f"{slide_id}_image_barcodes.txt")
        save_path = os.path.join(SAVE_DIR, f"{slide_id}.pt")

        if os.path.exists(save_path) and not REBUILD_EXISTING:
            results["skip"] += 1
            tqdm.write(f"⏩ [Skip] {slide_id:<25} | .pt exists")
            continue

        if not os.path.exists(h5ad_path):
            results["fail"] += 1
            tqdm.write(f"❌ [Fail] {slide_id:<25} | Missing h5ad")
            continue

        success, msg = process_single_slide(
            slide_id,
            h5ad_path,
            img_features_path,
            img_barcode_path,
            save_path,
            net,
            all_pathways,
        )

        if success:
            results["success"] += 1
            tqdm.write(f"✅ [OK] {slide_id:<25} | {msg}")
        else:
            results["fail"] += 1
            tqdm.write(f"❌ [Reject] {slide_id:<25} | {msg.splitlines()[0]}")

    print("\n" + "=" * 60)
    print("✨ Graph construction finished.")
    print(f"✅ Success: {results['success']}")
    print(f"⏩ Skipped: {results['skip']}")
    print(f"❌ Failed/rejected: {results['fail']}")
    print(f"📁 New pathway graphs saved in: {SAVE_DIR}")
    print(f"📄 Used pathway list: {used_pathway_txt}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    batch_build_graphs()
