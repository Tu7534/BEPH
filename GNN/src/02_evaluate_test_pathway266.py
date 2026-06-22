#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate MorphGAT pathway266 on held-out gold-standard test slides.

For each slide:
1. Load the graph and best checkpoint.
2. Extract one embedding vector per graph node/spot.
3. Align h5ad gold labels to graph node order by barcode.
4. Cluster the slide independently with KMeans.
5. Calculate ARI and NMI over repeated seeds.
6. Save embeddings, spot predictions, metrics, and spatial figures.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


INVALID_LABELS = {
    "", "nan", "none", "na", "unknown", "unlabeled", "undefined", "not applicable"
}


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def import_training_module(train_script):
    train_script = Path(train_script).resolve()
    sys.path.insert(0, str(train_script.parent))
    spec = importlib.util.spec_from_file_location("pathway266_training_module", train_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training script: {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(module, checkpoint, device):
    saved_args = checkpoint.get("args", {})
    in_dim = int(checkpoint.get("in_dim", saved_args.get("in_dim", 266) or 266))
    hidden_dim = int(saved_args.get("hidden_dim", 256))
    out_dim = int(saved_args.get("out_dim", 64))

    model = module.GCLModel_Morph(
        in_channels=in_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, {"in_dim": in_dim, "hidden_dim": hidden_dim, "out_dim": out_dim}


def get_graph_barcodes(graph):
    candidate_attrs = [
        "barcodes", "barcode", "spot_ids", "spot_id",
        "obs_names", "node_names", "cell_ids"
    ]
    for attr in candidate_attrs:
        if not hasattr(graph, attr):
            continue
        values = getattr(graph, attr)
        if values is None:
            continue
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        values = np.asarray(values).reshape(-1)
        if len(values) == graph.x.shape[0]:
            return [str(x) for x in values]
    raise AttributeError(
        "No graph barcode field was found. Expected one of: "
        + ", ".join(candidate_attrs)
    )


def normalize_barcode(value):
    text = str(value).strip()
    if text.endswith("-1"):
        text = text[:-2]
    return text


def align_h5ad_to_graph(adata, graph_barcodes, label_col):
    obs_names = pd.Index(adata.obs_names.astype(str))
    labels = adata.obs[label_col].copy()
    labels.index = obs_names

    exact = labels.reindex(graph_barcodes)
    exact_match_count = int(exact.notna().sum())

    # Use exact matching whenever it covers any labels. This avoids unnecessary
    # barcode normalization and preserves the original h5ad indexing.
    if exact_match_count > 0:
        return exact, "exact"

    # Fallback for a common 10x difference: one side includes "-1".
    norm_obs = pd.Index([normalize_barcode(x) for x in obs_names])
    if norm_obs.duplicated().any():
        raise ValueError(
            "Normalized h5ad barcodes are not unique; cannot safely align by stripped '-1'."
        )

    normalized_series = pd.Series(labels.to_numpy(), index=norm_obs)
    normalized_graph = [normalize_barcode(x) for x in graph_barcodes]
    aligned = normalized_series.reindex(normalized_graph)
    return aligned, "strip_-1"


def clean_gold_labels(series):
    result = series.astype("object")
    text = result.astype(str).str.strip()
    invalid = result.isna() | text.str.lower().isin(INVALID_LABELS)
    text[invalid] = np.nan
    return text


def get_spatial_coordinates(graph, adata, graph_barcodes):
    # Preferred: graph coordinates are already in graph node order.
    if hasattr(graph, "pos") and graph.pos is not None:
        pos = graph.pos
        if torch.is_tensor(pos):
            pos = pos.detach().cpu().numpy()
        pos = np.asarray(pos)
        if pos.ndim == 2 and pos.shape[0] == len(graph_barcodes) and pos.shape[1] >= 2:
            return pos[:, :2], "graph.pos"

    for key in ["spatial", "X_spatial"]:
        if key not in adata.obsm:
            continue
        coords = np.asarray(adata.obsm[key])
        coord_df = pd.DataFrame(
            coords[:, :2],
            index=adata.obs_names.astype(str),
            columns=["x", "y"],
        )
        aligned = coord_df.reindex(graph_barcodes)
        if not aligned.isna().any().any():
            return aligned.to_numpy(), f"adata.obsm['{key}'] exact"

        norm_index = pd.Index([normalize_barcode(x) for x in coord_df.index])
        if not norm_index.duplicated().any():
            coord_df.index = norm_index
            norm_barcodes = [normalize_barcode(x) for x in graph_barcodes]
            aligned = coord_df.reindex(norm_barcodes)
            if not aligned.isna().any().any():
                return aligned.to_numpy(), f"adata.obsm['{key}'] strip_-1"

    return None, "missing"


def extract_embedding(module, model, graph, device, representation):
    graph = module.sanitize_graph(graph)
    graph = graph.to(device)
    with torch.no_grad():
        z, node_emb, _ = model(graph.x, graph.edge_index, graph.edge_attr)
    output = z if representation == "z" else node_emb
    output = output.detach().cpu().numpy()
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def encode_categories(values):
    categories = pd.Categorical(pd.Series(values).astype(str))
    return categories.codes, list(categories.categories)


def save_categorical_spatial_plot(coords, labels, title, out_path):
    if coords is None:
        return

    codes, categories = encode_categories(labels)
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=9)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Spatial x")
    ax.set_ylabel("Spatial y")

    # A compact legend with one marker per category.
    handles = []
    for idx, category in enumerate(categories):
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=scatter.cmap(scatter.norm(idx)),
                markeredgecolor="none",
                label=str(category),
                markersize=6,
            )
        )
    ax.legend(
        handles=handles,
        title="Domain",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_kmeans(embedding, n_clusters, seeds):
    scaled = StandardScaler().fit_transform(embedding)
    outputs = {}
    for seed in seeds:
        model = KMeans(
            n_clusters=n_clusters,
            n_init=20,
            random_state=int(seed),
        )
        outputs[int(seed)] = model.fit_predict(scaled)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_script", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--representation", choices=["z", "node_emb"], default="z")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--plot_seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    embeddings_dir = out_dir / "embeddings"
    predictions_dir = out_dir / "predictions"
    figures_dir = out_dir / "figures"
    for folder in [metrics_dir, embeddings_dir, predictions_dir, figures_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu={args.gpu} is invalid; PyTorch sees "
                f"{torch.cuda.device_count()} GPU(s)."
            )
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using {device}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("CUDA unavailable; evaluation will run on CPU.")

    module = import_training_module(args.train_script)
    checkpoint = safe_torch_load(args.checkpoint, map_location="cpu")
    model, model_config = build_model(module, checkpoint, device)

    split_df = pd.read_csv(args.split_csv)
    split_df["slide_id"] = (
        split_df["slide_id"].astype(str)
        .map(lambda x: os.path.basename(x).replace(".pt", ""))
    )
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()
    test_df = split_df.loc[split_df["split"].eq("test")].copy()

    manifest = pd.read_csv(args.manifest)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.set_index("slide_id", drop=False)

    missing_manifest = sorted(set(test_df["slide_id"]) - set(manifest.index))
    if missing_manifest:
        raise ValueError(f"Test slides missing from manifest: {missing_manifest}")

    bad_manifest = manifest.loc[
        manifest.index.isin(test_df["slide_id"])
        & ~manifest["status"].astype(str).str.upper().eq("OK")
    ]
    if not bad_manifest.empty:
        raise ValueError(
            "Manifest contains non-OK test rows. Fix these rows first:\n"
            + bad_manifest[["slide_id", "status", "h5ad_path", "label_col"]].to_string(index=False)
        )

    seeds = list(range(args.n_runs))
    if args.plot_seed not in seeds:
        seeds.append(args.plot_seed)

    run_metric_rows = []
    slide_summary_rows = []
    skipped_rows = []

    for _, split_row in test_df.iterrows():
        slide_id = str(split_row["slide_id"])
        group = str(split_row.get("group", "UNKNOWN"))
        info = manifest.loc[slide_id]

        graph_path = Path(args.graph_dir) / f"{slide_id}.pt"
        h5ad_path = Path(str(info["h5ad_path"]))
        label_col = str(info["label_col"])

        print(f"\n[{slide_id}] group={group}")

        try:
            if not graph_path.exists():
                raise FileNotFoundError(f"Missing graph: {graph_path}")
            if not h5ad_path.exists():
                raise FileNotFoundError(f"Missing h5ad: {h5ad_path}")

            graph = safe_torch_load(graph_path, map_location="cpu")
            graph_barcodes = get_graph_barcodes(graph)
            embedding = extract_embedding(
                module, model, graph, device, args.representation
            )

            if embedding.shape[0] != len(graph_barcodes):
                raise ValueError(
                    f"Embedding rows ({embedding.shape[0]}) do not match "
                    f"barcode count ({len(graph_barcodes)})."
                )

            adata = sc.read_h5ad(h5ad_path)
            adata.obs_names = adata.obs_names.astype(str)
            if label_col not in adata.obs.columns:
                raise KeyError(
                    f"Label column '{label_col}' is absent from {h5ad_path}. "
                    f"Available columns include: {list(adata.obs.columns)[:30]}"
                )

            aligned_labels, barcode_mode = align_h5ad_to_graph(
                adata, graph_barcodes, label_col
            )
            gold_all = clean_gold_labels(aligned_labels)
            valid = gold_all.notna().to_numpy()

            n_valid = int(valid.sum())
            if n_valid < 2:
                raise ValueError("Fewer than two graph spots have valid gold labels.")

            gold = gold_all.iloc[np.flatnonzero(valid)].astype(str).to_numpy()
            embedding_valid = embedding[valid]
            n_clusters = int(pd.Series(gold).nunique())

            if n_clusters < 2:
                raise ValueError(f"Gold labels contain only {n_clusters} category.")
            if n_clusters >= n_valid:
                raise ValueError(
                    f"n_clusters={n_clusters} is not valid for n_spots={n_valid}."
                )

            pred_by_seed = run_kmeans(embedding_valid, n_clusters, seeds)

            for seed in range(args.n_runs):
                pred = pred_by_seed[seed]
                run_metric_rows.append({
                    "slide_id": slide_id,
                    "group": group,
                    "seed": seed,
                    "representation": args.representation,
                    "n_clusters": n_clusters,
                    "n_spots_total": len(graph_barcodes),
                    "n_spots_labeled": n_valid,
                    "barcode_alignment": barcode_mode,
                    "ARI": adjusted_rand_score(gold, pred),
                    "NMI": normalized_mutual_info_score(gold, pred),
                })

            slide_runs = pd.DataFrame(
                [x for x in run_metric_rows if x["slide_id"] == slide_id]
            )
            slide_summary_rows.append({
                "slide_id": slide_id,
                "group": group,
                "representation": args.representation,
                "label_col": label_col,
                "h5ad_path": str(h5ad_path),
                "n_clusters": n_clusters,
                "n_spots_total": len(graph_barcodes),
                "n_spots_labeled": n_valid,
                "barcode_alignment": barcode_mode,
                "ARI_mean": slide_runs["ARI"].mean(),
                "ARI_std": slide_runs["ARI"].std(ddof=0),
                "NMI_mean": slide_runs["NMI"].mean(),
                "NMI_std": slide_runs["NMI"].std(ddof=0),
            })

            # Save the fixed-seed output; never select a seed using the gold ARI.
            plot_pred = pred_by_seed[args.plot_seed]
            coords_all, coord_source = get_spatial_coordinates(
                graph, adata, graph_barcodes
            )
            coords_valid = coords_all[valid] if coords_all is not None else None

            np.save(embeddings_dir / f"{slide_id}_{args.representation}.npy", embedding)

            spot_output = pd.DataFrame({
                "barcode": np.asarray(graph_barcodes)[valid],
                "gold_label": gold,
                "pred_cluster": plot_pred,
            })
            if coords_valid is not None:
                spot_output["spatial_x"] = coords_valid[:, 0]
                spot_output["spatial_y"] = coords_valid[:, 1]
            spot_output.to_csv(
                predictions_dir / f"{slide_id}_spot_predictions.csv",
                index=False,
            )

            save_categorical_spatial_plot(
                coords_valid,
                gold,
                f"{slide_id}: ground truth",
                figures_dir / f"{slide_id}_ground_truth.png",
            )
            save_categorical_spatial_plot(
                coords_valid,
                plot_pred,
                f"{slide_id}: predicted domains (seed={args.plot_seed})",
                figures_dir / f"{slide_id}_prediction.png",
            )

            print(
                f"  labels={n_clusters}, labeled spots={n_valid}, "
                f"ARI={slide_summary_rows[-1]['ARI_mean']:.4f}"
                f"±{slide_summary_rows[-1]['ARI_std']:.4f}, "
                f"NMI={slide_summary_rows[-1]['NMI_mean']:.4f}"
                f"±{slide_summary_rows[-1]['NMI_std']:.4f}, "
                f"coords={coord_source}"
            )

        except Exception as exc:
            skipped_rows.append({
                "slide_id": slide_id,
                "group": group,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    metrics_by_seed = pd.DataFrame(run_metric_rows)
    slide_metrics = pd.DataFrame(slide_summary_rows)
    skipped = pd.DataFrame(skipped_rows)

    metrics_by_seed.to_csv(metrics_dir / "metrics_by_seed.csv", index=False)
    slide_metrics.to_csv(metrics_dir / "slide_metrics.csv", index=False)
    skipped.to_csv(metrics_dir / "skipped_slides.csv", index=False)

    summary_lines = [
        f"Checkpoint: {args.checkpoint}",
        f"Representation: {args.representation}",
        f"Model config: {json.dumps(model_config, ensure_ascii=False)}",
        f"Completed slides: {len(slide_metrics)}",
        f"Failed slides: {len(skipped)}",
    ]

    if not slide_metrics.empty:
        group_summary = (
            slide_metrics.groupby("group", as_index=False)
            .agg(
                n_slides=("slide_id", "count"),
                ARI_mean=("ARI_mean", "mean"),
                ARI_between_slide_std=("ARI_mean", lambda x: x.std(ddof=0)),
                NMI_mean=("NMI_mean", "mean"),
                NMI_between_slide_std=("NMI_mean", lambda x: x.std(ddof=0)),
            )
        )
        group_summary.to_csv(metrics_dir / "group_summary.csv", index=False)

        summary_lines.append("\nGroup summary:")
        summary_lines.append(group_summary.to_string(index=False))

        summary_lines.append("\nAll test slides:")
        summary_lines.append(
            f"Mean slide ARI = {slide_metrics['ARI_mean'].mean():.4f} "
            f"± {slide_metrics['ARI_mean'].std(ddof=0):.4f}"
        )
        summary_lines.append(
            f"Mean slide NMI = {slide_metrics['NMI_mean'].mean():.4f} "
            f"± {slide_metrics['NMI_mean'].std(ddof=0):.4f}"
        )

    if not skipped.empty:
        summary_lines.append("\nFailed slides:")
        summary_lines.append(skipped.to_string(index=False))

    summary_text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    print(f"\nAll outputs were saved under: {out_dir}")


if __name__ == "__main__":
    main()
