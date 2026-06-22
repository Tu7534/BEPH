#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

DLPFC_LABELS = [
    "layer_guess_reordered", "layer_guess", "Layer", "layer",
    "ground_truth", "Ground Truth", "manual_annotation", "annotation",
]
BREAST_LABELS = [
    "annotation", "manual_annotation", "ground_truth", "Ground Truth",
    "region", "Region", "spatial_domain", "domain", "label", "Label",
]
COMMON_LABELS = [
    "ground_truth", "Ground Truth", "manual_annotation", "annotation",
    "Annotation", "region", "Region", "spatial_domain", "domain",
    "label", "Label",
]
FORBIDDEN_LABEL_TOKENS = [
    "leiden", "louvain", "mclust", "pred", "cluster", "kmeans", "domain_pred",
]


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_graph_barcodes(graph):
    attrs = [
        "barcodes", "barcode", "spot_ids", "spot_id",
        "obs_names", "node_names", "cell_ids",
    ]
    for attr in attrs:
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
    raise AttributeError("No graph barcode field found.")


def normalize_barcode(value):
    text = str(value).strip()
    return text[:-2] if text.endswith("-1") else text


def overlap_ratio(graph_barcodes, obs_names):
    graph_exact = set(map(str, graph_barcodes))
    obs_exact = set(map(str, obs_names))
    exact = len(graph_exact & obs_exact) / max(1, len(graph_exact))

    graph_norm = {normalize_barcode(x) for x in graph_barcodes}
    obs_norm = {normalize_barcode(x) for x in obs_names}
    normalized = len(graph_norm & obs_norm) / max(1, len(graph_norm))
    return exact, normalized, max(exact, normalized)


def valid_label_values(series):
    values = series.dropna().astype(str).str.strip()
    invalid = {"", "nan", "none", "na", "unknown", "unlabeled", "undefined"}
    return values[~values.str.lower().isin(invalid)]


def choose_label_column(adata, group):
    group_upper = str(group).upper()
    if "DLPFC" in group_upper:
        ordered = DLPFC_LABELS + COMMON_LABELS
    elif "BREAST" in group_upper:
        ordered = BREAST_LABELS + COMMON_LABELS
    else:
        ordered = COMMON_LABELS

    seen = set()
    ordered = [x for x in ordered if not (x in seen or seen.add(x))]

    for col in ordered:
        if col not in adata.obs.columns:
            continue
        values = valid_label_values(adata.obs[col])
        n_unique = int(values.nunique())
        if 2 <= n_unique <= 30:
            return col, n_unique, "preferred"

    fallback = []
    for col in adata.obs.columns:
        low = str(col).lower()
        if any(token in low for token in FORBIDDEN_LABEL_TOKENS):
            continue
        series = adata.obs[col]
        if not (
            pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_object_dtype(series)
        ):
            continue
        values = valid_label_values(series)
        n_unique = int(values.nunique())
        if 2 <= n_unique <= 30:
            fallback.append((col, n_unique))

    if not fallback:
        return "", 0, "none"

    fallback.sort(key=lambda x: (x[1], x[0]))
    return fallback[0][0], fallback[0][1], "fallback"


def parse_candidate_paths(text):
    if pd.isna(text):
        return []
    return [Path(x.strip()) for x in str(text).split(" | ") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--report_csv", required=True)
    parser.add_argument("--min_overlap", type=float, default=0.95)
    parser.add_argument("--min_margin", type=float, default=0.02)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    resolved_rows = []
    report_rows = []

    for _, row in manifest.iterrows():
        slide_id = str(row["slide_id"])
        group = str(row.get("group", "UNKNOWN"))
        graph_path = Path(args.graph_dir) / f"{slide_id}.pt"
        out_row = row.to_dict()

        if not graph_path.exists():
            out_row["status"] = "MISSING_GRAPH"
            resolved_rows.append(out_row)
            continue

        graph = safe_torch_load(graph_path, map_location="cpu")
        graph_barcodes = get_graph_barcodes(graph)

        candidates = parse_candidate_paths(row.get("candidate_paths", ""))
        current_path = row.get("h5ad_path", "")
        if isinstance(current_path, str) and current_path.strip():
            current = Path(current_path.strip())
            if current not in candidates:
                candidates.append(current)

        scored = []
        for candidate in candidates:
            result = {
                "slide_id": slide_id,
                "group": group,
                "candidate_path": str(candidate),
                "exists": candidate.exists(),
                "exact_overlap": 0.0,
                "normalized_overlap": 0.0,
                "best_overlap": 0.0,
                "label_col": "",
                "n_labels": 0,
                "label_source": "",
                "n_obs": 0,
                "score": -1.0,
                "error": "",
            }

            if not candidate.exists():
                result["error"] = "missing file"
                report_rows.append(result)
                continue

            try:
                adata = sc.read_h5ad(candidate, backed="r")
                exact, normalized, best = overlap_ratio(
                    graph_barcodes, adata.obs_names.astype(str)
                )
                label_col, n_labels, label_source = choose_label_column(adata, group)

                label_bonus = 0.05 if label_source == "preferred" else (
                    0.01 if label_source == "fallback" else 0.0
                )
                result.update({
                    "exact_overlap": exact,
                    "normalized_overlap": normalized,
                    "best_overlap": best,
                    "label_col": label_col,
                    "n_labels": n_labels,
                    "label_source": label_source,
                    "n_obs": int(adata.n_obs),
                    "score": best + label_bonus,
                })

                if getattr(adata, "file", None) is not None:
                    adata.file.close()
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"

            report_rows.append(result)
            scored.append(result)

        valid = [
            x for x in scored
            if not x["error"] and x["exists"] and x["label_col"]
        ]
        valid.sort(key=lambda x: (x["score"], x["best_overlap"]), reverse=True)

        if not valid:
            out_row["status"] = "NO_VALID_CANDIDATE"
            resolved_rows.append(out_row)
            continue

        best = valid[0]
        second_score = valid[1]["score"] if len(valid) > 1 else -1.0
        margin = best["score"] - second_score

        out_row["h5ad_path"] = best["candidate_path"]
        out_row["label_col"] = best["label_col"]
        out_row["n_spots"] = best["n_obs"]
        out_row["n_labels"] = best["n_labels"]
        out_row["best_overlap"] = best["best_overlap"]
        out_row["selection_margin"] = margin

        if best["best_overlap"] >= args.min_overlap and (
            len(valid) == 1 or margin >= args.min_margin
        ):
            out_row["status"] = "OK"
        else:
            out_row["status"] = "REVIEW_TIE"

        resolved_rows.append(out_row)

    resolved = pd.DataFrame(resolved_rows)
    report = pd.DataFrame(report_rows)

    out_path = Path(args.out_csv)
    report_path = Path(args.report_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    resolved.to_csv(out_path, index=False)
    report.sort_values(
        ["slide_id", "score"], ascending=[True, False]
    ).to_csv(report_path, index=False)

    print("\nResolved manifest status:")
    print(resolved["status"].value_counts(dropna=False).to_string())
    print(f"\nResolved manifest: {out_path}")
    print(f"Candidate report:  {report_path}")

    review = resolved.loc[~resolved["status"].eq("OK")]
    if not review.empty:
        print("\nRows requiring manual review:")
        cols = [
            "slide_id", "group", "status", "h5ad_path",
            "label_col", "best_overlap", "selection_margin",
        ]
        print(review[cols].to_string(index=False))
    else:
        print("\nAll rows resolved confidently.")


if __name__ == "__main__":
    main()
