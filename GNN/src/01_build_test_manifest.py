#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build and inspect a manifest for held-out test h5ad files.

The output CSV is reviewed manually before final evaluation. Each row contains:
slide_id, group, h5ad_path, label_col, n_spots, n_labels, status, label_examples
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import scanpy as sc


COMMON_LABEL_COLUMNS = [
    "layer_guess_reordered",
    "layer_guess",
    "ground_truth",
    "Ground Truth",
    "manual_annotation",
    "annotation",
    "Annotation",
    "region",
    "Region",
    "spatial_domain",
    "domain",
    "label",
    "Label",
    "cluster",
    "Cluster",
]


def normalize_slide_id(value):
    return os.path.basename(str(value).strip()).replace(".pt", "").replace(".h5ad", "")


def find_h5ad_candidates(slide_id, h5ad_files):
    slide_low = slide_id.lower()

    exact = [p for p in h5ad_files if p.stem.lower() == slide_low]
    if exact:
        return exact

    # Fallback for filenames containing the slide ID plus a prefix/suffix.
    partial = [p for p in h5ad_files if slide_low in p.stem.lower()]
    return partial


def choose_label_column(adata, group):
    ordered = list(COMMON_LABEL_COLUMNS)
    if "DLPFC" in str(group).upper():
        ordered = [
            "layer_guess_reordered",
            "layer_guess",
            "Layer",
            "layer",
            "ground_truth",
            "manual_annotation",
            "annotation",
        ] + ordered
    elif "BREAST" in str(group).upper():
        ordered = [
            "annotation",
            "manual_annotation",
            "ground_truth",
            "region",
            "spatial_domain",
            "label",
        ] + ordered

    seen = set()
    ordered = [x for x in ordered if not (x in seen or seen.add(x))]

    for col in ordered:
        if col not in adata.obs.columns:
            continue
        values = adata.obs[col].dropna().astype(str).str.strip()
        values = values[~values.str.lower().isin(
            {"", "nan", "none", "na", "unknown", "unlabeled", "undefined"}
        )]
        n_unique = values.nunique()
        if 2 <= n_unique <= 30:
            return col

    # Conservative fallback: only categorical/object columns with 2-30 groups.
    fallback = []
    for col in adata.obs.columns:
        series = adata.obs[col]
        if not (
            pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_object_dtype(series)
        ):
            continue
        values = series.dropna().astype(str).str.strip()
        values = values[~values.str.lower().isin(
            {"", "nan", "none", "na", "unknown", "unlabeled", "undefined"}
        )]
        n_unique = values.nunique()
        if 2 <= n_unique <= 30:
            fallback.append((col, n_unique))

    if not fallback:
        return ""
    fallback.sort(key=lambda x: (x[1], x[0]))
    return fallback[0][0]


def inspect_h5ad(path, group):
    adata = sc.read_h5ad(path, backed="r")
    label_col = choose_label_column(adata, group)

    n_labels = ""
    examples = ""
    if label_col:
        values = adata.obs[label_col].dropna().astype(str).str.strip()
        values = values[~values.str.lower().isin(
            {"", "nan", "none", "na", "unknown", "unlabeled", "undefined"}
        )]
        n_labels = int(values.nunique())
        examples = " | ".join(values.drop_duplicates().head(10).tolist())

    result = {
        "label_col": label_col,
        "n_spots": int(adata.n_obs),
        "n_labels": n_labels,
        "label_examples": examples,
    }
    if getattr(adata, "file", None) is not None:
        adata.file.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--h5ad_root", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    split_df = pd.read_csv(args.split_csv)
    required = {"slide_id", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"split_csv is missing columns: {sorted(missing)}")

    split_df["slide_id"] = split_df["slide_id"].map(normalize_slide_id)
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()
    test_df = split_df.loc[split_df["split"].eq("test")].copy()

    if test_df.empty:
        raise RuntimeError("No test rows were found in split_csv.")

    h5ad_root = Path(args.h5ad_root)
    h5ad_files = sorted(h5ad_root.rglob("*.h5ad"))
    print(f"Found {len(h5ad_files)} h5ad files under {h5ad_root}")

    rows = []
    for _, row in test_df.iterrows():
        slide_id = row["slide_id"]
        group = str(row.get("group", "UNKNOWN"))
        candidates = find_h5ad_candidates(slide_id, h5ad_files)

        result = {
            "slide_id": slide_id,
            "group": group,
            "h5ad_path": "",
            "label_col": "",
            "n_spots": "",
            "n_labels": "",
            "status": "",
            "label_examples": "",
            "candidate_paths": " | ".join(str(p) for p in candidates[:10]),
        }

        if len(candidates) == 0:
            result["status"] = "MISSING_H5AD"
        elif len(candidates) > 1:
            result["status"] = "MULTIPLE_H5AD"
        else:
            path = candidates[0]
            result["h5ad_path"] = str(path)
            try:
                info = inspect_h5ad(path, group)
                result.update(info)
                result["status"] = "OK" if info["label_col"] else "NO_LABEL_COLUMN"
            except Exception as exc:
                result["status"] = f"READ_ERROR: {type(exc).__name__}: {exc}"

        rows.append(result)

    manifest = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    print("\nManifest status:")
    print(manifest["status"].value_counts(dropna=False).to_string())
    print(f"\nSaved manifest to: {out_path}")
    print(
        "\nBefore evaluation, open the CSV and verify every row has:\n"
        "  status=OK, correct h5ad_path, and the correct gold label_col.\n"
        "For MULTIPLE_H5AD rows, manually choose one path and change status to OK."
    )


if __name__ == "__main__":
    main()
