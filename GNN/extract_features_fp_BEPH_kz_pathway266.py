#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
BEPH image patch feature extraction pipeline - pathway266 compatible version

Purpose:
    1. Extract pathology image features for each spatial spot.
    2. Save features as .npy for downstream multi-modal graph construction.
    3. Save barcode/coordinate metadata to ensure strict alignment with h5ad spots.

Important changes compared with the previous version:
    - Uses the same spot QC rule as graph construction to avoid image/gene count mismatch.
    - Supports png/jpg/jpeg/tif/tiff image extensions.
    - Saves *_image_barcodes.txt and *_image_coords.npy for alignment checking.
    - pin_memory is set to False to reduce CUDA pinned-memory/OOM risk.
    - Existing features are checked against current QC-filtered barcodes before skipping.
=============================================================================
"""

import os
import gc
import traceback

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from scipy import sparse
from mmengine.config import Config as MMConfig
from mmengine.registry import init_default_scope
from mmselfsup.apis import init_model


# ==========================================
# 1. Global configuration
# ==========================================
class Config:
    ROOT_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data"
    H5AD_DIR = os.path.join(ROOT_DIR, "Raw_Data", "h5ad_files")
    IMAGE_DIR = os.path.join(ROOT_DIR, "Raw_Data", "images")
    CSV_PATH = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/process_list.csv"

    # Image features are still saved in Graph_pt because the graph construction script reads them here.
    OUTPUT_DIR = os.path.join(ROOT_DIR, "Graph_pt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_CONFIG = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/mmselfsup/configs/tsne/beitv2_base.py"
    MODEL_CHECKPOINT = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/checkpoints/BEPH_backbone.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SRC_SIZE = 80
    TARGET_SIZE = 224

    # Safer default for single-GPU batch extraction.
    # You can increase this after confirming GPU memory is sufficient.
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    PIN_MEMORY = False

    # If True, existing .npy features will be overwritten.
    # Keep False for normal runs; set True only when old features were generated with different QC rules.
    FORCE_REWRITE = False

    IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


# ==========================================
# 2. Utility functions
# ==========================================
def safe_x_max(adata):
    """Safely get max value from dense/sparse adata.X."""
    x = adata.X
    if sparse.issparse(x):
        if x.nnz == 0:
            return 0.0
        return float(x.max())
    return float(np.nanmax(x))


def apply_spot_qc_for_alignment(adata):
    """
    Apply exactly the same spot-level QC rule used before graph construction.
    This keeps image patch order/count aligned with gene/pathway graph nodes.
    """
    adata.var_names_make_unique()
    if safe_x_max(adata) > 50:
        sc.pp.filter_cells(adata, min_genes=200)
    return adata


def find_image_file(slide_id):
    for ext in Config.IMAGE_EXTS:
        p = os.path.join(Config.IMAGE_DIR, f"{slide_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def save_metadata(out_barcode_txt, out_coord_npy, barcodes, coords):
    with open(out_barcode_txt, "w") as f:
        for b in barcodes:
            f.write(str(b) + "\n")
    np.save(out_coord_npy, coords.astype(np.int32))


# ==========================================
# 3. Coordinate extractor
# ==========================================
class CoordinateExtractor:
    @staticmethod
    def get_valid_coords_and_barcodes(h5ad_path, img_w, img_h):
        adata = sc.read_h5ad(h5ad_path)
        adata = apply_spot_qc_for_alignment(adata)

        if "spatial" in adata.obsm.keys():
            coord_key = "spatial"
        elif "X_spatial" in adata.obsm.keys():
            coord_key = "X_spatial"
        else:
            raise ValueError(f"No spatial coords found in obsm of {h5ad_path}")

        coords = np.asarray(adata.obsm[coord_key], dtype=float)
        barcodes = adata.obs_names.astype(str).tolist()

        scale_factor = 1.0
        spatial_uns = adata.uns.get("spatial", {})
        library_id = list(spatial_uns.keys())[0] if spatial_uns else None
        if library_id:
            scalefactors = spatial_uns[library_id].get("scalefactors", {})
            if "tissue_hires_scalef" in scalefactors:
                scale_factor = scalefactors["tissue_hires_scalef"]
            elif "tissue_lowres_scalef" in scalefactors:
                scale_factor = scalefactors["tissue_lowres_scalef"]

        pixel_coords = (coords * scale_factor).astype(int)
        top_left_coords = pixel_coords - (Config.SRC_SIZE // 2)

        valid_coords = []
        for x, y in top_left_coords:
            x_safe = max(0, min(int(x), max(0, img_w - Config.SRC_SIZE)))
            y_safe = max(0, min(int(y), max(0, img_h - Config.SRC_SIZE)))
            valid_coords.append([x_safe, y_safe])

        del adata
        gc.collect()
        return np.asarray(valid_coords, dtype=np.int32), barcodes


# ==========================================
# 4. Dataset
# ==========================================
class SpatialPatchDataset(Dataset):
    def __init__(self, image_path, coords):
        Image.MAX_IMAGE_PIXELS = None
        self.img = Image.open(image_path).convert("RGB")
        self.coords = coords
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = self.img.crop((int(x), int(y), int(x) + Config.SRC_SIZE, int(y) + Config.SRC_SIZE))
        patch = patch.resize((Config.TARGET_SIZE, Config.TARGET_SIZE), Image.Resampling.LANCZOS)
        patch_tensor = self.transform(patch)
        return patch_tensor, self.coords[idx]


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return imgs, coords


# ==========================================
# 5. Feature extractor
# ==========================================
class FeatureExtractor:
    def __init__(self):
        print(f"\n⚙️ Initializing BEPH model | device: {Config.DEVICE} ...")
        mm_cfg = MMConfig.fromfile(Config.MODEL_CONFIG)
        init_default_scope(mm_cfg.get("default_scope", "mmselfsup"))
        self.model = init_model(mm_cfg, Config.MODEL_CHECKPOINT, device=Config.DEVICE)
        self.model.eval()
        print("✅ BEPH model loaded.\n")

    def extract(self, dataloader, slide_id):
        features_list = []
        use_amp = Config.DEVICE.type == "cuda"

        with torch.no_grad():
            for batch_imgs, _ in tqdm(dataloader, desc=f"🔍 Extract {slide_id}", leave=False):
                batch_imgs = batch_imgs.to(Config.DEVICE, non_blocking=False)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    feat = self.model.extract_feat(batch_imgs, stage="backbone")[0]

                if len(feat.shape) == 3:
                    feat = feat[:, 0, :]

                features_list.append(feat.detach().cpu().numpy().astype(np.float32))

                del batch_imgs, feat
                if Config.DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        return np.concatenate(features_list, axis=0)


# ==========================================
# 6. Pipeline manager
# ==========================================
class BEPHPipeline:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def process_slide(self, slide_id):
        h5ad_path = os.path.join(Config.H5AD_DIR, f"{slide_id}.h5ad")
        image_path = find_image_file(slide_id)

        out_npy = os.path.join(Config.OUTPUT_DIR, f"{slide_id}_image_features.npy")
        out_barcode_txt = os.path.join(Config.OUTPUT_DIR, f"{slide_id}_image_barcodes.txt")
        out_coord_npy = os.path.join(Config.OUTPUT_DIR, f"{slide_id}_image_coords.npy")

        if not os.path.exists(h5ad_path):
            return "Failed (Missing h5ad)"
        if image_path is None:
            return "Failed (Missing image)"

        try:
            Image.MAX_IMAGE_PIXELS = None
            img_w, img_h = Image.open(image_path).size
            valid_coords, barcodes = CoordinateExtractor.get_valid_coords_and_barcodes(h5ad_path, img_w, img_h)

            # If feature exists, verify shape and save metadata.
            if os.path.exists(out_npy) and not Config.FORCE_REWRITE:
                old_features = np.load(out_npy, mmap_mode="r")
                if old_features.shape[0] == len(barcodes):
                    save_metadata(out_barcode_txt, out_coord_npy, barcodes, valid_coords)
                    return f"Skipped (Existing features matched: {old_features.shape[0]} spots)"
                return (
                    f"Failed (Existing feature count {old_features.shape[0]} != QC spots {len(barcodes)}; "
                    f"delete {out_npy} or set FORCE_REWRITE=True)"
                )

            dataset = SpatialPatchDataset(image_path, valid_coords)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
            )

            features = self.feature_extractor.extract(loader, slide_id)
            if features.shape[0] != len(barcodes):
                raise RuntimeError(f"feature count {features.shape[0]} != barcode count {len(barcodes)}")

            np.save(out_npy, features)
            save_metadata(out_barcode_txt, out_coord_npy, barcodes, valid_coords)

            del dataset, loader, features, valid_coords
            gc.collect()
            if Config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            return f"Success ({len(barcodes)} patches)"

        except Exception as e:
            print(f"\n[!!!] Sample {slide_id} error:\n{traceback.format_exc()}")
            return f"Error ({str(e)})"

    def run(self):
        if not os.path.exists(Config.CSV_PATH):
            print(f"❌ Missing process list: {Config.CSV_PATH}")
            return

        df = pd.read_csv(Config.CSV_PATH)
        ids = df["slide_id"].tolist() if "slide_id" in df.columns else df.iloc[:, 0].tolist()

        print(f"🚀 Start BEPH image feature pipeline, {len(ids)} slides.\n")
        print(f"Output dir: {Config.OUTPUT_DIR}")
        print(f"Batch size: {Config.BATCH_SIZE}, workers: {Config.NUM_WORKERS}, pin_memory: {Config.PIN_MEMORY}\n")

        results = {"success": 0, "skip": 0, "fail": 0}
        pbar = tqdm(ids, desc="Overall Progress")

        for slide_id in pbar:
            slide_id = str(slide_id)
            status_msg = self.process_slide(slide_id)

            if "Success" in status_msg:
                results["success"] += 1
            elif "Skipped" in status_msg:
                results["skip"] += 1
            else:
                results["fail"] += 1

            pbar.set_postfix({"status": status_msg[:40]})

        print("\n" + "=" * 50)
        print("✨ Image feature extraction finished.")
        print(f"✅ Success: {results['success']}")
        print(f"⏩ Skipped: {results['skip']}")
        print(f"❌ Failed: {results['fail']}")
        print(f"📁 NPY features saved in: {Config.OUTPUT_DIR}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    pipeline = BEPHPipeline()
    pipeline.run()
