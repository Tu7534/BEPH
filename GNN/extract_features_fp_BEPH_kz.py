"""
=============================================================================
【BEPH 工业级端到端特征流水线】 (Single-GPU NPY Edition)
功能描述:
提取 ST 平台图像特征，内存防爆优化，并直接输出建图所需的 .npy 矩阵文件。
=============================================================================
"""

import os
import gc
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from mmengine.config import Config as MMConfig
from mmengine.registry import init_default_scope
from mmselfsup.apis import init_model


# ==========================================
# 1. 全局配置类 (Configuration)
# ==========================================
class Config:
    # 基础目录
    ROOT_DIR = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data'
    H5AD_DIR = os.path.join(ROOT_DIR, 'Raw_Data', 'h5ad_files')
    IMAGE_DIR = os.path.join(ROOT_DIR, 'Raw_Data', 'images')
    CSV_PATH = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/process_list.csv'

    # 输出目录 (直接存到建图脚本需要的目录)
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'Graph_pt')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 模型配置
    MODEL_CONFIG = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/mmselfsup/configs/tsne/beitv2_base.py'
    MODEL_CHECKPOINT = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/checkpoints/BEPH_backbone.pth'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 图像与批处理参数
    SRC_SIZE = 80             # 从原图裁剪的 Patch 大小
    TARGET_SIZE = 224         # 缩放后喂给模型的 Patch 大小
    BATCH_SIZE = 128          # 单卡标准 Batch Size
    NUM_WORKERS = 4           # 数据加载线程


# ==========================================
# 2. 坐标提取器 (Coordinate Extractor)
# ==========================================
class CoordinateExtractor:
    @staticmethod
    def get_valid_coords(h5ad_path, img_w, img_h):
        adata = sc.read_h5ad(h5ad_path)
        
        # 🌟 核心同步：在这里加上相同的质控规则，保证提出来的坐标就是质控后的细胞坐标！ 🌟
        if adata.X.max() > 50:
             sc.pp.filter_cells(adata, min_genes=200)
        
        # 兼容性寻找空间坐标
        coord_key = 'spatial' if 'spatial' in adata.obsm.keys() else ('X_spatial' if 'X_spatial' in adata.obsm.keys() else None)
        
        if coord_key is None:
            raise ValueError(f"No spatial coords found in obsm of {h5ad_path}")

        coords = adata.obsm[coord_key]
        scale_factor = 1.0
        
        library_id = list(adata.uns.get('spatial', {}).keys())[0] if adata.uns.get('spatial') else None
        if library_id:
            scalefactors = adata.uns['spatial'][library_id].get('scalefactors', {})
            if 'tissue_hires_scalef' in scalefactors:
                scale_factor = scalefactors['tissue_hires_scalef']
            elif 'tissue_lowres_scalef' in scalefactors:
                scale_factor = scalefactors['tissue_lowres_scalef']

        pixel_coords = (coords * scale_factor).astype(int)
        top_left_coords = pixel_coords - (Config.SRC_SIZE // 2)
        
        # 边界防爆检测：允许存在少量越界，将其强行拉回边界内，保证与基因 Spot 一一对应
        valid_coords = []
        for x, y in top_left_coords:
            x_safe = max(0, min(x, img_w - Config.SRC_SIZE))
            y_safe = max(0, min(y, img_h - Config.SRC_SIZE))
            valid_coords.append([x_safe, y_safe])
                
        return np.array(valid_coords)


# ==========================================
# 3. 数据集加载器 (PyTorch Dataset)
# ==========================================
class SpatialPatchDataset(Dataset):
    def __init__(self, image_path, coords):
        Image.MAX_IMAGE_PIXELS = None
        self.img = Image.open(image_path).convert('RGB')
        self.coords = coords
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = self.img.crop((x, y, x + Config.SRC_SIZE, y + Config.SRC_SIZE))
        patch = patch.resize((Config.TARGET_SIZE, Config.TARGET_SIZE), Image.Resampling.LANCZOS)
        patch_tensor = self.transform(patch)
        return patch_tensor, self.coords[idx]

def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return imgs, coords


# ==========================================
# 4. 深度学习推理引擎 (Feature Extractor)
# ==========================================
class FeatureExtractor:
    def __init__(self):
        print(f"\n⚙️ 正在初始化 BEPH 模型引擎 | 设备: {Config.DEVICE}...")
        mm_cfg = MMConfig.fromfile(Config.MODEL_CONFIG)
        init_default_scope(mm_cfg.get('default_scope', 'mmselfsup'))
        
        self.model = init_model(mm_cfg, Config.MODEL_CHECKPOINT, device=Config.DEVICE)
        self.model.eval()
        print("✅ 单卡模型加载成功！\n")

    def extract(self, dataloader, slide_id):
        features_list = []
        
        # 自动选择混合精度以节省显存加速推理 (AMP)
        use_amp = Config.DEVICE.type == 'cuda'
        
        with torch.no_grad():
            for batch_imgs, _ in tqdm(dataloader, desc=f"🔍 提取 {slide_id}", leave=False):
                batch_imgs = batch_imgs.to(Config.DEVICE)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    feat = self.model.extract_feat(batch_imgs, stage='backbone')[0]
                
                if len(feat.shape) == 3: 
                    feat = feat[:, 0, :] 
                
                features_list.append(feat.cpu().numpy().astype(np.float32))
                
        all_features = np.concatenate(features_list, axis=0)
        return all_features


# ==========================================
# 5. 流水线控制器 (Pipeline Manager)
# ==========================================
class BEPHPipeline:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def process_slide(self, slide_id):
        h5ad_path = os.path.join(Config.H5AD_DIR, f"{slide_id}.h5ad")
        image_path = os.path.join(Config.IMAGE_DIR, f"{slide_id}.png")
        
        out_npy = os.path.join(Config.OUTPUT_DIR, f"{slide_id}_image_features.npy")

        if os.path.exists(out_npy):
            return "Skipped (Already extracted)"
            
        if not os.path.exists(h5ad_path) or not os.path.exists(image_path):
            return "Failed (Missing files)"

        try:
            Image.MAX_IMAGE_PIXELS = None
            img_w, img_h = Image.open(image_path).size
            
            valid_coords = CoordinateExtractor.get_valid_coords(h5ad_path, img_w, img_h)

            dataset = SpatialPatchDataset(image_path, valid_coords)
            loader = DataLoader(
                dataset, 
                batch_size=Config.BATCH_SIZE, 
                num_workers=Config.NUM_WORKERS, 
                pin_memory=True, 
                collate_fn=collate_fn
            )

            # 推理提取
            features = self.feature_extractor.extract(loader, slide_id)

            # 直接保存为 NPY，完美对接多模态融合建图脚本
            np.save(out_npy, features)
            
            # 清理内存与显存防 OOM
            del dataset, loader, features
            gc.collect()
            torch.cuda.empty_cache()
            
            return f"Success ({len(valid_coords)} patches)"

        except Exception as e:
            return f"Error ({str(e)})"

    def run(self):
        if not os.path.exists(Config.CSV_PATH):
            print(f"❌ 找不到名单文件: {Config.CSV_PATH}")
            return
            
        df = pd.read_csv(Config.CSV_PATH)
        ids = df['slide_id'].tolist() if 'slide_id' in df.columns else df.iloc[:, 0].tolist()
        
        print(f"🚀 启动 BEPH 工业级流水线，共 {len(ids)} 个样本待处理...\n")
        
        results = {'success': 0, 'skip': 0, 'fail': 0}
        pbar = tqdm(ids, desc="Overall Progress")
        
        for slide_id in pbar:
            slide_id = str(slide_id)
            status_msg = self.process_slide(slide_id)
            
            if "Success" in status_msg:
                results['success'] += 1
            elif "Skipped" in status_msg:
                results['skip'] += 1
            else:
                results['fail'] += 1
                
            pbar.set_postfix({'status': status_msg[:25]})
            
        print("\n" + "="*50)
        print(f"✨ 流水线执行完毕！")
        print(f"✅ 成功提取: {results['success']}")
        print(f"⏩ 跳过已有: {results['skip']}")
        print(f"❌ 提取失败: {results['fail']}")
        print(f"📁 最终 Numpy 矩阵保存在: {Config.OUTPUT_DIR}")
        print("="*50 + "\n")


# ==========================================
# 6. 启动器 (Main)
# ==========================================
if __name__ == "__main__":
    pipeline = BEPHPipeline()
    pipeline.run()