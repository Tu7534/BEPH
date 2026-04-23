"""
=============================================================================
【BEPH 工业级端到端特征流水线】 (End-to-End Visual Feature Pipeline)

功能描述:
本流水线将“坐标生成”与“深度学习特征提取”完美融合。
一键实现：读取 .h5ad 坐标 -> 智能缩放 -> 图像边界检测 -> PIL切图转换 -> 
MMSelfSup (BEiT-v2) 特征提取 -> 打包保存 .pt 和 .h5。

架构设计 (OOP):
- `Config`: 全局配置中心，所有的路径和参数都在这里修改。
- `CoordinateExtractor`: 空间坐标数学计算模块。
- `SpatialPatchDataset`: PyTorch 标准数据加载器，负责图像的按需裁剪与预处理。
- `FeatureExtractor`: 深度学习推理引擎。
- `BEPHPipeline`: 流水线调度控制器，负责批处理、进度条和断点续传。
=============================================================================
"""

import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
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
    H5AD_DIR = os.path.join(ROOT_DIR, 'Raw_Data', 'Log1p')
    IMAGE_DIR = os.path.join(ROOT_DIR, 'Raw_Data', 'images')
    CSV_PATH = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/process_list.csv'

    # 输出目录
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'BEPH_Features')
    COORD_SAVE_DIR = os.path.join(OUTPUT_DIR, 'patches')  # 保存临时坐标
    H5_SAVE_DIR = os.path.join(OUTPUT_DIR, 'h5_files')    # 保存最终特征与坐标
    PT_SAVE_DIR = os.path.join(OUTPUT_DIR, 'pt_files')    # 保存纯特征向量

    # 模型配置
    MODEL_CONFIG = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/mmselfsup/configs/tsne/beitv2_base.py'
    MODEL_CHECKPOINT = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/checkpoints/BEPH_backbone.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 图像与批处理参数
    SRC_SIZE = 80             # 从原图裁剪的 Patch 大小
    TARGET_SIZE = 224         # 缩放后喂给模型的 Patch 大小
    BATCH_SIZE = 128          # DataLoader 批次大小
    NUM_WORKERS = 4           # 多线程读取


# ==========================================
# 2. 坐标提取器 (Coordinate Extractor)
# ==========================================
class CoordinateExtractor:
    """负责从 H5AD 解析坐标，计算缩放因子，并剔除越界 Patch"""
    
    @staticmethod
    def get_valid_coords(h5ad_path, img_w, img_h):
        adata = sc.read_h5ad(h5ad_path)
        if 'spatial' not in adata.obsm.keys():
            raise ValueError("No 'spatial' coords in obsm")

        coords = adata.obsm['spatial']
        scale_factor = 1.0
        
        # 智能获取缩放因子
        library_id = list(adata.uns.get('spatial', {}).keys())[0] if adata.uns.get('spatial') else None
        if library_id:
            scalefactors = adata.uns['spatial'][library_id].get('scalefactors', {})
            if 'tissue_hires_scalef' in scalefactors:
                scale_factor = scalefactors['tissue_hires_scalef']
            elif 'tissue_lowres_scalef' in scalefactors:
                scale_factor = scalefactors['tissue_lowres_scalef']

        # 映射到像素坐标并计算左上角
        pixel_coords = (coords * scale_factor).astype(int)
        top_left_coords = pixel_coords - (Config.SRC_SIZE // 2)
        
        # 边界防爆检测
        valid_coords = []
        for x, y in top_left_coords:
            if x >= 0 and y >= 0 and x + Config.SRC_SIZE <= img_w and y + Config.SRC_SIZE <= img_h:
                valid_coords.append([x, y])
                
        valid_coords = np.array(valid_coords)
        if len(valid_coords) == 0:
            raise ValueError("0 valid patches after boundary check.")
            
        return valid_coords


# ==========================================
# 3. 数据集加载器 (PyTorch Dataset)
# ==========================================
class SpatialPatchDataset(Dataset):
    """根据提取的坐标，动态从 PIL 图像中裁剪 Patch 并转为 Tensor"""
    
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
        # 1. 裁剪 SRC_SIZE (例如 80x80)
        patch = self.img.crop((x, y, x + Config.SRC_SIZE, y + Config.SRC_SIZE))
        
        # 2. Lanczos 高质量缩放到 TARGET_SIZE (224x224)
        patch = patch.resize((Config.TARGET_SIZE, Config.TARGET_SIZE), Image.Resampling.LANCZOS)
        
        # 3. 转 Tensor
        patch_tensor = self.transform(patch)
        return patch_tensor, self.coords[idx]

def collate_fn(batch):
    """自定义 Collate Function，将 batch 打包"""
    imgs = torch.stack([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return imgs, coords


# ==========================================
# 4. 深度学习推理引擎 (Feature Extractor)
# ==========================================
class FeatureExtractor:
    """负责加载 BEiT 模型，吞吐 DataLoader 并提取特征"""
    
    def __init__(self):
        print("\n⚙️ 正在初始化 BEPH 模型引擎...")
        mm_cfg = MMConfig.fromfile(Config.MODEL_CONFIG)
        init_default_scope(mm_cfg.get('default_scope', 'mmselfsup'))
        self.model = init_model(mm_cfg, Config.MODEL_CHECKPOINT, device=Config.DEVICE)
        self.model.eval()
        print("✅ 模型加载成功！\n")

    def extract(self, dataloader, slide_id):
        features_list = []
        coords_list = []
        
        # 禁用梯度计算以节省显存
        with torch.no_grad():
            for batch_imgs, batch_coords in tqdm(dataloader, desc=f"🔍 提取 {slide_id}", leave=False):
                batch_imgs = batch_imgs.to(Config.DEVICE)
                
                # 兼容 DataParallel 或单卡
                if hasattr(self.model, 'module'):
                    feat = self.model.module.extract_feat(batch_imgs, stage='backbone')[0]
                else:
                    feat = self.model.extract_feat(batch_imgs, stage='backbone')[0]
                
                # 如果是 3D 张量 (Batch, 1, Dim)，压缩为 (Batch, Dim)
                if len(feat.shape) == 3: 
                    feat = feat[:, 0, :] 
                
                features_list.append(feat.cpu().numpy().astype(np.float32))
                coords_list.append(batch_coords)
                
        all_features = np.concatenate(features_list, axis=0)
        all_coords = np.concatenate(coords_list, axis=0)
        return all_features, all_coords


# ==========================================
# 5. 流水线控制器 (Pipeline Manager)
# ==========================================
class BEPHPipeline:
    """统筹整个前处理与特征提取流程"""
    
    def __init__(self):
        # 自动创建所有必要的输出文件夹
        for d in [Config.COORD_SAVE_DIR, Config.H5_SAVE_DIR, Config.PT_SAVE_DIR]:
            os.makedirs(d, exist_ok=True)
            
        self.feature_extractor = FeatureExtractor()

    def process_slide(self, slide_id):
        """处理单个样本的核心流水线"""
        h5ad_path = os.path.join(Config.H5AD_DIR, f"{slide_id}.h5ad")
        image_path = os.path.join(Config.IMAGE_DIR, f"{slide_id}.png")
        
        out_h5 = os.path.join(Config.H5_SAVE_DIR, f"{slide_id}.h5")
        out_pt = os.path.join(Config.PT_SAVE_DIR, f"{slide_id}.pt")
        out_coord = os.path.join(Config.COORD_SAVE_DIR, f"{slide_id}.h5")

        # 1. 检查断点续传
        if os.path.exists(out_h5) and os.path.exists(out_pt):
            return "Skipped (Already extracted)"
            
        if not os.path.exists(h5ad_path) or not os.path.exists(image_path):
            return "Failed (Missing source files)"

        try:
            # 2. 坐标提取
            Image.MAX_IMAGE_PIXELS = None
            img_w, img_h = Image.open(image_path).size
            valid_coords = CoordinateExtractor.get_valid_coords(h5ad_path, img_w, img_h)
            
            # (可选) 备份一份纯坐标文件，以备不时之需
            with h5py.File(out_coord, 'w') as f:
                f.create_dataset('coords', data=valid_coords)
                f.attrs['patch_size'] = Config.SRC_SIZE

            # 3. 构建 Dataset 和 DataLoader
            dataset = SpatialPatchDataset(image_path, valid_coords)
            loader = DataLoader(
                dataset, 
                batch_size=Config.BATCH_SIZE, 
                num_workers=Config.NUM_WORKERS, 
                pin_memory=True, 
                collate_fn=collate_fn
            )

            # 4. 模型特征提取
            features, coords = self.feature_extractor.extract(loader, slide_id)

            # 5. 双格式保存
            with h5py.File(out_h5, 'w') as f:
                f.create_dataset('features', data=features)
                f.create_dataset('coords', data=coords)
                
            torch.save(torch.from_numpy(features), out_pt)
            
            return f"Success ({len(coords)} patches)"

        except Exception as e:
            return f"Error ({str(e)})"

    def run(self):
        """主执行入口"""
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
        print(f"📁 最终特征保存在: {Config.OUTPUT_DIR}")
        print("="*50 + "\n")


# ==========================================
# 6. 启动器 (Main)
# ==========================================
if __name__ == "__main__":
    pipeline = BEPHPipeline()
    pipeline.run()