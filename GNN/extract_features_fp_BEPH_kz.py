"""
=============================================================================
【BEPH 工业级提取器】空间转录组定制版形态学特征提取流水线

核心功能:
摒弃了臃肿的 CLAM 依赖和 Openslide 报错隐患，使用最基础的 `PIL.Image` 直接操作高清 .png 图片。
专为空间转录组 (ST) 的 Spot 切片数据量身定制，高效提取 BEPH (BEiT-v2) 模型的视觉特征。

工程化优势与亮点:
1. 极简配置区 (Global Config): 将所有难记的路径 (根目录、模型权重、输入输出) 
   集中在代码最上方，修改极度方便，告别冗长的命令行参数。
2. PIL 替代 OpenSlide: 新建了 `PILSlide` 辅助类。因为空间转录组的切片大多已经是
   单一的 .png 高清图，不需要用沉重的 openslide 去读金字塔层级，彻底解决了由于 
   Libvips 导致的环境安装报错。
3. 智能图像缩放 (Resize): 在 Dataset 中自动将 10x 的 48x48 Spot 放大至模型
   所需要的 224x224 (Lanczos重采样)，保证特征提取不失真。
4. 防崩塌与断点续传: `process_one_slide` 函数内置了强健的 try-except 保护。
   自带 `os.path.exists` 检查，如果在 30 张切片里跑到第 15 张断电了，
   再次运行会瞬间跳过前 14 张，直接从第 15 张继续提取。
5. 双格式同步保存: 提取完毕后，不仅保存了附带坐标信息的 `features.h5`，还直接 
   `torch.save` 保存了纯特征的 `.pt` 文件。这就为你下一步直接运行 `构建图结构.py` 
   准备好了完美的食材！
=============================================================================
"""

# cd /data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/
import os
import time
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmselfsup.apis import init_model

# ================= 1. 全局配置区域 =================
# 数据根目录 (请根据上一部的设置确认)
ROOT_DIR = './kz_data' 

# 输入目录
PATCHES_DIR = os.path.join(ROOT_DIR, 'Segmentation', 'patches') # .h5 坐标文件
IMAGES_DIR = os.path.join(ROOT_DIR, 'Raw_Data', 'images')       # .png 原始图片
CSV_PATH = 'process_list.csv'                                   # 样本列表

# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, 'BEPH_Features')
os.makedirs(os.path.join(OUTPUT_DIR, 'h5_files'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'pt_files'), exist_ok=True)

# 模型配置 (保持你提供的路径不变)
MODEL_CONFIG = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/mmselfsup/configs/tsne/beitv2_base.py'
MODEL_CHECKPOINT = '/data/home/wangzz_group/zhaipengyuan/BEPH-main/checkpoints/BEPH_backbone.pth'

# 参数设置
TARGET_PATCH_SIZE = 224  # 模型输入大小
BATCH_SIZE = 128         # 显存允许的话建议调大 (48 -> 128 或 256) 加速
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# =======================================================

# --- 辅助类：模拟 OpenSlide ---
class PILSlide:
    def __init__(self, path):
        Image.MAX_IMAGE_PIXELS = None
        self.img = Image.open(path).convert('RGB')
        self.level_dimensions = [self.img.size] 
        
    def read_region(self, location, level, size):
        x, y = location
        w, h = size
        return self.img.crop((x, y, x + w, y + h))

# --- Dataset 定义 ---
def eval_transforms(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, file_path, wsi, target_patch_size=224):
        self.wsi = wsi
        self.file_path = file_path
        self.target_patch_size = (target_patch_size, target_patch_size)
        self.roi_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_size = f['coords'].attrs['patch_size']
            self.patch_level = f['coords'].attrs['patch_level']
            self.length = len(dset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        
        # 1. 从原图切取 (48x48)
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        
        # 2. 放大到模型需求 (224x224)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size, Image.Resampling.LANCZOS)
            
        img = self.roi_transforms(img)
        return img, coord

def collate_features(batch):
    img = torch.stack([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

# ================= 核心处理逻辑 (修正版：增加保存 .pt) =================
def process_one_slide(model, slide_id, h5_path, slide_path):
    output_h5_name = os.path.join(OUTPUT_DIR, 'h5_files', slide_id + '.h5')
    output_pt_name = os.path.join(OUTPUT_DIR, 'pt_files', slide_id + '.pt') # <--- 新增
    
    # 跳过检查 (如果两个都存在才跳过)
    if os.path.exists(output_h5_name) and os.path.exists(output_pt_name):
        print(f"⏩ [跳过] {slide_id} 已存在。")
        return

    try:
        # 1. 准备数据
        wsi = PILSlide(slide_path)
        dataset = Whole_Slide_Bag_FP(file_path=h5_path, wsi=wsi, target_patch_size=TARGET_PATCH_SIZE)
        
        if len(dataset) == 0:
            print(f"⚠️ [跳过] {slide_id} 没有任何 patch。")
            return

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, collate_fn=collate_features)
        
        # 2. 提取特征
        features_list = []
        coords_list = []
        
        for batch, coords in tqdm(loader, desc=f"Ext: {slide_id}", leave=False):
            with torch.no_grad():
                batch = batch.to(DEVICE)
                
                # mmselfsup 提取特征
                if hasattr(model, 'module'):
                    feat = model.module.extract_feat(batch, stage='backbone')[0]
                else:
                    feat = model.extract_feat(batch, stage='backbone')[0]
                
                if len(feat.shape) == 3: 
                    feat = feat[:, 0, :] 
                
                features_list.append(feat.cpu().numpy().astype(np.float32)) 
                coords_list.append(coords)

        # 3. 整合
        all_features = np.concatenate(features_list, axis=0)
        all_coords = np.concatenate(coords_list, axis=0)
        
        # 4. 保存 .h5 (特征 + 坐标)
        with h5py.File(output_h5_name, 'w') as f:
            f.create_dataset('features', data=all_features)
            f.create_dataset('coords', data=all_coords)
            
        # 5. 保存 .pt (纯特征，用于快速训练) <--- 新增部分
        torch.save(torch.from_numpy(all_features), output_pt_name)
            
        print(f"✅ 完成: {slide_id} -> Shape: {all_features.shape}")

    except Exception as e:
        print(f"❌ 错误 {slide_id}: {e}")
# ================= 主程序入口 =================
if __name__ == "__main__":
    # 1. 加载模型 (只加载一次！)
    print("正在加载 BEPH 模型...")
    try:
        cfg = Config.fromfile(MODEL_CONFIG)
        init_default_scope(cfg.get('default_scope', 'mmselfsup'))
        model = init_model(cfg, MODEL_CHECKPOINT, device=DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()

    # 2. 读取列表
    if not os.path.exists(CSV_PATH):
        print(f"CSV文件未找到: {CSV_PATH}")
        exit()
        
    df = pd.read_csv(CSV_PATH)
    ids = df.iloc[:, 0].tolist() if 'slide_id' not in df.columns else df['slide_id'].tolist()
    
    print(f"🚀 开始批量提取特征，共 {len(ids)} 个样本...")
    
    # 3. 循环处理
    for slide_id in tqdm(ids, desc="Total Progress"):
        # 构造路径
        h5_file = os.path.join(PATCHES_DIR, f"{slide_id}.h5")
        img_file = os.path.join(IMAGES_DIR, f"{slide_id}.png")
        
        if not os.path.exists(h5_file):
            print(f"❌ 缺失 Patch 坐标文件: {h5_file}")
            continue
        if not os.path.exists(img_file):
            print(f"❌ 缺失图片文件: {img_file}")
            continue
            
        process_one_slide(model, str(slide_id), h5_file, img_file)
        
    print("\n✨ 全部任务结束！")
    print(f"特征已保存在: {OUTPUT_DIR}")