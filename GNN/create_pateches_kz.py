# /data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/
"""
=============================================================================
【BEPH 前置处理】H&E 图像切割坐标生成器 (Patch Coordinate Extractor)

核心功能: 
它不提取特征！它的唯一任务是计算并保存“切蛋糕的位置”。
它读取空间转录组基因矩阵 (.h5ad) 中的物理坐标，结合 H&E 图像的缩放因子，计算出在高清图片上切割 80x80 小图像块 (Patch) 时所需的左上角像素坐标，并打包存为 .h5 文件。

流程简述:
1. 读取基因矩阵: 使用 Scanpy 读取 .h5ad，提取其中保存的 `adata.obsm['spatial']` 物理坐标。
2. 智能缩放计算: 深度解析 adata 内部隐藏的 `scalefactors` (缩放因子)，自动判断你提供的是高清 (hires) 还是低清图，将物理坐标精准转换为图片上的实际像素点。
3. 边界防爆检测: 计算出 80x80 方块的四个角，如果发现某个方块切到了图片的外面（越界），直接无情丢弃，防止后续图像读取代码崩溃。
4. 打包坐标: 将所有合法的左上角坐标保存为 `{slide_id}.h5` 文件。

与上下文的关系:
它是 `extract_features.py` (刚刚我给你加注释的那个脚本) 的前置！
你必须先运行这个脚本生成坐标 (`.h5`文件)，`extract_features.py` 才能知道要去图片的哪里切图并提取特征。
=============================================================================
"""
import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
ROOT_DIR = './kz_data' 
H5AD_DIR = os.path.join(ROOT_DIR, 'Raw_Data', 'Log1p')
IMAGE_DIR = os.path.join(ROOT_DIR, 'Raw_Data', 'images')
CSV_PATH = 'process_list.csv'

OUTPUT_DIR = os.path.join(ROOT_DIR, 'Segmentation')
PATCHES_SAVE_DIR = os.path.join(OUTPUT_DIR, 'patches')

SRC_SIZE = 80
TARGET_SIZE = 224

# ================= 修复后的处理函数 =================
def process_single_sample(slide_id, h5ad_path, image_path):
    if not os.path.exists(h5ad_path) or not os.path.exists(image_path):
        return False

    try:
        adata = sc.read_h5ad(h5ad_path)
        Image.MAX_IMAGE_PIXELS = None 
        full_image = Image.open(image_path)
        img_w, img_h = full_image.size

        if 'spatial' not in adata.obsm.keys():
            return False

        coords = adata.obsm['spatial']
        
        # --- 🛠️ 关键修改开始：更强健的缩放因子获取逻辑 ---
        scale_factor = 1.0
        library_id = list(adata.uns.get('spatial', {}).keys())[0] if adata.uns.get('spatial') else None
        
        if library_id:
            scalefactors = adata.uns['spatial'][library_id].get('scalefactors', {})
            
            # 【强制逻辑】我们默认使用的图片就是 hires 图片
            # 不再通过文件名判断，而是直接取 tissue_hires_scalef
            if 'tissue_hires_scalef' in scalefactors:
                scale_factor = scalefactors['tissue_hires_scalef']
                print(f"   Using hires scale factor: {scale_factor:.4f}")
            elif 'tissue_lowres_scalef' in scalefactors:
                # 如果没有hires因子，尝试用lowres（极少见）
                scale_factor = scalefactors['tissue_lowres_scalef']
                print(f"   Warning: Using lowres scale factor: {scale_factor:.4f}")
        else:
            print("   ⚠️ Warning: No spatial metadata found in uns. Using scale=1.0")
        # ------------------------------------------------

        # 计算像素坐标
        pixel_coords = (coords * scale_factor).astype(int)
        top_left_coords = pixel_coords - (SRC_SIZE // 2)
        
        # --- 🛠️ 调试打印 (只打印第一个点) ---
        if len(coords) > 0:
            print(f"   [Debug] 图片尺寸: {img_w}x{img_h}")
            print(f"   [Debug] 原始坐标示例: {coords[0]}")
            print(f"   [Debug] 缩放后坐标示例: {pixel_coords[0]}")
        # ----------------------------------

        valid_coords = []
        for x, y in top_left_coords:
            # 这里的判断非常严格，只要有一个角出界就丢弃
            if x >= 0 and y >= 0 and x + SRC_SIZE <= img_w and y + SRC_SIZE <= img_h:
                valid_coords.append([x, y])
        
        valid_coords = np.array(valid_coords)

        if len(valid_coords) == 0:
            print(f"   ❌ 错误：有效 Patch 数量为 0！请检查缩放因子是否正确。")
            return False

        # 保存
        os.makedirs(PATCHES_SAVE_DIR, exist_ok=True)
        h5_save_path = os.path.join(PATCHES_SAVE_DIR, f'{slide_id}.h5')
        
        with h5py.File(h5_save_path, 'w') as f:
            dset = f.create_dataset('coords', data=valid_coords)
            dset.attrs['patch_size'] = SRC_SIZE
            dset.attrs['patch_level'] = 0
            f.attrs['n_patches'] = len(valid_coords)
            
        print(f"   ✅ 保存成功: {len(valid_coords)} patches -> {os.path.basename(h5_save_path)}")
        return True

    except Exception as e:
        print(f"❌ 错误 {slide_id}: {e}")
        return False

# ================= 执行逻辑 =================
if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if 'slide_id' in df.columns:
            ids = df['slide_id'].tolist()
        else:
            ids = df.iloc[:, 0].tolist()
        
        print(f"🚀 开始处理 {len(ids)} 个样本...")
        
        for slide_id in tqdm(ids):
            # 这里的名字匹配非常重要，确保你的文件叫 SampleID.h5ad 和 SampleID.png
            h5ad_file = os.path.join(H5AD_DIR, f"{slide_id}.h5ad")
            img_file = os.path.join(IMAGE_DIR, f"{slide_id}.png") # 确保你的图片已经去掉后缀了
            
            process_single_sample(str(slide_id), h5ad_file, img_file)
    else:
        print("CSV not found.")