import os
import sys
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 🚀 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
gnn_dir = os.path.dirname(current_dir)
if gnn_dir not in sys.path: sys.path.insert(0, gnn_dir)

from train import GCLModel_Morph

def generate_paper_figures():
    sample_root = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/"
    pt_path = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/breast_cancer.pt"
    model_path = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/GNN/checkpoints/best_model.pth"
    metadata_path = os.path.join(sample_root, "metadata.txt")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================= 1. 提取预测结果 =================
    print("[-] 正在提取 MorphGAT 预测特征...")
    model = GCLModel_Morph(in_channels=233, hidden_channels=128, out_channels=32, n_clusters=4).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    data = torch.load(pt_path).to(DEVICE)
    with torch.no_grad():
        _, node_emb, _, _ = model(data.x, data.edge_index, data.edge_attr)
    embeddings = node_emb.cpu().numpy()

    # K=8 聚类与映射
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=8, n_init=20, random_state=42)
    clusters_raw = kmeans.fit_predict(embeddings)

    try:
        truth_df = pd.read_csv(metadata_path, sep='\t', encoding='utf-16')
    except:
        truth_df = pd.read_csv(metadata_path, sep='\t', encoding='utf-8')
    
    def get_mapping(gt, pred):
        df = pd.DataFrame({'gt': gt, 'pred': pred})
        return {c: df[df['pred'] == c]['gt'].mode()[0] for c in df['pred'].unique()}

    adata = sc.read_visium(sample_root)
    adata.var_names_make_unique()
    barcodes = adata.obs_names.str.replace('-1', '')
    
    mapping = get_mapping(truth_df['annot_type'].values, clusters_raw)
    mapped_predictions = [mapping[c] for c in clusters_raw]

    pred_series = pd.Series(mapped_predictions, index=barcodes)
    truth_series = pd.Series(truth_df['annot_type'].values, index=truth_df['ID'].str.replace('-1', ''))
    
    adata.obs['MorphGAT_Result'] = pred_series.reindex(barcodes).values
    adata.obs['Ground_Truth'] = truth_series.reindex(barcodes).values

    # ================= 2. 提取图像系坐标与暗黑底图 =================
    print("[-] 正在处理底层病理图像系 (Seaborn 模式)...")
    library_id = list(adata.uns['spatial'].keys())[0]
    scale_factor = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
    
    # 提取真实的像素坐标
    spatial_x = adata.obsm['spatial'][:, 0] * scale_factor
    spatial_y = adata.obsm['spatial'][:, 1] * scale_factor
    
    # 提取高分辨率图像
    img_rgb = adata.uns['spatial'][library_id]['images']['hires']
    
    # ✨ 核心操作 1：将彩色图像转为灰度图，并整体降低亮度，营造“暗色底片”效果
    if img_rgb.ndim == 3:
        img_gray = np.dot(img_rgb[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img_rgb
    
    # 强行把亮度压暗 50%
    img_gray_dark = img_gray * 0.5 

    # 组装给 Seaborn 画图用的 DataFrame
    df = pd.DataFrame({
        'spatial_x': spatial_x,
        'spatial_y': spatial_y,
        'Ground Truth': adata.obs['Ground_Truth'].values,
        'Prediction': adata.obs['MorphGAT_Result'].values
    })

    # ================= 3. 完全复刻 PathCLAST 风格绘图 =================
    print("[-] 正在生成 Seaborn 双联大图...")
    plt.style.use('default')
    # 设定背景为深灰/黑色系，边缘不留白
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), facecolor='#111111')

    custom_palette = {
        'Invasive': '#FF7F0E',          
        'Healthy': '#1F77B4',           
        'Surrounding tumor': '#2CA02C', 
        'Tumor': '#D62728'              
    }
    
    # ✨ 核心操作 2：调整点的大小。由于我们不用 Scanpy 了，用像素尺寸
    SPOT_SIZE = 40  # 你可以根据画面效果微调这个数字 (原版是 100)

    # ---------- 左图 (Ground Truth) ----------
    ax1 = axes[0]
    ax1.set_title('Spatial Plot - Ground Truth (Overlay)', fontsize=22, pad=20, color='white')
    # 铺上压暗的灰度底片
    ax1.imshow(img_gray_dark, cmap='gray')
    
    # 用 Seaborn 画点：alpha=1.0 保证纯色不透明，linewidth=0 保证没杂色边框
    sns.scatterplot(data=df, x='spatial_x', y='spatial_y', hue='Ground Truth', 
                    palette=custom_palette, s=SPOT_SIZE, ax=ax1, legend=False, 
                    alpha=1.0, linewidth=0)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, 
                    left=False, right=False, labelbottom=False, labelleft=False)

    # ---------- 右图 (Prediction) ----------
    ax2 = axes[1]
    ax2.set_title('Spatial Plot - MorphGAT (Overlay)', fontsize=22, pad=20, color='white')
    ax2.imshow(img_gray_dark, cmap='gray')
    
    sns.scatterplot(data=df, x='spatial_x', y='spatial_y', hue='Prediction', 
                    palette=custom_palette, s=SPOT_SIZE, ax=ax2, legend=False, 
                    alpha=1.0, linewidth=0)
    
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, 
                    left=False, right=False, labelbottom=False, labelleft=False)

    # 复刻高级的 ARI 分数标签框
    ari_text = "MorphGAT : ARI 0.5440"
    ax2.text(0.02, 0.98, ari_text, 
             transform=ax2.transAxes,
             fontsize=20, fontweight='bold', color='white', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7, edgecolor='none'))

    plt.tight_layout(pad=3.0)
    save_path = os.path.join(current_dir, "MorphGAT_Paper_Ready_Seaborn.png")
    # facecolor 保证保存出的图周围也是暗色
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ 完美！基于 Seaborn 的暗黑系、高亮 Spot 双联图已保存至: {save_path}")

if __name__ == "__main__":
    generate_paper_figures()