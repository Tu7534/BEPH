import os
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

# ================= 🚀 路径魔法 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
gnn_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(gnn_dir)

for path in [gnn_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)
# =================================================

def map_predict_to_gt(gt_labels, pred_labels):
    """
    将高维预测簇 (如 8 类) 映射到低维 Ground Truth 标签上 (如 4 类)
    逻辑：看这个预测出来的簇里，哪种真实的标签最多，就把它归为哪一类。
    """
    df = pd.DataFrame({'gt': gt_labels, 'pred': pred_labels})
    mapping = {}
    for cluster in df['pred'].unique():
        # 找到该预测簇中，占比最高的真实标签
        majority_gt = df[df['pred'] == cluster]['gt'].mode()[0]
        mapping[cluster] = majority_gt
        
    print(f"[-] 自动亚群合并字典: {mapping}")
    return df['pred'].map(mapping).values

# 导入最新版的双 Loss + DEC 模型框架
from train import GCLModel_Morph

def plot_spatial_comparison(pt_path, expr_path, model_path, truth_path, hidden_dim, out_dim, n_clusters=8, save_name="comparison.png"):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ================= 1. 加载模型与特征 =================
    print(f"[-] 正在从 {model_path} 加载模型权重...")
    
    # 🌟 n_clusters 这里一定要传入你训练时设定的 8
    model = GCLModel_Morph(in_channels=15, hidden_channels=hidden_dim, out_channels=out_dim, n_clusters=n_clusters).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    print("[-] 正在提取 MorphGAT 图节点嵌入 (Embeddings)...")
    data = torch.load(pt_path).to(DEVICE)
    with torch.no_grad():
        _, node_emb, _, _ = model(data.x, data.edge_index, data.edge_attr)
    embeddings_gnn = node_emb.cpu().numpy()

    # ================= 2. 加载原始表达矩阵 =================
    print("[-] 正在加载原始 H5 表达矩阵 (用于基因基线)...")
    adata = sc.read_h5ad(expr_path) if expr_path.endswith('.h5ad') else sc.read_10x_h5(expr_path)
    barcodes = adata.obs_names.astype(str).str.strip().str.replace('-1', '', regex=False)
    
    coord_df = pd.DataFrame({
        'barcode': barcodes,
        'idx': range(len(barcodes))
    })

    # ================= 3. 读取大表与对齐 =================
    try:
        truth_full = pd.read_csv(truth_path, sep='\t', encoding='utf-16')
    except:
        truth_full = pd.read_csv(truth_path, sep='\t', encoding='utf-8')
    
    truth_full['ID'] = truth_full['ID'].astype(str).str.strip().str.replace('-1', '', regex=False)
    
    cols_lower = {c.lower(): c for c in truth_full.columns}
    x_col = next((cols_lower[c] for c in ['x', 'imagerow', 'image_row', 'pxl_col_in_fullres', 'array_col'] if c in cols_lower), None)
    y_col = next((cols_lower[c] for c in ['y', 'imagecol', 'image_col', 'pxl_row_in_fullres', 'array_row'] if c in cols_lower), None)
    
    merged = pd.merge(coord_df, truth_full[['ID', x_col, y_col, 'annot_type']], left_on='barcode', right_on='ID', how='inner')
    merged = merged.rename(columns={x_col: 'x', y_col: 'y'}) 
    
    # ================= 4. 计算不同特征的聚类 ARI =================
    print(f"\n[-] 开始执行 K-Means 聚类 (K={n_clusters})...")
    # 🌟 这里的 K 必须是你设定的 8 (n_clusters)，而不是 GT 的 4 类
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)

    # 4.1 MorphGAT 聚类与映射
    aligned_emb_gnn = embeddings_gnn[merged['idx'].values]
    merged['pred_gnn_raw'] = kmeans.fit_predict(aligned_emb_gnn)
    
    # 🌟 关键映射步骤：把跑出来的 8 个簇，映射回真实的标签文字
    merged['pred_gnn_mapped'] = map_predict_to_gt(merged['annot_type'], merged['pred_gnn_raw'])
    ari_gnn = adjusted_rand_score(merged['annot_type'], merged['pred_gnn_mapped'])

    # 4.2 纯基因表达聚类 (为了公平，基线也要过一遍相同的映射逻辑)
    print("[-] 正在计算纯基因表达 (PCA-50) 聚类与映射...")
    raw_X = adata.X
    if sp.issparse(raw_X):
        raw_X = raw_X.toarray()
    aligned_genes = raw_X[merged['idx'].values]
    
    pca = PCA(n_components=50, random_state=42)
    aligned_genes_pca = pca.fit_transform(aligned_genes)
    merged['pred_genes_raw'] = kmeans.fit_predict(aligned_genes_pca)
    
    # 基线映射
    merged['pred_genes_mapped'] = map_predict_to_gt(merged['annot_type'], merged['pred_genes_raw'])
    ari_genes = adjusted_rand_score(merged['annot_type'], merged['pred_genes_mapped'])

    print("\n" + "="*40)
    print(f"📊 映射后 ARI 对比报告 (Over-clustering to Mapping)")
    print(f"🧬 纯基因表达 (PCA): {ari_genes:.4f}")
    print(f"🔥 MorphGAT (GNN)  : {ari_gnn:.4f}")
    print("="*40 + "\n")

    # ================= 5. 三联画可视化 =================
    print("[-] 正在生成三联大图...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 9))
    for ax in [ax1, ax2, ax3]:
        ax.invert_yaxis()

    types = merged['annot_type'].unique()
    type_to_color = {t: i for i, t in enumerate(types)}
    colors_true = [type_to_color[t] for t in merged['annot_type']]
    
    # 真实图
    ax1.scatter(merged['x'], merged['y'], c=colors_true, cmap='Set1', s=30, edgecolor='none', alpha=0.9)
    ax1.set_title("Ground Truth (Manual Annotation)", fontsize=18, pad=15)
    ax1.set_aspect('equal')
    ax1.axis('off') 

    # 纯基因预测图 (使用映射后的颜色，保证和真实图颜色对应)
    colors_genes_mapped = [type_to_color[t] for t in merged['pred_genes_mapped']]
    ax2.scatter(merged['x'], merged['y'], c=colors_genes_mapped, cmap='Set1', s=30, edgecolor='none', alpha=0.9)
    ax2.set_title(f"Pure Gene (K={n_clusters}->Mapped) ARI: {ari_genes:.4f}", fontsize=18, pad=15)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # MorphGAT 预测图 (使用映射后的颜色)
    colors_gnn_mapped = [type_to_color[t] for t in merged['pred_gnn_mapped']]
    ax3.scatter(merged['x'], merged['y'], c=colors_gnn_mapped, cmap='Set1', s=30, edgecolor='none', alpha=0.9)
    ax3.set_title(f"MorphGAT (K={n_clusters}->Mapped) ARI: {ari_gnn:.4f}", fontsize=18, pad=15)
    ax3.set_aspect('equal')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化结果已保存至: {save_name}")

if __name__ == "__main__":
    CONFIG = {
        "pt_path": os.path.join(project_root, "DATA_DIRECTORY/kz_data/Graph_pt/breast_cancer.pt"),
        "expr_path": os.path.join(project_root, "DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/filtered_feature_bc_matrix.h5"),
        "truth_path": os.path.join(project_root, "DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/metadata.txt"),
        "model_path": os.path.join(project_root, "GNN/checkpoints/best_model.pth"),
        "hidden_dim": 32,
        "out_dim": 32,
        "n_clusters": 8, # 🌟 核心修改点：这里必须改成 8！
        "save_name": os.path.join(current_dir, "breast_cancer_comparison.png")
    }

    plot_spatial_comparison(**CONFIG)