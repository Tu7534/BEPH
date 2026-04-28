import os
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ================= 🚀 路径魔法：适配 GNN/Visualization 结构 =================
# 当前脚本：GNN/Visualization/visualize_results.py
current_dir = os.path.dirname(os.path.abspath(__file__))
gnn_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(gnn_dir)

# 把 GNN 目录和根目录都加入搜索路径
for path in [gnn_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)
# =====================================================================

# 导入模型类
from train import GCLModel_Morph

def plot_spatial_comparison(pt_path, expr_path, model_path, truth_path, hidden_dim, out_dim, save_name="comparison.png"):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ================= 1. 加载模型并提取特征 (刚刚被不小心删掉的部分) =================
    print(f"[-] 正在从 {model_path} 加载模型权重...")
    model = GCLModel_Morph(in_channels=15, hidden_channels=hidden_dim, out_channels=out_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print("[-] 正在提取图节点嵌入特征 (Embeddings)...")
    data = torch.load(pt_path).to(DEVICE)
    with torch.no_grad():
        _, node_emb = model(data.x, data.edge_index, data.edge_attr)
    embeddings = node_emb.cpu().numpy()  # 🌟 关键变量在这里诞生！

    # ================= 2. 从表达矩阵获取 Barcode 顺序 =================
    print("[-] 提取 Barcode 顺序...")
    adata = sc.read_h5ad(expr_path) if expr_path.endswith('.h5ad') else sc.read_10x_h5(expr_path)
    barcodes = adata.obs_names.astype(str).str.strip().str.replace('-1', '', regex=False)
    
    coord_df = pd.DataFrame({
        'barcode': barcodes,
        'idx': range(len(barcodes))
    })

    # ================= 3. 读取超级大表，同时获取标签和坐标 =================
    print("[-] 正在从表格中加载真实标签与坐标...")
    try:
        truth_full = pd.read_csv(truth_path, sep='\t', encoding='utf-16')
    except:
        truth_full = pd.read_csv(truth_path, sep='\t', encoding='utf-8')
    
    truth_full['ID'] = truth_full['ID'].astype(str).str.strip().str.replace('-1', '', regex=False)
    
    # 🕵️ 智能寻找坐标列
    cols_lower = {c.lower(): c for c in truth_full.columns}
    x_candidates = ['x', 'imagerow', 'image_row', 'pxl_col_in_fullres', 'array_col']
    y_candidates = ['y', 'imagecol', 'image_col', 'pxl_row_in_fullres', 'array_row']
    
    x_col = next((cols_lower[c] for c in x_candidates if c in cols_lower), None)
    y_col = next((cols_lower[c] for c in y_candidates if c in cols_lower), None)
    
    if x_col and y_col:
        print(f"[-] 成功在表格中找到坐标列: X='{x_col}', Y='{y_col}'")
    else:
        print(f"❌ 找不到坐标！请检查表格。现有的列名有: {truth_full.columns.tolist()}")
        return

    # ================= 4. 对齐并合并数据 =================
    print("[-] 正在对齐真实标签与特征...")
    merged = pd.merge(coord_df, truth_full[['ID', x_col, y_col, 'annot_type']], left_on='barcode', right_on='ID', how='inner')
    merged = merged.rename(columns={x_col: 'x', y_col: 'y'}) 
    
    # 提取对齐后的 embeddings
    aligned_embeddings = embeddings[merged['idx'].values]
    n_clusters = len(np.unique(merged['annot_type']))
    
    # ================= 5. 聚类 =================
    print(f"[-] 执行 K-Means (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    merged['pred'] = kmeans.fit_predict(aligned_embeddings)
    
    ari_val = adjusted_rand_score(merged['annot_type'], merged['pred'])

    # ================= 6. 绘图 =================
    print(f"[-] 正在生成对比图, 当前 ARI: {ari_val:.4f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    types = merged['annot_type'].unique()
    type_to_color = {t: i for i, t in enumerate(types)}
    colors_true = [type_to_color[t] for t in merged['annot_type']]
    
    # 真实图
    ax1.scatter(merged['x'], merged['y'], c=colors_true, cmap='Set1', s=30, edgecolor='none', alpha=0.8)
    ax1.set_title("Ground Truth (Manual Annotation)", fontsize=16)
    ax1.set_aspect('equal')

    # 预测图
    ax2.scatter(merged['x'], merged['y'], c=merged['pred'], cmap='Set1', s=30, edgecolor='none', alpha=0.8)
    ax2.set_title(f"MorphGAT Prediction (ARI: {ari_val:.4f})", fontsize=16)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"✅ 可视化结果已保存至: {save_name}")

if __name__ == "__main__":
    CONFIG = {
        "pt_path": os.path.join(project_root, "DATA_DIRECTORY/kz_data/Graph_pt/breast_cancer.pt"),
        "expr_path": os.path.join(project_root, "DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/filtered_feature_bc_matrix.h5"),
        "truth_path": os.path.join(project_root, "DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/metadata.txt"),
        "model_path": os.path.join(project_root, "GNN/checkpoints/best_model.pth"),
        "hidden_dim": 32,
        "out_dim": 32,
        "save_name": os.path.join(current_dir, "breast_cancer_comparison.png")
    }

    plot_spatial_comparison(**CONFIG)