import os
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_dense_adj

# 🌟 新增导入多种聚类算法
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# ================= 1. 路径配置 =================
BASE_DIR   = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data"
SAMPLE_ID  = "151673"
PT_PATH    = f"{BASE_DIR}/Graph_pt/{SAMPLE_ID}.pt"
H5AD_PATH  = f"{BASE_DIR}/Raw_Data/h5ad_files/{SAMPLE_ID}.h5ad"
LABEL_CSV  = f"{BASE_DIR}/10X/{SAMPLE_ID}/{SAMPLE_ID}_labels.csv"
MODEL_PATH = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/GNN/breast_3/checkpoints/run_20260523_111944/best_model.pth"

# ================= 2. 导入模型定义 =================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_3 import GCLModel_Morph

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[-] 设备: {DEVICE}")

    # --- A. 加载数据与对齐 ---
    print("[-] 加载数据并对齐...")
    data = torch.load(PT_PATH).to(DEVICE)
    adata = sc.read_h5ad(H5AD_PATH)
    df_labels = pd.read_csv(LABEL_CSV).set_index('barcode')
    
    # 统一去掉 barcode 的 '-1' 后缀以防万一
    if '-' in adata.obs_names[0] and '-' not in data.barcodes[0]:
        adata.obs_names = [b.split('-')[0] for b in adata.obs_names]
        df_labels.index = [b.split('-')[0] for b in df_labels.index]
    
    common_barcodes = sorted(list(set(data.barcodes) & set(adata.obs_names) & set(df_labels.index)))
    print(f"[*] 最终对齐公共 Spot 数: {len(common_barcodes)}")
    
    if len(common_barcodes) == 0:
        raise ValueError("❌ 对齐失败，请检查 Barcode 格式！")

    adata = adata[common_barcodes].copy()
    adata.obs['label'] = df_labels.loc[common_barcodes, 'layer']
    
    # --- B. 提取 Embedding ---
    print("[-] 提取模型特征...")
    model = GCLModel_Morph(in_channels=233, hidden_channels=256, out_channels=64).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model'], strict=False)
    model.eval()
    
    with torch.no_grad():
        _, node_emb, _, _ = model(data.x, data.edge_index, data.edge_attr)
    
    emb_df = pd.DataFrame(node_emb.cpu().numpy(), index=data.barcodes)
    embeddings = emb_df.loc[common_barcodes].values
    
    # --- C. 空间特征平滑 ---
    print("[-] 执行空间特征平滑...")
    adj_all = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].cpu().numpy()
    adj_df = pd.DataFrame(adj_all, index=data.barcodes, columns=data.barcodes)
    adj_aligned = adj_df.loc[common_barcodes, common_barcodes].values
    
    adj_norm = adj_aligned / (adj_aligned.sum(axis=1, keepdims=True) + 1e-6)
    embeddings_smoothed = 0.25 * embeddings + 0.75 * (adj_norm @ embeddings)
    
    # --- D. 强效空间融合 ---
    print("[-] 融合物理空间坐标...")
    coords = adata.obsm['spatial'].copy()
    coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
    
    # 空间权重 (DLPFC 推荐使用 1.0 ~ 2.0，我们这里取中 1.5)
    SPATIAL_WEIGHT = 1.5 
    feats = StandardScaler().fit_transform(embeddings_smoothed)
    X_combined = np.hstack([feats, coords_norm * SPATIAL_WEIGHT])
    adata.obsm['X_combined'] = X_combined
    
    # ================= 聚类大比武 =================
    print("\n[-] 开始执行多种聚类算法...")
    n_targets = 7 # DLPFC 真实类别数
    
    # 1. K-Means
    print("    -> 正在执行 KMeans...")
    kmeans = KMeans(n_clusters=n_targets, random_state=42)
    adata.obs['kmeans'] = kmeans.fit_predict(X_combined).astype(str)
    
    # 2. Gaussian Mixture Model (GMM)
    print("    -> 正在执行 GMM...")
    gmm = GaussianMixture(n_components=n_targets, covariance_type='tied', random_state=42)
    adata.obs['gmm'] = gmm.fit_predict(X_combined).astype(str)
    
    # 3. Agglomerative Clustering (层次聚类)
    print("    -> 正在执行 Agglomerative (层次聚类)...")
    agg = AgglomerativeClustering(n_clusters=n_targets)
    adata.obs['agg'] = agg.fit_predict(X_combined).astype(str)

    # 4. Leiden (图聚类，带自动搜索)
    print("    -> 正在执行 Leiden (自动搜索 7 簇)...")
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_combined')
    best_res = 0.5
    for res in np.linspace(0.05, 1.0, 40):
        sc.tl.leiden(adata, resolution=res, key_added='leiden_temp')
        if len(adata.obs['leiden_temp'].unique()) == n_targets:
            best_res = res
            break
    sc.tl.leiden(adata, resolution=best_res, key_added='leiden')
    
    # --- 计算指标并打印表格 ---
    methods = ['kmeans', 'gmm', 'agg', 'leiden']
    results = []
    true_labels = adata.obs['label'].values
    
    print(f"\n{'='*50}")
    print(f"{'Method':<12} | {'Clusters':<8} | {'ARI':<8} | {'NMI':<8}")
    print(f"{'-'*50}")
    
    for m in methods:
        preds = adata.obs[m].values
        n_c = len(np.unique(preds))
        ari = adjusted_rand_score(true_labels, preds)
        nmi = normalized_mutual_info_score(true_labels, preds)
        results.append({'Method': m, 'ARI': ari, 'NMI': nmi})
        print(f"{m:<12} | {n_c:<8} | {ari:.4f}   | {nmi:.4f}")
    print(f"{'='*50}\n")
    
    # --- 绘图 (UMAP 与 空间图) ---
    print("[-] 正在生成多方法对比可视化图像...")
    # 强制算一下 UMAP 以防报错
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)
        
    # 将聚类列转为 category 类型，Scanpy 才会分配离散颜色
    for m in methods:
        adata.obs[m] = adata.obs[m].astype('category')
        
    # 绘制 UMAP 对比
    sc.pl.umap(adata, color=['label'] + methods, ncols=3, show=False)
    plt.savefig("DLPFC_Multi_UMAP.png", dpi=200, bbox_inches='tight')
    
    # 绘制 Spatial 对比 (直接用坐标散点图，无需原图背景即可看出分层)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    methods_plot = ['label', 'kmeans', 'gmm', 'agg', 'leiden']
    titles = ['Ground Truth', 'K-Means', 'GMM', 'Agglomerative', 'Leiden']
    
    for i, (m, title) in enumerate(zip(methods_plot, titles)):
        sc.pl.embedding(adata, basis="spatial", color=m, ax=axes[i], show=False, title=title)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig("DLPFC_Multi_Spatial.png", dpi=200, bbox_inches='tight')
    
    print("✅ 分析完成！请查看 DLPFC_Multi_UMAP.png 和 DLPFC_Multi_Spatial.png")

if __name__ == "__main__":
    main()