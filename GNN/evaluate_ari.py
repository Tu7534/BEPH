"""
=============================================================================
【工业级通用 ARI 评估脚本】 (文件内配参版)

核心功能:
- 支持任意空间转录组数据集的 ARI 评估。
- 自动识别读取 `.h5` (10x 官方格式) 或 `.h5ad` (Scanpy 标准格式)。
- 内置“暴力清洗”逻辑，自动抹平不同数据集中 Barcode 后缀和隐藏空格的差异，实现完美对齐。
=============================================================================
"""

import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# 导入你的模型类
from GNN.train import GCLModel_Morph 

def evaluate_ari(pt_path, expr_path, model_path, truth_path, hidden_dim, out_dim):
    """
    通用型 Barcode 对齐与 ARI 计算函数
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    print(f"[-] 正在加载模型权重: {model_path}")
    model = GCLModel_Morph(in_channels=15, hidden_channels=hidden_dim, out_channels=out_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 提取模型预测的嵌入 (Embeddings)
    print(f"[-] 正在加载图结构数据: {pt_path}")
    data = torch.load(pt_path).to(DEVICE)
    with torch.no_grad():
        _, node_emb = model(data.x, data.edge_index, data.edge_attr)
    embeddings = node_emb.cpu().numpy()

    # 3. 智能读取表达矩阵获取 Barcode
    print(f"[-] 正在加载表达矩阵: {expr_path}")
    if expr_path.endswith('.h5ad'):
        adata = sc.read_h5ad(expr_path)
    else:
        adata = sc.read_10x_h5(expr_path)
    
    barcodes = adata.obs_names.tolist()
    pred_df = pd.DataFrame({
        'barcode': barcodes,
        'embedding_idx': range(len(barcodes))
    })

    # 4. 读取真实标签文件 (尝试 utf-16 兼容 Excel 导出的文件，失败则回退 utf-8)
    print(f"[-] 正在加载真实标签: {truth_path}")
    try:
        truth_full = pd.read_csv(truth_path, sep='\t', encoding='utf-16')
        if len(truth_full.columns) < 2:  # 如果切错分隔符，尝试逗号
            truth_full = pd.read_csv(truth_path, sep=',', encoding='utf-16')
    except UnicodeError:
        truth_full = pd.read_csv(truth_path, sep='\t', encoding='utf-8')
        if len(truth_full.columns) < 2:
            truth_full = pd.read_csv(truth_path, sep=',', encoding='utf-8')

    # 从大表中精确提取我们需要的两列 (根据你的表格截图，列名是 ID 和 ground_truth)
    truth_df = pd.DataFrame()
    truth_df['barcode'] = truth_full['ID']
    truth_df['label'] = truth_full['ground_truth']  
    # 💡 提示：如果你想测试四大核心功能区(粗分类)的 ARI，可以把上一行改成 truth_full['annot_type']
    
    # ================= 🚀 暴力清洗 Barcode 保证绝对对齐 =================
    # 强制转字符串 -> 去掉前后空格回车 -> 去掉 '-1' 后缀
    pred_df['barcode'] = pred_df['barcode'].astype(str).str.strip().str.replace('-1', '', regex=False)
    truth_df['barcode'] = truth_df['barcode'].astype(str).str.strip().str.replace('-1', '', regex=False)
    
    print(f"🩺 H5 里的 Barcode 示例: {pred_df['barcode'].iloc[:3].tolist()}")
    print(f"🩺 标签里的 Barcode 示例: {truth_df['barcode'].iloc[:3].tolist()}")
    # ====================================================================

    # 5. 精确对齐
    merged_df = pd.merge(truth_df, pred_df, on='barcode', how='inner')
    merged_df = merged_df.dropna(subset=['label'])
    merged_df['label'] = merged_df['label'].astype(str)
    
    if len(merged_df) == 0:
        print("❌ 错误：合并后的 Spot 数量为 0。请检查上述打印的 Barcode 示例，或者确认分隔符是否为制表符！")
        return

    aligned_embeddings = embeddings[merged_df['embedding_idx'].values]
    true_labels = merged_df['label'].values

    # 6. 执行 K-Means 聚类
    n_clusters = len(np.unique(true_labels))
    print(f"[-] 正在对 {len(aligned_embeddings)} 个匹配的点进行聚类 (K={n_clusters})...")
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    
    emb_variance = np.var(aligned_embeddings, axis=0).mean()
    print(f"🩺 特征方差 (Variance): {emb_variance:.6f}")
    
    pred_labels = kmeans.fit_predict(aligned_embeddings)

    # 7. 计算 ARI
    ari_score = adjusted_rand_score(true_labels, pred_labels)

    print("\n" + "="*50)
    print(f"📊 最终对齐评估结果")
    print(f"🔥 Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"✅ 参与评估的 Spot 总数: {len(merged_df)}")
    print("="*50)

    # 裸跑基线测试
    raw_features = data.x.cpu().numpy() 
    aligned_raw_features = raw_features[merged_df['embedding_idx'].values]
    pred_labels_raw = kmeans.fit_predict(aligned_raw_features)
    ari_score_raw = adjusted_rand_score(true_labels, pred_labels_raw)
    
    print("\n" + "-"*50)
    print(f"裸跑基线 (仅使用原始 15 维特征) ARI: {ari_score_raw:.4f}")
    print("-"*50 + "\n")
    
    return ari_score

if __name__ == "__main__":
    # ============================================================
    # 🎯 配置区域 (每次跑不同数据时，只改这里即可) 🎯
    # ============================================================
    
    # 1. 图结构文件 (.pt)
    PT_FILE = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/breast_cancer.pt"
    
    # 2. 表达矩阵文件 (.h5 或 .h5ad)
    EXPR_FILE = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/filtered_feature_bc_matrix.h5"
    
    # 3. 真实标签文件 (.txt 或 .tsv)
    TRUTH_FILE = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/metadata.txt"
    
    # 4. 训练好的模型权重
    MODEL_WEIGHTS = "checkpoints/best_model.pth"

    # 5. 模型结构参数 (必须和刚才训练时设置的 --hidden_dim 保持一致)
    HIDDEN_DIM = 32  # 修复点：这里必须改成 32 以匹配你的训练权重
    OUT_DIM = 32
    # ============================================================

    evaluate_ari(
        pt_path=PT_FILE,
        expr_path=EXPR_FILE,
        model_path=MODEL_WEIGHTS,
        truth_path=TRUTH_FILE,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM
    )