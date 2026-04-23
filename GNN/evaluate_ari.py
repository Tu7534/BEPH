import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# 导入你的模型类
from 模型框架搭建 import GCLModel_Morph 

def evaluate_ari_aligned_breast_cancer(pt_path, h5_path, model_path, truth_path):
    """
    针对乳腺癌数据集进行 Barcode 对齐后的 ARI 计算
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型 (注意，这里的 hidden_channels 和 out_channels 必须和你训练时对应)
    # 如果你刚才用 --hidden_dim 32 训练了，这里就得是 32
    model = GCLModel_Morph(in_channels=15, hidden_channels=32, out_channels=32).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 提取模型预测的嵌入 (Embeddings)
    data = torch.load(pt_path).to(DEVICE)
    with torch.no_grad():
        _, node_emb = model(data.x, data.edge_index, data.edge_attr)
    embeddings = node_emb.cpu().numpy()

    # 3. 获取原始 Barcode 顺序
    # 使用 sc.read_10x_h5 读取 10x 官方的 H5 文件
    print(f"[-] 正在加载表达矩阵: {h5_path}")
    adata = sc.read_10x_h5(h5_path)
    
    # scanpy读取10x数据时，barcode通常是带 '-1' 后缀的，这和真实标签文件通常是一致的
    barcodes = adata.obs_names.tolist()

    # 创建一个包含预测结果的 DataFrame
    pred_df = pd.DataFrame({
        'barcode': barcodes,
        'embedding_idx': range(len(barcodes))
    })

    # 4. 读取真实标签文件
    # 根据之前生成的 metadata.txt，格式为: Barcode \t 标签
    truth_df = pd.read_csv(truth_path, sep='\t', header=None, names=['barcode', 'label'])
    print(f"[-] 成功加载真实标签: {len(truth_df)} 个 spot")

    # 5. 精确对齐：将预测的索引与真实标签合并
    merged_df = pd.merge(truth_df, pred_df, on='barcode', how='inner')
    
    # 清洗：剔除掉真实标签为空 (NaN) 的行，强制转换为字符串
    merged_df = merged_df.dropna(subset=['label'])
    merged_df['label'] = merged_df['label'].astype(str)
    
    if len(merged_df) == 0:
        print("❌ 错误：Barcode 无法匹配，请检查 H5 和 txt 文件的 Barcode 格式！")
        return

    # 提取清洗对齐后的 embeddings 
    aligned_embeddings = embeddings[merged_df['embedding_idx'].values]
    true_labels = merged_df['label'].values

    # 6. 执行 K-Means 聚类
    n_clusters = len(np.unique(true_labels))
    print(f"[-] 正在对 {len(aligned_embeddings)} 个匹配的点进行聚类 (K={n_clusters})...")
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    
    # === 特征健康度体检 ===
    emb_variance = np.var(aligned_embeddings, axis=0).mean()
    print(f"🩺 特征方差 (Variance): {emb_variance:.6f}")
    print(f"🩺 前 3 个点的特征预览:\n{aligned_embeddings[:3, :5]}")
    # ==========================
    
    pred_labels = kmeans.fit_predict(aligned_embeddings)

    # 7. 计算 ARI
    ari_score = adjusted_rand_score(true_labels, pred_labels)

    print("\n" + "="*40)
    print(f"📊 乳腺癌数据集最终对齐评估结果")
    print(f"🔥 Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"✅ 参与评估的 Spot 总数: {len(merged_df)}")
    print("="*40)

    # 基线测试：直接用原始特征跑
    raw_features = data.x.cpu().numpy() 
    aligned_raw_features = raw_features[merged_df['embedding_idx'].values]
    
    print("\n[-] 正在对原始的 15 维特征进行裸聚类...")
    pred_labels_raw = kmeans.fit_predict(aligned_raw_features)
    ari_score_raw = adjusted_rand_score(true_labels, pred_labels_raw)
    
    print("="*40)
    print(f"裸跑基线 Adjusted Rand Index (ARI): {ari_score_raw:.4f}")
    print("="*40)
    
    return ari_score

if __name__ == "__main__":
    # === 修改为乳腺癌数据的路径 ===
    DATA_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)"
    
    # 注意：你需要先跑一边你的图构建脚本，生成乳腺癌的 .pt 文件
    # 这里假设你生成的 pt 文件名为 breast_cancer.pt，保存在之前的 Graph_pt 文件夹下
    PT_FILE = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/breast_cancer.pt" 
    
    # 官方的表达矩阵
    H5_FILE = os.path.join(DATA_DIR, "filtered_feature_bc_matrix.h5")
    
    # 你刚刚生成的包含病理学标签的文本文件
    TRUTH_FILE = os.path.join(DATA_DIR, "metadata.txt")
    
    MODEL_WEIGHTS = "checkpoints/best_model.pth"

    evaluate_ari_aligned_breast_cancer(PT_FILE, H5_FILE, MODEL_WEIGHTS, TRUTH_FILE)