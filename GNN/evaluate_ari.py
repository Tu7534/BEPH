import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# 导入你的模型类
from 模型框架搭建 import GCLModel_Morph 

def evaluate_ari_aligned_151673(pt_path, h5ad_path, model_path, truth_path):
    """
    针对 151673 进行 Barcode 对齐后的 ARI 计算
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    model = GCLModel_Morph(in_channels=15, hidden_channels=128, out_channels=32).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 提取模型预测的嵌入 (Embeddings)
    data = torch.load(pt_path).to(DEVICE)
    with torch.no_grad():
        _, node_emb = model(data.x, data.edge_index, data.edge_attr)
    embeddings = node_emb.cpu().numpy()

    # 3. 获取原始 Barcode 顺序 (通过读取 h5ad)
    # 这一步至关重要，确保我们知道 embeddings 每一行对应哪个 spot
    adata = sc.read_h5ad(h5ad_path)
    barcodes = adata.obs_names.tolist()

    # 创建一个包含预测结果的 DataFrame
    pred_df = pd.DataFrame({
        'barcode': barcodes,
        'embedding_idx': range(len(barcodes))
    })

    # 4. 读取真实标签文件
    # 根据你提供的示例：AAACAAGTATCTCCCA-1  Layer_3 (空格或制表符分隔)
    truth_df = pd.read_csv(truth_path, sep='\s+', header=None, names=['barcode', 'label'])
    print(f"[-] 成功加载真实标签: {len(truth_df)} 个 spot")

    # 5. 精确对齐：将预测的索引与真实标签合并
    merged_df = pd.merge(truth_df, pred_df, on='barcode', how='inner')
    
    # 🌟 新增修复：剔除掉真实标签为空 (NaN) 的行
    merged_df = merged_df.dropna(subset=['label'])
    # 🌟 新增修复：强制将所有标签转换为字符串，防止数字和文本混杂
    merged_df['label'] = merged_df['label'].astype(str)
    
    if len(merged_df) == 0:
        print("❌ 错误：Barcode 无法匹配，请检查 h5ad 和 txt 文件的 Barcode 格式！")
        return

    # 提取清洗对齐后的 embeddings 
    aligned_embeddings = embeddings[merged_df['embedding_idx'].values]
    true_labels = merged_df['label'].values

    # 6. 执行 K-Means 聚类
    n_clusters = len(np.unique(true_labels))
    print(f"[-] 正在对 {len(aligned_embeddings)} 个匹配的点进行聚类 (K={n_clusters})...")
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    pred_labels = kmeans.fit_predict(aligned_embeddings)

    # 7. 🌟 计算 ARI 🌟
    ari_score = adjusted_rand_score(true_labels, pred_labels)

    print("\n" + "="*40)
    print(f"📊 样本 151673 最终对齐评估结果")
    print(f"🔥 Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"✅ 参与评估的 Spot 总数: {len(merged_df)}")
    print("="*40)

    return ari_score

if __name__ == "__main__":
    # 配置路径
    SAMPLE_ID = "151673"
    # 修改为你的实际路径
    BASE_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/"
    
    PT_FILE = os.path.join(BASE_DIR, "Graph_pt", f"{SAMPLE_ID}.pt")
    H5AD_FILE = os.path.join(BASE_DIR, "Raw_Data/Log1p", f"{SAMPLE_ID}.h5ad") # 原始空转数据
    MODEL_WEIGHTS = "checkpoints/best_model.pth"
    TRUTH_FILE = os.path.join(BASE_DIR, "151673/raw", f"{SAMPLE_ID}_truth.txt") 

    evaluate_ari_aligned_151673(PT_FILE, H5AD_FILE, MODEL_WEIGHTS, TRUTH_FILE)