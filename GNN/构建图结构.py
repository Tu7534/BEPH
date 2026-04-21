# cd /data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/
# 构造GNN可以直接使用的（.pt）图数据结构
"""
================================================================================
脚本名称: 构建图结构.py (Spatial Graph Constructor)

功能描述:
本项目 (MorphGAT) 的核心数据预处理脚本。主要用于将多模态数据融合成 GNN 
可以直接读取的 PyTorch Geometric (PyG) 图数据结构 (.pt 文件)。

处理逻辑:
1. 节点特征 (Node Features, x): 从计算好的 TPscore CSV 文件中读取 
   (包含 14 个通路得分 + 1 个综合 TPscore，共 15 维特征)。
2. 图拓扑结构 (Edge Index): 根据 .h5ad 文件中的 spot 物理空间坐标 (spatial)，
   利用 KNN 算法 (K=6) 构建物理空间邻接图。
3. 边权重 (Edge Attributes): 从 BEPH 提取的 .h5 形态学特征文件中读取每个 spot 
   的图像特征，计算相连节点间的余弦相似度，作为后续形态学随机删边 (Edge Dropping) 
   的置信度依据。

输出: 
生成包含 x, edge_index, edge_attr, pos 的 PyG Data 对象，并保存为 .pt 文件，
供 DataLoader 直接进行 Batch 训练。
================================================================================
"""

import os
import glob
import torch
import h5py
import numpy as np
import scanpy as sc
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from torch_geometric.data import Data
from tqdm import tqdm


# 引入我们在 utils.py 中写的 CSV 读取函数
from utils import load_spot_features 

class SpatialGraphBuilder:
    def __init__(self, k_neighbors=6):
        self.k = k_neighbors

    def load_and_normalize_morph_features(self, h5_path, key='features'):
        """读取 BEPH 提取的形态学特征，并进行 L2 归一化"""
        with h5py.File(h5_path, 'r') as f:
            if key not in f.keys():
                key = list(f.keys())[0]
            feat_numpy = f[key][:]
        feat_tensor = torch.from_numpy(feat_numpy).float()
        feat_normalized = F.normalize(feat_tensor, p=2, dim=1)
        return feat_normalized

    def build_physical_topology(self, coords, symmetric=True):
        """基于空间坐标构建 KNN 邻接图"""
        N = coords.shape[0]
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='kd_tree', metric='euclidean').fit(coords)
        _, indices = nbrs.kneighbors(coords)
        indices = indices[:, 1:] # 排除节点自己
        row = np.repeat(np.arange(N), self.k)
        col = indices.flatten()
        data = np.ones(len(row))
        
        adj = sparse.csr_matrix((data, (row, col)), shape=(N, N))
        if symmetric:
            adj = adj.maximum(adj.transpose()) # 确保图是对称的（无向图）
            
        adj_coo = adj.tocoo()
        edge_index = torch.stack([
            torch.from_numpy(adj_coo.row),
            torch.from_numpy(adj_coo.col)
        ], dim=0).long()
        return edge_index

    def compute_morphological_weights(self, edge_index, morph_features, mode='relu'):
        """计算相连边的形态学置信度 (余弦相似度)"""
        src_feat = morph_features[edge_index[0]]
        dst_feat = morph_features[edge_index[1]]
        # 因为前面已经做过 L2 normalize，这里的点积等价于余弦相似度
        weights = (src_feat * dst_feat).sum(dim=1) 
        
        if mode == 'relu':
            weights = torch.relu(weights) # 砍掉负相关的边权重
        elif mode == 'scaled':
            weights = (weights + 1) / 2   # 映射到 0~1 区间
        return weights

    def process(self, h5_path, csv_path, coords_array, barcodes):
        """
        总控流程：将通路特征、形态学权重和空间拓扑组合成 Data 对象
        """
        # 1. 提取形态学特征 (仅用于计算边权重)
        morph_features = self.load_and_normalize_morph_features(h5_path)
        
        # 2. 🌟提取通路特征 (作为节点真正的输入特征 x)🌟
        node_features = load_spot_features(csv_path, barcodes)
        
        # 3. 构建物理拓扑
        edge_index = self.build_physical_topology(coords_array)
        
        # 4. 计算边的形态学权重
        edge_attr = self.compute_morphological_weights(edge_index, morph_features, mode='scaled')
        
        # 5. 打包为 PyG Data
        # 建议把物理坐标也存进 pos 里，以后做可视化或计算空间距离惩罚时非常有用
        data = Data(x=node_features, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    pos=torch.tensor(coords_array, dtype=torch.float32))
        return data


# ==========================================
# 批量处理
# ==========================================
def run_batch_processing(h5ad_dir, feature_dir, csv_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[-] 输出目录已就绪: {output_dir}")

    h5ad_files = glob.glob(os.path.join(h5ad_dir, "*.h5ad"))
    if not h5ad_files:
        print(f"❌ 错误: 在 {h5ad_dir} 下未找到任何 .h5ad 文件！")
        return

    print(f"[-] 找到 {len(h5ad_files)} 个样本待处理...")
    builder = SpatialGraphBuilder(k_neighbors=6)
    
    success_count, fail_count = 0, 0

    for h5ad_path in tqdm(h5ad_files, desc="Processing Graphs"):
        filename = os.path.basename(h5ad_path)
        sample_name = os.path.splitext(filename)[0]
        
        # 路径拼接
        feature_path = os.path.join(feature_dir, sample_name + ".h5")
        # 🌟 新增：指向你之前统一生成的 TPscore CSV 文件
        csv_path = os.path.join(csv_dir, sample_name + "_stmeta.data.csv") 
        save_path = os.path.join(output_dir, sample_name + ".pt")

        # 检查文件完整性
        if not os.path.exists(feature_path):
            print(f"\n⚠️ 跳过: 找不到 H5 形态特征 ({feature_path})")
            fail_count += 1; continue
        if not os.path.exists(csv_path):
            print(f"\n⚠️ 跳过: 找不到 CSV 通路特征 ({csv_path})")
            fail_count += 1; continue

        try:
            # 读取空转原始数据，获取坐标和 Barcode 顺序
            adata = sc.read_h5ad(h5ad_path)
            
            # 兼容不同平台存放坐标的键名
            spatial_keys = [k for k in adata.obsm.keys() if 'spatial' in k.lower()]
            if not spatial_keys:
                raise KeyError(f"No spatial coords found in {filename}")
            coords = adata.obsm[spatial_keys[0]]
            if hasattr(coords, 'values'):
                coords = coords.values

            # 获取当前切片的 spot barcodes，保证特征顺序绝对对齐
            barcodes = adata.obs_names.tolist()

            # 核心构建
            data = builder.process(feature_path, csv_path, coords, barcodes)
            data.sample_name = sample_name 

            torch.save(data, save_path)
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 失败: {sample_name} - 错误信息: {e}")
            fail_count += 1

    print("\n" + "="*30)
    print(f"处理汇总: ✅ 成功: {success_count} | ❌ 失败: {fail_count}")
    print("="*30)

if __name__ == "__main__":
    # 配置路径
    H5AD_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Raw_Data/Log1p/"
    FEATURE_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/BEPH_Features/h5_files/"
    # 🌟 新增：之前集中存放 TPscore 的文件夹
    CSV_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/TPscore/" 
    OUTPUT_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/"

    run_batch_processing(H5AD_DIR, FEATURE_DIR, CSV_DIR, OUTPUT_DIR)