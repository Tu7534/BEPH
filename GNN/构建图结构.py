# cd /data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/
# 构造GNN可以直接使用的（.pt）图数据结构

import os
import sys
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


class SpatialGraphBuilder:
    def __init__(self, k_neighbors=6):
        self.k = k_neighbors

    def load_and_normalize_features(self, h5_path, key='features'):
        with h5py.File(h5_path, 'r') as f:
            if key not in f.keys():
                key = list(f.keys())[0]
            feat_numpy = f[key][:]
        feat_tensor = torch.from_numpy(feat_numpy).float()
        feat_normalized = F.normalize(feat_tensor, p=2, dim=1)
        return feat_normalized

    def build_physical_topology(self, coords, symmetric=True):
        N = coords.shape[0]
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='kd_tree', metric='euclidean').fit(coords)
        _, indices = nbrs.kneighbors(coords)
        indices = indices[:, 1:]
        row = np.repeat(np.arange(N), self.k)
        col = indices.flatten()
        data = np.ones(len(row))
        adj = sparse.csr_matrix((data, (row, col)), shape=(N, N))
        if symmetric:
            adj = adj.maximum(adj.transpose())
        adj_coo = adj.tocoo()
        edge_index = torch.stack([
            torch.from_numpy(adj_coo.row),
            torch.from_numpy(adj_coo.col)
        ], dim=0).long()
        return edge_index

    def compute_morphological_weights(self, edge_index, features, mode='relu'):
        # 计算形态学权重
        src_feat = features[edge_index[0]]
        dst_feat = features[edge_index[1]]
        weights = (src_feat * dst_feat).sum(dim=1)
        if mode == 'relu':
            weights = torch.relu(weights)
        elif mode == 'scaled':
            weights = (weights + 1) / 2
        return weights

    def process(self, h5_path, coords_array):
        """
        总控，将上面三个步骤串联起来，最后打包成一个 torch_geometric.data.Data 对象
        x:节点特征矩阵
        edge_index:边的连接关系
        edge_attr: 边的权重
        """
        x = self.load_and_normalize_features(h5_path)
        edge_index = self.build_physical_topology(coords_array)
        edge_attr = self.compute_morphological_weights(edge_index, x)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

# ==========================================
# 批量处理
# ==========================================

def run_batch_processing(h5ad_dir, feature_dir, output_dir):
    # 1. 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[-] 已创建输出目录: {output_dir}")
    else:
        print(f"[-] 输出目录已存在: {output_dir}")

    # 2. 获取所有 h5ad 文件列表
    # 假设文件名是 "SampleA.h5ad"
    h5ad_files = glob.glob(os.path.join(h5ad_dir, "*.h5ad"))
    
    if len(h5ad_files) == 0:
        print(f"❌ 错误: 在 {h5ad_dir} 下未找到任何 .h5ad 文件！")
        return

    print(f"[-] 找到 {len(h5ad_files)} 个样本待处理...")
    
    # 初始化构建器
    builder = SpatialGraphBuilder(k_neighbors=6)
    
    # 成功和失败计数
    success_count = 0
    fail_count = 0

    # 3. 开始循环处理 (使用 tqdm 显示进度条)
    for h5ad_path in tqdm(h5ad_files, desc="Processing"):
        
        # 获取文件名 (不带扩展名)，例如 "ColorectalCancer_10x"
        filename = os.path.basename(h5ad_path)
        sample_name = os.path.splitext(filename)[0]
        
        # 构建对应的特征文件路径
        # 假设特征文件和 h5ad 文件同名，只是后缀是 .h5
        feature_path = os.path.join(feature_dir, sample_name + ".h5")
        
        # 构建输出文件路径
        save_path = os.path.join(output_dir, sample_name + ".pt")

        # 检查特征文件是否存在
        if not os.path.exists(feature_path):
            print(f"\n⚠️ 跳过: {sample_name} - 找不到对应的 .h5 特征文件 ({feature_path})")
            fail_count += 1
            continue

        try:
            # --- 步骤 A: 读取坐标 ---
            adata = sc.read_h5ad(h5ad_path)
            if 'spatial' in adata.obsm.keys():
                coords = adata.obsm['spatial']
            else:
                # 尝试其他常见键名
                possible_keys = [k for k in adata.obsm.keys() if 'spatial' in k]
                if possible_keys:
                    coords = adata.obsm[possible_keys[0]]
                else:
                    raise KeyError(f"No spatial coords found in {filename}")

            if hasattr(coords, 'values'):
                coords = coords.values

            # --- 步骤 B: 处理数据 ---
            # 这一步会调用我们之前写的完整流程
            data = builder.process(feature_path, coords)
            
            # 附加元数据 (可选，方便以后知道这是哪个样本)
            data.sample_name = sample_name 

            # --- 步骤 C: 保存 ---
            torch.save(data, save_path)
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 失败: {sample_name} - 错误信息: {e}")
            fail_count += 1
            continue

    print("\n" + "="*30)
    print(f"处理汇总:")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📂 结果保存在: {output_dir}")
    print("="*30)

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置路径
    H5AD_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Raw_Data/Log1p/"
    FEATURE_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/BEPH_Features/h5_files/"
    
    # 输出路径
    OUTPUT_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/"

    # 运行
    run_batch_processing(H5AD_DIR, FEATURE_DIR, OUTPUT_DIR)