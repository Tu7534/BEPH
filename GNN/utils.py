import scanpy as sc
import os
import pandas as pd

def load_ST_file(file_path):
    print(f"Loading data from {file_path}...")
    adata = sc.read_h5ad(file_path)
    adata.var_names_make_unique()

    if 'spatial' not in adata.obsm.keys():
        coord_cols = ['pxl_row_in_fullres', 'pxl_col_in_fullres']
        if all(col in adata.obs.columns for col in coord_cols):
            adata.obsm['spatial'] = adata.obs[coord_cols].to_numpy()
            adata.obs.drop(columns=coord_cols, inplace=True)
        else:
            print("Warning: 未找到空间坐标，但这不影响基因特征提取。")

    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


import pandas as pd
import torch

def load_spot_features(csv_path, spot_barcodes):
    """
    读取 TPscore 文件，并严格按照图中节点的 barcode 顺序对齐特征
    
    :param csv_path: 你生成的 _stmeta.data.csv 文件路径
    :param spot_barcodes: 当前切片构建图时，所有节点对应的 barcode 列表 
    :return: PyTorch Tensor, 形状为 [N, num_features]
    """
    # 1. 读取 CSV，自动将第一列（Barcode）设为索引
    df = pd.read_csv(csv_path, index_col=0)
    
    # 2. 我们使用方案B：提取 14 个通路得分 + 综合 TPscore = 15 维特征
    pathways = ['Angiogenesis', 'Apoptosis', 'Cell_Cycle', 'DNA_damage', 'DNA_repair', 
                'Differentiation', 'EMT', 'Hypoxia', 'Inflammation', 'Invasion', 
                'Metastasis', 'Proliferation', 'Quiescence', 'Stemness', 'TPscore']
    
    # 核心对齐：强制按照图节点的顺序提取数据，防止特征错位
    features_df = df.loc[spot_barcodes, pathways]
    
    # 3. 检查是否有缺失值并填充
    if features_df.isnull().values.any():
        features_df = features_df.fillna(0.0)

    # 4. 转换为 PyTorch 支持的 FloatTensor
    x = torch.tensor(features_df.values, dtype=torch.float32)
    
    return x