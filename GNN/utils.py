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