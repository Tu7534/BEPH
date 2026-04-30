"""
=============================================================================================
脚本名称: 多模态融合建图.py (Multi-modal Graph Construction)

💡 核心架构创新 (Key Innovations):
    1. 摒弃传统的“原始基因表达”建图法，使用 Decoupler (MLM) 将 ST 数据提炼为【233 维泛癌微环境特征】。
    2. 摒弃传统的“物理距离”定边法，使用【深度学习微观图像特征的余弦相似度】作为连通权重。

🕸️ GNN 图网络拓扑定义 (Graph Topology):
    - 节点特征 (Node Features, X)      <- 233 维高阶通路得分 (囊括 Hallmark / 泛免疫 / 基质重塑)
    - 空间连边 (Edge Index)            <- 空间转录组 Hexagonal Grid (6 邻居原位物理拓扑)
    - 边权重 (Edge Attributes)         <- 基于 BEiT 等模型提取的病理 Patch 形态学相似度
=============================================================================================
"""



import os
import gc
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import squidpy as sq
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

def process_single_slide(slide_id, h5ad_path, img_features_path, save_path, net):
    try:
        # 1. 基因分支
        adata = sc.read_h5ad(h5ad_path)
        adata.var_names_make_unique()

        if 'spatial' not in adata.obsm.keys():
            if 'X_spatial' in adata.obsm.keys():
                adata.obsm['spatial'] = adata.obsm['X_spatial']
            else:
                return False, f"找不到空间坐标 'spatial'"
        
        if adata.X.max() > 50: 
            # 过滤掉表达基因少于 200 的垃圾 Spot（这步必须在提取图像特征之前或同步进行）
            sc.pp.filter_cells(adata, min_genes=200) 
            sc.pp.normalize_total(adata, target_sum=1e4) 
            sc.pp.log1p(adata)

        # MLM 打分
        dc.run_mlm(mat=adata, net=net, source='source', target='target', weight=None, use_raw=False)
        
        all_pathways = sorted(net['source'].unique().tolist()) 
        pathway_scores_df = adata.obsm['mlm_estimate']
        
        original_dim = pathway_scores_df.shape[1]
        target_dim = len(all_pathways)
        
        # ========================================================
        # 🌟 严苛质量控制 (QC)：剔除小鼠或测序质量极差的切片 🌟
        # ========================================================
        if original_dim < 150:
            return False, f"QC未通过! 仅算出了 {original_dim}/{target_dim} 个通路，作为劣质/小鼠样本丢弃。"
        
        # 强行对齐列名，对少量缺失的通路补 0
        pathway_scores_df = pathway_scores_df.reindex(columns=all_pathways, fill_value=0.0)
        pathway_scores = pathway_scores_df.values
        
        scaler = StandardScaler()
        scaled_pathways = scaler.fit_transform(pathway_scores)
        x_tensor = torch.tensor(scaled_pathways, dtype=torch.float32)

        # 2. 图像分支
        sq.gr.spatial_neighbors(adata, n_rings=1, coord_type='grid', n_neighs=6)
        adj_matrix = adata.obsp['spatial_connectivities']
        edge_index = torch.tensor(np.vstack(adj_matrix.nonzero()), dtype=torch.long)

        if not os.path.exists(img_features_path):
            return False, f"未找到图像特征文件: {os.path.basename(img_features_path)}"

        image_features = np.load(img_features_path)
        if image_features.shape[0] != x_tensor.shape[0]:
            return False, f"维度不匹配! 图像特征数:{image_features.shape[0]} vs 基因Spot数:{x_tensor.shape[0]}"

        edge_weights = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            img_u = image_features[u]
            img_v = image_features[v]
            
            cos_sim = np.dot(img_u, img_v) / (np.linalg.norm(img_u) * np.linalg.norm(img_v) + 1e-8)
            weight = np.exp(cos_sim) / np.exp(1.0)
            edge_weights.append(weight)
            
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)

        # 3. 融合打包
        graph_data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
        torch.save(graph_data, save_path)
        
        del adata, x_tensor, edge_index, edge_attr, image_features, graph_data
        gc.collect()
        
        return True, f"通路对齐: {original_dim}维 -> {target_dim}维"
        
    except Exception as e:
        return False, str(e)

def batch_build_graphs():
    print("="*60)
    print("🚀 启动 [多模态融合] 泛癌图结构 批量构建流水线 (严苛 QC 版)")
    print("="*60)
    
    project_root = "/data/home/wangzz_group/zhaipengyuan/BEPH-main"
    h5ad_dir = os.path.join(project_root, "DATA_DIRECTORY/kz_data/Raw_Data/h5ad_files")
    img_feat_dir = os.path.join(project_root, "DATA_DIRECTORY/kz_data/Graph_pt")
    save_dir = os.path.join(project_root, "DATA_DIRECTORY/kz_data/Graph_pt")
    csv_path = os.path.join(project_root, "DATA_DIRECTORY/process_list.csv")
    
    net_path = os.path.join(project_root, "DATA_DIRECTORY/Pathway/pancancer_microenvironment_net.csv")
    if not os.path.exists(net_path):
        net_path = os.path.join(project_root, "Pathway/pancancer_microenvironment_net.csv")

    os.makedirs(save_dir, exist_ok=True)

    print(f"[-] 🎯 正在加载泛癌特征字典...")
    net = pd.read_csv(net_path)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        slide_ids = df['slide_id'].tolist() if 'slide_id' in df.columns else df.iloc[:, 0].tolist()
        print(f"[-] 从 process_list.csv 读取到 {len(slide_ids)} 个样本")
    else:
        slide_ids = [f.replace('.h5ad', '') for f in os.listdir(h5ad_dir) if f.endswith('.h5ad')]

    results = {'success': 0, 'skip': 0, 'fail': 0}
    pbar = tqdm(slide_ids, desc="批量建图进度")
    
    for slide_id in pbar:
        slide_id = str(slide_id)
        h5ad_path = os.path.join(h5ad_dir, f"{slide_id}.h5ad")
        img_features_path = os.path.join(img_feat_dir, f"{slide_id}_image_features.npy")
        save_path = os.path.join(save_dir, f"{slide_id}.pt")
        
        if os.path.exists(save_path):
            results['skip'] += 1
            tqdm.write(f"⏩ [跳过] 样本 {slide_id: <20} | 原因: .pt 文件已存在")
            continue
            
        if not os.path.exists(h5ad_path):
            results['fail'] += 1
            tqdm.write(f"❌ [失败] 样本 {slide_id: <20} | 原因: 找不到 .h5ad")
            continue

        success, msg = process_single_slide(slide_id, h5ad_path, img_features_path, save_path, net)
        
        if success:
            results['success'] += 1
            tqdm.write(f"✅ [成功] 样本 {slide_id: <20} | 状态: {msg}")
        else:
            results['fail'] += 1
            tqdm.write(f"❌ [拒绝] 样本 {slide_id: <20} | 原因: {msg}")

    print("\n" + "="*50)
    print(f"✨ 批量图数据构建全部完成！")
    print(f"✅ 成功生成 (完美可用): {results['success']} 个")
    print(f"⏩ 跳过已有: {results['skip']} 个")
    print(f"❌ 剔除垃圾/报错: {results['fail']} 个")
    print("="*50 + "\n")

if __name__ == "__main__":
    batch_build_graphs()