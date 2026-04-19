import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

class MorphologicalDropEdge(object):
    def __init__(self, p_overall=0.4):
        """
        基于边权重的形态学丢边增强模块。
        
        Args:
            p_overall (float): 整体丢边的基准概率。
                               权重越小的边，实际丢弃概率 > p_overall
                               权重越大的边，实际丢弃概率 < p_overall
        """
        self.p_overall = p_overall

    def __call__(self, data: Data) -> Data:
        """
        每次调用都会基于当前的概率分布，动态生成一个新的损坏图。
        """
        # 必须 clone，避免修改原始缓存的数据
        aug_data = data.clone()
        
        if aug_data.edge_attr is None:
            # 如果没有边权重，直接返回原图或抛出警告
            return aug_data
            
        edge_attr = aug_data.edge_attr.squeeze()
        edge_index = aug_data.edge_index

        row, col = edge_index
        # 只处理上三角矩阵（避免无向图重复计算）
        mask_upper = row <= col
        
        w_upper = edge_attr[mask_upper]
        row_upper = row[mask_upper]
        col_upper = col[mask_upper]
        
        # --- 1. Min-Max 归一化 ---
        w_min = edge_attr.min()
        w_max = edge_attr.max()
        
        if w_max - w_min < 1e-6:
            s_ij = w_upper
        else:
            s_ij = (w_upper - w_min) / (w_max - w_min)
            
        # --- 2. 计算动态丢边概率 ---
        # 权重 s_ij 越大（重要结构），(1-s_ij) 越小，p_drop 越低 -> 保留
        p_drop = self.p_overall * (1.0 - s_ij)
        p_drop = torch.clamp(p_drop, min=0.0, max=1.0)
        
        # --- 3. 伯努利采样 ---
        # 这里不设定 seed，利用 torch 默认的随机状态，保证每次 epoch 结果不同
        p_keep = 1.0 - p_drop
        keep_mask = torch.bernoulli(p_keep).to(torch.bool)
        
        # 筛选保留的边
        kept_row = row_upper[keep_mask]
        kept_col = col_upper[keep_mask]
        kept_attr = w_upper[keep_mask]
        kept_edge_index = torch.stack([kept_row, kept_col], dim=0)
        
        # --- 4. 恢复无向图对称性 ---
        new_edge_index, new_edge_attr = to_undirected(
            kept_edge_index, 
            kept_attr, 
            num_nodes=data.num_nodes
        )
        
        aug_data.edge_index = new_edge_index
        aug_data.edge_attr = new_edge_attr
        
        return aug_data