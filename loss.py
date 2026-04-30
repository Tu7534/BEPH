import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, to_dense_batch



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def spatial_contrastive_loss(z1, z2, edge_index, x_raw, batch, temperature=0.2, hard_weight=3.0):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z1_b, mask = to_dense_batch(z1, batch)      
    z2_b, _ = to_dense_batch(z2, batch)         
    x_b, _ = to_dense_batch(x_raw, batch)       
    adj_b = to_dense_adj(edge_index, batch=batch)  

    batch_size, M, D = z1_b.size()
    total_loss = 0.0
    total_nodes = 0

    for i in range(batch_size):
        valid = mask[i].bool()
        ni = valid.sum().item()
        if ni == 0:
            continue

        zi1 = z1_b[i, valid]   
        zi2 = z2_b[i, valid]
        xi = x_b[i, valid]
        adj = adj_b[i][:ni, :ni]
        adj.fill_diagonal_(1.0)

        # 自适应困难负样本挖掘
        x_norm = F.normalize(xi, dim=1)
        raw_sim = torch.matmul(x_norm, x_norm.T)
        neighbors_only = adj - torch.eye(ni, device=adj.device)
        
        valid_neighbor_sims = raw_sim[neighbors_only.bool()]
        
        if valid_neighbor_sims.numel() > 0:
            adaptive_thresh = torch.quantile(valid_neighbor_sims.float(), 0.10)
            hard_neg_mask = (neighbors_only * (raw_sim < adaptive_thresh)).float()
        else:
            hard_neg_mask = torch.zeros_like(adj)

        # ✨ 优化 6: 构建 Multi-Positive 掩码
        # 正样本 = 自己 + 空间邻居 (必须剔除那些形态差异过大的困难负样本)
        pos_mask = adj - hard_neg_mask

        sim = torch.exp(torch.matmul(zi1, zi2.T) / temperature)  
        easy_neg_mask = (adj == 0).float()
        
        pos_sim_sum = (sim * pos_mask).sum(dim=-1)
        easy_neg_sim_sum = (sim * easy_neg_mask).sum(dim=-1)
        hard_neg_sim_sum = (sim * hard_neg_mask * hard_weight).sum(dim=-1)

        denom = pos_sim_sum + easy_neg_sim_sum + hard_neg_sim_sum + 1e-8
        
        # ✨ Multi-Positive InfoNCE 核心公式计算
        log_prob = torch.log(sim / denom.unsqueeze(-1) + 1e-8)
        loss_i = - (pos_mask * log_prob).sum(dim=-1) / (pos_mask.sum(dim=-1) + 1e-8)

        total_loss += loss_i.sum()
        total_nodes += ni

    if total_nodes == 0:
        return torch.tensor(0.0, device=z1.device)

    loss = total_loss / total_nodes
    return loss

def reconstruction_loss(reconstructed_x, original_x):
    return nn.MSELoss()(reconstructed_x, original_x)
