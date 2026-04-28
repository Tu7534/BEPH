"""
================================================================================
脚本名称: 模型框架搭建.py (MorphGAT - Hard Negative & Raw Residual Edition)
级别: 工业级标准 (Production-Ready)
功能描述:
【究极升级版】：
1. GNN 架构升级：引入 "跨维度原始特征残差连接" (Raw Feature Residuals)，抵抗过度同化。
2. Loss 架构升级：在 Spatial InfoNCE 中加入 "困难负样本挖掘" (Hard Negative Mining)，锐化空间边界。
3. 继承模块：DEC 端到端聚类、特征重建损失、特征掩码。
================================================================================
"""

import os
import glob
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, to_dense_adj
from torch_geometric.nn.inits import glorot, zeros
from sklearn.cluster import KMeans

from corrupted_graph import MorphologicalDropEdge

# ==========================================
# 工具函数
# ==========================================
def setup_logger(log_dir):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = logging.getLogger("MorphGAT")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(); ch.setFormatter(formatter); logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log')); fh.setFormatter(formatter); logger.addHandler(fh)
    return logger

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# 1. 增强模块与数据集
# ==========================================
def apply_feature_masking(x, drop_prob=0.2):
    mask = torch.rand(x.size(0), device=x.device) > drop_prob
    x_masked = x.clone()
    x_masked[~mask] = 0.0
    return x_masked

class ContrastiveGraphDataset(Dataset):
    def __init__(self, root_dir, p_overall=0.4, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.augmentor = MorphologicalDropEdge(p_overall=p_overall)
        super().__init__(root_dir, transform, pre_transform)

    def len(self): return len(self.file_list)
    def get(self, idx):
        data_orig = torch.load(self.file_list[idx])
        if data_orig.x.shape[1] != 15:
            raise ValueError(f"\n❌ 发现脏数据！\n文件: {self.file_list[idx]}\n节点特征不是 15 维！")
        
        data_corr = self.augmentor(data_orig)
        data_orig.x = apply_feature_masking(data_orig.x, drop_prob=0.1)
        data_corr.x = apply_feature_masking(data_corr.x, drop_prob=0.2)
        return data_orig, data_corr

# ==========================================
# 2. 模型定义 (🚀 新增原始残差连接)
# ==========================================
class MorphGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        super().__init__(node_dim=0, aggr='add') 
        self.heads, self.out_channels, self.concat, self.dropout = heads, out_channels, concat, dropout
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels)) 
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        self.bias_lambda = nn.Parameter(torch.tensor(2.0)) 
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight); glorot(self.att_src); glorot(self.att_dst); zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        out = self.propagate(edge_index, x=x, alpha_src=alpha_src, alpha_dst=alpha_dst, edge_attr=edge_attr)
        out = out.view(-1, H * C) if self.concat else out.mean(dim=1)
        return out + self.bias

    def message(self, x_j, alpha_src_i, alpha_dst_j, edge_attr, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_src_i + alpha_dst_j, 0.2)
        if edge_attr is not None:
            alpha = alpha + self.bias_lambda * torch.log(edge_attr.view(-1, 1) + 1e-6)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)

class GCLModel_Morph(nn.Module):
    def __init__(self, in_channels=15, hidden_channels=128, out_channels=32, n_clusters=4):
        super().__init__()
        
        self.feature_proj = nn.Sequential(
            nn.Linear(in_channels, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU()
        )
        
        self.heads1 = 4 
        self.conv1 = MorphGATConv(hidden_channels, hidden_channels, heads=self.heads1, concat=True)
        self.skip_proj = nn.Linear(hidden_channels, hidden_channels * self.heads1)
        
        # 🚀 策略三：为原始特征准备的残差投影层
        self.raw_proj1 = nn.Linear(in_channels, hidden_channels * self.heads1)
        self.raw_proj2 = nn.Linear(in_channels, hidden_channels)
        
        self.conv2 = MorphGATConv(hidden_channels * self.heads1, hidden_channels, heads=1, concat=False)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, in_channels)
        )
        
        self.n_clusters = n_clusters
        self.alpha = 1.0 
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, out_channels))
        zeros(self.cluster_centers)

    def forward(self, x, edge_index, edge_attr):
        x_raw = x.clone() # 🚀 保留最纯净的 15 维生物学特征
        
        x = self.feature_proj(x)
        x_in = x  
        
        # 🚀 第一层卷积 + 常规残差 + 原始特征残差 (强行保留生物学底色)
        x = F.elu(self.conv1(x, edge_index, edge_attr)) + self.skip_proj(x_in) + self.raw_proj1(x_raw)
        
        # 🚀 第二层卷积 + 原始特征残差
        x = F.dropout(x, p=0.4, training=self.training)
        node_emb = self.conv2(x, edge_index, edge_attr) + self.raw_proj2(x_raw)
        
        z = self.proj_head(node_emb)
        rec_x = self.decoder(z)
        
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centers, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return z, node_emb, rec_x, q

# ==========================================
# 3. 损失函数 (🚀 新增困难负样本挖掘)
# ==========================================
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def spatial_contrastive_loss(z1, z2, edge_index, x_raw, temperature=0.2, hard_weight=3.0):
    """
    带困难负样本挖掘的 InfoNCE。
    x_raw: 原始 15 维特征，用于衡量细胞之间的真实生物学差异。
    hard_weight: 困难负样本的惩罚倍数，撕开边界的关键。
    """
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    N = z1.size(0)
    
    # 1. 基础相似度矩阵
    sim_matrix = torch.exp(torch.matmul(z1, z2.T) / temperature)
    pos_sim = torch.diag(sim_matrix)
    
    # 2. 空间邻接矩阵
    spatial_adj = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0)
    spatial_adj.fill_diagonal_(1.0)
    
    # 3. 简单负样本 (空间上离得远的)
    easy_neg_mask = (spatial_adj == 0).float()
    easy_neg_sim = (sim_matrix * easy_neg_mask).sum(dim=-1)
    
    # 🚀 4. 困难负样本 (Hard Negatives) 挖掘
    # 计算原始特征的相似度
    x_norm = F.normalize(x_raw, dim=1)
    raw_sim = torch.matmul(x_norm, x_norm.T)
    
    # 找出真正的空间邻居 (排除自己)
    neighbors_only = spatial_adj - torch.eye(N, device=z1.device)
    
    # 定义困难负样本：是空间邻居，但在 15 维原始特征上差异很大 (余弦相似度 < 0.5)
    hard_neg_mask = (neighbors_only * (raw_sim < 0.5)).float()
    
    # 对困难负样本施加数倍的推斥力 (hard_weight)
    hard_neg_sim = (sim_matrix * hard_neg_mask * hard_weight).sum(dim=-1)
    
    # 5. 计算终极 Loss
    loss = -torch.log(pos_sim / (pos_sim + easy_neg_sim + hard_neg_sim + 1e-8)).mean()
    return loss, torch.diag(torch.matmul(z1, z2.T)).mean().item(), 0.0

def reconstruction_loss(reconstructed_x, original_x):
    return nn.MSELoss()(reconstructed_x, original_x)

# ==========================================
# 4. 训练主引擎
# ==========================================
def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed); logger = setup_logger(args.save_dir)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 初始化终极训练任务 (HardNeg + Residual) | Device: {DEVICE}")

    full_dataset = ContrastiveGraphDataset(args.data_dir, p_overall=0.5)
    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GCLModel_Morph(in_channels=15, hidden_channels=args.hidden_dim, out_channels=32, n_clusters=args.n_clusters).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    history = {'pre_loss': [], 'fine_loss': []}

    # ---------------------------------------------------------
    # 阶段一：预训练
    # ---------------------------------------------------------
    logger.info(f"🔥 Phase 1: 预训练特征 ({args.pretrain_epochs} Epochs)...")
    for epoch in range(args.pretrain_epochs):
        model.train(); total_loss = 0
        for b_orig, b_corr in train_loader:
            b_orig, b_corr = b_orig.to(DEVICE), b_corr.to(DEVICE)
            optimizer.zero_grad()
            
            z1, _, rec_x1, _ = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
            z2, _, _, _ = model(b_corr.x, b_corr.edge_index, b_corr.edge_attr)
            
            # 🚀 传入原始特征 b_orig.x 用于困难负样本判别
            loss_cl, _, _ = spatial_contrastive_loss(z1, z2, b_orig.edge_index, b_orig.x, temperature=args.temp)
            loss_rec = reconstruction_loss(rec_x1, b_orig.x)
            loss = loss_cl + args.lambda_rec * loss_rec
            
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history['pre_loss'].append(avg_loss)
        if epoch % 5 == 0: logger.info(f"Pre-train Epoch {epoch} | Loss: {avg_loss:.4f}")

    # ---------------------------------------------------------
    # 阶段二：初始化聚类中心
    # ---------------------------------------------------------
    logger.info("🎯 Phase 2: 使用 KMeans 初始化 DEC 聚类中心...")
    model.eval(); all_z = []
    with torch.no_grad():
        for b_orig, _ in train_loader:
            z, _, _, _ = model(b_orig.to(DEVICE).x, b_orig.edge_index, b_orig.edge_attr)
            all_z.append(z.cpu())
    
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=args.seed)
    kmeans.fit(torch.cat(all_z).numpy())
    model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_).to(DEVICE)

    # ---------------------------------------------------------
    # 阶段三：联合微调
    # ---------------------------------------------------------
    logger.info(f"🚀 Phase 3: 联合微调 ({args.epochs} Epochs)...")
    best_val_loss = float('inf'); patience_counter = 0

    for epoch in range(args.epochs):
        model.train(); total_train_loss = 0
        for b_orig, b_corr in train_loader:
            b_orig, b_corr = b_orig.to(DEVICE), b_corr.to(DEVICE)
            optimizer.zero_grad()
            
            z1, _, rec_x1, q1 = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
            z2, _, _, _ = model(b_corr.x, b_corr.edge_index, b_corr.edge_attr)
            
            # 🚀 再次传入原始特征进行对比学习
            loss_cl, _, _ = spatial_contrastive_loss(z1, z2, b_orig.edge_index, b_orig.x, temperature=args.temp)
            loss_rec = reconstruction_loss(rec_x1, b_orig.x)
            p1 = target_distribution(q1.detach())
            loss_dec = F.kl_div(q1.log(), p1, reduction='batchmean')
            
            loss = loss_cl + args.lambda_rec * loss_rec + args.lambda_dec * loss_dec
            loss.backward(); optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval(); total_val_loss = 0
        with torch.no_grad():
            for b_orig, b_corr in val_loader:
                b_orig, b_corr = b_orig.to(DEVICE), b_corr.to(DEVICE)
                z1, _, rec_x1, q1 = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
                z2, _, _, _ = model(b_corr.x, b_corr.edge_index, b_corr.edge_attr)
                
                v_loss_cl, _, _ = spatial_contrastive_loss(z1, z2, b_orig.edge_index, b_orig.x, temperature=args.temp)
                v_loss_rec = reconstruction_loss(rec_x1, b_orig.x)
                p1 = target_distribution(q1)
                v_loss_dec = F.kl_div(q1.log(), p1, reduction='batchmean')
                total_val_loss += (v_loss_cl + args.lambda_rec * v_loss_rec + args.lambda_dec * v_loss_dec).item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        history['fine_loss'].append(avg_val_loss)
        
        if epoch % 5 == 0:
            logger.info(f"Fine-tune Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            patience_counter = 0 
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.warning(f"✋ 触发早停机制！(Epoch {epoch})")
            break

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(history['pre_loss'])), history['pre_loss'], label='Phase 1: Pre-train Loss')
    offset = len(history['pre_loss'])
    plt.plot(range(offset, offset + len(history['fine_loss'])), history['fine_loss'], label='Phase 3: Fine-tune Val Loss')
    plt.title('Training Loss Pipeline (Hard Negatives)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'training_pipeline.png'), dpi=300)
    logger.info("✅ 训练完成，模型已保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MorphGAT Industrial Pipeline")
    parser.add_argument("--data_dir", type=str, default="/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--gpu", type=str, default="5")
    parser.add_argument("--seed", type=int, default=42)
    
    # 轮次设置 (这里已经为你设好了较高的容忍度)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--patience", type=int, default=100)
    
    # 核心模块开关参数
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_dec", type=float, default=1.0)

    args = parser.parse_args()
    train(args)