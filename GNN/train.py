"""
================================================================================
脚本名称: train.py (MorphGAT - 泛癌微环境 233 维专版)
级别: 工业级标准 (Production-Ready)
功能描述:
【泛癌终极版】：
1. 动态维度自适应：自动识别 233 维输入特征。
2. GNN 架构升级：引入 "跨维度原始特征残差连接" (Raw Feature Residuals)，抵抗过度同化。
3. Loss 架构升级：在 Spatial InfoNCE 中加入 "困难负样本挖掘" (Hard Negative Mining)，锐化空间边界。
4. 继承模块：DEC 端到端聚类、特征重建损失、特征掩码。
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
from torch_geometric.utils import softmax, to_dense_adj, to_dense_batch
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
        # 🌟 修复：移除硬编码的 15 维限制，改为动态判断是否为空
        if data_orig.x.shape[1] < 2:
            raise ValueError(f"\n❌ 发现脏数据！文件: {self.file_list[idx]}\n节点特征维度异常: {data_orig.x.shape[1]}")
        
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
        self.bias_lambda = nn.Parameter(torch.tensor(5.0)) 
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
    def __init__(self, in_channels=233, hidden_channels=128, out_channels=32, n_clusters=8):
        super().__init__()
        
        self.feature_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU()
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
        x_raw = x.clone() # 🚀 保留最纯净的生物学特征 (233维)
        
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

def spatial_contrastive_loss(z1, z2, edge_index, x_raw, batch, temperature=0.2, hard_weight=3.0, hard_threshold=0.5):
    """
    带困难负样本挖掘的 InfoNCE。
    x_raw: 原始特征，用于衡量细胞之间的真实生物学差异。
    hard_weight: 困难负样本的惩罚倍数，撕开边界的关键。
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z1_b, mask = to_dense_batch(z1, batch)      # (B, M, D), (B, M)
    z2_b, _ = to_dense_batch(z2, batch)         # (B, M, D)
    x_b, _ = to_dense_batch(x_raw, batch)       # (B, M, F)
    adj_b = to_dense_adj(edge_index, batch=batch)  # (B, M, M)

    batch_size, M, D = z1_b.size()
    total_loss = 0.0
    total_pos_sim = 0.0
    total_nodes = 0

    for i in range(batch_size):
        valid = mask[i].bool()
        ni = valid.sum().item()
        if ni == 0:
            continue

        zi1 = z1_b[i, valid]   # (ni, D)
        zi2 = z2_b[i, valid]
        xi = x_b[i, valid]
        adj = adj_b[i][:ni, :ni]
        adj.fill_diagonal_(1.0)

        sim = torch.exp(torch.matmul(zi1, zi2.T) / temperature)  # (ni, ni)
        pos_sim = torch.diag(sim)

        # easy negatives: nodes within same graph but not adjacent
        easy_neg_mask = (adj == 0).float()
        easy_neg_sim = (sim * easy_neg_mask).sum(dim=-1)

        # hard negatives: spatial neighbors but raw feature similarity < threshold
        x_norm = F.normalize(xi, dim=1)
        raw_sim = torch.matmul(x_norm, x_norm.T)
        neighbors_only = adj - torch.eye(ni, device=adj.device)
        hard_neg_mask = (neighbors_only * (raw_sim < hard_threshold)).float()
        hard_neg_sim = (sim * hard_neg_mask * hard_weight).sum(dim=-1)

        denom = pos_sim + easy_neg_sim + hard_neg_sim + 1e-8
        loss_i = -torch.log(pos_sim / denom).mean()

        total_loss += loss_i * ni
        total_pos_sim += pos_sim.mean().item() * ni
        total_nodes += ni

    if total_nodes == 0:
        return torch.tensor(0.0, device=z1.device), 0.0, 0.0

    loss = total_loss / total_nodes
    avg_pos = total_pos_sim / total_nodes
    return loss, avg_pos, 0.0

def reconstruction_loss(reconstructed_x, original_x):
    return nn.MSELoss()(reconstructed_x, original_x)

# ==========================================
# 4. 训练主引擎
# ==========================================
def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed); logger = setup_logger(args.save_dir)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 初始化泛癌训练任务 (HardNeg + Residual) | Device: {DEVICE}")

    full_dataset = ContrastiveGraphDataset(args.data_dir, p_overall=0.5)
    logger.info(f"📊 成功载入 {len(full_dataset)} 个样本数据。")
    
    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 动态检测输入维度
    if args.in_dim is None:
        sample_pt = full_dataset.file_list[0]
        tmp = torch.load(sample_pt)
        in_dim = tmp.x.shape[1]
        logger.info(f"✅ 自动检测到输入维度: {in_dim}")
    else:
        in_dim = args.in_dim

    model = GCLModel_Morph(in_channels=in_dim, hidden_channels=args.hidden_dim, out_channels=32, n_clusters=args.n_clusters).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    history = {'pre_loss': [], 'fine_loss': []}

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # ---------------------------------------------------------
    # 阶段一：预训练
    # ---------------------------------------------------------
    logger.info(f"🔥 Phase 1: 预训练特征 ({args.pretrain_epochs} Epochs)...")
    for epoch in range(args.pretrain_epochs):
        model.train(); total_loss = 0
        for b_orig, b_corr in train_loader:
            b_orig, b_corr = b_orig.to(DEVICE), b_corr.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                z1, _, rec_x1, _ = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
                z2, _, _, _ = model(b_corr.x, b_corr.edge_index, b_corr.edge_attr)
                loss_cl, _, _ = spatial_contrastive_loss(z1, z2, b_orig.edge_index, b_orig.x, b_orig.batch if hasattr(b_orig, 'batch') else torch.zeros(b_orig.x.size(0), dtype=torch.long, device=b_orig.x.device), temperature=args.temp)
                loss_rec = reconstruction_loss(rec_x1, b_orig.x)
                loss = loss_cl + args.lambda_rec * loss_rec

            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
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
            b_orig = b_orig.to(DEVICE)
            z, _, _, _ = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
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

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                z1, _, rec_x1, q1 = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
                z2, _, _, _ = model(b_corr.x, b_corr.edge_index, b_corr.edge_attr)
                loss_cl, _, _ = spatial_contrastive_loss(z1, z2, b_orig.edge_index, b_orig.x, b_orig.batch if hasattr(b_orig, 'batch') else torch.zeros(b_orig.x.size(0), dtype=torch.long, device=b_orig.x.device), temperature=args.temp)
                loss_rec = reconstruction_loss(rec_x1, b_orig.x)
                p1 = target_distribution(q1.detach())
                loss_dec = F.kl_div(q1.log(), p1, reduction='batchmean')
                loss = loss_cl + args.lambda_rec * loss_rec + args.lambda_dec * loss_dec

            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval(); total_val_loss = 0
        with torch.no_grad():
            for b_orig, b_corr in val_loader:
                b_orig, b_corr = b_orig.to(DEVICE), b_corr.to(DEVICE)
                z1, _, rec_x1, q1 = model(b_orig.x, b_orig.edge_index, b_orig.edge_attr)
                z2, _, _, _ = model(b_corr.x, b_corr.edge_index, b_corr.edge_attr)
                
                v_loss_cl, _, _ = spatial_contrastive_loss(z1, z2, b_orig.edge_index, b_orig.x, b_orig.batch if hasattr(b_orig, 'batch') else torch.zeros(b_orig.x.size(0), dtype=torch.long, device=b_orig.x.device), temperature=args.temp)
                v_loss_rec = reconstruction_loss(rec_x1, b_orig.x)
                p1 = target_distribution(q1)
                v_loss_dec = F.kl_div(q1.log(), p1, reduction='batchmean')
                total_val_loss += (v_loss_cl + args.lambda_rec * v_loss_rec + args.lambda_dec * v_loss_dec).item()
                
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else avg_train_loss # 防止验证集为空
        history['fine_loss'].append(avg_val_loss)
        
        if epoch % 5 == 0:
            logger.info(f"Fine-tune Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 保存权重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': scaler.state_dict()
            }
            torch.save(ckpt, os.path.join(args.save_dir, "best_model.pth"))
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
    plt.title('Training Loss Pipeline (Pan-Cancer Hard Negatives)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'training_pipeline.png'), dpi=300)
    logger.info("✅ 训练完成，最佳模型已保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MorphGAT Pan-Cancer Pipeline")
    parser.add_argument("--data_dir", type=str, default="/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--gpu", type=str, default="5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard_weight", type=float, default=1.5)
    parser.add_argument("--hard_threshold", type=float, default=0.3)
    
    # 🌟 提高预训练轮次以充分吸收 233 维复杂特征
    parser.add_argument("--pretrain_epochs", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=500)
    
    # 🌟 调整 Batch Size 适配多样本
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--temp", type=float, default=0.5)
    
    # 🌟 扩大隐藏层维度
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=150)
    
    # 聚类数默认 8
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_dec", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--in_dim", type=int, default=None)

    args = parser.parse_args()
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir, exist_ok=True)
    train(args)