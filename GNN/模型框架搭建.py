"""
================================================================================
脚本名称: 模型框架搭建.py (MorphGAT Model & Training Framework)
级别: 工业级标准 (Production-Ready)
功能描述:
本脚本定义了 MorphGAT 的核心网络结构，并实现了节点级别的图对比学习 (Graph Contrastive Learning)。
包含：标准日志系统、随机种子固定、命令行参数配置、早停机制以及自动化训练监控曲线。
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
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros

from corrupted_graph import MorphologicalDropEdge

# ==========================================
# 工具函数：初始化日志与随机种子
# ==========================================
def setup_logger(log_dir):
    """配置全局日志系统"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger("MorphGAT")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def set_seed(seed=42):
    """固定所有随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 定义数据集 (集成动态增强)
# ==========================================
class ContrastiveGraphDataset(Dataset):
    def __init__(self, root_dir, p_overall=0.4, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.augmentor = MorphologicalDropEdge(p_overall=p_overall)
        super().__init__(root_dir, transform, pre_transform)

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        file_path = self.file_list[idx]
        data_orig = torch.load(file_path)
        
        # 强制特征维度校验锁
        if data_orig.x.shape[1] != 15:
            raise ValueError(f"\n❌ 发现脏数据！\n文件: {file_path}\n"
                             f"节点特征维度是 {data_orig.x.shape[1]} 维，而非预期的 15 维！")
        
        # 实时生成损坏图
        data_corr = self.augmentor(data_orig)
        return data_orig, data_corr

# ==========================================
# 2. 模型定义 (MorphGAT)
# ==========================================
class MorphGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, bias=True):
        super().__init__(node_dim=0, aggr='add') 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat  
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels)) 
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.bias_lambda = nn.Parameter(torch.tensor(2.0)) 
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        out = self.propagate(edge_index, x=x, alpha_src=alpha_src, alpha_dst=alpha_dst, edge_attr=edge_attr)
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, alpha_src_i, alpha_dst_j, edge_attr, index, ptr, size_i):
        alpha = alpha_src_i + alpha_dst_j
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        if edge_attr is not None:
            w_ij = edge_attr.view(-1, 1)
            epsilon = 1e-6
            morphological_bias = torch.log(w_ij + epsilon)
            bias_term = self.bias_lambda * morphological_bias
            alpha = alpha + bias_term
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class GCLModel_Morph(nn.Module):
    def __init__(self, in_channels=15, hidden_channels=128, out_channels=32):
        super().__init__()
        
        self.feature_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU()
        )
        
        self.heads1 = 4 
        self.conv1 = MorphGATConv(hidden_channels, hidden_channels, heads=self.heads1, concat=True)
        self.skip_proj = nn.Linear(hidden_channels, hidden_channels * self.heads1)
        dim_after_conv1 = hidden_channels * self.heads1
        self.conv2 = MorphGATConv(dim_after_conv1, hidden_channels, heads=1, concat=False)
        
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.feature_proj(x)
        x_in = x  
        
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x) 
        x = x + self.skip_proj(x_in) 
        x = F.dropout(x, p=0.4, training=self.training) 
        
        x = self.conv2(x, edge_index, edge_attr)
        node_emb = x 
        
        z = self.proj_head(node_emb)
        return z, node_emb 

# ==========================================
# 3. 损失函数定义
# ==========================================
def contrastive_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    sim_matrix = torch.matmul(z1, z2.T) 
    
    pos_sim = torch.diag(sim_matrix).mean().item()
    n = sim_matrix.size(0)
    neg_sim = (sim_matrix.sum() - torch.diag(sim_matrix).sum()) / max((n * (n - 1)), 1)
    neg_sim = neg_sim.item()
    
    sim_matrix_scaled = sim_matrix / temperature
    labels = torch.arange(n).to(z1.device)
    loss = F.cross_entropy(sim_matrix_scaled, labels)
    
    return loss, pos_sim, neg_sim

# ==========================================
# 4. 训练主引擎 (隔离作用域)
# ==========================================
def train(args):
    # 初始化环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    logger = setup_logger(args.save_dir)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 初始化训练任务 | PID: {os.getpid()} | Device: {DEVICE}")

    # 数据集加载
    logger.info(f"📂 加载数据集: {args.data_dir}")
    full_dataset = ContrastiveGraphDataset(args.data_dir, p_overall=0.5)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型构建
    model = GCLModel_Morph(in_channels=15, hidden_channels=args.hidden_dim, out_channels=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 训练监控组件
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'pos_sim': [], 'neg_sim': []}

    logger.info("🔥 开始对比学习训练...")
    pbar = tqdm(range(args.epochs), desc="Training", dynamic_ncols=True)

    for epoch in pbar:
        # --- Train ---
        model.train()
        total_train_loss, total_pos_sim, total_neg_sim = 0, 0, 0
        
        for batch_orig, batch_corr in train_loader:
            batch_orig, batch_corr = batch_orig.to(DEVICE), batch_corr.to(DEVICE)
            
            optimizer.zero_grad()
            z1, _ = model(batch_orig.x, batch_orig.edge_index, batch_orig.edge_attr)
            z2, _ = model(batch_corr.x, batch_corr.edge_index, batch_corr.edge_attr)
            
            loss, pos_sim, neg_sim = contrastive_loss(z1, z2, temperature=args.temp)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            torch.cuda.empty_cache() 
        
        num_batches = len(train_loader)
        avg_train_loss = total_train_loss / num_batches
        avg_pos_sim = total_pos_sim / num_batches
        avg_neg_sim = total_neg_sim / num_batches

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_orig, batch_corr in val_loader:
                batch_orig, batch_corr = batch_orig.to(DEVICE), batch_corr.to(DEVICE)
                z1, _ = model(batch_orig.x, batch_orig.edge_index, batch_orig.edge_attr)
                z2, _ = model(batch_corr.x, batch_corr.edge_index, batch_corr.edge_attr)
                val_loss, _, _ = contrastive_loss(z1, z2, temperature=args.temp)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # 记录与更新
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['pos_sim'].append(avg_pos_sim)
        history['neg_sim'].append(avg_neg_sim)
        
        pbar.set_postfix({
            'TrL': f"{avg_train_loss:.2f}", 
            'VaL': f"{avg_val_loss:.2f}",
            'P_Sim': f"{avg_pos_sim:.3f}"
        })
        
        # 早停与模型保存机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            patience_counter = 0 
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.warning(f"✋ 触发早停机制！验证集 Loss 连续 {args.patience} 轮未下降。")
            break

    logger.info("✅ 训练周期结束！正在生成分析曲线图...")

    # --- 绘制曲线图 ---
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    plt.title('Contrastive Loss Curve (InfoNCE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['pos_sim'], label='Positive Sim', color='green', linewidth=2)
    plt.plot(history['neg_sim'], label='Negative Sim', color='red', linewidth=2)
    plt.title('Feature Similarity Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"📊 训练曲线已保存至: {plot_path}")

# ==========================================
# 5. 命令行入口 (核心拦截器)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MorphGAT Industrial Training Pipeline")
    
    # 基础配置
    parser.add_argument("--data_dir", type=str, default="/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/", help="图数据存储路径")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型和日志保存路径")
    parser.add_argument("--gpu", type=str, default="5", help="指定的物理显卡 ID")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    
    # 超参数
    parser.add_argument("--epochs", type=int, default=200, help="最大训练轮次")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--temp", type=float, default=0.5, help="InfoNCE 温度系数")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GAT 隐藏层维度")
    parser.add_argument("--patience", type=int, default=50, help="早停容忍轮次")

    args = parser.parse_args()
    
    # 执行训练
    train(args)