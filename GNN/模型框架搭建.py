# 在对比学习中，每个 Epoch 动态生成不同的损坏图（On-the-fly Augmentation）效果通常远好于预先生成固定的图。
"""
================================================================================
脚本名称: 模型框架搭建.py (MorphGAT Model & Training Framework)

功能描述:
本脚本定义了 MorphGAT 的核心网络结构，并实现了节点级别的图对比学习 (Graph Contrastive Learning) 训练流程。

核心机制:
1. 动态图增强 (On-the-fly Augmentation): 每次取数据时，通过 MorphologicalDropEdge
   根据形态学置信度，动态生成不同的损坏图 (Corrupted Graph)。
2. 特征升维 (Feature Projection): 将 15 维的致密通路信号 (TPscore) 映射至高维空间 (128维)，
   打破 GAT 的信息瓶颈。
3. 形态学注意力 (MorphGATConv): 自定义的图卷积层，将形态学相似度作为先验偏差 (Bias) 
   融入到注意力权重的计算中。
4. 对比损失优化 (InfoNCE Loss): 最大化同一节点在原图和损坏图中的表示相似度，
   促使模型学到鲁棒的肿瘤微环境空间表征。
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
from torch.utils.data import random_split
import os
import glob
from tqdm import tqdm

# ==========================================
# 【关键】导入刚才写的增强模块
# ==========================================
from corrupted_graph import MorphologicalDropEdge

# ==========================================
# 1. 定义数据集 (集成动态增强)
# ==========================================
class ContrastiveGraphDataset(Dataset):
    def __init__(self, root_dir, p_overall=0.4, transform=None, pre_transform=None):
        """
        root_dir: 原图路径 (包含 .pt 文件)
        p_overall: 传递给增强模块的基础删边概率
        """
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        
        # 初始化增强器
        self.augmentor = MorphologicalDropEdge(p_overall=p_overall)
        super().__init__(root_dir, transform, pre_transform)

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        # 1. 读取原图
        file_path = self.file_list[idx]
        data_orig = torch.load(file_path)
        
        # 2. 实时生成损坏图 (每次训练 epoch 调用时，结果都会随机变化)
        data_corr = self.augmentor(data_orig)
        
        return data_orig, data_corr
    
# ==========================================
# 2. 模型定义
# ==========================================
class MorphGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, bias=True):
        super().__init__(node_dim=0, aggr='add') 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat  # 是否拼接多个注意力头的输出（True=拼接，False=平均）
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
        """
        in_channels: 初始节点特征维度 (14个通路 + 1个TPscore = 15)
        """
        super().__init__()
        
        # 🌟 1. 动态升维模块 (15 -> 64 -> 128)
        self.feature_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU()
        )
        
        self.heads1 = 4 # 注意力头数
        
        # 2. 第一层图卷积 (此时输入已经是 hidden_channels)
        self.conv1 = MorphGATConv(hidden_channels, hidden_channels, heads=self.heads1, concat=True)
        
        # 跳连接的线性投影层：匹配升维后的隐层维度和第一层卷积的输出维度
        self.skip_proj = nn.Linear(hidden_channels, hidden_channels * self.heads1)
        
        # 3. 第二层图卷积 （输出维度=hidden_channels）
        dim_after_conv1 = hidden_channels * self.heads1
        self.conv2 = MorphGATConv(dim_after_conv1, hidden_channels, heads=1, concat=False)
        
        # 4. 投影头：将hidden维度的节点嵌入映射到最终对比学习的输出维度 out_channels
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # 🌟 先对 15 维的特征进行升维解压缩
        x = self.feature_proj(x)
        x_in = x  # 保存升维后的特征，用于跳连接
        
        # 图卷积与聚合
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x) # ELU激活函数
        x = x + self.skip_proj(x_in) # 跳连接
        x = F.dropout(x, p=0.4, training=self.training) # Dropout正则化
        
        x = self.conv2(x, edge_index, edge_attr)
        node_emb = x # 用于下游任务的最终节点表征
        
        z = self.proj_head(node_emb) # 用于计算对比损失的潜向量
        return z, node_emb 
    

# ==========================================
# 3. 训练流程 
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[-] 使用设备: {DEVICE}")

EPOCHS = 200
BATCH_SIZE = 8
LR = 0.0005
TEMP = 0.2
ROOT_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/"
SAVE_DIR = "checkpoints"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 数据准备 ---
full_dataset = ContrastiveGraphDataset(ROOT_DIR, p_overall=0.5)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 🌟 修正：将 in_channels 设为 15，hidden 设为 128
model = GCLModel_Morph(in_channels=15, hidden_channels=128, out_channels=32).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# 对比损失函数
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # ⚠️注意：如果单个 Batch 内的 spot 点总数非常巨大(>30000)，这里可能会占用较大显存
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.shape[0]).to(z1.device)
    
    return F.cross_entropy(sim_matrix, labels)

# --- 训练循环 ---
best_val_loss = float('inf')
pbar = tqdm(range(EPOCHS), desc="Training")

for epoch in pbar:
    # 1. Train
    model.train()
    total_train_loss = 0
    for batch_orig, batch_corr in train_loader:
        batch_orig = batch_orig.to(DEVICE)
        batch_corr = batch_corr.to(DEVICE)
        
        optimizer.zero_grad()
        z1, _ = model(batch_orig.x, batch_orig.edge_index, batch_orig.edge_attr)
        z2, _ = model(batch_corr.x, batch_corr.edge_index, batch_corr.edge_attr)
        
        loss = contrastive_loss(z1, z2, temperature=TEMP)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)

    # 2. Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_orig, batch_corr in val_loader:
            batch_orig = batch_orig.to(DEVICE)
            batch_corr = batch_corr.to(DEVICE)
            
            z1, _ = model(batch_orig.x, batch_orig.edge_index, batch_orig.edge_attr)
            z2, _ = model(batch_corr.x, batch_corr.edge_index, batch_corr.edge_attr)
            
            val_loss = contrastive_loss(z1, z2, temperature=TEMP)
            total_val_loss += val_loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    
    # 3. Log & Save
    pbar.set_postfix({'Train': f"{avg_train_loss:.4f}", 'Val': f"{avg_val_loss:.4f}"})
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")

print("✅ 完成！")