# 在对比学习中，每个 Epoch 动态生成不同的损坏图（On-the-fly Augmentation）效果通常远好于预先生成固定的图。
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
        root_dir: 原图路径 (不需要 corr_dir 了)
        p_overall: 传递给增强模块的参数
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
# 2. 模型定义 (保持不变)
# ==========================================
class MorphGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, bias=True):
        super().__init__(node_dim=0, aggr='add') # max可尝试
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat  # 是否拼接多个注意力头的输出（True=拼接，False=平均）
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels)) # 可学习参数
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
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.heads1 = 4 #注意力头数
        # 第一层图卷积
        self.conv1 = MorphGATConv(in_channels, hidden_channels, heads=self.heads1, concat=True)
        # 跳连接的线性投影层：匹配输入和第一层输出的维度
        self.skip_proj = nn.Linear(in_channels, hidden_channels * self.heads1)
        # 计算第一层卷积后的输出维度
        dim_after_conv1 = hidden_channels * self.heads1
        # 第二层图卷积 （输出维度=hidden_channels）
        self.conv2 = MorphGATConv(dim_after_conv1, hidden_channels, heads=1, concat=False)
        # 投影头：将hidden维度的节点嵌入映射到最终输出维度out_channels
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )# 全连接层

    def forward(self, x, edge_index, edge_attr):
        x_in = x
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)# ELU激活函数（比ReLU更友好，减少梯度消失）
        x = x + self.skip_proj(x_in)
        # Dropout正则化：随机丢弃40%的神经元，防止过拟合
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        node_emb = x
        z = self.proj_head(node_emb)
        return z, node_emb # 返回最终输出z和中间节点嵌入node_emb
    


# ==========================================
# 3. 训练流程 
# 模型分别对两张图提取节点嵌入，通过对比损失让 “同一节点在不同图中的嵌入尽可能相似，不同节点的嵌入尽可能不同
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
# 这里 p_overall=0.5 意味着每次训练，图结构都会有 50% 左右的变动（基于权重调整）
full_dataset = ContrastiveGraphDataset(ROOT_DIR, p_overall=0.5)
# 划分训练集（80%）和验证集（20%）
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# 数据加载器：按批次加载数据，训练集shuffle=True（打乱顺序），验证集shuffle=False（无需打乱）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = GCLModel_Morph(in_channels=768, hidden_channels=256, out_channels=128).to(DEVICE)
# 优化器：Adam（最常用的自适应学习率优化器），带权重衰减（L2正则化）防止过拟合
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# 对比损失函数
def contrastive_loss(z1, z2, temperature=0.5):
    # 步骤1：L2归一化 → 让嵌入的模长为1，相似度仅由夹角决定（余弦相似度）
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # 步骤2：计算相似度矩阵 → z1和z2的余弦相似度 / 温度系数
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    # 步骤3：标签：每个样本的正例是自己（比如第i个样本的正例是z2的第i个）
    labels = torch.arange(z1.shape[0]).to(z1.device)
    # 步骤4：交叉熵损失 → 让正例的相似度尽可能高，负例尽可能低
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
