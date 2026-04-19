#  训练完后提取特征

import torch
import os
import glob
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from torch_geometric.utils import softmax
# ==========================================
# 1. 导入你的模型类 (必须和训练时定义的一模一样)
# ==========================================
# 假设你的模型类定义在 train_gcl.py 里，或者你直接把模型类的代码粘贴在这里
# 这里为了独立运行，建议把 GCLModel_Morph 和 MorphGATConv 复制过来，
# 或者从 train_gcl import * (如果都在同一目录下)
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
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.heads1 = 4
        self.conv1 = MorphGATConv(in_channels, hidden_channels, heads=self.heads1, concat=True)
        self.skip_proj = nn.Linear(in_channels, hidden_channels * self.heads1)
        dim_after_conv1 = hidden_channels * self.heads1
        self.conv2 = MorphGATConv(dim_after_conv1, hidden_channels, heads=1, concat=False)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
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
# 2. 配置路径
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 原始数据路径
DATA_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_pt/"

# 模型权重路径
MODEL_PATH = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/GNN/checkpoints/best_model.pth"

# 结果保存路径 (特征保存到这里)
OUTPUT_DIR = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Graph_embeddings/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[-] 创建输出目录: {OUTPUT_DIR}")

# ==========================================
# 3. 加载模型
# ==========================================
print("[-] 正在加载模型...")
# 参数必须与训练时一致 (in=768, hidden=256, out=128)
model = GCLModel_Morph(in_channels=768, hidden_channels=256, out_channels=128).to(DEVICE)

# 加载训练好的权重
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"[√] 成功加载权重: {MODEL_PATH}")
else:
    raise FileNotFoundError(f"找不到模型权重文件: {MODEL_PATH}")

# 开启评估模式 (非常重要！这会关闭 Dropout)
model.eval()

# ==========================================
# 4. 提取并保存特征
# ==========================================
file_list = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))
print(f"[-] 准备处理 {len(file_list)} 个文件...")

with torch.no_grad(): # 不计算梯度，节省显存
    for file_path in tqdm(file_list, desc="Extracting Features"):
        filename = os.path.basename(file_path)
        
        # 1. 加载单个数据
        data = torch.load(file_path)
        data = data.to(DEVICE)
        
        # 2. 模型前向传播
        # 注意：这里我们只要 node_emb，不要最后的 z (投影头输出)
        # z 是为了计算 Loss 用的，node_emb 才是保留了原始信息的特征
        _, node_emb = model(data.x, data.edge_index, data.edge_attr)
        
        # 3. 转移到 CPU
        embedding_cpu = node_emb.cpu()
        
        # 4. 保存
        # 结果保存为: 原始文件名.pt (内容是 Tensor [N, 256])
        save_path = os.path.join(OUTPUT_DIR, filename)
        torch.save(embedding_cpu, save_path)

print("="*30)
print("✅ 特征提取完成！")
print(f"📂 结果保存在: {OUTPUT_DIR}")
print("你可以用这些文件进行聚类或可视化分析了。")