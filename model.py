import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros

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
            # ✨ 优化 5: 对 edge_attr 做安全截断，彻底防止 log(0) 引发 NaN 梯度爆炸
            safe_edge_attr = torch.clamp(edge_attr.view(-1, 1), min=1e-4)
            alpha = alpha + self.bias_lambda * torch.log(safe_edge_attr)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)

class GCLModel_Morph(nn.Module):
    def __init__(self, in_channels=233, hidden_channels=128, out_channels=32, n_clusters=4):
        super().__init__()
        self.heads1 = 4 
        
        self.gate1 = nn.Sequential(nn.Linear(hidden_channels * self.heads1, 1), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())
        
        self.feature_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels), nn.GELU()
        )
        
        self.conv1 = MorphGATConv(hidden_channels, hidden_channels, heads=self.heads1, concat=True)
        self.skip_proj = nn.Linear(hidden_channels, hidden_channels * self.heads1)
        
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
        x_raw = x.clone() 
        
        x = self.feature_proj(x)
        x_in = x  
        
        conv1_out = F.elu(self.conv1(x, edge_index, edge_attr)) + self.skip_proj(x_in)
        raw1 = self.raw_proj1(x_raw)
        g1 = self.gate1(conv1_out)  
        x = conv1_out * g1 + raw1 * (1.0 - g1)
        
        x = F.dropout(x, p=0.4, training=self.training)
        conv2_out = self.conv2(x, edge_index, edge_attr)
        raw2 = self.raw_proj2(x_raw)
        g2 = self.gate2(conv2_out)  
        node_emb = conv2_out * g2 + raw2 * (1.0 - g2)
        
        z = self.proj_head(node_emb)
        rec_x = self.decoder(z)
        
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centers, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return z, node_emb, rec_x, q
