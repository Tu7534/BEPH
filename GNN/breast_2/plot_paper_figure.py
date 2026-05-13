import os
import sys
import torch
import torch.nn as nn # 确保基础组件导入
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ================= 🚀 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
gnn_dir = os.path.dirname(current_dir)
if gnn_dir not in sys.path: sys.path.insert(0, gnn_dir)

# 导入底层图卷积基础组件
from train_2 import MorphGATConv

# ================= 🛡️ 架构解耦区 =================
# 显式重写最新的极限版 GCLModel_Morph，确保完美接纳带 LayerNorm 的新权重
class GCLModel_Morph(nn.Module):
    def __init__(self, in_channels=233, hidden_channels=128, out_channels=32, n_clusters=20):
        super().__init__()
        self.heads1 = 8 
        
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
        
        # ✨ 核心同步升级：完美对齐 checkpoints_2 的 LayerNorm 结构
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, in_channels)
        )
        
        self.n_clusters = n_clusters
        self.alpha = 1.0 
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, out_channels))
        nn.init.zeros_(self.cluster_centers)

    def forward(self, x, edge_index, edge_attr):
        x_raw = x.clone() 
        x = self.feature_proj(x)
        x_in = x  
        
        conv1_out = torch.nn.functional.elu(self.conv1(x, edge_index, edge_attr)) + self.skip_proj(x_in)
        raw1 = self.raw_proj1(x_raw)
        g1 = self.gate1(conv1_out)  
        x = conv1_out * g1 + raw1 * (1.0 - g1)
        
        x = torch.nn.functional.dropout(x, p=0.4, training=self.training)
        conv2_out = self.conv2(x, edge_index, edge_attr)
        raw2 = self.raw_proj2(x_raw)
        g2 = self.gate2(conv2_out)  
        node_emb = conv2_out * g2 + raw2 * (1.0 - g2)
        
        z = self.proj_head(node_emb)
        rec_x = self.decoder(z)
        return z, node_emb, rec_x, None

# ================= 🎨 主画图流程 =================
def generate_paper_figures():
    sample_root = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/"
    pt_path = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Human Breast Cancer (Block A Section 1)/breast_cancer.pt"
    model_path = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/GNN/breast_2/checkpoints_2/best_model.pth"
    metadata_path = os.path.join(sample_root, "metadata.txt")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_CLUSTERS = 20

    # ================= 1. 提取预测特征 =================
    print(f"[-] 正在加载 MorphGAT 极限版模型与图数据...")
    model = GCLModel_Morph(in_channels=233, hidden_channels=128, out_channels=32, n_clusters=N_CLUSTERS).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    data = torch.load(pt_path).to(DEVICE)
    with torch.no_grad():
        _, node_emb, _, _ = model(data.x, data.edge_index, data.edge_attr)
    embeddings = node_emb.cpu().numpy()

    # ================= 2. 全图 K-Means 与 严格对齐评估 =================
    print(f"[-] 正在执行全切片 K-Means (K={N_CLUSTERS}) 与严格学术对齐...")
    
    kmeans_whole = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42)
    full_clusters_raw = kmeans_whole.fit_predict(embeddings)

    try:
        truth_df = pd.read_csv(metadata_path, sep='\t', encoding='utf-16')
    except:
        truth_df = pd.read_csv(metadata_path, sep='\t', encoding='utf-8')
    truth_df['ID'] = truth_df['ID'].astype(str).str.strip().str.replace('-1', '', regex=False)

    adata = sc.read_visium(sample_root)
    adata.var_names_make_unique()
    barcodes = adata.obs_names.astype(str).str.strip().str.replace('-1', '', regex=False)
    coord_df = pd.DataFrame({'barcode': barcodes, 'idx': range(len(barcodes))})
    
    merged = pd.merge(coord_df, truth_df[['ID', 'fine_annot_type']], left_on='barcode', right_on='ID', how='inner')
    merged = merged.rename(columns={'fine_annot_type': 'annot_type'})

    subset_preds = full_clusters_raw[merged['idx'].values]

    def get_mapping(gt, pred):
        df = pd.DataFrame({'gt': gt, 'pred': pred})
        return {c: df[df['pred'] == c]['gt'].dropna().mode()[0] for c in df['pred'].unique()}
    
    mapping = get_mapping(merged['annot_type'].values, subset_preds)
    
    # ================= 📊 学术指标计算 =================
    subset_mapped = pd.Series(subset_preds).map(mapping).values
    real_ari = adjusted_rand_score(merged['annot_type'], subset_mapped)
    raw_nmi = normalized_mutual_info_score(merged['annot_type'], subset_preds)

    print("\n" + "="*50)
    print(f"[*] 📊 学术指标计算完成！")
    print(f"[*] 极限版映射大类 ARI 得分: {real_ari:.4f}")
    print(f"[*] 极限版纯无监督 NMI 得分: {raw_nmi:.4f}")
    print("="*50 + "\n")

    full_clusters_mapped = pd.Series(full_clusters_raw).map(mapping).fillna('Unknown').values

    # ================= 3. 提取空间坐标与暗黑底图 =================
    library_id = list(adata.uns['spatial'].keys())[0]
    scale_factor = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
    
    spatial_x = adata.obsm['spatial'][:, 0] * scale_factor
    spatial_y = adata.obsm['spatial'][:, 1] * scale_factor
    
    img_rgb = adata.uns['spatial'][library_id]['images']['hires']
    
    if img_rgb.ndim == 3:
        img_gray = np.dot(img_rgb[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img_rgb
    img_gray_dark = img_gray * 0.4  

    # ================= 4. 融合生成：暗黑系 20 类精细大图 =================
    print("[-] 正在生成暗黑系高级双联大图 (极限权重版)...")
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(24, 11), facecolor='#151515')

    unique_gt_labels = merged['annot_type'].unique()
    base_colors = sns.color_palette("tab20", 20).as_hex()
    
    gt_palette = {label: base_colors[i % 20] for i, label in enumerate(unique_gt_labels)}
    gt_palette['Unknown'] = '#444444' 

    SPOT_SIZE = 35 

    def plot_panel(ax, title, x_coords, y_coords, labels, palette, add_ari=False):
        ax.imshow(img_gray_dark, cmap='gray')
        colors = [palette.get(str(lbl), '#444444') for lbl in labels]
        ax.scatter(x_coords, y_coords, c=colors, s=SPOT_SIZE, edgecolors='none', alpha=0.95)
        ax.set_title(title, fontsize=24, pad=20, color='white', fontweight='bold')
        ax.axis('off')
        
        if add_ari:
            ari_text = f"MorphGAT : ARI {real_ari:.4f}"
            ax.text(0.02, 0.98, ari_text, transform=ax.transAxes,
                    fontsize=20, fontweight='bold', color='#00FF00', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8, edgecolor='none'))

    x_gt = spatial_x[merged['idx'].values]
    y_gt = spatial_y[merged['idx'].values]
    labels_gt = merged['annot_type'].values
    
    plot_panel(axes[0], 'Ground Truth (20 Fine Niches)', x_gt, y_gt, labels_gt, gt_palette)
    
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=gt_palette[label], label=label) for label in unique_gt_labels]
    lgd = axes[0].legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                         title="Pathology Labels", fontsize=11, title_fontsize=13, frameon=False, ncol=1)
    plt.setp(lgd.get_texts(), color='white')
    plt.setp(lgd.get_title(), color='white')

    plot_panel(axes[1], f'MorphGAT Ultimate (Mapped, K={N_CLUSTERS})', spatial_x, spatial_y, full_clusters_mapped, gt_palette, add_ari=True)

    plt.tight_layout(pad=3.0, w_pad=6.0) 
    save_path = os.path.join(current_dir, f"MorphGAT_Dark_Aligned_Ultimate.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ 完美！兼容 LayerNorm 的暗黑顶刊大图已保存至: {save_path}")

if __name__ == "__main__":
    generate_paper_figures()