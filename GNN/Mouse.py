import os
import glob

print("="*50)
print("🧹 启动 [无效样本/小鼠数据] 自动清理程序")
print("="*50)

# 配置目录
project_root = "/data/home/wangzz_group/zhaipengyuan/BEPH-main"
h5ad_dir = os.path.join(project_root, "DATA_DIRECTORY/kz_data/Raw_Data/Log1p")
graph_dir = os.path.join(project_root, "DATA_DIRECTORY/kz_data/Graph_pt")

# 获取所有的 h5ad 文件
h5ad_files = glob.glob(os.path.join(h5ad_dir, "*.h5ad"))
deleted_count = 0

for h5ad_path in h5ad_files:
    slide_id = os.path.basename(h5ad_path).replace('.h5ad', '')
    
    # 对应的 pt 和 npy 路径
    pt_path = os.path.join(graph_dir, f"{slide_id}.pt")
    npy_path = os.path.join(graph_dir, f"{slide_id}_image_features.npy")

    # 核心逻辑：如果这个样本没有成功生成 .pt (说明是小鼠或异常)，就把它相关的源文件全删了
    if not os.path.exists(pt_path):
        print(f"🗑️ 发现异常/小鼠样本: {slide_id: <20} -> 正在执行物理删除...")
        
        # 删除 h5ad
        os.remove(h5ad_path)
        
        # 删除对应的图片特征 npy (如果存在的话)
        if os.path.exists(npy_path):
            os.remove(npy_path)
            
        deleted_count += 1

print("\n" + "="*50)
print(f"✨ 空间清理完成！")
print(f"✅ 共彻底删除了 {deleted_count} 个无效样本（.h5ad 及对应 .npy）")
print("="*50 + "\n")