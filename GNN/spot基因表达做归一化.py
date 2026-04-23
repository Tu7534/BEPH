# cd /data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Raw_Data/
# 对原始数据基因表达做Log1p Normalization（对数加一归一化）

import scanpy as sc
import os
import glob
from pathlib import Path

# ================= 配置区域 =================
# 输入文件夹路径
input_folder = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Raw_Data/h5ad_files"
# 输出文件夹路径
output_folder = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Raw_Data/Log1p"

# 质量控制参数 (根据你的数据情况微调)
MIN_COUNTS = 100  # 一个 Spot 至少要有 100 个 Count，否则视为垃圾点剔除
MIN_GENES = 10    # 一个 Spot 至少要检测到 10 个基因

# ================= 主程序 =================

# 1. 确保输出文件夹存在，不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已创建输出文件夹: {output_folder}")
else:
    print(f"输出文件夹已存在: {output_folder}")

# 2. 获取所有 h5ad 文件路径
# 支持 .h5ad 后缀
files = glob.glob(os.path.join(input_folder, "*.h5ad"))

print(f"共发现 {len(files)} 个 h5ad 文件，准备开始处理...\n")

for i, file_path in enumerate(files):
    file_name = os.path.basename(file_path)
    print(f"[{i+1}/{len(files)}] 正在处理: {file_name} ...")
    
    try:
        # --- A. 读取数据 ---
        adata = sc.read_h5ad(file_path)
        original_n_obs = adata.n_obs
        
        # --- B. 质量控制 (QC) - 关键步骤 ---
        # 过滤掉质量极差的点，防止 Log1p 后出现异常高值
        sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
        sc.pp.filter_cells(adata, min_genes=MIN_GENES)
        
        # 打印过滤信息
        filtered_n_obs = adata.n_obs
        print(f"    - QC过滤: 移除 {original_n_obs - filtered_n_obs} 个低质量 Spots (剩余 {filtered_n_obs})")
        
        # 如果过滤后没点了，跳过
        if filtered_n_obs == 0:
            print("    - [警告] 该文件过滤后为空，跳过保存。")
            continue

        # --- C. 标准化处理 ---
        # 1. 总计数归一化 (Normalize to CPM/TPM like)
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        # 2. Log1p 变换
        sc.pp.log1p(adata)
        
        # (可选) 可以在这里加入找高变基因，如果你希望保存的文件里带有高变基因标记
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
        adata = adata[:, adata.var['highly_variable']]
        # --- D. 保存文件 ---
        # 构造输出路径，保持文件名不变
        output_path = os.path.join(output_folder, file_name)
        
        # 保存 (开启压缩可以减小文件体积)
        adata.write(output_path, compression="gzip")
        print(f"    - 已保存至: {output_path}")
        
    except Exception as e:
        print(f"    - [错误] 处理文件 {file_name} 时出错: {e}")

print("\n" + "="*30)
print("所有文件处理完成！")