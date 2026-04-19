import scanpy as sc
import pandas as pd
import gseapy as gp
import numpy as np
import scipy.sparse as sp
import os
import glob
import traceback

# ==========================================
# 1. 基础路径设置
# ==========================================
# 输入数据根目录 (h5ad 文件所在位置)
base_data_dir = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/Raw_Data/h5ad_files"
# 通路基因集文件路径
gmt_file = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/TP评分/CancerSEA_14_states.gmt"
# 🌟 新增：统一的输出文件夹路径 🌟
output_base_dir = "/data/home/wangzz_group/zhaipengyuan/BEPH-main/DATA_DIRECTORY/kz_data/TPscore"

def process_single_dataset(file_path, gmt_file, output_dir):
    """
    处理单个 .h5ad 空间转录组数据集，并将结果保存到指定文件夹
    """
    # 🌟 核心修改：提取原始文件名，并重定向到新的输出文件夹
    file_name = os.path.basename(file_path) # 例如：提取出 "GSM5621965.h5ad"
    output_name = file_name.replace(".h5ad", "_stmeta.data.csv") # 变更为 "GSM5621965_stmeta.data.csv"
    output_csv = os.path.join(output_dir, output_name) # 拼接为最终的绝对输出路径
    
    # 即使结果存在，也强制重新运行并覆盖
    if os.path.exists(output_csv):
        print(f"🔄 发现旧的结果文件，即将重新计算并覆盖: {output_csv}")

    print(f"\n{'='*80}")
    print(f"🚀 开始处理数据集: {file_name} (原路径: {file_path})")
    print(f"{'='*80}")
    
    try:
        # --- 1. 读取 h5ad 数据 ---
        print("1. 正在读取 .h5ad 数据...")
        adata = sc.read_h5ad(file_path)
        adata.var_names_make_unique()

        # --- 2. 自动化质控与预处理 ---
        print("2. 正在计算质控指标与标准化...")
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        # 兼容性处理：如果存在 total_counts，则重命名为 nCount_Spatial
        rename_dict = {}
        if 'total_counts' in adata.obs.columns:
            rename_dict['total_counts'] = 'nCount_Spatial'
        if 'n_genes_by_counts' in adata.obs.columns:
            rename_dict['n_genes_by_counts'] = 'nFeature_Spatial'
        if rename_dict:
            adata.obs.rename(columns=rename_dict, inplace=True)

        if sp.issparse(adata.X):
            data_vals = adata.X.data
        else:
            data_vals = adata.X.flatten()

        # 智能判断是否需要标准化
        if np.all(np.mod(data_vals, 1) == 0) or data_vals.max() > 50:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            print("   -> 识别到原始计数 (Raw counts)，已完成 normalize_total 和 log1p。")
        else:
            print("   -> 识别到数据似乎已标准化，跳过 normalize 步骤。")

        # 提取表达矩阵转为 DataFrame，供 ssGSEA 使用
        expr_df = pd.DataFrame(
            adata.X.toarray().T if sp.issparse(adata.X) else adata.X.T, 
            index=adata.var_names, 
            columns=adata.obs_names
        )

        # --- 3. 运行 ssGSEA 计算 ---
        print(f"3. 开始运行 ssGSEA (多线程计算中，请耐心等待)...")
        ss_res = gp.ssgsea(
            data=expr_df,
            gene_sets=gmt_file,
            outdir=None,              
            sample_norm_method='rank',
            no_plot=True,
            threads=8   # 使用8线程加速
        )

        # --- 4. 提取结果并计算综合 TPscore ---
        print("4. 正在整理计算结果...")
        results_df = ss_res.res2d.pivot(index='Name', columns='Term', values='NES')
        
        # 把列名中的空格替换为下划线 (Cell Cycle -> Cell_Cycle)
        results_df.columns = results_df.columns.str.replace(' ', '_')
        pathway_cols = results_df.columns.tolist()
        
        adata.obs = adata.obs.join(results_df)
        
        # 计算综合TPscore及高低分组
        adata.obs['TPscore'] = adata.obs[pathway_cols].mean(axis=1)
        median_score = adata.obs['TPscore'].median()
        adata.obs['group'] = np.where(adata.obs['TPscore'] >= median_score, 'high', 'low')

        # --- 5. 导出 CSV ---
        print("5. 正在导出 CSV...")
        
        # 安全导出：只导出 adata.obs 中实际存在的列
        desired_cols = ['nCount_Spatial', 'nFeature_Spatial'] + pathway_cols + ['TPscore', 'group']
        cols_to_export = [col for col in desired_cols if col in adata.obs.columns]
        
        export_df = adata.obs[cols_to_export]
        export_df.to_csv(output_csv)
        
        print(f"✅ 完美结束！结果已统一保存至: {output_csv}")
        return True
        
    except Exception as e:
        print(f"❌ 处理样本 {file_path} 时发生严重错误！")
        print(traceback.format_exc())
        return False

# ==========================================
# 自动扫描并批量执行
# ==========================================
if __name__ == "__main__":
    # 🌟 新增：确保全局的集中输出目录存在，如果不存在则自动创建
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"📁 已就绪输出文件夹: {output_base_dir}")

    # 使用 glob 匹配所有 .h5ad 文件
    search_pattern = os.path.join(base_data_dir, "**", "*.h5ad")
    
    # 获取所有 h5ad 文件列表
    all_h5ad_files = glob.glob(search_pattern, recursive=True)
    
    print(f"🔍 扫描完毕！在 '{base_data_dir}' 下共找到 {len(all_h5ad_files)} 个 .h5ad 文件。")
    
    success_count = 0
    fail_count = 0
    
    for file_path in all_h5ad_files:
        # 将 output_base_dir 传入函数
        is_success = process_single_dataset(file_path, gmt_file, output_base_dir)
        if is_success:
            success_count += 1
        else:
            fail_count += 1
            
    print("\n" + "="*80)
    print("🎉 批量任务全部执行完毕！")
    print(f"📊 统计信息: 成功 {success_count} 个样本，失败 {fail_count} 个样本。")
    print(f"📂 所有生成的 CSV 文件都已存放在: {output_base_dir}")
    print("="*80)