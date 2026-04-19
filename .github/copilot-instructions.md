## 目标
为 AI 编码代理（如 Copilot/Agents）提供一份精简、可执行的指南，帮助其在本仓库中快速完成常见任务：数据预处理（特征->图）、特征提取、模型训练与评估。

## 项目概要（一眼看懂）
- 主要语言/框架：Python，PyTorch，PyTorch-Geometric，Scanpy，h5py。
- 主要模块：
  - `CLAM_Feature/`：包含特征提取与 MIL/CLAM 相关模型（如 `model_clam.py`、`extract_features_fp_BEPH.py`）。
  - `GNN/`：图构建与 GNN 实验（核心 notebook: `GNN/构建图结构.ipynb`）。
  - `DATA_DIRECTORY/` 与 `FEATURES_DIRECTORY/`：原始 h5ad、h5 特征与生成的 .pt 图文件的默认位置。
  - `checkpoints/`：预训练/已保存模型权重（例如 `BEPH_backbone.pth`）。

## 关键开发工作流（可直接运行的命令）
- 安装依赖：项目根目录含 `requirement.txt`（注意文件名为 `requirement.txt`，不是 `requirements.txt`）。
```bash
python -m pip install -r requirement.txt
```
- 将 notebook 转为脚本并批量构建图（在 `GNN/构建图结构.ipynb` 中有 `run_batch_processing` 主入口）：
```bash
jupyter nbconvert --to script GNN/构建图结构.ipynb
python GNN/构建图结构.py
```
（上面脚本会默认使用 notebook 中的路径变量：H5AD_DIR、FEATURE_DIR、OUTPUT_DIR；可在生成的脚本中修改这些常量或直接在 notebook 中运行最后的 cell。）
- 特征提取示例：查看 `CLAM_Feature/extract_features_fp_BEPH.py` 或 `CLAM_Feature/extract_features_fp.py`，这些脚本会将切片/patch 图像或 embedding 导出到 `FEATURES_DIRECTORY/h5_files/`。

## 数据契约 & 约定（重要）
- 输入：AnnData `.h5ad`（样本名为 `SampleName.h5ad`）和对应的特征文件 `SampleName.h5`。两者通过 basename 绑定（同名不同后缀）。
- 坐标：`AnnData.obsm['spatial']`（或包含 `spatial` 的其它 obsm 键）被用来构建物理拓扑。
- 特征 h5 文件：默认 key 为 `features`；若找不到，则取文件中的第一个 dataset。
- 输出：每个样本会生成一个 `SampleName.pt`（`torch_geometric.data.Data`），存放在 `DATA_DIRECTORY/.../Graph_pt/`（见 notebook 中 `OUTPUT_DIR`）。

## 代码库中可直接复用的实现细节（对自动化很重要）
- 图构建逻辑（来自 `GNN/构建图结构.ipynb`）:
  - 使用 sklearn.NearestNeighbors（kd_tree）构造 kNN 图（默认 k=6），并对称化邻接矩阵。
  - 边权重通过节点特征点乘得到（dot product），默认经过 ReLU。
  - 最终数据结构为 PyG 的 `Data(x, edge_index, edge_attr)`。
- 错误/跳过策略：若找不到对应的 `.h5` 特征文件或没有 spatial 坐标，脚本会跳过该样本并计入失败数（可在 `run_batch_processing` 中看到）。

## 常见工程约定与模式
- 命名：样本名与文件名严格按 basename 关联；新增数据时请确保 `.h5ad` 与 `.h5` 同名。
- Notebook-first：许多数据处理逻辑保存在 notebook 中（例如 `GNN/构建图结构.ipynb`、`GNN/构建图结构.ipynb` 的 main cell），在自动化时优先将 notebook 转为脚本并在 Clean 环境下运行。
- 模型与检查点：预训练权重存于 `checkpoints/`，训练脚本会寻找该目录下的 checkpoint 文件（例如 `BEPH_backbone.pth`）。

## 快速调试提示（面向代理）
- 若处理单个样本以重现问题：在 notebook 顶部把 `H5AD_DIR` 指向包含单个 `.h5ad` 的目录，`FEATURE_DIR` 指向对应 `.h5`，把 `OUTPUT_DIR` 指到临时目录，然后运行最后 cell 并观察异常堆栈。
- 常见异常定位：
  - KeyError: No spatial coords → 检查 `adata.obsm.keys()`，可能键名不同（如 `spatial_1`），代理应尝试匹配包含 `spatial` 的键名。
  - h5 key not found → 代理应读取 h5 keys 列表并使用第一个 dataset 作兜底。

## 编辑/扩展建议（供后续代理修改）
- 将 notebook 中的批处理逻辑抽成 `GNN/graph_builder.py`（可导入）会方便 CI 与自动化运行；现阶段代理可以先通过 `nbconvert` 转脚本并验证行为。
- 若要加入单元测试，请以小样本 h5/h5ad 的临时数据为目标，测试 graph 输出的 shape 与 edge_index 合法性。

---
如果你希望我把其中某节展开成可运行脚本（例如把 `GNN/构建图结构.ipynb` 转成模块并写单元测），或将 README 中的使用示例补全为 CI job，我可以继续实现；请指出你优先的项。 
