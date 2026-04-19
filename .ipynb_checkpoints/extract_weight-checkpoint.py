import torch
import os

# 自动检测可用GPU（优先使用分配的逻辑0卡）
if torch.cuda.is_available():
    # 查看可见的GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"可用GPU数量: {gpu_count}, 分配的GPU编号: {os.environ.get('CUDA_VISIBLE_DEVICES', '默认所有')}")
    # 使用逻辑0卡（无论物理卡是哪张，这是分配到的第一张卡）
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    print("CUDA不可用，使用CPU")

# 加载权重到指定设备
ck = torch.load("./BEPH_weight.pth", map_location=device)
outPath = "./BEPH_backbone.pth"
output_dict = dict(state_dict=dict(), author='Yzc')
has_backbone = False

for key, value in ck['state_dict'].items():
    if key.startswith('backbone'):
        output_dict['state_dict'][key] = value.to(device)  # 显式移到设备（可选，map_location已处理）
        has_backbone = True

if not has_backbone:
    raise Exception('Cannot find a backbone module in the checkpoint.')

torch.save(output_dict, outPath)
print("Backbone weights saved to:", outPath)