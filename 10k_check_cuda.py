import torch
 
# 查看CUDA是否可用
if torch.cuda.is_available():
    # 列出所有可用的CUDA设备
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    # 获取当前设备索引
    current_device = torch.cuda.current_device()
    print(f"Current Device: {current_device}")
else:
    print("CUDA is not available.")