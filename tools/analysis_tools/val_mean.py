import torch
from torchvision import transforms

# 假设你的 mean 和 std 是基于 [0, 255] 计算的
mean = [59.8,  162.13, 160.19]
std = [44.82 ,46.41, 48.16]

# 创建一个随机张量模拟图像数据（范围 [0, 255]）
fake_img = torch.rand(3, 640, 640) * 255

# 标准化
normalize = transforms.Normalize(mean=mean, std=std)
normalized_img = normalize(fake_img)

print("Normalized values (should be around 0):", normalized_img.mean())
print("Normalized std (should be around 1):", normalized_img.std())