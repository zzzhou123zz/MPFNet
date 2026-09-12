import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityWeightingLayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ModalityWeightingLayer, self).__init__()
        
        # 通道注意力机制的全连接层
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=1)  # 降低维度
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels * 2, kernel_size=1)  # 恢复维度
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor):
        # 将输入拼接在一起形成多模态的组合
        combined = torch.cat([inputs1, inputs2], dim=1)  # 在通道维度拼接
        # 全局平均池化
        se_weight = F.adaptive_avg_pool2d(combined, (1, 1))  # 输出为 (batch_size, in_channels * 2, 1, 1)
        
        # 通过全连接层生成注意力
        se_weight = F.relu(self.fc1(se_weight))
        se_weight = self.sigmoid(self.fc2(se_weight))  # 输出为 (batch_size, in_channels * 2, 1, 1)

        # 分成 alpha 和 beta
        alpha, beta = torch.split(se_weight, inputs1.size(1), dim=1)

        # 返回加权后的输入
        return (inputs1 * alpha, inputs2 * beta)
    
class SimpleWeightingLayer(nn.Module):
    def __init__(self):
        super(SimpleWeightingLayer, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor):
        # 返回加权后的输入
        return (inputs1 * self.w1, inputs2 * self.w2)