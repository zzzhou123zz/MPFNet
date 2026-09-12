import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from typing import List, Tuple, Union


# 添加适配器动态调整通道数
class AdaptiveChannelNorm(nn.Module):
    """自适应通道归一化，能处理不同通道数的输入"""

    def __init__(self, base_channels: int):
        super().__init__()
        self.base_channels = base_channels
        self.norms = nn.ModuleDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        norm_key = f"norm_{C}"

        if norm_key not in self.norms:
            self.norms[norm_key] = nn.BatchNorm2d(C).to(x.device)

        return self.norms[norm_key](x)


# 正样本提取阶段
class LightweightDenoiseConvNeXt(nn.Module):
    # 带有动态适配器的正样本特征提取器
    def __init__(self, in_channels: int = 256, hidden_dim: int = None,
                 drop_path_rate: float = 0.1, dims: List[int] = None):
        super().__init__()
        # 基础通道数，但实际运行时会动态适应
        self.base_in_channels = in_channels

        if hidden_dim is None:
            hidden_dim = max(64, in_channels // 2)
        self.hidden_dim = hidden_dim

        # 存储不同通道数的网络模块
        self.denoise_networks = nn.ModuleDict()
        self.gates = nn.ModuleDict()

        print(f"初始化去噪器，基础通道数: {in_channels}, 隐藏维度: {hidden_dim}")

    def _create_denoise_network(self, in_channels: int) -> nn.Module:
        """为特定通道数创建去噪网络"""
        return nn.Sequential(
            # 1. 输入投影
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),

            # 2. 深度可分离卷积块1
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=7, padding=3,
                     groups=self.hidden_dim, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),

            # 3. 点卷积扩展
            nn.Conv2d(self.hidden_dim, self.hidden_dim * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.GELU(),

            # 4. Dropout
            nn.Dropout2d(p=0.1),

            # 5. 点卷积收缩
            nn.Conv2d(self.hidden_dim * 4, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),

            # 6. 第二个深度可分离卷积块
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2,
                     groups=self.hidden_dim, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),

            # 7. 输出投影
            nn.Conv2d(self.hidden_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def _create_gate_network(self, in_channels: int) -> nn.Module:
        """为特定通道数创建门控网络"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(16, in_channels // 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(16, in_channels // 8), in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """前向传播"""
        if isinstance(x, list):
            return [self._denoise_single_feature(feat) for feat in x]
        else:
            return self._denoise_single_feature(x)

    def _denoise_single_feature(self, feat: torch.Tensor) -> torch.Tensor:
        """对单个特征图进行去噪处理"""
        B, C, H, W = feat.shape

        # 为当前通道数创建或获取网络
        net_key = f"net_{C}"
        gate_key = f"gate_{C}"

        if net_key not in self.denoise_networks:
            print(f"为通道数 {C} 创建新的去噪网络")
            self.denoise_networks[net_key] = self._create_denoise_network(C).to(feat.device)
            self.gates[gate_key] = self._create_gate_network(C).to(feat.device)

            # 初始化权重
            self.denoise_networks[net_key].apply(self._init_weights)
            self.gates[gate_key].apply(self._init_weights)

        # 获取对应的网络
        denoise_net = self.denoise_networks[net_key]
        gate_net = self.gates[gate_key]

        # 去噪处理
        try:
            denoised = denoise_net(feat)
        except Exception as e:
            print(f"去噪网络处理失败 (通道数 {C}): {e}")
            return feat  # 返回原始特征

        # 门控融合
        try:
            gate_weight = gate_net(denoised)
            output = feat * (1 - gate_weight) + denoised * gate_weight
        except Exception as e:
            print(f"门控融合失败 (通道数 {C}): {e}")
            output = (feat + denoised) * 0.5  # 简单平均

        return output


# 负样本提取阶段
def get_spatial_negative_by_bbox(orig_feats: List[torch.Tensor],
                                 strategy: str = 'spatial_shuffle') -> List[torch.Tensor]:
    """生成负样本特征"""
    negative_feats = []

    for feat in orig_feats:
        try:
            B, C, H, W = feat.shape

            if strategy == 'spatial_shuffle':
                # 空间打乱
                feat_flat = feat.view(B, C, -1)
                indices = torch.randperm(H * W, device=feat.device)
                negative_feat = feat_flat[:, :, indices].view(B, C, H, W)

            elif strategy == 'feature_corrupt':
                # 特征损坏
                noise = torch.randn_like(feat) * 0.15

                # 随机遮挡
                mask = torch.ones_like(feat)
                if H > 8 and W > 8:  # 确保特征图足够大
                    mask_h = torch.randint(0, max(1, H // 4), (1,)).item()
                    mask_w = torch.randint(0, max(1, W // 4), (1,)).item()
                    mask_size_h = min(H // 4, H - mask_h)
                    mask_size_w = min(W // 4, W - mask_w)
                    mask[:, :, mask_h:mask_h + mask_size_h, mask_w:mask_w + mask_size_w] = 0

                negative_feat = feat * mask + noise

            else:
                # 默认：简单噪声
                negative_feat = feat + torch.randn_like(feat) * 0.1

        except Exception as e:
            print(f"负样本生成失败: {e}")
            negative_feat = feat + torch.randn_like(feat) * 0.05

        negative_feats.append(negative_feat)

    return negative_feats


def create_yolov8_denoiser(model_size: str = 'small',
                           custom_channels: int = None) -> LightweightDenoiseConvNeXt:
    """创建YOLOv8去噪器"""
    if custom_channels is not None:
        base_channels = custom_channels
        hidden_dim = max(64, base_channels // 2)
    else:
        # 基于模型大小设置基础通道数
        channel_configs = {
            'nano': 256,
            'small': 512,
            'medium': 768,
            'large': 512,
            'xlarge': 640,
        }
        base_channels = channel_configs.get(model_size, 512)
        hidden_dim = max(64, base_channels // 2)

    return LightweightDenoiseConvNeXt(
        in_channels=base_channels,
        hidden_dim=hidden_dim,
        drop_path_rate=0.1
    )


# def get_spatial_negative_by_bbox(feat, gt_bboxes, spatial_scale=1.0):
#     """
#     基于目标框提取空间维度的负样本（非目标区域）
#     Args:
#         feat: 单模态特征 [B, C, H, W]（RGB或红外的某一尺度特征）
#         gt_bboxes: 目标框标注 [B, N, 4]（格式：x1, y1, x2, y2，像素坐标）
#         spatial_scale: 特征图与原始图像的缩放比例（如特征图是原图的1/16，则scale=1/16）
#     Returns:
#         pos_feat: 目标区域特征 [B, C, H, W]（仅目标框内有值，其他区域为0）
#         neg_feat: 非目标区域特征 [B, C, H, W]（仅目标框外有值，其他区域为0）
#     """
#     B, C, H, W = feat.shape
#     pos_mask = torch.zeros((B, 1, H, W), device=feat.device)  # 目标区域掩码（1表示目标）
#     neg_mask = torch.ones((B, 1, H, W), device=feat.device)  # 非目标区域掩码（1表示背景）
#
#     for b in range(B):
#         bboxes = gt_bboxes[b]  # 当前样本的所有目标框 [N, 4]
#         if bboxes.numel() == 0:
#             # 无目标时，全区域为负样本
#             pos_mask[b] = 0
#             neg_mask[b] = 1
#             continue
#
#         # 目标框坐标转换到特征图尺度（原始坐标 × 缩放比例）
#         bboxes_feat = bboxes * spatial_scale
#         # 取整得到特征图上的坐标（x1, y1, x2, y2）
#         x1 = torch.clamp(bboxes_feat[:, 0].long(), 0, W - 1)
#         y1 = torch.clamp(bboxes_feat[:, 1].long(), 0, H - 1)
#         x2 = torch.clamp(bboxes_feat[:, 2].long(), 0, W - 1)
#         y2 = torch.clamp(bboxes_feat[:, 3].long(), 0, H - 1)
#
#         # 标记目标区域（掩码置1）
#         for i in range(bboxes.shape[0]):
#             pos_mask[b, 0, y1[i]:y2[i] + 1, x1[i]:x2[i] + 1] = 1
#         # 非目标区域 = 1 - 目标区域
#         neg_mask[b] = 1 - pos_mask[b]
#
#     # 应用掩码提取正/负样本特征（保留空间结构，便于后续池化）
#     pos_feat = feat * pos_mask  # 目标区域特征
#     neg_feat = feat * neg_mask  # 非目标区域特征（负样本）
#     return pos_feat, neg_feat
