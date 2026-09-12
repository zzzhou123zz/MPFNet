import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np

from typing import Tuple , List
from torch import Tensor

from mmengine.model import BaseModule
from mmyolo.models.utils import make_divisible, make_round

from mmyolo.registry import MODELS

# 多尺度特征相加，保持元组结构
#@MODELS.register_module()
# class BaseFusion2(BaseModule): # 对应层元素逐个相加
#     def __init__(
#         self,
#     ) -> None:
#         super().__init__()
#
#     def forward(self, inputs1: Tuple[Tensor],inputs2: Tuple[Tensor]):
#         """Forward function."""
#         outs = []
#         for i , (ir , rgb) in enumerate(zip(inputs1 , inputs2)):
#             outs.append(ir + rgb)
#         return tuple(outs)
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)
#concat方法的融合
@MODELS.register_module()
class BaseFusion2(BaseModule):
    """基于特征concat的双模态融合模块"""

    def __init__(self,
                 fusion_strategy: str = 'concat',
                 use_conv: bool = True,
                 conv_cfg: dict = None,
                 norm_cfg: dict = None,
                 act_cfg: dict = None,
                 init_cfg: dict = None):
        super().__init__(init_cfg=init_cfg)

        self.fusion_strategy = fusion_strategy
        self.use_conv = use_conv

        # 设置默认配置
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        # 不预先定义fusion_convs，在第一次forward时动态创建
        self.fusion_convs = None
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.is_initialized = False

    def _build_fusion_convs(self, feat1: torch.Tensor, feat2: torch.Tensor, level_idx: int):
        """动态构建融合卷积层"""
        if self.fusion_convs is None:
            self.fusion_convs = nn.ModuleList()

        # 获取输入通道数
        in_channels1 = feat1.size(1)
        in_channels2 = feat2.size(1)

        # 根据融合策略确定输入通道数
        if self.fusion_strategy == 'concat':
            conv_in_ch = in_channels1 + in_channels2
        else:
            conv_in_ch = in_channels1  # 假设两个模态通道数相同

        # 输出通道数可以保持与单个模态相同或调整
        # 这里我们保持与单个模态相同的通道数
        out_ch = in_channels1

        # 构建卷积层
        conv_layers = []

        # 添加卷积层
        conv_layers.append(
            nn.Conv2d(conv_in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # 添加归一化层
        if self.norm_cfg is not None:
            norm_type = self.norm_cfg.get('type', 'BatchNorm2d')
            if norm_type == 'BatchNorm2d':
                conv_layers.append(nn.BatchNorm2d(out_ch))
            elif norm_type == 'GroupNorm':
                groups = self.norm_cfg.get('groups', 32)
                conv_layers.append(nn.GroupNorm(groups, out_ch))

        # 添加激活函数
        if self.act_cfg is not None:
            act_type = self.act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                conv_layers.append(nn.ReLU(inplace=True))
            elif act_type == 'LeakyReLU':
                negative_slope = self.act_cfg.get('negative_slope', 0.01)
                conv_layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

        # 确保ModuleList有足够的元素
        while len(self.fusion_convs) <= level_idx:
            self.fusion_convs.append(nn.Identity())

        # 替换对应位置的卷积层
        self.fusion_convs[level_idx] = nn.Sequential(*conv_layers)

        # 将卷积层移动到正确的设备
        device = feat1.device
        self.fusion_convs[level_idx] = self.fusion_convs[level_idx].to(device)

    def forward(self, feats1: List[torch.Tensor], feats2: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播"""
        assert len(feats1) == len(feats2), "两个模态的特征数量必须相同"

        fused_feats = []

        for i, (f1, f2) in enumerate(zip(feats1, feats2)):
            # 检查特征形状
            assert f1.shape[2:] == f2.shape[2:], f"第{i}层特征空间尺寸不匹配: {f1.shape} vs {f2.shape}"

            # 根据融合策略进行特征融合
            if self.fusion_strategy == 'concat':
                # 特征concat融合
                fused = torch.cat([f1, f2], dim=1)
            elif self.fusion_strategy == 'add':
                # 确保通道数相同才能相加
                assert f1.shape[1] == f2.shape[1], f"第{i}层通道数不匹配，无法相加: {f1.shape[1]} vs {f2.shape[1]}"
                fused = f1 + f2
            elif self.fusion_strategy == 'max':
                # 确保通道数相同才能取最大值
                assert f1.shape[1] == f2.shape[1], f"第{i}层通道数不匹配，无法取最大值: {f1.shape[1]} vs {f2.shape[1]}"
                fused = torch.maximum(f1, f2)
            elif self.fusion_strategy == 'avg':
                # 确保通道数相同才能平均
                assert f1.shape[1] == f2.shape[1], f"第{i}层通道数不匹配，无法平均: {f1.shape[1]} vs {f2.shape[1]}"
                fused = (f1 + f2) / 2
            else:
                raise ValueError(f"不支持的融合策略: {self.fusion_strategy}")

            # 如果有融合卷积层，则应用
            if self.use_conv:
                # 动态构建或获取卷积层
                if self.fusion_convs is None or i >= len(self.fusion_convs) or isinstance(self.fusion_convs[i], nn.Identity):
                    self._build_fusion_convs(f1, f2, i)

                fused = self.fusion_convs[i](fused)

            fused_feats.append(fused)

        self.is_initialized = True
        return fused_feats

    def simple_concat_fusion(self, feats1: List[torch.Tensor], feats2: List[torch.Tensor]) -> List[torch.Tensor]:
        """简化的concat融合，不使用卷积层"""
        assert len(feats1) == len(feats2), "两个模态的特征数量必须相同"

        fused_feats = []
        for f1, f2 in zip(feats1, feats2):
            # 直接concat融合
            fused = torch.cat([f1, f2], dim=1)
            fused_feats.append(fused)

        return fused_feats

@MODELS.register_module()
class BaseFusion(BaseModule):
    def __init__(
        self,
        in_channels : int = 256,
    ) -> None:
        super().__init__()
        # self.block1 = nn.Conv2d(in_channels * 2 , in_channels , 1, 1,0)
        # self.block2 = nn.Conv2d(in_channels * 2 , in_channels , 1, 1,0)
        # self.fusion_blocks = nn.Sequential(*blocks)

    def forward(self, input1: Tuple[Tensor],input2: Tuple[Tensor]):
        # ir_fea = self.block1(torch.cat([input1,input2] , dim = 1))
        # vis_fea = self.block2(torch.cat([input2,input1] , dim = 1))
        """Forward function."""
        fusion_fea = input1 + input2
        return [fusion_fea]

@MODELS.register_module()
class Add2(BaseModule):
    def __init__(
            self,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = None,
            norm_cfg: dict = None,
            act_cfg: dict = None,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # 动态创建卷积层字典
        self.fusion_convs = nn.ModuleDict()

    def forward(self, inputs1: Tuple[Tensor], inputs2: Tuple[Tensor]):
        """Forward function."""
        outs = []
        for i, (ir, rgb) in enumerate(zip(inputs1, inputs2)):
            # 获取当前层级的实际通道数
            in_channels = ir.shape[1] + rgb.shape[1]  # 拼接后的总通道数
            out_channels = ir.shape[1]  # 输出通道数与单个输入相同

            # 动态创建或获取对应的卷积层
            layer_key = f'level_{i}_c{in_channels}'
            if layer_key not in self.fusion_convs:
                self.fusion_convs[layer_key] = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, self.kernel_size,
                              self.stride, self.padding),
                    nn.BatchNorm2d(out_channels) if self.norm_cfg else nn.Identity(),
                    nn.ReLU(inplace=True) if self.act_cfg else nn.Identity()
                )
                # 将新层移到正确的设备
                self.fusion_convs[layer_key] = self.fusion_convs[layer_key].to(ir.device)

            # 在通道维度拼接两个模态的特征
            concatenated = torch.cat([ir, rgb], dim=1)
            # 通过卷积学习融合权重
            fused = self.fusion_convs[layer_key](concatenated)
            outs.append(fused)

        return tuple(outs)