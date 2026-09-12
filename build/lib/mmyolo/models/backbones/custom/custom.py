# mkp模块
import torch.nn as nn
import torch
from mmengine.registry import MODELS
# from ..backbones import CSPDarknet
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

# ================== 修复并适配你的Pzconv卷积层 ==================

class Conv(nn.Module):
    """标准卷积层 - 用于Pzconv内部"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1,
                 bias=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Pzconv(nn.Module):
    """你的自定义Pzconv卷积层 - 修复版"""

    def __init__(self, dim, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 多尺度深度卷积序列：3x3 -> 1x1 -> 5x5 -> 1x1 -> 7x7
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)  # 3x3深度卷积
        self.conv2 = Conv(dim, dim, kernel_size=1, stride=1)  # 1x1逐点卷积
        self.conv3 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)  # 5x5深度卷积
        self.conv4 = Conv(dim, dim, 1, 1)  # 1x1逐点卷积
        self.conv5 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)  # 7x7深度卷积

    def forward(self, x):
        identity = x  # 保存输入用于残差连接
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = x5 + identity  # 残差连接
        return x6


# ================== 适配YOLOv8的Pzconv包装层 ==================

class PzconvModule(BaseModule):
    """适配YOLOv8的Pzconv包装层"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 norm_cfg: dict = None,
                 act_cfg: dict = None,
                 **kwargs):
        super().__init__()

        # 如果输入输出通道不同，需要先调整通道数
        if in_channels != out_channels:
            self.channel_align = Conv(in_channels, out_channels, 1, 1)
        else:
            self.channel_align = None

        # 如果需要下采样(stride>1)，先进行下采样
        if stride > 1:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        else:
            self.downsample = None
            self.downsample_bn = None

        # 核心Pzconv层
        self.pzconv = Pzconv(out_channels)

        # 最终激活函数
        if act_cfg and act_cfg.get('type') == 'SiLU':
            self.final_act = nn.SiLU(inplace=act_cfg.get('inplace', True))
        else:
            self.final_act = None

    def forward(self, x):
        # 1. 通道对齐
        if self.channel_align is not None:
            x = self.channel_align(x)

        # 2. 下采样处理
        if self.downsample is not None:
            x = self.downsample_bn(self.downsample(x))

        # 3. Pzconv处理
        x = self.pzconv(x)

        # 4. 最终激活
        if self.final_act is not None:
            x = self.final_act(x)

        return x

class PzconvCSPLayerWithTwoConv(BaseModule):
    """CSP层内部也使用Pzconv的版本"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 norm_cfg: dict = None,
                 act_cfg: dict = None):
        super().__init__()

        mid_channels = out_channels // 2

        # CSP分支卷积
        self.main_conv = Conv(in_channels, mid_channels, 1, 1)
        self.short_conv = Conv(in_channels, mid_channels, 1, 1)
        self.final_conv = Conv(out_channels, out_channels, 1, 1)

        # Bottleneck blocks - 内部使用Pzconv
        self.blocks = nn.ModuleList([
            PzconvBottleneck(mid_channels, mid_channels, add_identity)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # 主分支通过blocks处理
        main_out = self.main_conv(x)
        for block in self.blocks:
            main_out = block(main_out)

        # 短连接分支
        short_out = self.short_conv(x)

        # 合并并最终处理
        final_out = torch.cat([main_out, short_out], dim=1)
        return self.final_conv(final_out)


class PzconvBottleneck(nn.Module):
    """使用Pzconv的Bottleneck块"""

    def __init__(self, in_channels, out_channels, add_identity=True):
        super().__init__()
        self.add_identity = add_identity and in_channels == out_channels

        # 使用Pzconv替代标准卷积
        self.pzconv = Pzconv(in_channels)

        # 如果通道数不匹配，添加1x1卷积调整
        if in_channels != out_channels:
            self.channel_conv = Conv(in_channels, out_channels, 1, 1)
        else:
            self.channel_conv = None

    def forward(self, x):
        identity = x

        # Pzconv处理
        out = self.pzconv(x)

        # 通道调整
        if self.channel_conv is not None:
            out = self.channel_conv(out)

        # 残差连接
        if self.add_identity:
            out = out + identity

        return out