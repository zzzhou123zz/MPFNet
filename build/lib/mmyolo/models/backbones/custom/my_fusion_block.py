# Ultralytics YOLO ??, AGPL-3.0 license
"""
Convolution modules
"""
from typing import Tuple
from torch import Tensor
from einops import rearrange
import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from mmyolo.registry import MODELS

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
        c2 = int(c2)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)




class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels):
        super().__init__()
        self.dconv = nn.Conv2d(
            in_channels, in_channels, 3,
            1, 1, groups=in_channels
        )

    def forward(self, x):
        x = self.dconv(x)
        return x


class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = self.dconv = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6


class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        # self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        # x5 = self.bn(x1)
        x6 = self.sigmoid(x1)

        return x6

# 分割比例：0.75：0.25；无额外卷积层；
# 原文中的配置为：3-2-1-fcm
class FCM_3(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        # 按比例分割，x1占3/4，x2占1/4
        self.one = dim - dim // 4
        self.two = dim // 4
        # 分支1卷积层处理x1：两次3x3卷积+一次1x1卷积
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        # 分支2卷积层处理x2：一次1x1卷积
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        # 空间注意力+通道注意力
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        # 特征融合：直接相加
        x5 = x33 + x44
        return x5

# 分割比例：0.75：0.25；无额外卷积层；
class FCM_2(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44

        return x5

# 分割比例：0.25：0.75；无额外卷积层；
class FCM_1(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()

        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44

        return x5

# 分割比例：0.25：0.75；有额外卷积层conv3
# class FCM(nn.Module):
#     def __init__(self, dim,dim_out):
#         super().__init__()
#         self.one = dim // 4
#         self.two = dim - dim // 4
#         self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
#         self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
#         self.conv123 = Conv(dim // 4, dim, 1, 1)
#
#         self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
#         self.conv3 = Conv(dim, dim, 1, 1)
#         self.spatial = Spatial(dim)
#         self.channel = Channel(dim)
#
#     def forward(self, x):
#         x1, x2 = torch.split(x, [self.one, self.two], dim=1)
#         x3 = self.conv1(x1)
#         x3 = self.conv12(x3)
#         x3 = self.conv123(x3)
#         x4 = self.conv2(x2)
#         x33 = self.spatial(x4) * x3
#         x44 = self.channel(x3) * x4
#         x5 = x33 + x44
#         x5 = self.conv3(x5)
#         return x5


class Down(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv2 = Conv(dim, dim, 3, 2, 1, g=dim // 2, act=False)
        self.conv4 = Conv(dim, dim_out, 1, 1)

    def forward(self, x):
        x2 = self.conv2(x)
        x2 = self.conv4(x2)
        return x2

import torch.nn as nn
from mmcv.cnn import ConvModule


class SELayer(nn.Module):
    """SE注意力模块（结合通道和空间注意力）"""

    def __init__(self, dim):
        super().__init__()
        self.channel_att = Channel(dim)
        self.spatial_att = Spatial(dim)

    def forward(self, x):
        # 先进行通道注意力，再进行空间注意力
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class FCM(nn.Module):
    def __init__(self, dim_out, align_method='adaptive_pool'):
        """
        自适应双模态融合模块 - 根据输入自动适配通道数
        Args:
            dim_out: 输出维度
            align_method: 尺度对齐方法 ('interpolate', 'adaptive_pool', 'conv_align')
        """
        super().__init__()
        self.dim_out = dim_out
        self.align_method = align_method

        # 这些层将在第一次前向传播时初始化
        self.conv1 = None
        self.conv12 = None
        self.conv123 = None
        self.conv2 = None
        self.conv22 = None
        self.conv223 = None
        self.conv3 = None
        self.att1 = None
        self.att2 = None
        self.selayer = None

        # 用于标记是否已初始化
        self.initialized = False

    def _initialize_layers(self, dim1, dim2):
        """根据输入维度初始化网络层"""
        # 第一个模态的处理分支
        self.conv1 = Conv(dim1, dim1, 3, 1, 1)
        self.conv12 = Conv(dim1, dim1, 3, 1, 1)
        self.conv123 = Conv(dim1, self.dim_out, 1, 1)

        # 第二个模态的处理分支
        self.conv2 = Conv(dim2, self.dim_out, 3, 1, 1)
        self.conv22 = Conv(dim1, dim1, 3, 1, 1)
        self.conv223 = Conv(dim1, self.dim_out, 1, 1)

        # RGB分支做通道+空间交叉注意力
        self.att1 = Channel(self.dim_out)
        self.att2 = Spatial(self.dim_out)

        # ir分支做空间注意力
        self.att2 = Spatial(self.dim_out)
        # 融合后的处理
        self.conv3 = Conv(self.dim_out, self.dim_out, 1, 1)
        # # 注意力模块
        # self.spatial = Spatial(self.dim_out)
        # self.channel = Channel(self.dim_out)
        # 新增SE层用于处理xm3
        self.selayer = SELayer(self.dim_out)  # 变量名拼写与使用处一致
        self.initialized = True

    def _align_features(self, feat1, feat2):
        """
        对齐两个特征的空间尺寸
        Args:
            feat1: 第一个特征 [B, C, H1, W1]
            feat2: 第二个特征 [B, C, H2, W2]
        Returns:
            对齐后的特征 (feat1_aligned, feat2_aligned)
        """
        B, C, H1, W1 = feat1.shape
        _, _, H2, W2 = feat2.shape

        if H1 == H2 and W1 == W2:
            return feat1, feat2

        if self.align_method == 'interpolate':
            # 将特征对齐到较大的尺寸
            target_h, target_w = max(H1, H2), max(W1, W2)
            if H1 != target_h or W1 != target_w:
                feat1 = F.interpolate(feat1, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)
            if H2 != target_h or W2 != target_w:
                feat2 = F.interpolate(feat2, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)

        elif self.align_method == 'adaptive_pool':
            # 将特征对齐到较小的尺寸
            target_h, target_w = min(H1, H2), min(W1, W2)
            if H1 != target_h or W1 != target_w:
                feat1 = F.adaptive_avg_pool2d(feat1, (target_h, target_w))
            if H2 != target_h or W2 != target_w:
                feat2 = F.adaptive_avg_pool2d(feat2, (target_h, target_w))

        return feat1, feat2

    def forward(self, modal1, modal2):
        """
        前向传播，支持不同尺度输入
        Args:
            modal1: 第一个模态输入 [B, C1, H1, W1]
            modal2: 第二个模态输入 [B, C2, H2, W2]
        Returns:
            融合后的特征 [B, dim_out, H_out, W_out]
        """
        # 第一次前向传播时初始化网络层
        if not self.initialized:
            dim1, dim2 = modal1.size(1), modal2.size(1)
            self._initialize_layers(dim1, dim2)
            # 将新初始化的层移动到正确的设备
            device = modal1.device
            self.conv1 = self.conv1.to(device)
            self.conv12 = self.conv12.to(device)
            self.conv123 = self.conv123.to(device)
            self.conv2 = self.conv2.to(device)
            self.conv22 = self.conv22.to(device)
            self.conv223 = self.conv223.to(device)
            self.conv3 = self.conv3.to(device)
            # self.spatial = self.spatial.to(device)
            # self.channel = self.channel.to(device)
            self.att1 = self.att1.to(device)
            self.att2 = self.att2.to(device)
            self.selayer = self.selayer.to(device)

        # 处理第一个模态
        x3 = self.conv1(modal1)
        xm1 = x3
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)

        # 处理第二个模态
        x4 = self.conv2(modal2)
        xm2 = x4
        x4 = self.conv12(x4)
        x4 = self.conv123(x4)

        # 对齐特征尺寸
        x3_aligned, x4_aligned = self._align_features(x3, x4)

        # # 双向注意力融合
        # x33 = self.spatial(x4_aligned) * x3_aligned  # 空间注意力加权的第一模态特征
        # x44 = self.channel(x3_aligned) * x4_aligned  # 通道注意力加权的第二模态特征

        x33 = self.att2(x4_aligned) * x3_aligned
        x44 = (self.att1(x3_aligned) + self.att2(x3_aligned)) * x4_aligned
        # 特征融合
        x5 = x33 + x44
        xm3 = xm1 + xm2
        xm3_se = self.selayer(xm3)
        x5 = x5 + xm3_se
        x5 = self.conv3(x5)

        return x5


# 直接用于单个特征融合
@MODELS.register_module()
class FCM_DualInput(nn.Module):
    def __init__(self, dim_out, align_method='interpolate'):
        """
        多尺度双模态融合模块（自适应通道数）
        Args:
            dim_out: 各尺度的输出维度（可以是单个值或列表）
            align_method: 尺度对齐方法
        """
        super().__init__()
        self.align_method = align_method
        self.fcm_blocks = nn.ModuleList()
        self.dim_out = dim_out if isinstance(dim_out, list) else [dim_out]
        self.initialized = False

    def _initialize_blocks(self, modal1_features, modal2_features):
        """根据输入特征初始化FCM块"""
        self.num_scales = len(modal1_features)
        if len(self.dim_out) == 1:
            self.dim_out = self.dim_out * self.num_scales

        self.fcm_blocks = nn.ModuleList([
            FCM(dim_out=d_out, align_method=self.align_method)
            for d_out in self.dim_out
        ])
        self.initialized = True

    def forward(self, modal1_features, modal2_features):
        """
        Args:
            modal1_features: 第一个模态的多尺度特征列表
            modal2_features: 第二个模态的多尺度特征列表
        Returns:
            融合后的多尺度特征列表
        """
        if not self.initialized:
            self._initialize_blocks(modal1_features, modal2_features)
            # 将FCM块移到正确设备
            device = modal1_features[0].device
            self.fcm_blocks = self.fcm_blocks.to(device)

        assert len(modal1_features) == len(modal2_features) == self.num_scales

        fused_features = []
        for i, (feat1, feat2) in enumerate(zip(modal1_features, modal2_features)):
            fused_feat = self.fcm_blocks[i](feat1, feat2)
            fused_features.append(fused_feat)

        return fused_features


class AttentionInteraction(nn.Module):
    """注意力交互模块 - 封装您的注意力逻辑"""

    def __init__(self, dim_out):
        super().__init__()
        self.spatial = Spatial(dim_out)
        self.channel = Channel(dim_out)

    def forward(self, inputs):
        # 输入是分支1和分支2的输出
        x3, x4 = inputs

        # 注意力交互
        x33 = self.spatial(x4) * x3  # 用x4的空间注意力增强x3
        x44 = self.channel(x3) * x4  # 用x3的通道注意力增强x4

        return x33 + x44


# # 保持您的原始注意力模块（需要您提供实现）
# class Spatial(nn.Module):
#     def __init__(self, dim_out):
#         super().__init__()
#         # 您的空间注意力实现
#
#     def forward(self, x):
#         # 您的空间注意力前向传播
#         return x
#
#
# class Channel(nn.Module):
#     def __init__(self, dim_out):
#         super().__init__()
#         # 您的通道注意力实现
#
#     def forward(self, x):
#         # 您的通道注意力前向传播
#         return x