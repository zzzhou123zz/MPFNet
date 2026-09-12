# YOLOv8 Pzconv自定义卷积替换方案
import torch
import torch.nn as nn
from typing import Union, List, Tuple
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from ...layers import CSPLayerWithTwoConv, SPPFBottleneck
from .custom import *
from ..base_backbone import BaseBackbone
import math


# ================== 修改后的YOLOv8类 ==================
def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


@MODELS.register_module()
class YOLOv8CSPDarknetPzconv(BaseBackbone):
    """使用Pzconv的YOLOv8 CSP-Darknet backbone"""
    # 第三个参数表示该stage中block数量，deepen_factor控制block缩放，则实际block数量为block*deepen_factor（进行四舍五入）
    # [in_channels, out_channels, num_blocks, add_identity（CSP 层中 block 是否添加残差连接（True 表示添加，增强残差连接））, use_spp]
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, None, 3, False, True]]
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),  # 骨干网络输出哪些阶段的特征进行融合
                 frozen_stages: int = -1,
                 norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg=None,
                 # 新增参数：选择是否在stage层使用Pzconv
                 use_pzconv_in_stage: bool = True,
                 use_pzconv_in_csp: bool = False):

        self.use_pzconv_in_stage = use_pzconv_in_stage
        self.use_pzconv_in_csp = use_pzconv_in_csp
        self.arch_settings[arch][-1][1] = last_stage_out_channels

        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Stem层保持不变 - 使用标准ConvModule"""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """构建stage层 - 可选择使用Pzconv"""
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []

        # 🎯 Stage层下采样卷积 - 可选择使用Pzconv
        if self.use_pzconv_in_stage:
            conv_layer = PzconvModule(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            print(f"Stage {stage_idx}: 使用Pzconv卷积 {in_channels}→{out_channels}")
        else:
            conv_layer = ConvModule(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        stage.append(conv_layer)

        # CSP层 - 可选择使用自定义版本
        if self.use_pzconv_in_csp:
            csp_layer = PzconvCSPLayerWithTwoConv(
                out_channels, out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            csp_layer = CSPLayerWithTwoConv(
                out_channels, out_channels, num_blocks=num_blocks,
                add_identity=add_identity, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        stage.append(csp_layer)

        # SPP层保持不变
        if use_spp:
            spp = SPPFBottleneck(
                out_channels, out_channels, kernel_sizes=5,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            stage.append(spp)

        return stage

# ================== 使用示例和测试 ==================

# def analyze_pzconv_structure():
#     """分析Pzconv的特点"""
#     print("🔍 Pzconv卷积层结构分析:")
#     print("━" * 50)
#     print("🔸 Conv1: 3×3深度卷积 (groups=dim)")
#     print("🔸 Conv2: 1×1逐点卷积 + BN + SiLU")
#     print("🔸 Conv3: 5×5深度卷积 (groups=dim)")
#     print("🔸 Conv4: 1×1逐点卷积 + BN + SiLU")
#     print("🔸 Conv5: 7×7深度卷积 (groups=dim)")
#     print("🔸 残差连接: x5 + x (跳跃连接)")
#     print("━" * 50)
#     print("✨ 特点: 多尺度感受野 (3×3, 5×5, 7×7) + 残差学习")
#
#
# if __name__ == "__main__":
#     print("🚀 YOLOv8 Pzconv自定义卷积替换示例")
#
#     # 分析Pzconv结构
#     analyze_pzconv_structure()
#
#     print("\n📌 使用方案:")
#
#     # 方案1: 只在Stage层使用Pzconv
#     print("\n1️⃣ Stage层使用Pzconv:")
#     model_stage = YOLOv8CSPDarknetPzconv(
#         use_pzconv_in_stage=True,
#         use_pzconv_in_csp=False
#     )
#
#     # 方案2: Stage层和CSP层都使用Pzconv
#     print("\n2️⃣ Stage + CSP层都使用Pzconv:")
#     model_full = YOLOv8CSPDarknetPzconv(
#         use_pzconv_in_stage=True,
#         use_pzconv_in_csp=True
#     )
#
#     # 测试模型
#     print("\n🧪 模型测试:")
#     x = torch.randn(1, 3, 416, 416)
#
#     try:
#         # 测试只使用Stage层Pzconv的模型
#         outputs = model_stage(x)
#         param_count = sum(p.numel() for p in model_stage.parameters())
#         print(f"✅ Stage-Pzconv模型: 参数量 {param_count:,}")
#         print(f"   输出形状: {[tuple(out.shape) for out in outputs]}")
#
#     except Exception as e:
#         print(f"❌ 模型测试失败: {str(e)}")
#
#     print("\n💡 Pzconv优势:")
#     print("- 🎯 多尺度感受野: 同时捕获3×3, 5×5, 7×7的特征")
#     print("- 🔄 残差学习: 有助于梯度流动和特征保持")
#     print("- ⚡ 深度卷积: 减少参数量，提高计算效率")
#     print("- 🎨 特征增强: 逐层增强特征表达能力")
#
#     print("\n🔧 使用建议:")
#     print("- 适合需要多尺度特征的检测任务")
#     print("- 可以替换YOLOv8中需要增强感受野的层")
#     print("- 建议先在Stage层测试效果，再考虑CSP层")
