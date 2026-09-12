import torch.nn as nn
from mmengine.registry import MODELS
from mmyolo.models.backbones import CSPDarknet


@MODELS.register_module()
class DualStreamBackbone(nn.Module):
    def __init__(self,
                 rgb_backbone_cfg,
                 ir_backbone_cfg,
                 fusion_cfg,
                 out_indices=(2, 3, 4)):
        super().__init__()

        # RGB流
        self.rgb_backbone = MODELS.build(rgb_backbone_cfg)

        # IR流
        self.ir_backbone = MODELS.build(ir_backbone_cfg)

        # 融合模块
        self.fusion = MODELS.build(fusion_cfg)

        self.out_indices = out_indices

    def forward(self, rgb_img, ir_img):
        """前向传播

        Args:
            rgb_img (Tensor): RGB图像张量
            ir_img (Tensor): 红外图像张量

        Returns:
            tuple[Tensor]: 多尺度融合特征
        """
        # 提取特征
        rgb_features = self.rgb_backbone(rgb_img)
        ir_features = self.ir_backbone(ir_img)

        # 融合特征
        fused_features = []
        for rgb_feat, ir_feat in zip(rgb_features, ir_features):
            fused_feat = self.fusion(rgb_feat, ir_feat)
            fused_features.append(fused_feat)

        # 返回指定尺度的特征
        return tuple(fused_features[i] for i in self.out_indices)