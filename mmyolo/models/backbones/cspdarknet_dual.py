from mmdet.registry import MODELS
from mmdet.models.backbones import CSPDarknet


@MODELS.register_module()
class DualCSPDarknet(nn.Module):
    def __init__(self,
                 rgb_cfg: dict,  # 可见光分支配置
                 thermal_cfg: dict,  # 红外分支配置
                 fusion_indices: list = [2, 3, 4]):  # 融合阶段(P3-P5)
        super().__init__()

        # 双分支Backbone
        self.rgb_backbone = CSPDarknet(**rgb_cfg)
        self.thermal_backbone = CSPDarknet(**thermal_cfg)

        # 特征融合层
        self.fusion_convs = nn.ModuleList([
            ConvModule(
                in_channels=rgb_cfg['arch'][i] * 2,
                out_channels=rgb_cfg['arch'][i],
                kernel_size=1,
                norm_cfg=dict(type='BN'))
            for i in fusion_indices
        ])

    def forward(self, rgb, thermal):
        rgb_features = self.rgb_backbone(rgb)
        thermal_features = self.thermal_backbone(thermal)

        fused_features = []
        for i, idx in enumerate(self.fusion_indices):
            fused = torch.cat([rgb_features[idx], thermal_features[idx]], dim=1)
            fused = self.fusion_convs[i](fused)
            fused_features.append(fused)

        return tuple(fused_features)