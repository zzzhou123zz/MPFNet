# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig
from copy import deepcopy

from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov5_pafpn import YOLOv5PAFPN


@MODELS.register_module()
class YOLOv8PAFPN_Attn(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 attnblock: ConfigType = None,
                 apply_atten_layers:List[int] = [0,0,0],
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        
        assert len(apply_atten_layers) <= len(in_channels) and max(apply_atten_layers) < len(in_channels), "len(apply_attn_layers) or max(apply_attn_layers) should be less than len(in_channels)"
        self.attn_list = [None for _ in in_channels]
        attnblock['inp'] = [make_divisible(x, self.widen_factor) for x in attnblock['inp']]
        attnblock['oup'] = [make_divisible(x, self.widen_factor) for x in attnblock['oup']]
        attnblock_configs = self.generate_configs(attnblock)
        for i in apply_atten_layers:
            self.attn_list[i] = MODELS.build(attnblock_configs[i])

    def generate_configs(self,base_config):
        # 找到所有需要展开的列表
        keys_with_lists = {k: v for k, v in base_config.items() if isinstance(v, list)}
        if not keys_with_lists:
            return base_config
        # 获取列表的长度，假设所有列表的长度相同
        num_configs = len(next(iter(keys_with_lists.values())))

        # 生成多个配置
        configs = []
        for i in range(num_configs):
            new_config = deepcopy(base_config)
            for k, v in keys_with_lists.items():
                new_config[k] = v[i]  # 取出每个列表的第i个值
            configs.append(new_config)
        
        return configs

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0] if self.attn_list[0] is None else self.attn_list[0](inner_out[0])]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out if self.attn_list[idx+1] == None else self.attn_list[idx+1](out))

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)

