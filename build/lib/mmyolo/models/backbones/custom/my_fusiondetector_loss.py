# Copyright (c) OpenMMLab. All rights reserved.
import torch
import copy
from typing import List, Tuple, Union, Optional
from mmdet.models.detectors.base import BaseDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmengine.dist import get_world_size
from mmengine.logging import print_log
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
import cv2

from mmyolo.registry import MODELS


class FeatureSSIMLoss(nn.Module):
    """
    适用于可见光和红外光特征图的SSIM损失函数
    考虑到两种模态的特性差异，使用自适应窗口和动态常数
    """

    def __init__(self,
                 window_size: int = 5,  # 特征图通常较小，使用5x5窗口
                 dynamic_constants: bool = True,  # 动态计算C1和C2
                 weight: float = 0.1):  # 损失权重
        super().__init__()
        self.window_size = window_size
        self.dynamic_constants = dynamic_constants
        self.weight = weight
        self.window = self._create_gaussian_window()

    def _create_gaussian_window(self):
        """创建高斯窗口，模拟局部感受野"""
        gauss = torch.Tensor([
            math.exp(-(x - self.window_size // 2) ** 2 / float(2 * 0.5 ** 2))
            for x in range(self.window_size)
        ])
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return window_2d  # 形状: [1, 1, window_size, window_size]

    def forward(self, vis_feats: List[torch.Tensor], ir_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        计算可见光和红外光特征图列表的SSIM损失

        Args:
            vis_feats: 可见光特征图列表，每个元素形状为[B, C, H, W]
            ir_feats: 红外光特征图列表，每个元素形状为[B, C, H, W]

        Returns:
            标量损失值
        """
        total_loss = 0.0
        num_layers = len(vis_feats)

        # 确保两种模态的特征图层数一致
        assert len(vis_feats) == len(ir_feats), \
            f"可见光特征图层数({len(vis_feats)})与红外光特征图层数({len(ir_feats)})不匹配"

        # 逐层计算SSIM损失
        for vis_feat, ir_feat in zip(vis_feats, ir_feats):
            # 确保特征图尺寸匹配
            assert vis_feat.shape == ir_feat.shape, \
                f"特征图尺寸不匹配: {vis_feat.shape} vs {ir_feat.shape}"

            # 将窗口移动到特征图所在设备
            window = self.window.to(vis_feat.device)
            B, C, H, W = vis_feat.shape

            # 如果通道数不匹配，扩展窗口以适应分组卷积
            if window.size(0) != C:
                window = window.expand(C, 1, self.window_size, self.window_size).contiguous()

            # 动态计算C1和C2（基于特征值范围）
            if self.dynamic_constants:
                # 对于可见光和红外光特征，使用各自的最大值计算常数
                max_val_vis = torch.max(torch.abs(vis_feat))
                max_val_ir = torch.max(torch.abs(ir_feat))
                max_val = torch.max(max_val_vis, max_val_ir)
                C1 = (0.01 * max_val) ** 2
                C2 = (0.03 * max_val) ** 2
            else:
                # 静态常数（如果特征已归一化）
                C1 = 1e-4
                C2 = 9e-4

            # 计算局部均值
            mu_vis = F.conv2d(vis_feat, window, padding=self.window_size // 2, groups=C)
            mu_ir = F.conv2d(ir_feat, window, padding=self.window_size // 2, groups=C)

            # 计算局部方差和协方差
            mu_vis_sq = mu_vis ** 2
            mu_ir_sq = mu_ir ** 2
            mu_vis_ir = mu_vis * mu_ir

            sigma_vis = F.conv2d(vis_feat ** 2, window, padding=self.window_size // 2, groups=C) - mu_vis_sq
            sigma_ir = F.conv2d(ir_feat ** 2, window, padding=self.window_size // 2, groups=C) - mu_ir_sq
            sigma_vis_ir = F.conv2d(vis_feat * ir_feat, window, padding=self.window_size // 2, groups=C) - mu_vis_ir

            # 计算SSIM
            ssim_map = ((2 * mu_vis_ir + C1) * (2 * sigma_vis_ir + C2)) / \
                       ((mu_vis_sq + mu_ir_sq + C1) * (sigma_vis + sigma_ir + C2))

            # 累加当前层的损失（1 - SSIM均值）
            total_loss += (1 - ssim_map.mean())

        # 返回平均损失并乘以权重
        return self.weight * (total_loss / num_layers)


@MODELS.register_module()
class my_fusiondetector_loss(BaseDetector):
    # 融合检测
    """
    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 fusion_block: ConfigType = dict(type='FCM_DualInput'),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True,
                 ssim_loss_cfg: dict = dict(weight=0.1, window_size=5)):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # 两个骨干网络
        self.backbone1 = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone)
        # 特征融合
        self.fusion_block = MODELS.build(fusion_block)  # BaseFusion2：对应层元素逐个相加

        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # 新增：初始化SSIM损失（用于衡量可见光和红外光特征差异）
        self.ssim_loss = FeatureSSIMLoss(**ssim_loss_cfg)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        # Find backbone parameters
        copy_ori = False
        ori_backbone_params = []
        ori_backbone_key = []
        for k, v in state_dict.items():
            if (k.startswith("backbone") and "backbone1" not in k and "backbone2" not in k):
                # or (k.startswith("neck")  and "neck1" not in k and "neck2" not in k):
                # Pretrained on original model
                ori_backbone_params += [v]
                ori_backbone_key += [k]
                copy_ori = True

        if copy_ori:
            for k, v in zip(ori_backbone_key, ori_backbone_params):
                state_dict[k.replace("backbone", "backbone1")] = v
                state_dict[k.replace("backbone", "backbone2")] = copy.deepcopy(v)
                # state_dict[k.replace("neck", "neck1")] = v
                # state_dict[k.replace("neck", "neck2")] = copy.deepcopy(v)
                del state_dict[k]
            # Force set the strict to "False"
            strict = False
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        res = super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                            strict, missing_keys, unexpected_keys,
                                            error_msgs)
        return res

    def forward(self,
                inputs: torch.Tensor,
                inputs2: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):

        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, inputs2, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, inputs2, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, inputs2, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    # 双骨干特征提取
    def extract_feat(self, batch_inputs: Tensor, batch_inputs2: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = list(self.backbone1(batch_inputs))  # 模态1特征
        y = list(self.backbone2(batch_inputs2))  # 模态2

        # 一次融合
        z = self.fusion_block(x, y)

        if self.with_neck:
            z = self.neck(z)
        return z

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_inputs2: Tensor,
                 batch_data_samples: OptSampleList = None):
        x = self.extract_feat(batch_inputs, batch_inputs2)
        results_list = self.bbox_head.forward(x)
        return results_list

    # 双模态融合特征计算损失
    def loss(self, batch_inputs: Tensor, batch_inputs2: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
        """
        x = self.extract_feat(batch_inputs, batch_inputs2)
        x1 = list(self.backbone1(batch_inputs))  # 模态1特征
        x2 = list(self.backbone2(batch_inputs2))  # 模态2
        det_losses = self.bbox_head.loss(x, batch_data_samples)
        ssim_loss_val = self.ssim_loss(x1, x2)
        det_losses['vis_ir_ssim_loss'] = ssim_loss_val
        # losses += ssim_loss_val
        return det_losses

    def predict(self,
                batch_inputs: Tensor, batch_inputs2: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs, batch_inputs2)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
