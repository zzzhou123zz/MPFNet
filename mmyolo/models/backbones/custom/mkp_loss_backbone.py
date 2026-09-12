import torch
import torch.nn as nn
from mmdet.models.detectors.base import BaseDetector
from typing import List, Tuple, Union
import torch.nn.functional as F
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from torch import Tensor
import cv2

from mmyolo.registry import MODELS
from ...layers import CSPLayerWithTwoConv, SPPFBottleneck
from .custom import *
from ..base_backbone import BaseBackbone
from .my_loss import *


@MODELS.register_module()
class PositiveGuidedInfoNCELoss(nn.Module):
    """正样本引导的InfoNCE损失：约束融合特征与双模态去噪特征的一致性"""

    def __init__(self, temperature=0.3):
        super().__init__()
        self.tau = temperature

    def global_feature_pool(self, feat_list):
        """多尺度特征池化为向量：[B, C, H, W] → [B, C]"""
        pooled_feats = []
        for feat in feat_list:
            avg_pool = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            pooled = avg_pool.view(feat.size(0), -1)  # [B, C]
            pooled_feats.append(pooled)
        return pooled_feats

    def l2_normalize(self, feat):
        """L2标准化：避免尺度差异影响相似度"""
        return torch.nn.functional.normalize(feat, p=2, dim=1)

    def compute_cosine_sim(self, anchor, positive, negative):
        """计算锚点与正/负样本的余弦相似度"""
        sim_pos = torch.sum(anchor * positive, dim=1, keepdim=True)  # [B, 1]
        sim_neg = torch.sum(anchor * negative, dim=1, keepdim=True)  # [B, 1]
        return torch.cat([sim_pos, sim_neg], dim=1) / self.tau  # [B, 2]

    def forward(self, fused_feats, rgb_deno_feats, ir_deno_feats):
        """
        Args:
            fused_feats: 融合特征列表（多尺度）
            rgb_deno_feats: 模态1去噪特征列表（如RGB）
            ir_deno_feats: 模态2去噪特征列表（如红外）
        Returns:
            损失字典（含融合-RGB/红外的正样本约束损失）
        """
        # 校验尺度数量一致
        assert len(fused_feats) == len(rgb_deno_feats) == len(ir_deno_feats), \
            "融合特征与去噪特征的尺度数量不匹配"

        # 特征池化+标准化
        fused_pooled = self.global_feature_pool(fused_feats)
        rgb_pooled = self.global_feature_pool(rgb_deno_feats)
        ir_pooled = self.global_feature_pool(ir_deno_feats)

        # 逐尺度计算损失
        total_fused_rgb = 0.0
        total_fused_ir = 0.0
        num_scales = len(fused_pooled)

        for i in range(num_scales):
            anchor = self.l2_normalize(fused_pooled[i])
            pos_rgb = self.l2_normalize(rgb_pooled[i])
            pos_ir = self.l2_normalize(ir_pooled[i])
            neg_ir = self.l2_normalize(ir_pooled[i])  # 负样本：红外去噪特征
            neg_rgb = self.l2_normalize(rgb_pooled[i])  # 负样本：RGB去噪特征

            # 融合-RGB去噪 损失
            sim_rgb = self.compute_cosine_sim(anchor, pos_rgb, neg_ir)
            total_fused_rgb += torch.nn.functional.cross_entropy(
                sim_rgb, torch.zeros_like(sim_rgb[:, 0], dtype=torch.long)
            )

            # 融合-红外去噪 损失
            sim_ir = self.compute_cosine_sim(anchor, pos_ir, neg_rgb)
            total_fused_ir += torch.nn.functional.cross_entropy(
                sim_ir, torch.zeros_like(sim_ir[:, 0], dtype=torch.long)
            )

        # 平均多尺度损失
        return {
            "total": (total_fused_rgb + total_fused_ir) / (2 * num_scales)
        }


@MODELS.register_module()
class NegativeInfoNCELoss(nn.Module):
    """
    负样本损失计算模块（基于指定公式）
    功能：对两个负样本特征先求平均，再通过公式 \( L_{\text{neg}} = -\log\left[1 - \exp\left(\frac{S(fused, neg)}{\tau}\right)\right] \) 计算损失
    其中 \( S(\cdot, \cdot) \) 表示余弦相似度
    """
    def __init__(self,
                 temperature: float = 0.1,  # 温度系数τ
                 reduction: str = 'mean'):  # 损失聚合方式
        super().__init__()
        self.tau = temperature
        self.reduction = reduction
        assert self.tau > 0, f"温度系数τ必须为正数，当前值为{self.tau}"

    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算两个特征向量的余弦相似度"""
        x_norm = F.normalize(x, p=2, dim=1)  # L2标准化 [B, C]
        y_norm = F.normalize(y, p=2, dim=1)  # L2标准化 [B, C]
        return torch.sum(x_norm * y_norm, dim=1)  # 点积即余弦相似度 [B]

    def _feature_pooling(self, feat: torch.Tensor) -> torch.Tensor:
        """将特征图[B, C, H, W]池化为向量[B, C]"""
        return F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # 全局平均池化+展平

    def forward(self,
                fused_feats: torch.Tensor,  # 融合特征 [B, C, H, W]
                neg_feat1: torch.Tensor,   # 第一个负样本特征 [B, C, H, W]
                neg_feat2: torch.Tensor) -> torch.Tensor:  # 第二个负样本特征 [B, C, H, W]
        """
        前向计算：两个负样本先平均，再代入公式计算损失
        Args:
            fused_feats: 融合特征图
            neg_feat1: 第一个负样本特征图（如RGB非目标区域）
            neg_feat2: 第二个负样本特征图（如红外非目标区域）
        Returns:
            计算得到的负样本损失
        """
        total_loss = 0.0
        num_scales = len(fused_feats)

        for i in range(num_scales):
            # 1. 特征池化（特征图→向量）
            fused_vec = self._feature_pooling(fused_feats[i])  # [B, C]
            neg_vec1 = self._feature_pooling(neg_feat1[i])  # [B, C]
            neg_vec2 = self._feature_pooling(neg_feat2[i])  # [B, C]

            # 2. 两个负样本特征取平均
            neg_vec_avg = (neg_vec1 + neg_vec2) / 2.0  # [B, C]

            # 3. 计算融合特征与平均负样本特征的余弦相似度 S(fused, neg_avg)
            similarity = self._cosine_similarity(fused_vec, neg_vec_avg)  # [B]

            # 4. 代入公式计算损失：L_neg = -log[1 - exp(S/τ)]
            exp_term = torch.exp(similarity / self.tau)  # exp(S/τ)
            # 防止数值不稳定（当exp_term接近1时，1 - exp_term可能下溢）
            loss = -torch.log(torch.clamp(1 - exp_term, min=1e-12))  # [B]

            # 5. 按指定方式聚合损失
            if self.reduction == 'mean':
                total_loss += loss.mean()
            elif self.reduction == 'sum':
                total_loss += loss.sum()

        # 如果是mean，则求平均；如果是sum，则已经是总和
        if self.reduction == 'mean':
            total_loss /= num_scales

        return total_loss


@MODELS.register_module()
class my_Detector_newloss(BaseDetector):
    # 融合检测（带去噪特征和InfoNCE损失）
    """
    Args:
        backbone (:obj:`ConfigDict` or dict): 骨干网络配置
        neck (:obj:`ConfigDict` or dict): 颈部网络配置
        bbox_head (:obj:`ConfigDict` or dict): 检测头配置
        fusion_block (:obj:`ConfigDict` or dict): 融合模块配置，默认FCM_DualInput
        info_nce_loss1 (:obj:`ConfigDict` or dict): InfoNCE损失配置
        info_nce_loss2 ：负样本
        train_cfg (:obj:`ConfigDict` or dict, optional): 训练配置
        test_cfg (:obj:`ConfigDict` or dict, optional): 测试配置
        data_preprocessor (:obj:`ConfigDict` or dict, optional): 数据预处理配置
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or list[dict], optional): 初始化配置
        use_syncbn (bool): 是否使用SyncBatchNorm，默认True
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 fusion_block: ConfigType = dict(type='BaseFusion2'),
                 info_nce_loss1: ConfigType = dict(
                     type='PositiveGuidedInfoNCELoss',
                     temperature=0.3
                 ),
                 info_nce_loss2:ConfigType = dict(
                     type='NegativeInfoNCELoss',
                     temperature=0.3
                 ),
                 denoiser_config: ConfigType = dict(
                     model_size='small',
                     drop_path_rate=0.1,
                     negative_strategy='feature_corrupt'
                 ),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # 1. 双模态骨干网络（返回两个值(原始特征, 去噪特征)）YOLOv8CSPDarknetPzconv
        self.backbone1 = MODELS.build(backbone)  # 模态1骨干（如RGB）
        self.backbone2 = MODELS.build(backbone)  # 模态2骨干（如红外）

        # 2. 特征融合模块（保持原有FCM_DualInput）
        #self.fusion_block = MODELS.build(fusion_block)
        if isinstance(fusion_block, dict) and fusion_block.get('type') == 'FCM_DualInput':
            from mmyolo.models.backbones.custom.my_fusion_block import FCM_DualInput
            self.fusion_block = FCM_DualInput(**{k: v for k, v in fusion_block.items() if k != 'type'})
        else:
            self.fusion_block = MODELS.build(fusion_block)
        # 3. 颈部网络（保持原有）
        if neck is not None:
            self.neck = MODELS.build(neck)

        # 4. 检测头
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        # 5. 正样本约束的InfoNCE损失模块
        self.info_nce_loss1 = MODELS.build(info_nce_loss1)

        # 6. 负样本
        self.info_nce_loss2 = MODELS.build(info_nce_loss2)

        self.denoiser_config = denoiser_config
        self.denoiser1 = None
        self.denoiser2 = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_denoisers(self, sample_feat: torch.Tensor):
        """延迟初始化去噪器（动态适应版本）"""
        if self.denoiser1 is None:
            print("初始化动态适应去噪器...")

            device = sample_feat[0].device if isinstance(sample_feat, list) else sample_feat.device

            # 创建动态适应的去噪器，不需要预先知道确切的通道数
            base_channels = 256  # 使用一个基础通道数，实际运行时动态适应
            hidden_dim = self.denoiser_config.get('hidden_dim', 128)

            self.denoiser1 = LightweightDenoiseConvNeXt(
                in_channels=base_channels,
                hidden_dim=hidden_dim,
                drop_path_rate=self.denoiser_config.get('drop_path_rate', 0.1)
            ).to(device)

            self.denoiser2 = LightweightDenoiseConvNeXt(
                in_channels=base_channels,
                hidden_dim=hidden_dim,
                drop_path_rate=self.denoiser_config.get('drop_path_rate', 0.1)
            ).to(device)

            print("动态适应去噪器初始化完成")

    def _init_denoisers_with_custom_channels(self, in_channels: int, device: torch.device):
        """使用自定义通道数初始化去噪器"""
        hidden_dim = max(64, in_channels // 2)
        drop_path_rate = self.denoiser_config.get('drop_path_rate', 0.1)

        self.denoiser1 = LightweightDenoiseConvNeXt(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            drop_path_rate=drop_path_rate
        ).to(device)

        self.denoiser2 = LightweightDenoiseConvNeXt(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            drop_path_rate=drop_path_rate
        ).to(device)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        # 加载预训练权重时，适配双骨干网络（复制单骨干权重到两个骨干）
        copy_ori = False
        ori_backbone_params = []
        ori_backbone_key = []

        for k, v in state_dict.items():
            if (k.startswith("backbone") and "backbone1" not in k and "backbone2" not in k):
                # 预训练权重来自单骨干模型，复制到两个骨干
                ori_backbone_params += [v]
                ori_backbone_key += [k]
                copy_ori = True

        if copy_ori:
            for k, v in zip(ori_backbone_key, ori_backbone_params):
                state_dict[k.replace("backbone", "backbone1")] = v
                state_dict[k.replace("backbone", "backbone2")] = copy.deepcopy(v)
                del state_dict[k]
            strict = False  # 允许权重键不严格匹配

        # 适配检测头权重（兼容两阶段模型权重）
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [k for k in state_dict.keys() if k.startswith(bbox_head_prefix)]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [k for k in state_dict.keys() if k.startswith(rpn_head_prefix)]

        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward(self,
                inputs: torch.Tensor,
                inputs2: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):
        if isinstance(inputs, list):
            inputs = inputs[0]
        if isinstance(inputs2, list):
            inputs2 = inputs2[0]
        if isinstance(data_samples, list) and len(data_samples) == 1:
            data_samples = data_samples[0]
        """统一前向接口，支持loss/predict/tensor三种模式"""
        if mode == 'loss':
            return self.loss(inputs, inputs2, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, inputs2, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, inputs2, data_samples)
        else:
            raise RuntimeError(f'无效模式 "{mode}"，仅支持loss、predict、tensor')

    def extract_feat(self, batch_inputs: Tensor, batch_inputs2: Tensor) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """提取特征（扩展为返回原始特征、去噪特征、融合特征）
        Returns:
            - fused_feats: 融合特征列表（送入neck和检测头）
            - denoise_feats1: 模态1去噪特征列表（用于InfoNCE损失）
            - denoise_feats2: 模态2去噪特征列表（用于InfoNCE损失）
            - neg1: 模态1负样本特征
            - neg2: 模态2负样本特征
        """
        if batch_inputs.dim() == 3:
            batch_inputs = batch_inputs.unsqueeze(0)  # 添加batch维度
        if batch_inputs2.dim() == 3:
            batch_inputs2 = batch_inputs2.unsqueeze(0)
        # 1. 双骨干提取特征：返回(原始特征, 去噪特征)
        orig_feats1 = list(self.backbone1(batch_inputs))  # 模态1：原始特征
        orig_feats2 = list(self.backbone2(batch_inputs2))  # 模态2：原始特征

        # 确保特征格式一致
        if not isinstance(orig_feats1, list):
            orig_feats1 = [orig_feats1]
        if not isinstance(orig_feats2, list):
            orig_feats2 = [orig_feats2]

        orig_feats12 = [feat.clone() for feat in orig_feats1]  # 复制模态1原始特征
        orig_feats22 = [feat.clone() for feat in orig_feats2]  # 复制模态2原始特征

        # 2. 特征融合（使用原始特征）
        fused_feats = self.fusion_block(orig_feats1, orig_feats2)

        # 3. 颈部网络处理（如果有）
        if self.with_neck:
            fused_feats = self.neck(fused_feats)

        # 4. 延迟初始化去噪器
        if self.denoiser1 is None:
           self._init_denoisers(orig_feats1)

        # 5. 去噪处理（即正样本）
        try:
            denoise_feats1 = self.denoiser1(orig_feats1)
            denoise_feats2 = self.denoiser2(orig_feats2)
        except Exception as e:
            print(f"去噪处理出错: {e}")
            # 应急处理：直接返回原始特征
            denoise_feats1 = orig_feats1
            denoise_feats2 = orig_feats2

        # 6. 得到负样本
        negative_strategy = self.denoiser_config.get('negative_strategy', 'spatial_shuffle')
        try:
            neg1 = get_spatial_negative_by_bbox(orig_feats12, strategy=negative_strategy)
            neg2 = get_spatial_negative_by_bbox(orig_feats22, strategy=negative_strategy)
        except Exception as e:
            print(f"负样本生成出错: {e}")
            # 应急处理：使用简单的噪声
            neg1 = [feat + torch.randn_like(feat) * 0.1 for feat in orig_feats12]
            neg2 = [feat + torch.randn_like(feat) * 0.1 for feat in orig_feats22]

        return fused_feats, denoise_feats1, denoise_feats2, neg1, neg2

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_inputs2: Tensor,
                 batch_data_samples: OptSampleList = None):
        """仅返回网络输出张量（用于推理中间过程）"""
        fused_feats, _, _, _, _ = self.extract_feat(batch_inputs, batch_inputs2)
        results_list = self.bbox_head.forward(fused_feats)
        return results_list

    def loss(self, batch_inputs: Tensor, batch_inputs2: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """计算总损失（原有检测损失 + 新增InfoNCE损失）"""
        # 1. 提取特征：融合特征 + 双模态去噪特征
        fused_feats, denoise_feats1, denoise_feats2, neg_feat1, neg_feat2 = self.extract_feat(batch_inputs,
                                                                                              batch_inputs2)

        # 2. 计算原有检测损失（分类+回归）
        det_losses = self.bbox_head.loss(fused_feats, batch_data_samples)

        # 3. 计算正样本约束的InfoNCE损失
        try:
            info_nce_losses1 = self.info_nce_loss1(
                fused_feats=fused_feats,
                rgb_deno_feats=denoise_feats1,
                ir_deno_feats=denoise_feats2
            )
        except Exception as e:
            print(f"正样本InfoNCE损失计算出错: {e}")
            info_nce_losses1 = {'info_nce': torch.tensor(0.0, device=batch_inputs.device)}

        # 4. 计算负样本约束的infonce损失
        try:
            info_nce_losses2 = self.info_nce_loss2(
                fused_feats=fused_feats,
                neg_feat1=neg_feat1,
                neg_feat2=neg_feat2
            )
        except Exception as e:
            print(f"负样本InfoNCE损失计算出错: {e}")
            info_nce_losses2 = {'info_nce': torch.tensor(0.0, device=batch_inputs.device)}

        # 5. 合并损失（InfoNCE损失作为辅助损失，权重可通过配置调整）
        total_losses = {**det_losses}

        # 添加INFONCE损失，默认权重为正样本0.2，负样本0.5
        pos_weight = self.denoiser_config.get('pos_loss_weight', 0.3)
        neg_weight = self.denoiser_config.get('neg_loss_weight', 0.3)

        if isinstance(info_nce_losses1, dict):
            for k, v in info_nce_losses1.items():
                total_losses[f"pos_info_nce_{k}"] = pos_weight * v
        else:
            total_losses["pos_info_nce"] = pos_weight * info_nce_losses1

        if isinstance(info_nce_losses2, dict):
            for k, v in info_nce_losses2.items():
                total_losses[f"neg_info_nce_{k}"] = neg_weight * v
        else:
            total_losses["neg_info_nce"] = neg_weight * info_nce_losses2

        return total_losses

    def predict(self,
                batch_inputs: Tensor, batch_inputs2: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """预测模式（保持原有逻辑，不涉及损失计算）"""
        fused_feats, _, _, _, _ = self.extract_feat(batch_inputs, batch_inputs2)  # 修复：接收所有5个返回值
        if not isinstance(batch_data_samples, list):
            batch_data_samples = [batch_data_samples]

        results_list = self.bbox_head.predict(
            fused_feats, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples