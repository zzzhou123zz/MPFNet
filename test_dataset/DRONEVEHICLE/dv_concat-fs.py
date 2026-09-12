_base_ = '/data/zfy/mmyolo/test_dataset/DRONEVEHICLE/dronev_v8.py'
import torch
custom_imports = dict(imports=[
    'projects.CO_DETR.codetr.codetr_dual_stream',
    'mmdet.datasets.transforms.my_loading',
    'mmdet.datasets.transforms.my_wrapper',
    'mmdet.datasets.transforms.my_formatting',
    'mmdet.models.data_preprocessors.my_data_preprocessor',
    'mmdet.datasets.transforms.my_transforms_possion',
    'mmdet.datasets.my_coco',
    'projects.CO_DETR.codetr',
    'projects.CO_DETR.codetr.codetr_dual_stream_reg',
], allow_failed_imports=False)

image_size = (640, 640)
num_classes = 5
classes = ()
# classes = ('Car', 'Truck', 'People', 'Bus', 'Lamp', 'Motorcycle')
# load_from = "/data/zfy/mmyolo/work_dirs/yolov8m_dual_stream_baseline/best_coco_bbox_mAP_50_epoch_72.pth"
# 声明一个骨干，在YOLODualFusionDetector内部创建两个相同骨干网络
model = dict(
    type='my_Detector_newloss',
    data_preprocessor=dict(
        type='DualInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknetPzconv',
        arch='P5',  # 输出三个尺度特征
        last_stage_out_channels=_base_.last_stage_out_channels,
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,  # 继承自_base_，控制通道数
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        use_pzconv_in_stage=True,
        use_pzconv_in_csp=False,
    ),
    # fusion_block=dict(
    #     type='BaseFusion2',
    #     # 一般情况下yolov8的通道数如下：
    #     # YOLOv8n: [128, 256, 512]
    #     # YOLOv8s: [256, 512, 1024]
    #     # YOLOv8m: [384, 768, 1536]
    #     # YOLOv8l: [512, 1024, 2048]
    #     #此处使用v8s，但widen_factor=0.75
    #     # dim_out=[192, 384, 768],  # 输出
    #     # align_method='adaptive_pool' #gap版
    # ),
    fusion_block=dict(
        type='BaseFusion2',
        fusion_strategy='concat', #concat
        use_conv=True,
    ),
    neck=dict(
        type='YOLOv8PAFPN',  # 路径聚合
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        in_channels=[256, 512, _base_.last_stage_out_channels],
        out_channels=[256, 512, _base_.last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=_base_.num_classes,
            in_channels=[256, 512, _base_.last_stage_out_channels],
            widen_factor=_base_.widen_factor,
            reg_max=16,
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=_base_.strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=_base_.strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=_base_.loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=_base_.loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=_base_.loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=_base_.num_classes,
            use_ciou=True,
            topk=_base_.tal_topk,
            alpha=_base_.tal_alpha,
            beta=_base_.tal_beta,
            eps=1e-9)),
    test_cfg=_base_.model_test_cfg)