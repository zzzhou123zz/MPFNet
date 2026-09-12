_base_ = '/data/zfy/mmyolo/test_dataset/LLVIP/llvip.py'
# work_dir = '/lab/hj/hujie004/zfy/mmyolo/work_dirs/yolov8m_baseline'
custom_imports = dict(imports=[
            'projects.CO_DETR.codetr.codetr_dual_stream',
            'mmdet.datasets.transforms.my_loading',
            'mmdet.datasets.transforms.my_wrapper',
            'mmdet.datasets.transforms.my_formatting',
            'mmdet.models.data_preprocessors.my_data_preprocessor',
            'mmdet.datasets.transforms.my_transforms_possion',
            'mmdet.datasets.my_coco',
            'projects.CO_DETR.codetr',
            'projects.CO_DETR.codetr.codetr_dual_stream_reg'
            ], allow_failed_imports=False)

image_size = (640, 640)
num_classes = 3
classes = ('person')

model = dict(
    type='YOLODualFusionDetector', #v8
    data_preprocessor=dict(
        type='DualInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=_base_.last_stage_out_channels,
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN',
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
