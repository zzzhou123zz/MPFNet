
"""
测试脚本：探查双模态检测模型的输出结构、数据格式等
运行此脚本后告诉我打印输出的具体内容
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmyolo.registry import MODELS

# ==================== 导入自定义模块 ====================
import sys
sys.path.insert(0, '/data/zfy/mmyolo')

# 先导入mmyolo确保所有模块被注册
import mmyolo
import mmyolo.models

# 只导入检测器类，预处理器已经通过mmyolo.models自动注册了
from mmyolo.models.backbones.custom.mkp_loss_backbone import my_Detector_newloss
from mmyolo.models.data_preprocessors.customs.double_data_preprocessor import DualInputDetDataPreprocessor
# ==================== 配置参数 ====================
CONFIG_PATH = "/data/zfy/mmyolo/test_dataset/FLIR/flir_mkp_fs_loss.py"  # 修改为你的配置文件路径
CHECKPOINT_PATH = "/data/zfy/mmyolo/work_dirs/flir_mkp_fs_loss/0.3-0.3-best_coco_bbox_mAP_50_epoch_97.pth"  # 修改为你的权重文件路径
RGB_IMG_PATH = "/data/zfy/dataset/FLIR-align-3class/images/visible/train/FLIR_00018.jpg"  # 修改为测试图像路径
IR_IMG_PATH = "/data/zfy/dataset/FLIR-align-3class/images/infrared/train/FLIR_00018.jpg"  # 修改为测试图像路径

# ==================== 加载模型 ====================
print("=" * 50)
print("加载模型中...")
print("=" * 50)

# 加载配置
cfg = Config.fromfile(CONFIG_PATH)
print(f"配置加载成功: {CONFIG_PATH}")
print(f"配置内容: {cfg.pretty_text[:500]}...")  # 打印配置前500个字符

# 构建模型
model = MODELS.build(cfg.model)
print(f"\n模型构建成功")
print(f"模型类型: {type(model)}")
print(f"模型结构:\n{model}")

# 加载权重
load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
print(f"\n权重加载成功: {CHECKPOINT_PATH}")

# 移至GPU（如果可用）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
print(f"模型已移至设备: {device}")

# ==================== 读取并预处理图像 ====================
print("\n" + "=" * 50)
print("读取和预处理图像中...")
print("=" * 50)

# 读取图像
rgb_img = cv2.imread(RGB_IMG_PATH)
ir_img = cv2.imread(IR_IMG_PATH)

if rgb_img is None or ir_img is None:
    raise FileNotFoundError(f"图像读取失败: RGB={RGB_IMG_PATH}, IR={IR_IMG_PATH}")

print(f"RGB图像原始尺寸: {rgb_img.shape}")
print(f"IR图像原始尺寸: {ir_img.shape}")

# 记录原始尺寸
original_h, original_w = rgb_img.shape[:2]

# 调整为640x640
rgb_resized = cv2.resize(rgb_img, (640, 640))
ir_resized = cv2.resize(ir_img, (640, 640))

print(f"调整后尺寸: {rgb_resized.shape}")

# 转为tensor并进行标准化处理
rgb_tensor = torch.from_numpy(rgb_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
ir_tensor = torch.from_numpy(ir_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

print(f"RGB Tensor形状: {rgb_tensor.shape}, 值范围: [{rgb_tensor.min():.4f}, {rgb_tensor.max():.4f}]")
print(f"IR Tensor形状: {ir_tensor.shape}, 值范围: [{ir_tensor.min():.4f}, {ir_tensor.max():.4f}]")

# 移至设备
rgb_tensor = rgb_tensor.to(device)
ir_tensor = ir_tensor.to(device)

# ==================== 前向推理 ====================
print("\n" + "=" * 50)
print("执行前向推理...")
print("=" * 50)

with torch.no_grad():
    # 尝试直接调用forward (推理模式，data_samples=None)
    try:
        result = model(rgb_tensor, ir_tensor, data_samples=None, mode='predict')
        print(f"\n推理成功！")
        print(f"返回类型: {type(result)}")
        print(f"返回长度: {len(result) if hasattr(result, '__len__') else 'N/A'}")

        # 探查第一个结果
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            print(f"\n第一个结果类型: {type(first_result)}")
            print(f"第一个结果属性: {dir(first_result)}")

            # 尝试访问常见属性
            if hasattr(first_result, 'pred_instances'):
                pred_inst = first_result.pred_instances
                print(f"\npred_instances类型: {type(pred_inst)}")
                print(f"pred_instances属性: {dir(pred_inst)}")

                if hasattr(pred_inst, 'bboxes'):
                    bboxes = pred_inst.bboxes
                    print(f"\nBboxes形状: {bboxes.shape}")
                    print(f"Bboxes数据类型: {type(bboxes)}")
                    print(f"第一个bbox (如果存在): {bboxes[0] if len(bboxes) > 0 else 'No bboxes'}")
                    print(f"Bbox值范围: [{bboxes.min():.4f}, {bboxes.max():.4f}]")

                if hasattr(pred_inst, 'scores'):
                    scores = pred_inst.scores
                    print(f"\nScores形状: {scores.shape}")
                    print(f"Scores: {scores}")

                if hasattr(pred_inst, 'labels'):
                    labels = pred_inst.labels
                    print(f"\nLabels形状: {labels.shape}")
                    print(f"Labels: {labels}")

                # 打印所有属性和值
                print(f"\n所有属性值:")
                for attr in dir(pred_inst):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(pred_inst, attr)
                            if not callable(val):
                                print(
                                    f"  {attr}: {type(val).__name__} = {val if not isinstance(val, torch.Tensor) else val.shape}")
                        except:
                            pass
        else:
            print(f"完整结果: {result}")

    except Exception as e:
        print(f"推理出错: {e}")
        import traceback

        traceback.print_exc()

# ==================== 检查data_preprocessor ====================
print("\n" + "=" * 50)
print("检查data_preprocessor配置...")
print("=" * 50)

if hasattr(model, 'data_preprocessor'):
    print(f"data_preprocessor类型: {type(model.data_preprocessor)}")
    print(f"data_preprocessor: {model.data_preprocessor}")
else:
    print("模型没有data_preprocessor属性")

# ==================== 检查模型各组件 ====================
print("\n" + "=" * 50)
print("模型组件检查...")
print("=" * 50)

print(f"bbox_head类型: {type(model.bbox_head) if hasattr(model, 'bbox_head') else 'N/A'}")
print(f"neck存在: {model.with_neck if hasattr(model, 'with_neck') else 'N/A'}")

print("\n" + "=" * 50)
print("测试完成！请告诉我以上输出内容")
print("=" * 50)