# /data/zfy/mmyolo/mmyolo/engine/hooks/safe_ema_hook.py

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from mmengine.hooks import EMAHook
from mmengine.registry import HOOKS
from mmengine.logging import print_log
from copy import deepcopy


@HOOKS.register_module()
class SafeEMAHook(EMAHook):
    """
    安全的EMA Hook，能处理模型结构变化的情况
    """

    def __init__(self,
                 ema_type: str = 'ExpMomentumEMA',
                 momentum: float = 0.0001,
                 update_buffers: bool = True,
                 priority: int = 49,
                 strict_load: bool = False,
                 **kwargs):
        self.strict_load = strict_load
        self.ema_initialized = False
        super().__init__(
            ema_type=ema_type,
            momentum=momentum,
            update_buffers=update_buffers,
            priority=priority,
            **kwargs
        )

    def before_run(self, runner) -> None:
        """在训练开始前初始化EMA模型"""
        try:
            super().before_run(runner)
            self.ema_initialized = True
            print_log("Standard EMA initialization successful", level='INFO')
        except Exception as e:
            print_log(f"Standard EMA initialization failed: {e}", level='WARNING')
            try:
                self._safe_ema_init(runner)
                self.ema_initialized = True
                print_log("Safe EMA initialization successful", level='INFO')
            except Exception as e2:
                print_log(f"Safe EMA initialization also failed: {e2}", level='ERROR')
                self.ema_initialized = False

    def _safe_ema_init(self, runner):
        """安全的EMA初始化"""
        print_log("Using safe EMA initialization...", level='INFO')

        # 尝试不同的方法创建EMA模型
        try:
            # 方法1：尝试使用父类的方法
            if hasattr(self, '_build_ema_model'):
                self.ema_model = self._build_ema_model()
            # 方法2：尝试使用build_ema_model
            elif hasattr(self, 'build_ema_model'):
                self.ema_model = self.build_ema_model()
            # 方法3：直接深拷贝模型
            else:
                self.ema_model = deepcopy(runner.model)
                print_log("Using deepcopy to create EMA model", level='INFO')
        except Exception as e:
            print_log(f"Failed to create EMA model: {e}", level='ERROR')
            # 方法4：最后的尝试 - 直接深拷贝
            try:
                self.ema_model = deepcopy(runner.model.cpu()).cuda() if torch.cuda.is_available() else deepcopy(
                    runner.model)
                print_log("Using fallback deepcopy method", level='INFO')
            except Exception as e2:
                print_log(f"All EMA initialization methods failed: {e2}", level='ERROR')
                self.ema_model = None
                return

        if self.ema_model is not None:
            # 安全地复制参数
            self._safe_copy_parameters(runner.model, self.ema_model)

    def _safe_copy_parameters(self, source_model, target_model):
        """安全地复制参数"""
        source_params = dict(source_model.named_parameters())
        target_params = dict(target_model.named_parameters())

        copied_count = 0
        skipped_count = 0

        for name, target_param in target_params.items():
            if name in source_params:
                source_param = source_params[name]
                if source_param.shape == target_param.shape:
                    target_param.data.copy_(source_param.data)
                    copied_count += 1
                else:
                    print_log(f"EMA: Shape mismatch for {name}: "
                              f"source {source_param.shape} vs target {target_param.shape}",
                              level='WARNING')
                    skipped_count += 1
            else:
                skipped_count += 1

        print_log(f"EMA parameter copy: {copied_count} copied, {skipped_count} skipped", level='INFO')

        # 处理buffers
        if self.update_buffers:
            source_buffers = dict(source_model.named_buffers())
            target_buffers = dict(target_model.named_buffers())

            for name, target_buffer in target_buffers.items():
                if name in source_buffers:
                    source_buffer = source_buffers[name]
                    if source_buffer.shape == target_buffer.shape:
                        target_buffer.data.copy_(source_buffer.data)

    def _safe_swap_parameters(self):
        """安全地交换参数"""
        if not self.ema_initialized or self.ema_model is None:
            print_log("EMA not properly initialized, skipping parameter swap", level='WARNING')
            return

        model_params = list(self.runner.model.parameters())
        ema_params = list(self.ema_model.parameters())

        swapped = 0
        skipped = 0

        for model_param, ema_param in zip(model_params, ema_params):
            if model_param.shape == ema_param.shape:
                # 形状匹配，可以安全交换
                temp_data = model_param.data.clone()
                model_param.data.copy_(ema_param.data)
                ema_param.data.copy_(temp_data)
                swapped += 1
            else:
                # 形状不匹配，跳过交换
                skipped += 1

        if skipped > 0:
            print_log(f"EMA swap: {swapped} swapped, {skipped} skipped due to shape mismatch",
                      level='DEBUG')

    def _safe_swap_buffers(self):
        """安全地交换buffers"""
        if not self.update_buffers or not self.ema_initialized or self.ema_model is None:
            return

        model_buffers = list(self.runner.model.buffers())
        ema_buffers = list(self.ema_model.buffers())

        for model_buffer, ema_buffer in zip(model_buffers, ema_buffers):
            if model_buffer.shape == ema_buffer.shape:
                temp_data = model_buffer.data.clone()
                model_buffer.data.copy_(ema_buffer.data)
                ema_buffer.data.copy_(temp_data)

    def _swap_ema_parameters(self):
        """覆盖原始的参数交换方法"""
        if not self.ema_initialized:
            print_log("EMA not initialized, skipping parameter swap", level='WARNING')
            return

        try:
            # 尝试使用原始方法
            super()._swap_ema_parameters()
        except RuntimeError as e:
            if "must match the size" in str(e):
                # 如果出现尺寸不匹配错误，使用安全交换
                print_log(f"EMA: Using safe parameter swap due to: {e}", level='WARNING')
                self._safe_swap_parameters()
                self._safe_swap_buffers()
            else:
                print_log(f"EMA: Unexpected error during parameter swap: {e}", level='ERROR')
        except Exception as e:
            print_log(f"EMA: General error during parameter swap: {e}", level='ERROR')
            # 尝试安全交换作为后备
            self._safe_swap_parameters()
            self._safe_swap_buffers()

    def before_val_epoch(self, runner) -> None:
        """验证前交换参数"""
        if not self.ema_initialized:
            return

        try:
            self._swap_ema_parameters()
        except Exception as e:
            print_log(f"EMA parameter swap failed before validation: {e}", level='ERROR')

    def after_val_epoch(self, runner, metrics: Optional[Dict] = None) -> None:
        """验证后恢复参数"""
        if not self.ema_initialized:
            return

        try:
            self._swap_ema_parameters()
        except Exception as e:
            print_log(f"EMA parameter restore failed after validation: {e}", level='ERROR')

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        """训练迭代后更新EMA"""
        if not self.ema_initialized:
            return

        try:
            super().after_train_iter(runner, batch_idx, data_batch, outputs)
        except Exception as e:
            print_log(f"EMA update failed: {e}", level='WARNING')