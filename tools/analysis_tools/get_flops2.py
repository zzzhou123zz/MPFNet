import torch
import torch.nn as nn
import argparse
import tempfile
import time
import numpy as np
from pathlib import Path
from mmdet.registry import MODELS
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmyolo.utils import switch_to_deploy

# 尝试导入可选的库
try:
    from thop import profile, clever_format

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = True
    print("Warning: thop not available. Install with: pip install thop")

try:
    from fvcore.nn import FlopCountMode, flop_count

    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not available. Install with: pip install fvcore")


def calculate_flops_with_thop(model, input_tensors):
    """使用 thop 库计算 FLOPS"""
    if not THOP_AVAILABLE:
        print("THOP not available")
        return None

    try:
        # 设置模型为评估模式
        model.eval()

        # 调试：打印输入张量的形状
        print(f"THOP输入张量形状: {[tensor.shape for tensor in input_tensors]}")

        # 使用no_grad避免梯度计算
        with torch.no_grad():
            flops, params = profile(model, inputs=input_tensors, verbose=False)

        flops_str, params_str = clever_format([flops, params], "%.3f")
        return {
            'flops': flops,
            'params': params,
            'flops_str': flops_str,
            'params_str': params_str,
            'method': 'thop'
        }
    except Exception as e:
        print(f"THOP calculation failed: {e}")
        # 尝试简化版本
        try:
            model.eval()
            with torch.no_grad():
                # 尝试单输入
                if len(input_tensors) > 1:
                    # 如果多输入失败，尝试只使用第一个输入
                    flops, params = profile(model, inputs=(input_tensors[0],), verbose=False)
                else:
                    flops, params = profile(model, inputs=input_tensors, verbose=False)

            flops_str, params_str = clever_format([flops, params], "%.3f")
            return {
                'flops': flops,
                'params': params,
                'flops_str': flops_str,
                'params_str': params_str,
                'method': 'thop_simplified'
            }
        except Exception as e2:
            print(f"THOP simplified also failed: {e2}")
            return None


def calculate_flops_with_fvcore(model, input_tensors):
    """使用 fvcore 库计算 FLOPS"""
    if not FVCORE_AVAILABLE:
        print("FVCore not available")
        return None

    try:
        model.eval()

        # 调试：打印输入张量的形状
        print(f"FVCore输入张量形状: {[tensor.shape for tensor in input_tensors]}")

        with torch.no_grad(), FlopCountMode(model) as flop_count_mode:
            # 执行前向传播
            _ = model(*input_tensors)

        flops = flop_count_mode.get_total_flops()
        params = sum(p.numel() for p in model.parameters())

        return {
            'flops': flops,
            'params': params,
            'flops_str': f"{flops / 1e9:.3f}G",
            'params_str': f"{params / 1e6:.3f}M",
            'method': 'fvcore'
        }
    except Exception as e:
        print(f"FVCore calculation failed: {e}")
        return None


def calculate_flops_manual_conv(model, input_tensors):
    """手动计算卷积层FLOPS的改进版本"""
    total_flops = 0
    total_params = 0
    layer_info = []

    def hook_fn(module, input, output):
        nonlocal total_flops, total_params

        module_name = module.__class__.__name__
        flops = 0
        params = 0

        # 跳过某些可能引起问题的模块
        skip_modules = ['Detect', 'YOLOv5Detect', 'YOLOXHead', 'PPYOLOEHead']
        if any(skip in module_name for skip in skip_modules):
            return

        if isinstance(module, nn.Conv2d):
            # 卷积层FLOPS计算
            if len(input) > 0 and input[0] is not None:
                try:
                    batch_size, in_channels, input_height, input_width = input[0].shape
                    output_height, output_width = output.shape[2], output.shape[3]

                    kernel_flops = module.kernel_size[0] * module.kernel_size[1] * in_channels
                    output_elements = batch_size * output_height * output_width * module.out_channels
                    flops = kernel_flops * output_elements

                    # 如果有偏置，额外加上偏置的操作
                    if module.bias is not None:
                        flops += output_elements
                except Exception as e:
                    print(f"Error calculating Conv2d FLOPs: {e}")

        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm FLOPS
            if len(input) > 0 and input[0] is not None:
                try:
                    flops = input[0].numel() * 2  # 标准化和缩放
                except Exception as e:
                    print(f"Error calculating BatchNorm FLOPs: {e}")

        elif isinstance(module, nn.ReLU) or isinstance(module, nn.SiLU):
            # 激活函数FLOPS
            if hasattr(output, 'numel'):
                try:
                    flops = output.numel()
                except Exception as e:
                    print(f"Error calculating Activation FLOPs: {e}")

        elif isinstance(module, nn.Linear):
            # 全连接层FLOPS计算
            try:
                batch_size = input[0].shape[0] if len(input) > 0 and input[0] is not None else 1
                flops = batch_size * module.in_features * module.out_features
                if module.bias is not None:
                    flops += batch_size * module.out_features
            except Exception as e:
                print(f"Error calculating Linear FLOPs: {e}")

        # 参数计算
        try:
            if hasattr(module, 'weight') and module.weight is not None:
                params += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                params += module.bias.numel()
        except Exception as e:
            print(f"Error calculating parameters: {e}")

        if flops > 0:
            layer_info.append({
                'name': module_name,
                'flops': flops,
                'params': params,
                'flops_str': f"{flops / 1e6:.2f}M" if flops > 1e6 else f"{flops / 1e3:.2f}K"
            })

        total_flops += flops
        total_params += params

    # 注册hook
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.SiLU)):
            hooks.append(module.register_forward_hook(hook_fn))

    # 执行前向传播
    try:
        model.eval()
        with torch.no_grad():
            print(f"Manual calculation forward pass with inputs: {[tensor.shape for tensor in input_tensors]}")
            _ = model(*input_tensors)
    except Exception as e:
        print(f"Forward pass failed during manual calculation: {e}")
        # 尝试单输入
        try:
            print("Trying single input for manual calculation...")
            _ = model(input_tensors[0])
        except Exception as e2:
            print(f"Single input also failed: {e2}")
        remove_hooks(hooks)
        return None

    # 移除hooks
    remove_hooks(hooks)

    return {
        'flops': total_flops,
        'params': total_params,
        'flops_str': f"{total_flops / 1e9:.3f}G" if total_flops > 1e9 else f"{total_flops / 1e6:.3f}M",
        'params_str': f"{total_params / 1e6:.3f}M",
        'method': 'manual',
        'layer_info': layer_info
    }


def remove_hooks(hooks):
    """移除注册的hooks"""
    for hook in hooks:
        hook.remove()


def manual_parameter_count(model):
    """手动计算参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_str': f"{total_params / 1e6:.3f}M",
        'trainable_params_str': f"{trainable_params / 1e6:.3f}M"
    }


def test_forward_pass(model, input_tensors):
    """测试前向传播是否正常"""
    print("Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            # 尝试多输入
            output = model(*input_tensors)
            print(f"Multi-input forward pass successful! Output type: {type(output)}")
            return True
    except Exception as e:
        print(f"Multi-input forward pass failed: {e}")

        # 尝试单输入
        try:
            with torch.no_grad():
                output = model(input_tensors[0])
                print(f"Single-input forward pass successful! Output type: {type(output)}")
                return True
        except Exception as e2:
            print(f"Single-input forward pass also failed: {e2}")
            return False


def measure_fps(model, input_tensors, num_runs=100, warmup_runs=10, batch_size=1):
    """
    测量模型的FPS（Frames Per Second）

    Args:
        model: 要测试的模型
        input_tensors: 输入张量
        num_runs: 测试运行次数
        warmup_runs: 预热运行次数
        batch_size: 批处理大小
    """
    print(f"Measuring FPS with {num_runs} runs, warmup {warmup_runs} runs, batch_size {batch_size}")

    # 准备批处理数据
    if batch_size > 1:
        # 扩展输入到指定批处理大小
        batched_inputs = []
        for tensor in input_tensors:
            batched_tensor = tensor.repeat(batch_size, 1, 1, 1)
            batched_inputs.append(batched_tensor)
        input_tensors = tuple(batched_inputs)
        print(f"Input shapes after batching: {[tensor.shape for tensor in input_tensors]}")

    model.eval()
    device = next(model.parameters()).device

    # GPU同步（如果使用GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 预热运行
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*input_tensors)

    # GPU同步
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 正式测量
    print("Measuring inference time...")
    timings = []

    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            _ = model(*input_tensors)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            timings.append((end_time - start_time) * 1000)  # 转换为毫秒

    # 计算统计信息
    timings = np.array(timings)
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)

    # 计算FPS
    fps = 1000 / mean_time * batch_size  # 考虑批处理大小
    fps_per_frame = 1000 / mean_time  # 每帧的FPS

    return {
        'fps': fps,
        'fps_per_frame': fps_per_frame,
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'batch_size': batch_size,
        'num_runs': num_runs,
        'device': str(device)
    }


def measure_fps_different_batch_sizes(model, input_tensors, batch_sizes=[1, 2, 4, 8, 16]):
    """
    测量不同批处理大小下的FPS
    """
    fps_results = {}

    for batch_size in batch_sizes:
        if batch_size == 1:
            # 对于批处理大小为1，使用原始输入
            current_inputs = input_tensors
        else:
            # 扩展输入到指定批处理大小
            batched_inputs = []
            for tensor in input_tensors:
                batched_tensor = tensor.repeat(batch_size, 1, 1, 1)
                batched_inputs.append(batched_tensor)
            current_inputs = tuple(batched_inputs)

        print(f"\nMeasuring FPS for batch_size={batch_size}")

        try:
            result = measure_fps(model, current_inputs, num_runs=50, warmup_runs=10, batch_size=batch_size)
            fps_results[batch_size] = result
        except Exception as e:
            print(f"Failed to measure FPS for batch_size={batch_size}: {e}")
            fps_results[batch_size] = None

    return fps_results


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops and FPS using alternative methods')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size')
    parser.add_argument(
        '--method',
        type=str,
        choices=['thop', 'fvcore', 'manual', 'all'],
        default='all',
        help='FLOPS calculation method')
    parser.add_argument(
        '--fps-batch-sizes',
        type=int,
        nargs='+',
        default=[1],
        help='batch sizes for FPS measurement')
    parser.add_argument(
        '--fps-runs',
        type=int,
        default=100,
        help='number of runs for FPS measurement')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    # 加载配置
    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')
        return

    cfg = Config.fromfile(args.config)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')

    # 构建模型
    print("Building model...")
    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    switch_to_deploy(model)

    print(f"Model built successfully!")

    # 创建输入数据
    device = next(model.parameters()).device

    # 对于双输入模型
    input1 = torch.randn(1, 3, h, w).to(device)
    input2 = torch.randn(1, 3, h, w).to(device)
    input_tensors = (input1, input2)

    print(f"Input shapes: {[tensor.shape for tensor in input_tensors]}")
    print("=" * 50)

    # 测试前向传播
    if not test_forward_pass(model, input_tensors):
        print("Forward pass test failed! Model might not be compatible with the provided inputs.")
        return

    results = {}

    # 手动参数计算（总是执行）
    print("Calculating parameters manually...")
    param_result = manual_parameter_count(model)
    results['manual_params'] = param_result
    print(f"Total Parameters: {param_result['total_params_str']}")
    print(f"Trainable Parameters: {param_result['trainable_params_str']}")
    print("=" * 50)

    # THOP 方法
    if args.method in ['thop', 'all'] and THOP_AVAILABLE:
        print("Calculating FLOPS with THOP...")
        try:
            thop_result = calculate_flops_with_thop(model, input_tensors)
            if thop_result and thop_result['flops'] > 0:
                results['thop'] = thop_result
                print(f"THOP - FLOPS: {thop_result['flops_str']}, Params: {thop_result['params_str']}")
            else:
                print("THOP method failed or returned zero FLOPS")
        except Exception as e:
            print(f"THOP method error: {e}")
        print("=" * 50)

    # FVCore 方法
    if args.method in ['fvcore', 'all'] and FVCORE_AVAILABLE:
        print("Calculating FLOPS with FVCore...")
        try:
            fvcore_result = calculate_flops_with_fvcore(model, input_tensors)
            if fvcore_result and fvcore_result['flops'] > 0:
                results['fvcore'] = fvcore_result
                print(f"FVCore - FLOPS: {fvcore_result['flops_str']}, Params: {fvcore_result['params_str']}")
            else:
                print("FVCore method failed or returned zero FLOPS")
        except Exception as e:
            print(f"FVCore method error: {e}")
        print("=" * 50)

    # Manual 方法
    if args.method in ['manual', 'all']:
        print("Calculating FLOPS manually...")
        try:
            manual_result = calculate_flops_manual_conv(model, input_tensors)
            if manual_result and manual_result['flops'] > 0:
                results['manual'] = manual_result
                print(f"Manual - FLOPS: {manual_result['flops_str']}, Params: {manual_result['params_str']}")
            else:
                print("Manual method failed or returned zero FLOPS")
        except Exception as e:
            print(f"Manual method error: {e}")
        print("=" * 50)

    # FPS 测量
    print("Measuring FPS...")
    fps_results = {}

    # 测量不同批处理大小的FPS
    if args.fps_batch_sizes:
        for batch_size in args.fps_batch_sizes:
            print(f"\nMeasuring FPS for batch_size={batch_size}")
            try:
                fps_result = measure_fps(model, input_tensors, num_runs=args.fps_runs, batch_size=batch_size)
                fps_results[batch_size] = fps_result
                print(f"Batch Size {batch_size}: {fps_result['fps']:.2f} FPS "
                      f"(mean: {fps_result['mean_time_ms']:.2f}ms, "
                      f"std: {fps_result['std_time_ms']:.2f}ms)")
            except Exception as e:
                print(f"FPS measurement failed for batch_size={batch_size}: {e}")
                fps_results[batch_size] = None

    results['fps'] = fps_results

    # 输出总结
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)

    # 检查是否有成功的FLOPS计算结果
    flops_results = {k: v for k, v in results.items() if 'flops' in v and v['flops'] > 0}

    if flops_results:
        for method, result in flops_results.items():
            print(f"{method.upper()}: FLOPS={result['flops_str']}, Params={result['params_str']}")
    else:
        print("No successful FLOPS calculations!")
        # 只显示参数结果
        for method, result in results.items():
            if 'total_params_str' in result:
                print(f"{method.upper()}: Params={result['total_params_str']}")

    # 输出FPS结果
    if fps_results:
        print("\nFPS Results:")
        print("-" * 40)
        for batch_size, result in fps_results.items():
            if result:
                print(f"Batch Size {batch_size}:")
                print(f"  FPS: {result['fps']:.2f}")
                print(f"  Inference Time: {result['mean_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
                print(f"  Range: {result['min_time_ms']:.2f} - {result['max_time_ms']:.2f} ms")
                print(f"  Device: {result['device']}")

    print("\nNote: Different methods may give different results due to:")
    print("- Different counting strategies for certain operations")
    print("- Handling of custom modules")
    print("- Treatment of activation functions")
    print("- Some methods may not support all operation types")

    # 如果所有FLOPS计算都失败了，提供更详细的调试信息
    if not flops_results:
        print("\n" + "!" * 50)
        print("WARNING: No FLOPS calculations succeeded!")
        print("Possible solutions:")
        print("1. Check if the model forward pass works correctly")
        print("2. Try different input shapes")
        print("3. Install thop: pip install thop")
        print("4. The model might use custom operations not supported by standard FLOPS counters")
        print("!" * 50)


if __name__ == '__main__':
    main()