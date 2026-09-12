import sys

sys.path.append('/data/zfy/mmyolo')  # 确保项目在路径中

from mmdet.registry import MODELS
# projects.CO_DETR.codetr.codetr_dual_stream
try:
    # 尝试导入自定义模块
    import projects.CO_DETR.configs.codino.codetr_dual_stream_reg

    print("✅ 成功导入自定义模块")

    # 检查是否注册
    if 'CoDETR_Dual_Reg' in MODELS.module_dict:
        print("✅ CoDETR_Dual_Reg 已注册")
    else:
        print("❌ CoDETR_Dual_Reg 未注册，请检查：")
        print("1. 文件顶部是否有 '@MODELS.register_module()'")
        print("2. 类名是否拼写正确")
        print("当前注册的模型:", list(MODELS.module_dict.keys()))
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请检查路径是否正确：/data/zfy/mmyolo/project/CO_DETR/models/codetr_dual_reg.py")