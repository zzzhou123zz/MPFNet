import os
import shutil
import argparse


def is_image_file(filename):
    """判断文件是否为图片格式"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.jfif'}
    return os.path.splitext(filename.lower())[1] in image_extensions


def merge_image_folders(folder1, folder2, output_folder, overwrite=False, keep_structure=False):
    """
    合并两个文件夹中的图片到指定文件夹

    参数:
        folder1 (str): 第一个源文件夹路径
        folder2 (str): 第二个源文件夹路径
        output_folder (str): 合并后保存的目标文件夹路径
        overwrite (bool): 若目标文件已存在，是否覆盖
        keep_structure (bool): 是否保留原始文件夹的目录结构

    返回:
        dict: 合并统计信息
    """
    # 初始化统计变量
    total_files = 0
    copied_files = 0
    skipped_files = 0

    # 创建输出文件夹（若不存在）
    os.makedirs(output_folder, exist_ok=True)

    def process_folder(source_folder):
        """处理单个源文件夹的图片复制逻辑"""
        nonlocal total_files, copied_files, skipped_files

        # 遍历源文件夹中的所有文件
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                # 仅处理图片文件
                if is_image_file(file):
                    total_files += 1
                    src_path = os.path.join(root, file)

                    # 确定目标文件路径
                    if keep_structure:
                        # 保留原始文件夹结构
                        relative_path = os.path.relpath(root, source_folder)
                        target_dir = os.path.join(output_folder, relative_path)
                        os.makedirs(target_dir, exist_ok=True)
                        target_path = os.path.join(target_dir, file)
                    else:
                        # 不保留结构，直接放到目标文件夹根目录
                        target_path = os.path.join(output_folder, file)

                    # 处理文件复制
                    if os.path.exists(target_path):
                        if overwrite:
                            # 覆盖已存在的文件
                            shutil.copy2(src_path, target_path)  # 保留文件元数据
                            copied_files += 1
                        else:
                            # 跳过已存在的文件
                            skipped_files += 1
                    else:
                        # 复制新文件
                        shutil.copy2(src_path, target_path)
                        copied_files += 1

    # 处理第一个文件夹
    process_folder(folder1)
    # 处理第二个文件夹
    process_folder(folder2)

    # 返回统计结果
    return {
        "total_images": total_files,
        "copied": copied_files,
        "skipped": skipped_files,
        "output_directory": output_folder
    }


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="合并两个文件夹中的图片到指定文件夹")
    parser.add_argument("--folder1", help="第一个包含图片的文件夹路径")
    parser.add_argument("--folder2", help="第二个包含图片的文件夹路径")
    parser.add_argument("--output", help="合并后图片的保存路径")
    parser.add_argument("--overwrite", action="store_true",
                        help="如果目标文件已存在，强制覆盖（默认不覆盖）")
    parser.add_argument("--keep-structure", action="store_true",
                        help="保留原始文件夹的目录结构（默认不保留）")

    args = parser.parse_args()

    # 验证输入文件夹是否存在
    for folder in [args.folder1, args.folder2]:
        if not os.path.isdir(folder):
            print(f"错误：文件夹不存在 - {folder}")
            exit(1)

    # 执行合并操作
    result = merge_image_folders(
        folder1=args.folder1,
        folder2=args.folder2,
        output_folder=args.output,
        overwrite=args.overwrite,
        keep_structure=args.keep_structure
    )

    # 打印合并结果
    print("\n===== 合并结果 =====")
    print(f"总图片数量：{result['total_images']}")
    print(f"成功复制：{result['copied']} 个文件")
    print(f"跳过（已存在）：{result['skipped']} 个文件")
    print(f"结果保存至：{result['output_directory']}")
    print("====================")