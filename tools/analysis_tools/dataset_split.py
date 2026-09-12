import os
import argparse
import random
from shutil import copy2


def split_dataset(image_dir, output_dir, train_ratio=0.8, random_seed=42):
    """
    划分数据集为训练集和测试集

    Args:
        image_dir: 原始图像文件夹路径
        output_dir: 划分后数据集的输出路径
        train_ratio: 训练集所占比例，默认0.8
        random_seed: 随机种子，保证划分结果可复现
    """
    # 设置随机种子，确保结果可复现
    random.seed(random_seed)

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif','.txt']
    image_files = []

    for file in os.listdir(image_dir):
        # 检查文件是否为图像
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    if not image_files:
        print(f"错误：在 {image_dir} 中未找到任何图像文件")
        return

    # 打乱文件顺序
    random.shuffle(image_files)

    # 计算划分点
    split_index = int(len(image_files) * train_ratio)

    # 分割为训练集和测试集
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 如果存在对应的标注文件，也一并复制（假设标注文件与图像文件同名，不同扩展名）
    label_extensions = ['.txt', '.xml', '.json']

    # 复制训练集文件
    for file in train_files:
        # 复制图像文件
        src = os.path.join(image_dir, file)
        dst = os.path.join(train_dir, file)
        copy2(src, dst)

        # 复制对应的标注文件
        file_name = os.path.splitext(file)[0]
        for ext in label_extensions:
            label_file = f"{file_name}{ext}"
            label_src = os.path.join(image_dir, label_file)
            if os.path.exists(label_src):
                label_dst_dir = os.path.join(output_dir, 'train', 'labels')
                os.makedirs(label_dst_dir, exist_ok=True)
                label_dst = os.path.join(label_dst_dir, label_file)
                copy2(label_src, label_dst)

    # 复制测试集文件
    for file in test_files:
        # 复制图像文件
        src = os.path.join(image_dir, file)
        dst = os.path.join(test_dir, file)
        copy2(src, dst)

        # 复制对应的标注文件
        file_name = os.path.splitext(file)[0]
        for ext in label_extensions:
            label_file = f"{file_name}{ext}"
            label_src = os.path.join(image_dir, label_file)
            if os.path.exists(label_src):
                label_dst_dir = os.path.join(output_dir, 'test', 'labels')
                os.makedirs(label_dst_dir, exist_ok=True)
                label_dst = os.path.join(label_dst_dir, label_file)
                copy2(label_src, label_dst)

    # 打印划分结果
    print(f"数据集划分完成！")
    print(f"总样本数: {len(image_files)}")
    print(f"训练集样本数: {len(train_files)} ({train_ratio * 100:.1f}%)")
    print(f"测试集样本数: {len(test_files)} ({(1 - train_ratio) * 100:.1f}%)")
    print(f"结果保存至: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将数据集按8:2比例划分为训练集和测试集')
    parser.add_argument('--image_dir', type=str, help='原始图像文件夹路径')
    parser.add_argument('--output_dir', type=str, help='划分后数据集的输出路径')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集所占比例，默认0.8')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，确保划分结果可复现，默认42')

    args = parser.parse_args()

    # 调用函数进行数据集划分
    split_dataset(args.image_dir, args.output_dir, args.train_ratio, args.seed)
