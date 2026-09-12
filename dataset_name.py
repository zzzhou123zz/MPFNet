import os
import argparse


def rename_images(folder_path, prefix, start_num=2):
    """
    重命名文件夹中的所有图片文件为统一前缀+顺序数字

    :param folder_path: 图片文件夹路径
    :param prefix: 文件名前缀（默认"img_"）

    """
    # 支持的图片扩展名
    image_extensions = ['.jpg']

    # 获取文件夹中所有图片文件
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in image_extensions]
    files.sort()  # 按原始文件名排序

    # 重命名文件
    for idx, filename in enumerate(files, start=start_num):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1]  # 保留原扩展名
        new_name = f"{prefix}{idx:05d}{ext}"  # 格式如 img_00001.jpg
        new_path = os.path.join(folder_path, new_name)

        # 避免覆盖已存在的文件
        while os.path.exists(new_path):
            idx += 1
            new_name = f"{prefix}{idx:05d}{ext}"
            new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default = '/data/zfy/dataset/FLIR-align-3class/images/infrared/test', help="包含图片的文件夹路径")
    parser.add_argument("--prefix", default="FILR_", help="文件名前缀（默认: img_）")
    parser.add_argument("--start", type=int, default=2, help="起始数字（默认: 2）")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"错误：文件夹 {args.folder} 不存在！")
    else:
        rename_images(args.folder, args.prefix)
        print("重命名完成！")