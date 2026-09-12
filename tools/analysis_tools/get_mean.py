import os
import shutil


def move_corresponding_txt_files(png_folder, txt_folder, output_folder):
    """
    将 png_folder 中的每个 .png 文件对应的 .txt 文件（来自 txt_folder）移动到 output_folder。

    参数:
        png_folder (str): 存放 .png 文件的文件夹路径。
        txt_folder (str): 存放 .txt 文件的文件夹路径。
        output_folder (str): 存放匹配的 .txt 文件的输出文件夹路径。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历 png 文件夹中的所有 .png 文件
    for png_file in os.listdir(png_folder):
        if png_file.lower().endswith('.png'):
            # 获取不带扩展名的文件名
            file_name = os.path.splitext(png_file)[0]

            # 构造对应的 .txt 文件路径
            txt_file = f"{file_name}.txt"
            txt_file_path = os.path.join(txt_folder, txt_file)

            # 检查 .txt 文件是否存在
            if os.path.exists(txt_file_path):
                # 构造目标路径
                dest_path = os.path.join(output_folder, txt_file)

                # 移动文件
                shutil.move(txt_file_path, dest_path)
                print(f"已移动: {txt_file} -> {dest_path}")
            else:
                print(f"未找到匹配的 .txt 文件: {txt_file}")


if __name__ == "__main__":
    # 示例用法
    png_folder = "/data/zfy/dataset/M3FD/val/rgb"  # 替换为你的 .png 文件夹路径
    txt_folder = "/data/zfy/dataset/M3FD/labels"  # 替换为你的 .txt 文件夹路径
    output_folder = "/data/zfy/dataset/M3FD/labels/val"  # 替换为你的输出文件夹路径

    move_corresponding_txt_files(png_folder, txt_folder, output_folder)
    print("操作完成！")