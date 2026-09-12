import os
import shutil
import re


def extract_eight_digit_numbers(txt_path):
    """从txt文件中提取所有8位数字编号（每行一个）"""
    if not os.path.exists(txt_path):
        print(f"错误：文件 {txt_path} 不存在")
        return []

    numbers = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 按行读取文件
            lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                # 去除每行的首尾空白字符（包括换行符）
                stripped_line = line.strip()

                # 检查是否为8位数字
                if re.fullmatch(r'\d{8}', stripped_line):
                    numbers.append(stripped_line)
                else:
                    print(f"警告：第 {line_num} 行 '{stripped_line}' 不是有效的8位数字编号，已跳过")

        # 去重处理
        numbers = list(set(numbers))
        print(f"从 {txt_path} 中提取到 {len(numbers)} 个有效的8位数字编号（这些编号的图片将被排除）")
    except Exception as e:
        print(f"读取文件时出错：{e}")

    return numbers


def copy_excluded_images(excluded_numbers, source_dir, target_dir):
    """复制除排除编号外的、以8位数字编号+_co.png命名的图片到目标文件夹"""
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.isdir(source_dir):
        print(f"错误：源文件夹 {source_dir} 不存在")
        return

    copied_count = 0
    excluded_count = 0

    # 遍历源文件夹中的所有文件
    for file in os.listdir(source_dir):
        # 检查文件名是否符合 "8位数字_co.png" 格式
        if re.fullmatch(r'\d{8}\.txt', file):
            # 提取8位数字部分
            number_part = file[:8]

            # 判断是否为排除的编号
            if number_part in excluded_numbers:
                excluded_count += 1
                print(f"已排除：{file}（存在于TXT文件中）")
            else:
                source_path = os.path.join(source_dir, file)
                target_path = os.path.join(target_dir, file)

                if os.path.exists(target_path):
                    print(f"文件 {file} 已存在于目标文件夹，跳过")
                    continue

                try:
                    shutil.copy2(source_path, target_path)
                    print(f"已复制：{file}")
                    copied_count += 1
                except Exception as e:
                    print(f"复制 {file} 时出错：{e}")

    print(f"操作完成，共复制了 {copied_count} 个文件，排除了 {excluded_count} 个文件")


if __name__ == "__main__":
    # 获取用户输入
    txt_file_path = input("请输入包含8位数字编号的txt文件路径：").strip()
    source_folder = input("请输入图片所在的源文件夹路径：").strip()
    target_folder = input("请输入目标文件夹路径：").strip()

    # 提取需要排除的8位数字编号
    excluded_numbers = extract_eight_digit_numbers(txt_file_path)

    # 复制排除后的图片（即使没有提取到编号也执行，复制所有符合格式的图片）
    copy_excluded_images(excluded_numbers, source_folder, target_folder)
