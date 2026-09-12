import os
import xml.etree.ElementTree as ET
import json
import argparse
from collections import defaultdict
from PIL import Image


def parse_xml_annotation(xml_path):
    """
    解析单个XML标注文件

    参数:
    xml_path: XML文件路径

    返回:
    解析后的标注信息字典
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 提取图像基本信息
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 提取所有对象标注
        objects = []
        for obj in root.findall('object'):
            # 提取类别名称
            class_name = obj.find('name').text

            # 提取边界框信息
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 计算COCO格式的bbox: [x, y, width, height]
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = (xmax - xmin) * (ymax - ymin)

            objects.append({
                'class_name': class_name,
                'bbox': bbox,
                'area': area
            })

        return {
            'filename': filename,
            'width': width,
            'height': height,
            'objects': objects
        }
    except Exception as e:
        print(f"解析XML文件 {xml_path} 时出错: {e}")
        return None


def load_classes(classes_file):
    """
    加载类别文件

    参数:
    classes_file: 类别文件路径

    返回:
    类别列表
    """
    try:
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    except Exception as e:
        print(f"加载类别文件 {classes_file} 时出错: {e}")
        return []


def convert_xml_to_coco(xml_dir, img_dir, classes_file, output_json):
    """
    将XML标注转换为COCO格式的JSON文件

    参数:
    xml_dir: XML标注文件夹路径
    img_dir: 图像文件夹路径
    classes_file: 类别文件路径
    output_json: 输出JSON文件路径
    """
    # 加载类别
    classes = load_classes(classes_file)
    if not classes:
        print("错误: 无法加载类别文件或类别文件为空")
        return False

    # 创建类别ID映射
    category_id_map = {name: i + 1 for i, name in enumerate(classes)}

    # 初始化COCO数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 添加类别信息
    for i, class_name in enumerate(classes):
        coco_data["categories"].append({
            "id": i + 1,
            "name": class_name,
            "supercategory": "none"
        })

    # 获取所有XML文件
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    if not xml_files:
        print(f"错误: 在文件夹 {xml_dir} 中未找到XML文件")
        return False

    print(f"找到 {len(xml_files)} 个XML文件")

    # 初始化计数器
    image_id = 1
    annotation_id = 1

    # 处理每个XML文件
    for i, xml_file in enumerate(xml_files):
        xml_path = os.path.join(xml_dir, xml_file)

        # 解析XML
        annotation = parse_xml_annotation(xml_path)
        if annotation is None:
            continue

        # 检查对应的图像文件是否存在
        img_path = os.path.join(img_dir, annotation['filename'])
        if not os.path.exists(img_path):
            # 尝试使用XML文件名（不带扩展名）加上常见图像扩展名
            base_name = os.path.splitext(xml_file)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                alt_img_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(alt_img_path):
                    img_path = alt_img_path
                    annotation['filename'] = base_name + ext
                    break
            else:
                print(f"警告: 未找到图像文件 {annotation['filename']}，跳过")
                continue

        # 验证图像尺寸（如果XML中的尺寸与图像实际尺寸不一致）
        try:
            with Image.open(img_path) as img:
                actual_width, actual_height = img.size
                if annotation['width'] != actual_width or annotation['height'] != actual_height:
                    print(f"警告: {annotation['filename']} 的尺寸信息不匹配，使用实际尺寸")
                    annotation['width'] = actual_width
                    annotation['height'] = actual_height
        except Exception as e:
            print(f"警告: 无法验证图像 {annotation['filename']} 的尺寸: {e}")

        # 添加图像信息
        image_info = {
            "id": image_id,
            "file_name": annotation['filename'],
            "width": annotation['width'],
            "height": annotation['height']
        }
        coco_data["images"].append(image_info)

        # 添加标注信息
        for obj in annotation['objects']:
            if obj['class_name'] not in category_id_map:
                print(f"警告: 未知类别 '{obj['class_name']}'，跳过")
                continue

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id_map[obj['class_name']],
                "bbox": obj['bbox'],
                "area": obj['area'],
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation_info)
            annotation_id += 1

        image_id += 1

        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个文件...")

    # 保存为JSON文件
    try:
        with open(output_json, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"转换完成! 结果已保存到 {output_json}")
        print(f"统计信息:")
        print(f"  图像数量: {len(coco_data['images'])}")
        print(f"  标注数量: {len(coco_data['annotations'])}")
        print(f"  类别数量: {len(coco_data['categories'])}")
        return True
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将XML标注转换为COCO格式的JSON文件')
    parser.add_argument('--xml-dir', required=True, help='XML标注文件夹路径')
    parser.add_argument('--img-dir', required=True, help='图像文件夹路径')
    parser.add_argument('--classes', required=True, help='类别文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')

    args = parser.parse_args()

    # 执行转换
    success = convert_xml_to_coco(args.xml_dir, args.img_dir, args.classes, args.output)

    if not success:
        print("转换失败!")


if __name__ == "__main__":
    main()