import json
import os
from collections import defaultdict


def coco_to_yolo(coco_json_path, output_dir, image_dir=None):
    """
    将COCO格式的标注转换为YOLO格式

    参数:
        coco_json_path: COCO格式的JSON文件路径
        output_dir: YOLO格式TXT文件的输出目录
        image_dir: 图片文件夹路径（可选，用于验证图片是否存在）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载COCO JSON文件
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 获取类别信息
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_ids = list(categories.keys())

    # 创建类别ID到YOLO类别的映射（从0开始）
    class_mapping = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    print("类别映射关系:")
    for cat_id, idx in class_mapping.items():
        print(f"  {categories[cat_id]}: {idx}")

    # 按图片ID分组标注
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # 获取图片信息
    image_info = {img['id']: img for img in coco_data['images']}

    # 处理每张图片的标注
    processed_count = 0
    for img_id, annotations in annotations_by_image.items():
        img = image_info[img_id]
        img_width = img['width']
        img_height = img['height']
        img_filename = os.path.splitext(img['file_name'])[0]

        # 检查图片是否存在（如果提供了图片目录）
        if image_dir:
            img_path = os.path.join(image_dir, img['file_name'])
            if not os.path.exists(img_path):
                print(f"警告: 图片 {img['file_name']} 不存在，跳过")
                continue

        # 创建YOLO格式的标注文件
        yolo_filename = f"{img_filename}.txt"
        yolo_path = os.path.join(output_dir, yolo_filename)

        with open(yolo_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                # 获取边界框信息
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']

                # 转换为YOLO格式 [class, x_center, y_center, width, height]（归一化）
                x, y, w, h = bbox

                # 计算中心点坐标
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height

                # 归一化宽高
                w_norm = w / img_width
                h_norm = h / img_height

                # 获取类别ID（从0开始）
                class_id = class_mapping[category_id]

                # 写入文件
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count} 张图片")

    # 生成类别名称文件
    with open(os.path.join(output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        for cat_id in category_ids:
            f.write(f"{categories[cat_id]}\n")

    print(f"\n转换完成！")
    print(f"- 处理了 {processed_count} 张图片")
    print(f"- 生成了 {len(annotations_by_image)} 个标注文件")
    print(f"- 类别数量: {len(categories)}")
    print(f"- 标注文件保存在: {output_dir}")


if __name__ == "__main__":
    # 配置参数
    COCO_JSON_PATH = "/data/zfy/dataset/dronevehicle/val.json"  # COCO格式的JSON文件路径
    OUTPUT_DIR = "/data/zfy/dataset/dronevehicle/images/labels/test"  # YOLO标注输出目录
    IMAGE_DIR = "/data/zfy/dataset/dronevehicle/images/visible/test"  # 图片文件夹路径（可选）

    # 执行转换
    coco_to_yolo(COCO_JSON_PATH, OUTPUT_DIR, IMAGE_DIR)