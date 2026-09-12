"""This script helps to convert yolo-style dataset to the coco format.

Usage:
    $ python yolo2coco.py /path/to/dataset --filter-dir /path/to/filter/images # 指定过滤文件夹

Note:
    1. 新增功能：通过--filter-dir指定图片文件夹，仅处理该文件夹中存在的图片及其标注
    2. 其他功能与原脚本一致，支持通过train.txt/val.txt/test.txt划分数据集
    3. 若不指定--filter-dir，则默认处理所有图片（与原脚本行为一致）
"""
import argparse
import os
import os.path as osp

import mmcv
import mmengine

IMG_EXTENSIONS = ('.jpg', '.png', '.jpeg')


def check_existence(file_path: str):
    """Check if target file is existed."""
    if not osp.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist!')


def get_image_info(yolo_image_dir, idx, file_name):
    """Retrieve image information."""
    img_path = osp.join(yolo_image_dir, file_name)
    check_existence(img_path)

    img = mmcv.imread(img_path)
    height, width = img.shape[:2]
    img_info_dict = {
        'file_name': file_name,
        'id': idx,
        'width': width,
        'height': height
    }
    return img_info_dict, height, width


def convert_bbox_info(label, idx, obj_count, image_height, image_width):
    """Convert yolo-style bbox info to the coco format."""
    label = label.strip().split()
    x = float(label[1])
    y = float(label[2])
    w = float(label[3])
    h = float(label[4])

    # convert x,y,w,h to x1,y1,x2,y2
    x1 = (x - w / 2) * image_width
    y1 = (y - h / 2) * image_height
    x2 = (x + w / 2) * image_width
    y2 = (y + h / 2) * image_height

    cls_id = int(label[0])
    width = max(0., x2 - x1)
    height = max(0., y2 - y1)
    coco_format_info = {
        'image_id': idx,
        'id': obj_count,
        'category_id': cls_id,
        'bbox': [x1, y1, width, height],
        'area': width * height,
        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
        'iscrowd': 0
    }
    obj_count += 1
    return coco_format_info, obj_count


def organize_by_existing_files(image_dir: str, existed_categories: list):
    """Format annotations by existing train/val/test files."""
    categories = ['train', 'val', 'test']
    image_list = []

    for cat in categories:
        if cat in existed_categories:
            txt_file = osp.join(image_dir, f'{cat}.txt')
            print(f'Start to read {cat} dataset definition')
            assert osp.exists(txt_file)

            with open(txt_file) as f:
                img_paths = f.readlines()
                img_paths = [
                    os.path.split(img_path.strip())[1]
                    for img_path in img_paths
                ]  # split the absolute path
                image_list.append(img_paths)
        else:
            image_list.append([])
    return image_list[0], image_list[1], image_list[2]


def get_filtered_images(filter_dir: str):
    """获取过滤文件夹中所有图片的文件名（不含路径）"""
    if not osp.exists(filter_dir):
        raise FileNotFoundError(f'过滤文件夹 {filter_dir} 不存在！')

    filter_images = set()
    for file in os.listdir(filter_dir):
        if file.lower().endswith(IMG_EXTENSIONS):
            filter_images.add(file)  # 仅保留图片文件名
    return filter_images


def convert_yolo_to_coco(image_dir: str, filter_dir: str = None):
    """Convert annotations from yolo style to coco style.

    Args:
        image_dir (str): 数据集根目录，包含labels、images、classes.txt等
        filter_dir (str, optional): 过滤文件夹，仅处理该文件夹中存在的图片
    """
    print(f'Start to load existing images and annotations from {image_dir}')
    check_existence(image_dir)

    # 检查基础文件是否存在
    yolo_label_dir = osp.join(image_dir, 'labels')
    yolo_image_dir = osp.join(image_dir, 'images')
    yolo_class_txt = osp.join(image_dir, 'classes.txt')
    check_existence(yolo_label_dir)
    check_existence(yolo_image_dir)
    check_existence(yolo_class_txt)
    print(f'All necessary files are located at {image_dir}')

    # 处理过滤文件夹（若指定）
    filtered_images = None
    if filter_dir:
        filtered_images = get_filtered_images(filter_dir)
        print(f'已加载过滤文件夹 {filter_dir} 中的图片，共 {len(filtered_images)} 张')

    # 检查train/val/test划分文件
    train_txt_path = osp.join(image_dir, 'train.txt')
    val_txt_path = osp.join(image_dir, 'val.txt')
    test_txt_path = osp.join(image_dir, 'test.txt')
    existed_categories = []
    print(f'Checking if train.txt, val.txt, and test.txt are in {image_dir}')
    if osp.exists(train_txt_path):
        print('Found train.txt')
        existed_categories.append('train')
    if osp.exists(val_txt_path):
        print('Found val.txt')
        existed_categories.append('val')
    if osp.exists(test_txt_path):
        print('Found test.txt')
        existed_categories.append('test')

    # 准备输出文件夹
    output_folder = osp.join(image_dir, 'annotations')
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
        check_existence(output_folder)

    # 加载类别信息
    with open(yolo_class_txt) as f:
        classes = f.read().strip().split()

    # 获取所有图片文件（并根据过滤文件夹筛选）
    all_images = os.listdir(yolo_image_dir)
    if filtered_images is not None:
        # 仅保留过滤文件夹中存在的图片
        indices = [img for img in all_images if img in filtered_images]
        print(f'通过过滤文件夹筛选后，剩余图片 {len(indices)} 张')
    else:
        indices = all_images  # 无过滤时使用所有图片
    total = len(indices)
    if total == 0:
        print('错误：未找到符合条件的图片，请检查过滤文件夹或图片路径')
        return

    # 初始化数据集结构
    if existed_categories == []:
        print('These files are not located, no need to organize separately.')
        dataset = {'images': [], 'annotations': [], 'categories': []}
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls})
    else:
        print('Need to organize the data accordingly.')
        train_dataset = {'images': [], 'annotations': [], 'categories': []}
        val_dataset = {'images': [], 'annotations': [], 'categories': []}
        test_dataset = {'images': [], 'annotations': [], 'categories': []}
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls})
            val_dataset['categories'].append({'id': i, 'name': cls})
            test_dataset['categories'].append({'id': i, 'name': cls})
        # 加载train/val/test划分（并自动过滤不在indices中的图片）
        train_img, val_img, test_img = organize_by_existing_files(
            image_dir, existed_categories)

    # 开始转换标注
    obj_count = 0
    skipped = 0
    converted = 0
    for idx, image in enumerate(mmengine.track_iter_progress(indices)):
        # 获取图片信息
        img_info_dict, image_height, image_width = get_image_info(
            yolo_image_dir, idx, image)

        # 确定当前处理的数据集（train/val/test或默认）
        if existed_categories != []:
            if image in train_img:
                dataset = train_dataset
            elif image in val_img:
                dataset = val_dataset
            elif image in test_img:
                dataset = test_dataset
            else:
                # 不在任何划分中的图片将被跳过
                skipped += 1
                continue
        dataset['images'].append(img_info_dict)

        # 处理标注文件
        img_name = osp.splitext(image)[0]
        label_path = f'{osp.join(yolo_label_dir, img_name)}.txt'
        if not osp.exists(label_path):
            print(f'WARNING: {label_path} 不存在，跳过该图片')
            skipped += 1
            continue

        # 解析并转换标注
        with open(label_path) as f:
            labels = f.readlines()
            for label in labels:
                coco_info, obj_count = convert_bbox_info(
                    label, idx, obj_count, image_height, image_width)
                dataset['annotations'].append(coco_info)
        converted += 1

    # 保存转换结果
    if existed_categories == []:
        out_file = osp.join(image_dir, 'annotations/result.json')
        print(f'Saving converted results to {out_file} ...')
        mmengine.dump(dataset, out_file)
    else:
        for category in existed_categories:
            out_file = osp.join(output_folder, f'{category}.json')
            print(f'Saving converted results to {out_file} ...')
            if category == 'train':
                mmengine.dump(train_dataset, out_file)
            elif category == 'val':
                mmengine.dump(val_dataset, out_file)
            elif category == 'test':
                mmengine.dump(test_dataset, out_file)

    # 转换统计信息
    print(f'Process finished! Please check at {output_folder} .')
    print(f'符合条件的图片总数: {total}, 成功转换: {converted}, 跳过: {skipped}')
    print(f'总标注数量: {obj_count}')
    print('You can use tools/analysis_tools/browse_coco_json.py to visualize!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        help='dataset directory with ./images and ./labels, classes.txt, etc.'
    )
    parser.add_argument(
        '--filter-dir',
        type=str,
        default=None,
        help='指定过滤文件夹，仅处理该文件夹中存在的图片（可选）'
    )
    args = parser.parse_args()
    convert_yolo_to_coco(args.image_dir, args.filter_dir)