import os
import glob
import xml.etree.ElementTree as ET
# 加载类别信息
def get_classes(classes_path):
   with open(classes_path, encoding='utf-8') as f:
       class_names = f.readlines()
   class_names = [c.strip() for c in class_names]
   print("Classes loaded:", class_names)
   return class_names
# 转换边界框坐标
def convert(size, box):
   dw = 1.0 / size[0]
   dh = 1.0 / size[1]
   x = (box[0] + box[1]) / 2.0
   y = (box[2] + box[3]) / 2.0
   w = box[1] - box[0]
   h = box[3] - box[2]
   return (x * dw, y * dh, w * dw, h * dh)
# XML转YOLO TXT格式
def convert_xml_to_yolo(xml_root_path, txt_save_path, classes_path):
   if not os.path.exists(txt_save_path):
       os.makedirs(txt_save_path)
   xml_paths = glob.glob(os.path.join(xml_root_path, '*.xml'))
   classes = get_classes(classes_path)
   for xml_file in xml_paths:
       txt_file = os.path.join(txt_save_path, os.path.basename(xml_file).replace('.xml', '.txt'))
       with open(txt_file, 'w') as txt:
           tree = ET.parse(xml_file)
           root = tree.getroot()
           size = root.find('size')
           w = int(size.find('width').text)
           h = int(size.find('height').text)
           for obj in root.iter('object'):
               cls_name = obj.find('name').text
               if cls_name not in classes:
                   continue
               cls_id = classes.index(cls_name)
               xmlbox = obj.find('bndbox')
               b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
               bbox = convert((w, h), b)
               txt.write(f"{cls_id} {' '.join(map(str, bbox))}\n")
# 用户输入路径
if __name__ == '__main__':
   xml_root_path = r"/data/zfy/dataset/dronevehicle/val/labels_xml"
   txt_save_path = r"/data/zfy/dataset/dronevehicle/val/labels"
   classes_path = r"/data/zfy/dataset/dronevehicle/val/classes.txt"
   convert_xml_to_yolo(xml_root_path, txt_save_path, classes_path)