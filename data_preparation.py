import os
import cv2
import random
import shutil
import xml.etree.ElementTree as ET
from glob import glob

def convert_annotation(xml_path, w, h, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in classes:
            continue
        cls_id = classes[cls_name]
        bnd = obj.find('bndbox')
        xmin = int(bnd.find('xmin').text)
        ymin = int(bnd.find('ymin').text)
        xmax = int(bnd.find('xmax').text)
        ymax = int(bnd.find('ymax').text)
        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines

def prepare_dataset(input_image_dir, input_annotation_dir, output_dir, classes, ratios, seed=42):
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # Get and shuffle image paths
    image_paths = sorted(glob(f"{input_image_dir}/*.jpg"))
    random.seed(seed)
    random.shuffle(image_paths)

    # Split dataset
    n = len(image_paths)
    train_end = int(n * ratios['train'])
    val_end = train_end + int(n * ratios['val'])
    
    splits = {
        'train': image_paths[:train_end],
        'val': image_paths[train_end:val_end],
        'test': image_paths[val_end:]
    }

    # Process each split
    for split, paths in splits.items():
        for img_path in paths:
            name = os.path.basename(img_path)
            xml_path = os.path.join(input_annotation_dir, name.replace('.jpg', '.xml'))
            
            img = cv2.imread(img_path)
            if img is None or not os.path.exists(xml_path):
                continue
                
            h, w = img.shape[:2]
            labels = convert_annotation(xml_path, w, h, classes)
            
            if not labels:
                continue
                
            # Copy image and write labels
            shutil.copy(img_path, f"{output_dir}/images/{split}/{name}")
            with open(f"{output_dir}/labels/{split}/{name.replace('.jpg', '.txt')}", 'w') as f:
                f.write('\n'.join(labels))

    print("Dataset prepared successfully")

    # Create data.yaml file
    data_yaml = f"""
train: {output_dir}/images/train
val: {output_dir}/images/val
test: {output_dir}/images/test

nc: {len(classes)}
names: {list(classes.keys())}
"""
    with open(f"{output_dir}/data.yaml", "w") as f:
        f.write(data_yaml)

if __name__ == "__main__":
    # Configuration
    config = {
        "input_image_dir": "dataset/images",  # Update with your path
        "input_annotation_dir": "dataset/annotations",  # Update with your path
        "output_dir": "dataset/yolo_dataset",
        "classes": {'MR': 0, 'NC': 1, 'WF': 2},
        "ratios": {'train': 0.7, 'val': 0.15, 'test': 0.15},
        "seed": 42
    }
    
    prepare_dataset(**config)