import cv2
import os
import random
import shutil
import xml.etree.ElementTree as ET
from glob import glob
import albumentations as A
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from ultralytics import YOLO
from IPython.display import Image, display
from sahi.utils.yolov8 import download_yolov8n_model


# --- 1. Thông số người dùng cần chỉnh ---
INPUT_IMAGE_DIR = "dataset/images"
INPUT_ANNOTATION_DIR = "dataset/annotations"
OUTPUT_DIR = "yolo_dataset"

CLASSES = {'MR': 0, 'NC': 1, 'WF': 2}
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
SEED = 42

# --- 2. Tạo thư mục ---
for split in ['train', 'val', 'test']:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# --- 3. Chuyển annotation từ XML -> YOLO format ---
def convert_annotation(xml_path, w, h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASSES:
            continue
        cls_id = CLASSES[cls_name]
        bnd = obj.find('bndbox')
        xmin, ymin = int(bnd.find('xmin').text), int(bnd.find('ymin').text)
        xmax, ymax = int(bnd.find('xmax').text), int(bnd.find('ymax').text)
        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        labels.append([cls_id, xc, yc, bw, bh])
    return labels

# --- 4. Augmentation (chỉ train) ---
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- 5. Chia tập ---
image_paths = sorted(glob(f"{INPUT_IMAGE_DIR}/*.jpg"))
random.seed(SEED)
random.shuffle(image_paths)

n = len(image_paths)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)
splits = {
    'train': image_paths[:train_end],
    'val': image_paths[train_end:val_end],
    'test': image_paths[val_end:]
}

# --- 6. Xử lý từng split ---
for split, paths in splits.items():
    for img_path in paths:
        name = os.path.basename(img_path)
        xml_path = os.path.join(INPUT_ANNOTATION_DIR, name.replace('.jpg', '.xml'))
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(xml_path):
            continue
        h, w = img.shape[:2]
        bboxes = convert_annotation(xml_path, w, h)
        if not bboxes:
            continue

        num_augs = 2 if split == 'train' else 1

        for i in range(num_augs):
            if split == 'train' and i > 0:
                class_labels = [b[0] for b in bboxes]
                bbox_only = [b[1:] for b in bboxes]
                transformed = augment(image=img, bboxes=bbox_only, class_labels=class_labels)
                transformed_img = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_labels = transformed['class_labels']
            else:
                transformed_img = img
                transformed_bboxes = [b[1:] for b in bboxes]
                transformed_labels = [b[0] for b in bboxes]

            if not transformed_bboxes:
                continue

            out_name = name.replace('.jpg', f'_{i}.jpg') if i > 0 else name
            out_txt = out_name.replace('.jpg', '.txt')
            cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{out_name}", transformed_img)
            with open(f"{OUTPUT_DIR}/labels/{split}/{out_txt}", 'w') as f:
                for cid, (x, y, w_, h_) in zip(transformed_labels, transformed_bboxes):
                    f.write(f"{cid} {x:.6f} {y:.6f} {w_:.6f} {h_:.6f}\n")

print("Dataset đã được chia và augmented xong.")

