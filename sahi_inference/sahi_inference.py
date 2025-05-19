import os
import json
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from data_preparation import *


# Cấu hình
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  

test_images_dir = os.path.join(ROOT_DIR, "yolo_dataset", "test")
model_path = os.path.join(ROOT_DIR, "yolov8n_insects\weights", "best.pt")
gt_coco_path = os.path.join(ROOT_DIR, "sahi_inference", "instances_test_gt.json")
output_json_path = os.path.join(ROOT_DIR, "sahi_inference", "sahi_results.json")

# SAHI slicing & postprocessing
slice_height = 128
slice_width = 128
overlap_height_ratio = 0.35
overlap_width_ratio = 0.35
perform_standard_pred = True
postprocess_type = "NMS"
postprocess_match_metric = "IOU"
postprocess_match_threshold = 0.45
postprocess_class_agnostic = False
verbose = 1
auto_slice_resolution = False
confidence_threshold = 0.5
device = "cuda"  # hoặc "cpu" nếu máy bạn không có GPU

# Load ground truth COCO
with open(gt_coco_path, 'r') as f:
    gt_coco = json.load(f)

filename_to_id = {img['file_name']: img['id'] for img in gt_coco['images']}
valid_category_ids = {cat['id'] for cat in gt_coco['categories']}
id2name = {cat['id']: cat['name'] for cat in gt_coco['categories']}  # ID → tên nhãn

# Load YOLOv8 model với SAHI wrapper
yolo_model = YOLO(model_path)
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model=yolo_model,
    confidence_threshold=confidence_threshold,
    device=device,
)

# Inference & lưu kết quả
coco_predictions = []

image_filenames = sorted([
    f for f in os.listdir(test_images_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for img_name in image_filenames:
    image_path = os.path.join(test_images_dir, img_name)
    print(f"Processing: {img_name}")

    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=perform_standard_pred,
        postprocess_type=postprocess_type,
        postprocess_match_metric=postprocess_match_metric,
        postprocess_match_threshold=postprocess_match_threshold,
        postprocess_class_agnostic=postprocess_class_agnostic,
        verbose=verbose,
        auto_slice_resolution=auto_slice_resolution,
    )

    image_id = filename_to_id.get(img_name)
    if image_id is None:
        print(f"Không tìm thấy image_id cho {img_name}, bỏ qua.")
        continue

    for obj_pred in result.object_prediction_list:
        category_id = obj_pred.category.id
        category_name = obj_pred.category.name
        if category_id not in valid_category_ids:
            print(f"category_id {category_id} không hợp lệ, bỏ qua.")
            continue

        x1, y1, x2, y2 = obj_pred.bbox.to_xyxy()
        bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO format

        coco_predictions.append({
            "image_id": image_id,
            "category_id": category_id,
            "category_name": category_name,
            "bbox": [float(round(v, 2)) for v in bbox],
            "score": float(round(obj_pred.score.value, 4))
        })

# Ghi file JSON kết quả
with open(output_json_path, "w") as f:
    json.dump(coco_predictions, f, indent=2)

print(f"\nSAHI inference hoàn tất. Kết quả với nhãn lưu tại: {output_json_path}")
