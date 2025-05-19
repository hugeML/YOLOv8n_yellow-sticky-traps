from data_preparation import * 

# Load model YOLOv8n (pretrained)
model = YOLO('yolov8n.pt')

# Train
model.train(
    data='data.yaml',
    epochs=250,
    imgsz=640,
    batch=32,
    device='cuda',              # hoặc 'cpu' nếu không có GPU
    project='main',             # thư mục lưu kết quả huấn luyện
    name='yolov8n_insects',
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.0,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    mosaic=1.0,
    mixup=0.0,
    patience=50,
    close_mosaic=15
)
# Load mô hình tốt nhất sau khi train
best_model = YOLO('main/yolov8n_insects/weights/best.pt')

# Đánh giá trên tập test
metrics = best_model.val(
    data='data.yaml',
    split='test',
    save=True,
    save_json=True
)

print("Evaluation hoàn tất!")
print(metrics)