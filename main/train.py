from ultralytics import YOLO
import argparse

def train_model(config):
    # Load a model
    model = YOLO(config['model_type'] + '.pt')  # load a pretrained model
    
    # Train the model
    results = model.train(
        data=config['data_yaml'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        name=config['name'],
        device=config['device']
    )
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--data', type=str, default='data/yolo_dataset/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=1024, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--model', type=str, default='yolov8n', help='Model type (yolov8n, yolov8s, etc.)')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--device', type=str, default='0', help='Device to use (e.g., 0 for GPU or "cpu")')
    
    args = parser.parse_args()
    
    config = {
        'data_yaml': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'model_type': args.model,
        'name': args.name,
        'device': args.device
    }
    
    train_model(config)