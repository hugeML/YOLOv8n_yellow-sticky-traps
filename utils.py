import os
import yaml

def check_dataset_structure(data_yaml_path):
    """Check if dataset structure matches YAML file"""
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)
    
    required_dirs = ['train', 'val', 'test']
    for split in required_dirs:
        img_dir = data.get(split, '')
        label_dir = img_dir.replace('images', 'labels')
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        
        print(f"✅ Found {split} set: {img_dir}")
    
    print(f"Classes: {data['names']}")
    return True

def count_images(data_yaml_path):
    """Count images in each split"""
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)
    
    counts = {}
    for split in ['train', 'val', 'test']:
        img_dir = data.get(split, '')
        if img_dir and os.path.exists(img_dir):
            counts[split] = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    print("Image counts:")
    for split, count in counts.items():
        print(f"{split}: {count}")
    
    return counts