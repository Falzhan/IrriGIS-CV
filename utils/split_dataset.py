import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(
    main_dir='data/raw/main',
    train_dir='data/raw/train',
    val_dir='data/raw/val',
    annotations_file='data/annotations/instances_default.json',
    val_size=0.2,
    random_state=42
):
    """Split dataset into training and validation sets."""
    
    print("Starting dataset split...")
    
    # Convert paths to absolute
    base_dir = Path(__file__).parent.parent
    main_dir = base_dir / main_dir
    train_dir = base_dir / train_dir
    val_dir = base_dir / val_dir
    annotations_file = base_dir / annotations_file
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Split images
    images = data['images']
    train_imgs, val_imgs = train_test_split(
        images, 
        test_size=val_size, 
        random_state=random_state
    )
    
    # Create new annotation files
    train_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': train_imgs,
        'annotations': [
            ann for ann in data['annotations'] 
            if ann['image_id'] in [img['id'] for img in train_imgs]
        ]
    }
    
    val_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': val_imgs,
        'annotations': [
            ann for ann in data['annotations'] 
            if ann['image_id'] in [img['id'] for img in val_imgs]
        ]
    }
    
    # Copy images to respective directories
    for img in train_imgs:
        src = main_dir / img['file_name']
        dst = train_dir / img['file_name']
        if src.exists():
            shutil.copy2(src, dst)
    
    for img in val_imgs:
        src = main_dir / img['file_name']
        dst = val_dir / img['file_name']
        if src.exists():
            shutil.copy2(src, dst)
    
    # Save split annotations
    with open(base_dir / 'data/annotations/instances_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(base_dir / 'data/annotations/instances_val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Split complete!")
    print(f"Training images: {len(train_imgs)}")
    print(f"Validation images: {len(val_imgs)}")
    print(f"Files saved to {train_dir} and {val_dir}")

if __name__ == '__main__':
    split_dataset()