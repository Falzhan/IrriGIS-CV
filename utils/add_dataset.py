##Bash
## python add_dataset.py --images "/path/to/new/dataset/images" --annotations "/path/to/new/dataset/annotations.json"



import json
import os
import shutil
from pathlib import Path

def add_dataset(
    new_dataset_path,
    new_annotation_file,
    main_dir='data/raw/main',
    annotations_file='data/annotations/instances_default.json',
    update_splits=True
):
    """
    Add a new dataset (images and annotations) to your existing project.
    
    Args:
        new_dataset_path (str): Path to the folder containing new images
        new_annotation_file (str): Path to the new annotation file (COCO format)
        main_dir (str): Main directory for all images in your project
        annotations_file (str): Path to your main annotation file
        update_splits (bool): Whether to update train/val splits after merging
    """
    print(f"Adding new dataset from {new_dataset_path} with annotations {new_annotation_file}")
    
    # Ensure main directory exists
    os.makedirs(main_dir, exist_ok=True)
    
    # Load existing annotations
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            existing_data = json.load(f)
            print(f"Loaded existing annotations with {len(existing_data['images'])} images")
    else:
        print(f"No existing annotations found at {annotations_file}, creating new")
        existing_data = {
            'info': {'description': 'Canal Monitoring Dataset'},
            'licenses': [],
            'categories': [{'id': 1, 'name': 'canal'}],
            'images': [],
            'annotations': []
        }
    
    # Load new annotations
    with open(new_annotation_file, 'r') as f:
        new_data = json.load(f)
        print(f"Loaded new annotations with {len(new_data['images'])} images")
    
    # Find maximum IDs in existing data to ensure uniqueness
    max_image_id = 0
    max_annotation_id = 0
    
    if existing_data['images']:
        max_image_id = max(img['id'] for img in existing_data['images'])
    
    if existing_data['annotations']:
        max_annotation_id = max(ann['id'] for ann in existing_data['annotations'])
    
    print(f"Maximum existing image ID: {max_image_id}")
    print(f"Maximum existing annotation ID: {max_annotation_id}")
    
    # Process and merge categories (if needed)
    category_mapping = {}
    
    # Check if categories need to be merged
    if 'categories' in new_data:
        existing_categories = {cat['name']: cat['id'] for cat in existing_data['categories']}
        
        for new_cat in new_data['categories']:
            if new_cat['name'] in existing_categories:
                # Map the new category ID to the existing one
                category_mapping[new_cat['id']] = existing_categories[new_cat['name']]
            else:
                # Add new category with an incremented ID
                new_id = max(existing_categories.values()) + 1
                existing_data['categories'].append({
                    'id': new_id, 
                    'name': new_cat['name']
                })
                category_mapping[new_cat['id']] = new_id
                existing_categories[new_cat['name']] = new_id
    
    # Copy images and update annotations
    added_images = 0
    added_annotations = 0
    
    # Process images
    for img in new_data['images']:
        # Update image ID to avoid conflicts
        old_img_id = img['id']
        img['id'] = max_image_id + 1
        max_image_id += 1
        
        # Fix file path to be just the filename
        orig_filename = img['file_name']
        img['file_name'] = os.path.basename(orig_filename)
        
        # Copy the image file to main directory
        src_path = os.path.join(new_dataset_path, os.path.basename(orig_filename))
        dst_path = os.path.join(main_dir, img['file_name'])
        
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")
            else:
                print(f"File already exists: {dst_path}")
            added_images += 1
        else:
            print(f"Warning: Source image not found: {src_path}")
            continue
        
        # Add the updated image info to existing dataset
        existing_data['images'].append(img)
        
        # Update corresponding annotations
        for ann in new_data['annotations']:
            if ann['image_id'] == old_img_id:
                # Update annotation ID and image ID
                ann['id'] = max_annotation_id + 1
                max_annotation_id += 1
                ann['image_id'] = img['id']
                
                # Update category ID if needed
                if ann['category_id'] in category_mapping:
                    ann['category_id'] = category_mapping[ann['category_id']]
                
                # Add to existing annotations
                existing_data['annotations'].append(ann)
                added_annotations += 1
    
    # Save updated annotations
    print(f"Saving updated annotations with {len(existing_data['images'])} images")
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
    
    with open(annotations_file, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"Added {added_images} images and {added_annotations} annotations")
    
    # Update train/val splits if requested
    if update_splits:
        try:
            from split_dataset import split_dataset
            print("Updating train/val splits...")
            split_dataset()
            print("Train/val splits updated")
        except ImportError:
            print("Warning: Could not import split_dataset function, splits not updated")
    
    print("Dataset addition complete!")

if __name__ == '__main__':
    # Example usage:
    # add_dataset('path/to/new/dataset', 'path/to/new/annotations.json')
    
    # Comment out the line below and replace with your actual paths
    # add_dataset('new_data/images', 'new_data/annotations.json')
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Add a new dataset to your project')
    parser.add_argument('--images', required=True, help='Path to folder containing new images')
    parser.add_argument('--annotations', required=True, help='Path to new annotation file')
    parser.add_argument('--main-dir', default='data/raw/main', help='Main directory for all images')
    parser.add_argument('--annotations-file', default='data/annotations/instances_default.json', 
                        help='Path to main annotation file')
    parser.add_argument('--no-update-splits', action='store_false', dest='update_splits',
                        help='Do not update train/val splits after merging')
    
    args = parser.parse_args()
    
    add_dataset(
        args.images, 
        args.annotations, 
        args.main_dir, 
        args.annotations_file,
        args.update_splits
    )