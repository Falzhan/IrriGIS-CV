import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms
import matplotlib.pyplot as plt

class CanalDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None, mask_size=(224, 224)):
        self.img_dir = img_dir
        self.mask_size = mask_size
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        self.annotations_by_image = self._organize_annotations()
        
        # Get category information
        self.categories = {cat['id']: cat['name'] for cat in self.annotations.get('categories', [])}
        print(f"Categories: {self.categories}")
        
        # Define image transformations
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to a standard size
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
        
        # Print debugging info about the first few images
        print(f"Image directory: {self.img_dir}")
        print(f"Found {len(self.images)} images in annotations")
        print(f"Found {len(self.annotations.get('annotations', []))} total annotations")
        
        valid_images = []
        for img in self.images:
            file_name = os.path.basename(img['file_name'])
            img_path = os.path.join(self.img_dir, file_name)
            if os.path.exists(img_path):
                valid_images.append(img)
            else:
                print(f"Warning: Image {img_path} not found, skipping")
        
        self.images = valid_images
        print(f"Found {len(self.images)} valid images after checking paths")
        
        # Print a few examples
        for i in range(min(3, len(self.images))):
            img_id = self.images[i]['id']
            file_name = os.path.basename(self.images[i]['file_name'])
            img_path = os.path.join(self.img_dir, file_name)
            ann_count = len(self.annotations_by_image.get(img_id, []))
            print(f"Image {i}: {img_path}, ID: {img_id}, has {ann_count} annotations")
    
    def _organize_annotations(self):
        """Organize annotations by image ID for easier access"""
        annotations_by_image = {}
        for ann in self.annotations.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        return annotations_by_image
    
    def _create_segmentation_mask(self, image_id, image_width, image_height):
        """Create a segmentation mask for an image based on annotations"""
        # Create an empty mask filled with 0 (background)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # Get annotations for this image
        anns = self.annotations_by_image.get(image_id, [])
        
        # Sort annotations by area (smaller on top) if 'area' is available
        # This ensures that larger objects don't completely cover smaller ones
        if anns and 'area' in anns[0]:
            anns = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)
        
        # Process each annotation
        for ann in anns:
            # Get category ID (add 1 because 0 is background)
            category_id = ann.get('category_id', 0)
            
            # Process segmentation data
            segmentation = ann.get('segmentation', [])
            
            # Segmentation could be in RLE format or polygon format
            if isinstance(segmentation, dict):  # RLE format
                # You'd need to implement RLE decoding here
                pass
            elif isinstance(segmentation, list):  # Polygon format
                # Create a temporary mask for this annotation
                temp_mask = Image.new('L', (image_width, image_height), 0)
                draw = ImageDraw.Draw(temp_mask)
                
                for polygon in segmentation:
                    if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                        # Convert flat list [x1,y1,x2,y2,...] to [(x1,y1), (x2,y2), ...]
                        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                        draw.polygon(points, fill=category_id)
                
                # Update the main mask
                temp_mask_np = np.array(temp_mask)
                mask = np.maximum(mask, temp_mask_np)
            
            # Handle 'bbox' format if 'segmentation' is not available
            elif 'bbox' in ann:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = [int(v) for v in bbox]
                mask[y:y+h, x:x+w] = category_id
        
        return mask
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        image_id = img_info['id']
        img_width = img_info.get('width', 0)
        img_height = img_info.get('height', 0)
        
        # Clean up the file name - extract just the base filename without any path components
        file_name = os.path.basename(img_info['file_name'])
        
        # Construct a clean path
        img_path = os.path.join(self.img_dir, file_name)
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # If we didn't get dimensions from annotations, get them from the image
            if img_width == 0 or img_height == 0:
                img_width, img_height = image.size
            
            # Create segmentation mask
            mask_array = self._create_segmentation_mask(image_id, img_width, img_height)
            mask_pil = Image.fromarray(mask_array)
            
            # Resize mask to match the target size
            mask_pil = mask_pil.resize(self.mask_size, Image.NEAREST)
            
            # Apply transformations to image
            image_tensor = self.transform(image)
            
            # Convert mask to tensor
            mask_tensor = torch.tensor(np.array(mask_pil), dtype=torch.long)
            
            return image_tensor, mask_tensor
            
        except FileNotFoundError:
            print(f"Error: Could not find file at {img_path}")
            print(f"Original file_name in annotation: {img_info['file_name']}")
            raise
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            raise

    def visualize_sample(self, idx):
        """Visualize an image and its mask for debugging"""
        image_tensor, mask_tensor = self[idx]
        
        # Convert tensor to numpy for visualization
        image_np = image_tensor.numpy().transpose(1, 2, 0)
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        mask_np = mask_tensor.numpy()
        
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot image
        ax1.imshow(image_np)
        ax1.set_title('Image')
        ax1.axis('off')
        
        # Plot mask with a colormap
        # Use a colormap with enough colors for all categories
        num_classes = len(self.categories) + 1  # +1 for background
        cmap = plt.cm.get_cmap('tab20', num_classes)
        
        im = ax2.imshow(mask_np, cmap=cmap, vmin=0, vmax=num_classes-1)
        ax2.set_title('Segmentation Mask')
        ax2.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        # Add class names to the colorbar if available
        if hasattr(self, 'categories'):
            # Create tick labels with class 0 as background
            tick_labels = ['Background']
            for i in range(1, num_classes):
                tick_labels.append(self.categories.get(i, f'Class {i}'))
            cbar.set_ticks(np.arange(num_classes) + 0.5)
            cbar.set_ticklabels(tick_labels[:num_classes])
        
        plt.tight_layout()
        plt.show()

def get_dataloader(img_dir, annotation_file, batch_size=4, num_workers=4):
    """Create a DataLoader with the CanalDataset"""
    # Make sure img_dir exists
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Make sure the annotation file exists
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
    # List the first few files in the image directory for debugging
    print(f"Contents of {img_dir}:")
    files = os.listdir(img_dir)
    for i, file in enumerate(files):
        if i >= 5:  # Print up to 5 files
            break
        print(f"  - {file}")
    print(f"Total files in directory: {len(files)}")
        
    dataset = CanalDataset(img_dir, annotation_file)
    
    # Create a DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

def test_dataset(img_dir, annotation_file):
    """Test the dataset by visualizing a few samples"""
    dataset = CanalDataset(img_dir, annotation_file)
    
    # Visualize a few samples
    num_samples = min(3, len(dataset))
    for i in range(num_samples):
        print(f"Visualizing sample {i}")
        dataset.visualize_sample(i)

    return dataset

if __name__ == "__main__":
    # Test the dataset - replace with your actual paths
    img_dir = "data/raw/train"
    annotation_file = "data/annotations/train.json"
    
    test_dataset(img_dir, annotation_file)