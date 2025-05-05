import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms

class CanalDataset(Dataset):
    def __init__(self, img_dir, annotation_file):
        self.img_dir = img_dir
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        
        # Define transformations for images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to a standard size
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
        
        # Print debugging info about the first few images
        print(f"Image directory: {self.img_dir}")
        for i in range(min(3, len(self.images))):
            print(f"Image {i} filename: {self.images[i]['file_name']}")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        
        # Clean up the file name - extract just the base filename without any path components
        file_name = os.path.basename(img_info['file_name'])
        
        # Construct a clean path
        img_path = os.path.join(self.img_dir, file_name)
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            # Apply transformations
            image_tensor = self.transform(image)
        except FileNotFoundError:
            print(f"Error: Could not find file at {img_path}")
            print(f"Original file_name in annotation: {img_info['file_name']}")
            raise
        
        # Create a dummy mask tensor of the same size as the image
        # This is a placeholder - you would normally load your real masks here
        H, W = 224, 224  # Using the same size as the resized image
        mask = torch.zeros((H, W), dtype=torch.long)
        
        return image_tensor, mask

def get_dataloader(img_dir, annotation_file, batch_size=4):
    # Make sure img_dir exists
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Make sure the annotation file exists
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
    # List the first few files in the image directory for debugging
    print(f"Contents of {img_dir}:")
    for i, file in enumerate(os.listdir(img_dir)):
        if i >= 5:  # Print up to 5 files
            break
        print(f"  - {file}")
        
    dataset = CanalDataset(img_dir, annotation_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)