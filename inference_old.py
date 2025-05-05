import torch
import yaml
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from models.detector import CanalMonitorNet
from models.metrics_old import calculate_levels

class CanalPredictor:
    def __init__(self, checkpoint_path, config_path='config/config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = CanalMonitorNet(num_classes=self.config['model']['num_classes'])
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully (trained for {checkpoint['epoch']+1} epochs)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get segmentation mask
        mask = torch.argmax(output, dim=1).squeeze(0).cpu()
        
        # Calculate condition levels
        levels = calculate_levels(mask, self.config['model']['num_classes'])
        
        print(f"Prediction for {os.path.basename(image_path)}:")
        print(f"  Water level: {levels['water_level']:.2f}/5.00 ({levels['water_percentage']:.1f}%)")
        print(f"  Silt level: {levels['silt_level']:.2f}/5.00 ({levels['silt_percentage']:.1f}%)")
        print(f"  Debris level: {levels['debris_level']:.2f}/5.00 ({levels['debris_percentage']:.1f}%)")
        
        return {
            'mask': mask,
            'levels': levels,
            'original_image': image
        }
    
    def visualize(self, result, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image
        ax1.imshow(result['original_image'])
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Create a colormap for the mask
        # Define custom colormap for different classes
        # Assuming: 0=background, 1=water, 2=silt, 3=debris
        cmap = plt.cm.colors.ListedColormap(['black', 'blue', 'brown', 'red'])
        bounds = [0, 1, 2, 3, 4]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot segmentation mask
        im = ax2.imshow(result['mask'], cmap=cmap, norm=norm)
        ax2.set_title('Segmentation Mask')
        ax2.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.set_ticklabels(['Background', 'Water', 'Silt', 'Debris'])
        
        # Add levels information
        levels = result['levels']
        info_text = (
            f"Water Level: {levels['water_level']:.1f}/5.0 ({levels['water_percentage']:.1f}%)\n"
            f"Silt Level: {levels['silt_level']:.1f}/5.0 ({levels['silt_percentage']:.1f}%)\n"
            f"Debris Level: {levels['debris_level']:.1f}/5.0 ({levels['debris_percentage']:.1f}%)"
        )
        
        plt.figtext(0.02, 0.02, info_text, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        
        plt.show()

def batch_predict(predictor, image_dir, output_dir='results'):
    """Process all images in a directory and save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.splitext(f.lower())[1] in valid_extensions]
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"Processing {img_file}...")
        
        try:
            # Predict
            result = predictor.predict(img_path)
            
            # Save visualization
            base_name = os.path.splitext(img_file)[0]
            save_path = os.path.join(output_dir, f"{base_name}_result.png")
            predictor.visualize(result, save_path=save_path)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

if __name__ == '__main__':
    # Initialize predictor with latest checkpoint
    predictor = CanalPredictor(
        checkpoint_path='models/checkpoint_epoch_50.pth'
    )
    
    # Process a single image
    image_path = 'data/raw/train/IMG_3202.jpg'  # Replace with your image path
    result = predictor.predict(image_path)
    predictor.visualize(result)
    
    # Uncomment to process all images in a directory
    # batch_predict(predictor, 'data/raw/val', 'results')

    ## python inference.py