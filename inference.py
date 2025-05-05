import torch
import yaml
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from models.detector import CanalMonitorNet
from models.metrics import calculate_levels

class CanalPredictor:
    def __init__(self, checkpoint_path, config_path='config/config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Analyze checkpoint to determine the number of classes
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Look for the classifier layer to determine number of classes
        classifier_weights = None
        for key, value in checkpoint['model_state_dict'].items():
            if 'classifier.3.weight' in key:
                classifier_weights = value
                break
        
        if classifier_weights is not None:
            self.num_classes = classifier_weights.shape[0]
            print(f"Detected {self.num_classes} classes in checkpoint")
        else:
            # Fallback to config or default
            self.num_classes = self.config.get('model', {}).get('num_classes', 12)
            print(f"Using {self.num_classes} classes from config")
            
        self.model = CanalMonitorNet(num_classes=self.num_classes)
        
        # Define category names (including background as class 0)
        self.category_names = [
            "Background",  # Assume class 0 is background
            "Water_Surface", 
            "Water_Line", 
            "Dry_Canal_Bed", 
            "Silt_Deposit", 
            "Water_Discoloration", 
            "Floating_Debris", 
            "Vegetation", 
            "Blockage", 
            "Canal", 
            "Canal_Bank", 
            "Side_Slope"
        ]
        
        # Ensure we have enough category names
        if len(self.category_names) < self.num_classes:
            for i in range(len(self.category_names), self.num_classes):
                self.category_names.append(f"Class_{i}")
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
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
        original_image = image.copy()  # Save original for visualization
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get segmentation mask
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Debug info - print unique classes found in the mask
        unique_classes = np.unique(mask)
        print(f"Unique classes in prediction: {unique_classes}")
        
        # Calculate condition levels (simplified for now)
        levels = {
            'water_level': 0,
            'water_percentage': 0,
            'silt_level': 0,
            'silt_percentage': 0,
            'debris_level': 0,
            'debris_percentage': 0
        }
        
        # Very simple calculation based on pixel counts
        # You'll want to replace this with your more sophisticated calculate_levels function
        total_pixels = mask.size
        
        # Map class indices to condition categories
        water_classes = [1]  # Assuming Water_Surface is class 1
        silt_classes = [4]   # Assuming Silt_Deposit is class 4
        debris_classes = [6] # Assuming Floating_Debris is class 6
        
        # Count pixels of each condition
        water_pixels = np.sum(np.isin(mask, water_classes))
        silt_pixels = np.sum(np.isin(mask, silt_classes))
        debris_pixels = np.sum(np.isin(mask, debris_classes))
        
        # Calculate percentages
        water_percent = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        silt_percent = (silt_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        debris_percent = (debris_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Convert to levels (0-5 scale)
        levels['water_percentage'] = water_percent
        levels['water_level'] = min(5.0, (water_percent / 20))
        
        levels['silt_percentage'] = silt_percent
        levels['silt_level'] = min(5.0, (silt_percent / 20))
        
        levels['debris_percentage'] = debris_percent
        levels['debris_level'] = min(5.0, (debris_percent / 20))
        
        print(f"Prediction for {os.path.basename(image_path)}:")
        print(f"  Water level: {levels['water_level']:.2f}/5.00 ({levels['water_percentage']:.1f}%)")
        print(f"  Silt level: {levels['silt_level']:.2f}/5.00 ({levels['silt_percentage']:.1f}%)")
        print(f"  Debris level: {levels['debris_level']:.2f}/5.00 ({levels['debris_percentage']:.1f}%)")
        
        return {
            'mask': mask,
            'levels': levels,
            'original_image': original_image
        }
    
    def visualize(self, result, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Plot original image
        ax1.imshow(result['original_image'])
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Create a colormap for the mask with distinct colors for each category
        # Generate a colormap with distinct colors for each class
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_classes))
        cmap = plt.cm.colors.ListedColormap(colors)
        
        # Plot segmentation mask
        im = ax2.imshow(result['mask'], cmap=cmap, vmin=0, vmax=self.num_classes-1)
        ax2.set_title('Segmentation Mask')
        ax2.axis('off')
        
        # Add colorbar with category names
        cbar = plt.colorbar(im, ax=ax2, ticks=np.arange(min(self.num_classes, len(self.category_names))))
        # Only use as many category names as we have classes (up to the length of category_names)
        display_categories = self.category_names[:self.num_classes]
        cbar.set_ticklabels(display_categories)
        
        # Add levels information
        levels = result['levels']
        if levels:
            info_text = (
                f"Water Level: {levels['water_level']:.1f}/5.0 ({levels['water_percentage']:.1f}%)\n"
                f"Silt Level: {levels['silt_level']:.1f}/5.0 ({levels['silt_percentage']:.1f}%)\n"
                f"Debris Level: {levels['debris_level']:.1f}/5.0 ({levels['debris_percentage']:.1f}%)"
            )
            
            plt.figtext(0.02, 0.02, info_text, 
                      bbox=dict(facecolor='white', alpha=0.8))
        
        # Add histogram to show distribution of classes
        ax_hist = fig.add_axes([0.92, 0.1, 0.03, 0.8])
        unique, counts = np.unique(result['mask'], return_counts=True)
        
        # Create a dictionary mapping class indices to counts
        class_counts = {i: 0 for i in range(self.num_classes)}
        for idx, count in zip(unique, counts):
            if idx < self.num_classes:
                class_counts[idx] = count
        
        # Convert to array for plotting
        count_data = np.zeros(self.num_classes)
        for idx, count in class_counts.items():
            count_data[idx] = count
            
        ax_hist.barh(np.arange(self.num_classes), count_data, color=colors)
        ax_hist.set_yticks(np.arange(self.num_classes))
        ax_hist.set_yticklabels([])
        ax_hist.set_title('Pixel Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
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

def analyze_checkpoint(checkpoint_path):
    """Analyze the checkpoint to understand its structure and parameters"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("\nCheckpoint Analysis:")
        print(f"Epochs trained: {checkpoint.get('epoch', 'N/A')}")
        
        if 'model_state_dict' in checkpoint:
            model_dict = checkpoint['model_state_dict']
            
            # Look at output layer to determine number of classes
            output_layer_keys = [k for k in model_dict.keys() if 'classifier.3.weight' in k]
            if output_layer_keys:
                out_layer = model_dict[output_layer_keys[0]]
                num_classes = out_layer.shape[0]
                print(f"Number of output classes in model: {num_classes}")
                
                # Print layer shapes for deeper analysis
                for key in output_layer_keys:
                    print(f"Output layer shape: {model_dict[key].shape}")
            
            # Print model structure summary
            print(f"Total layers: {len(model_dict.keys())}")
            
        if 'optimizer_state_dict' in checkpoint:
            print("Optimizer state is present in the checkpoint")
            
        if 'best_metric' in checkpoint:
            print(f"Best metric achieved: {checkpoint.get('best_metric')}")
            
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")

if __name__ == '__main__':
    checkpoint_path = 'models/checkpoints_20250504_023838/final_model.pth'
##    checkpoint_path = 'models/checkpoint_epoch_50.pth'
    
    # First analyze the checkpoint
    print("Analyzing checkpoint...")
    analyze_checkpoint(checkpoint_path)
    
    # Initialize predictor with checkpoint
    predictor = CanalPredictor(
        checkpoint_path=checkpoint_path
    )
    
    # Process a single image
    image_path = 'data/raw/train/665a96ccb8993.image.jpg'  # Replace with your image path
    result = predictor.predict(image_path)
    predictor.visualize(result)
    
    # Uncomment to process all images in a directory
    # batch_predict(predictor, 'data/raw/val', 'results')

    ## python inference.py