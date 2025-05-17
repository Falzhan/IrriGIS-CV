import torch
import yaml
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from torchvision import transforms
from models.detector import CanalMonitorNet
from models.metrics import calculate_levels
from utils.descriptor import CanalDescriptor

class CanalPredictor:
    def __init__(self, checkpoint_path, config_path='config/config.yaml', max_percentage=50):
        """
        Initialize the Canal Predictor.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            config_path: Path to the configuration file
            max_percentage: The percentage threshold for maximum level (5) rating (default 50%)
        """
        # Store the max percentage threshold for ratings
        self.max_percentage = max_percentage
        print(f"Using {self.max_percentage}% as threshold for maximum level ratings")
        
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
            self.num_classes = self.config.get('model', {}).get('num_classes', 10)
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
            # First, analyze the checkpoint keys
            checkpoint_keys = list(checkpoint['model_state_dict'].keys())
            model_keys = list(self.model.state_dict().keys())
            
            print(f"Checkpoint has {len(checkpoint_keys)} keys")
            print(f"Model has {len(model_keys)} keys")
            
            # Check for "module." prefix (common when training with DataParallel)
            has_module_prefix = any(k.startswith('module.') for k in checkpoint_keys)
            
            # Create a new state dict with matching keys
            new_state_dict = {}
            missing_keys = []
            
            if has_module_prefix:
                print("Detected 'module.' prefix in checkpoint, removing prefix")
                # Remove 'module.' prefix
                for k, v in checkpoint['model_state_dict'].items():
                    new_key = k[7:] if k.startswith('module.') else k
                    new_state_dict[new_key] = v
            else:
                new_state_dict = checkpoint['model_state_dict']
            
            # Check if we need to modify the model keys or checkpoint keys
            if not any(k.startswith('deeplab.') for k in new_state_dict.keys()) and any(k.startswith('deeplab.') for k in model_keys):
                print("Model expects 'deeplab.' prefix but checkpoint doesn't have it. Using strict=False to load partial state")
                # Use strict=False to load whatever matches
                self.model.load_state_dict(new_state_dict, strict=False)
            else:
                # Try to load with strict=False as a fallback
                self.model.load_state_dict(new_state_dict, strict=False)
            
            print(f"Model loaded successfully (trained for {checkpoint.get('epoch', 'unknown')+1} epochs)")
            print("Note: Some keys from the checkpoint may not have been loaded due to architecture mismatch")
            
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
        
        self.descriptor = CanalDescriptor()
    
    def percentage_to_level(self, percentage, level_type):
        """
        Convert percentage to level rating based on the lenient scale and type.
        
        Args:
            percentage: Percentage value (0-100)
            level_type: 'water', 'silt', or 'debris'
            
        Returns:
            Integer level rating from 1-5
        """
        # Scale percentage relative to max_percentage threshold
        scaled_percentage = (percentage / self.max_percentage) * 100
        
        # Handle zero or very low percentages specially
        if percentage < 1.0:
            if level_type == 'water':
                return 1  # Dry
            elif level_type in ['silt', 'debris']:
                return 3  # Normal (no sediment/obstruction)
        
        # Different thresholds per level type
        if level_type == 'water':
            # Water level: 1-Dry, 2-Low, 3-Normal, 4-High, 5-Overflow
            if scaled_percentage < 5:
                return 1  # Dry
            elif scaled_percentage < 25:
                return 2  # Low
            elif scaled_percentage < 60:
                return 3  # Normal
            elif scaled_percentage < 90:
                return 4  # High
            else:
                return 5  # Overflow
        
        elif level_type == 'silt':
            # Silt level: 1-Cleaning, 2-Light, 3-Normal, 4-Dirty, 5-Heavily Silted
            if percentage < 1:
                return 3  # Normal (no sediment)
            elif scaled_percentage < 25:
                return 2  # Light
            elif scaled_percentage < 60:
                return 4  # Dirty
            else:
                return 5  # Heavily Silted
        
        else:  # debris
            # Debris level: 1-Clearing, 2-Light, 3-Normal, 4-Heavy, 5-Blocked
            if percentage < 1:
                return 3  # Normal (no obstruction)
            elif scaled_percentage < 25:
                return 2  # Light
            elif scaled_percentage < 60:
                return 4  # Heavy
            else:
                return 5  # Blocked
    
    def predict(self, image_path):
        """
        Predict canal conditions from an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with results
        """
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
        
        # Calculate condition levels
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
        
        # Convert to levels based on new rating system
        water_level = self.percentage_to_level(water_percent, 'water')
        silt_level = self.percentage_to_level(silt_percent, 'silt')
        debris_level = self.percentage_to_level(debris_percent, 'debris')
        
        levels = {
            'water_percentage': water_percent,
            'water_level': water_level,
            'silt_percentage': silt_percent,
            'silt_level': silt_level,
            'debris_percentage': debris_percent,
            'debris_level': debris_level
        }
        
        # Get descriptions for each level
        water_desc = self._get_level_description('water', water_level)
        silt_desc = self._get_level_description('silt', silt_level)
        debris_desc = self._get_level_description('debris', debris_level)
        
        print(f"Prediction for {os.path.basename(image_path)}:")
        print(f"  Water level: {water_level}/5 ({water_percent:.1f}%) - {water_desc}")
        print(f"  Silt level: {silt_level}/5 ({silt_percent:.1f}%) - {silt_desc}")
        print(f"  Debris level: {debris_level}/5 ({debris_percent:.1f}%) - {debris_desc}")
        
        return {
            'mask': mask,
            'levels': levels,
            'original_image': original_image
        }
    
    def _get_level_description(self, level_type, value):
        """
        Get descriptive text for level values based on the rating scale.
        
        Args:
            level_type: Type of level ('water', 'silt', or 'debris')
            value: Integer level value (1-5)
            
        Returns:
            String description of the level
        """
        # Updated descriptions based on rating scale
        water_desc = {
            1: 'Dry',
            2: 'Low',
            3: 'Normal',
            4: 'High',
            5: 'Overflow'
        }
        
        silt_desc = {
            1: 'Cleaning',
            2: 'Light',
            3: 'Normal',
            4: 'Dirty',
            5: 'Heavily Silted'
        }
        
        debris_desc = {
            1: 'Clearing',
            2: 'Light',
            3: 'Normal',
            4: 'Heavy',
            5: 'Blocked'
        }
        
        # Get the appropriate description based on level type
        if level_type == 'water':
            return water_desc.get(value, 'Normal')
        elif level_type == 'silt':
            return silt_desc.get(value, 'Normal')
        else:  # debris
            return debris_desc.get(value, 'Normal')
            
    def _get_level_color(self, level_type, value):
        """
        Get color for visualization based on level value.
        
        Args:
            level_type: Type of level ('water', 'silt', or 'debris')
            value: Numeric level value
            
        Returns:
            RGB color tuple for visualization
        """
        # Define colors for different levels
        # Format: (R, G, B) where each value is between 0 and 1
        
        # Water level colors: blue scale (dry to overflow)
        water_colors = {
            1: (0.9, 0.9, 0.9),  # Light gray for dry
            2: (0.7, 0.7, 1.0),  # Light blue for low
            3: (0.0, 0.5, 1.0),  # Medium blue for normal
            4: (0.0, 0.0, 0.8),  # Deep blue for high
            5: (0.5, 0.0, 0.5)   # Purple for overflow
        }
        
        # Silt level colors: brown scale (light to heavily silted)
        silt_colors = {
            1: (0.8, 0.8, 0.3),  # Yellow for cleaning
            2: (0.9, 0.8, 0.6),  # Light tan for light
            3: (0.8, 0.7, 0.5),  # Medium tan for normal
            4: (0.6, 0.5, 0.4),  # Dark tan for dirty
            5: (0.4, 0.3, 0.1)   # Brown for heavily silted
        }
        
        # Debris level colors: green to red scale (light to blocked)
        debris_colors = {
            1: (0.5, 0.9, 0.5),  # Green for clearing
            2: (0.7, 1.0, 0.7),  # Light green for light
            3: (1.0, 1.0, 0.5),  # Yellow for normal
            4: (1.0, 0.6, 0.4),  # Orange for heavy
            5: (0.8, 0.2, 0.2)   # Red for blocked
        }
        
        # Return appropriate color based on level type
        if level_type == 'water':
            return water_colors.get(value, water_colors[3])  # Default to normal
        elif level_type == 'silt':
            return silt_colors.get(value, silt_colors[3])    # Default to normal
        else:  # debris
            return debris_colors.get(value, debris_colors[3])  # Default to normal

    def visualize(self, result, save_path=None, enhance_water_line=True):
        """
        Visualize segmentation results with metrics display.
        
        Args:
            result: Dictionary containing segmentation results
            save_path: Path to save the visualization
            enhance_water_line: Whether to enhance water line visualization
        """
        # Create a figure with 2 rows, 2 columns
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])
        
        # Plot original image
        ax_orig = plt.subplot(gs[0, 0])
        ax_orig.imshow(result['original_image'])
        ax_orig.set_title('Original Image', fontsize=14)
        ax_orig.axis('off')
        
        # Create a colormap for the mask with distinct colors for each category
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_classes))
        cmap = plt.cm.colors.ListedColormap(colors)
        
        # Plot segmentation mask
        ax_mask = plt.subplot(gs[0, 1])
        im = ax_mask.imshow(result['mask'], cmap=cmap, vmin=0, vmax=self.num_classes-1)
        ax_mask.set_title('Segmentation Mask', fontsize=14)
        ax_mask.axis('off')
        
        # Add colorbar with category names
        cbar_ax = fig.add_axes([0.93, 0.55, 0.02, 0.3])
        cbar = plt.colorbar(im, cax=cbar_ax, ticks=np.arange(min(self.num_classes, len(self.category_names))))
        display_categories = self.category_names[:self.num_classes]
        cbar.set_ticklabels(display_categories)
        
        # Create histogram of class distribution
        ax_hist = plt.subplot(gs[1, 1])
        unique, counts = np.unique(result['mask'], return_counts=True)
        
        # Create a dictionary mapping class indices to counts
        class_counts = {i: 0 for i in range(self.num_classes)}
        for idx, count in zip(unique, counts):
            if idx < self.num_classes:
                class_counts[idx] = count
        
        # Convert to array for plotting
        indices = np.arange(self.num_classes)
        count_data = np.zeros(self.num_classes)
        for idx, count in class_counts.items():
            count_data[idx] = count
        
        # Plot horizontal bars for class distribution    
        bars = ax_hist.barh(indices, count_data, color=colors)
        ax_hist.set_yticks(indices)
        ax_hist.set_yticklabels(display_categories, fontsize=8)
        ax_hist.set_title('Class Distribution (Pixel Count)', fontsize=12)
        
        # Add text labels with pixel counts
        for i, v in enumerate(count_data):
            if v > 0:  # Only show labels for classes that are present
                percentage = (v / np.sum(count_data)) * 100
                ax_hist.text(v + 0.1 * np.max(count_data), i, f"{int(v)} ({percentage:.1f}%)", 
                            va='center', fontsize=8)
        
        # Create metrics visualization as a simple table
        ax_metrics = plt.subplot(gs[1, 0])
        ax_metrics.axis('off')
        
        # Get levels information
        levels = result.get('levels', {})
        
        if levels:
            # Get descriptions for each level
            water_desc = self._get_level_description('water', levels.get('water_level', 3))
            silt_desc = self._get_level_description('silt', levels.get('silt_level', 3))
            debris_desc = self._get_level_description('debris', levels.get('debris_level', 3))
            
            # Create a simple table for metrics with descriptions
            metrics_data = [
                ["Water Level", f"{levels.get('water_level', 3)}/5", f"{levels.get('water_percentage', 0):.1f}%", water_desc],
                ["Silt Level", f"{levels.get('silt_level', 3)}/5", f"{levels.get('silt_percentage', 0):.1f}%", silt_desc],
                ["Debris Level", f"{levels.get('debris_level', 3)}/5", f"{levels.get('debris_percentage', 0):.1f}%", debris_desc]
            ]
            
            # Create table
            table = ax_metrics.table(
                cellText=metrics_data,
                colLabels=["Metric", "Level", "Percentage", "Description"],
                loc='center',
                cellLoc='center',
                colWidths=[0.2, 0.15, 0.15, 0.3]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(14)
            table.scale(1, 2)  # Make cells taller
            
            # Add title
            ax_metrics.set_title('Canal Condition Metrics', fontsize=16, pad=20)
            
            # Add note about threshold
            threshold_note = f"* Maximum level (5) threshold: {self.max_percentage}%"
            plt.figtext(0.02, 0.12, threshold_note, fontsize=10, style='italic')
            
            # Get remarks
            remark = self.descriptor.get_remark(
                levels['water_level'],
                levels['silt_level'],
                levels['debris_level']
            )
            short_remark = self.descriptor.get_short_remark(
                levels['water_level'],
                levels['silt_level'],
                levels['debris_level']
            )
            
            # Add remarks to the visualization
            plt.figtext(0.02, 0.02, remark, 
                       bbox=dict(facecolor='white', alpha=0.8, 
                               edgecolor='red' if "CRITICAL" in remark else 'black'))
            
            # Add short remark at the top
            plt.figtext(0.5, 0.98, short_remark, 
                       ha='center', va='top',
                       bbox=dict(facecolor='yellow', alpha=0.5),
                       fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
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
    # Default settings
    checkpoint_path = 'models/checkpoints_20250506_112241/best_model.pth'
    max_percentage_threshold = 50 
    
    # Create predictor with configurable threshold
    predictor = CanalPredictor(
        checkpoint_path=checkpoint_path,
        max_percentage=max_percentage_threshold
    )
    
    # Process a single image
    image_path = 'data/raw/train/Canal-1.jpg'  
    result = predictor.predict(image_path)
    predictor.visualize(result)
    
    # Uncomment to process all images in a directory
    # batch_predict(predictor, 'data/raw/val', 'results')