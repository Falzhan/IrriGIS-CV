import torch
import torch.nn as nn
import numpy as np
import io
import PIL.Image
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from roboflow import Roboflow
import tempfile
import os

class CanalMonitorNet(nn.Module):
    def __init__(self, num_classes=11):
        super(CanalMonitorNet, self).__init__()
        
        # Load pretrained DeepLabV3 with ResNet101 backbone
        self.deeplab = deeplabv3_resnet101(pretrained=True)
        
        # Replace the classifier head with our custom one
        self.deeplab.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        return self.deeplab(x)['out']

class HybridCanalMonitorNet(nn.Module):
    def __init__(self, num_classes=11, roboflow_api_key="BATlZ8b1AQuon0tnMKIm"):
        super(HybridCanalMonitorNet, self).__init__()
        
        # Original CanalMonitorNet
        self.canal_net = CanalMonitorNet(num_classes=num_classes)
        
        # Roboflow model initialization (lazy loading to avoid API calls during import)
        self.roboflow_api_key = roboflow_api_key
        self.rf = None
        self.water_model = None
        
        # Define image transforms
        self.to_pil = transforms.ToPILImage()
        
        # Category indices - assuming Water_Surface is index 2 based on order in your category list
        self.water_surface_idx = 2
        
        # Flag to enable/disable Roboflow enhancement (useful for training vs inference)
        self.use_roboflow = True
        
    def _initialize_roboflow(self):
        """Initialize Roboflow model if not already done"""
        if self.rf is None:
            self.rf = Roboflow(api_key=self.roboflow_api_key)
            self.water_model = self.rf.workspace().project("water-uzqur").version(1).model
    
    def _create_water_mask(self, water_pred, image_shape):
        """
        Create a water mask from Roboflow predictions
        
        Args:
            water_pred: JSON prediction from Roboflow
            image_shape: Tuple (height, width) of the original image
        
        Returns:
            torch.Tensor: Binary mask highlighting water areas
        """
        height, width = image_shape
        mask = torch.zeros((height, width), dtype=torch.float32)
        
        # Extract water detections from predictions
        predictions = water_pred.get('predictions', [])
        
        for pred in predictions:
            # Skip if not water
            if pred.get('class') != 'water':
                continue
            
            # Extract bounding box coordinates
            x_center = int(pred.get('x', 0))
            y_center = int(pred.get('y', 0))
            box_width = int(pred.get('width', 0))
            box_height = int(pred.get('height', 0))
            confidence = float(pred.get('confidence', 0))
            
            # Calculate box coordinates
            x_min = max(0, int(x_center - box_width / 2))
            y_min = max(0, int(y_center - box_height / 2))
            x_max = min(width, int(x_center + box_width / 2))
            y_max = min(height, int(y_center + box_height / 2))
            
            # Fill the detected water area with confidence value
            mask[y_min:y_max, x_min:x_max] = confidence
        
        return mask
    
    def forward(self, x):
        # Get primary segmentation from CanalMonitorNet
        canal_output = self.canal_net(x)
        
        # Skip Roboflow enhancement if disabled or during training
        if not self.use_roboflow or not self.training:
            return canal_output
        
        # Initialize Roboflow if needed
        self._initialize_roboflow()
        
        # Process each image in the batch with Roboflow
        batch_size = x.size(0)
        height, width = x.shape[2], x.shape[3]
        
        for i in range(batch_size):
            with torch.no_grad():
                # Convert tensor to PIL image
                img = self.to_pil(x[i].cpu())
                
                # Create a temporary file to save the image
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    img_path = temp_file.name
                    img.save(img_path)
                
                try:
                    # Get water predictions from Roboflow
                    water_pred = self.water_model.predict(img_path).json()
                    
                    # Create water mask from predictions
                    water_mask = self._create_water_mask(water_pred, (height, width))
                    water_mask = water_mask.to(x.device)
                    
                    # Enhance Water_Surface class by combining with the water mask
                    # Use a weighted combination strategy
                    alpha = 0.7  # Weight for original prediction
                    beta = 0.3   # Weight for Roboflow enhancement
                    
                    enhanced = alpha * canal_output[i, self.water_surface_idx] + beta * water_mask
                    canal_output[i, self.water_surface_idx] = enhanced
                
                except Exception as e:
                    print(f"Roboflow prediction failed: {e}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(img_path):
                        os.remove(img_path)
        
        return canal_output

    def enable_roboflow(self, enable=True):
        """Enable or disable Roboflow enhancement"""
        self.use_roboflow = enable
        return self

# Helper function to perform inference with both models and compare results
def compare_predictions(image_path, hybrid_model):
    """
    Run inference with both the base model and hybrid model and compare results
    
    Args:
        image_path: Path to input image
        hybrid_model: Initialized HybridCanalMonitorNet
    
    Returns:
        tuple: (base_prediction, hybrid_prediction)
    """
    # Set up image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess image
    img = PIL.Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    
    # Run inference with base model
    hybrid_model.enable_roboflow(False)
    with torch.no_grad():
        base_output = hybrid_model(input_tensor)
    
    # Run inference with hybrid model
    hybrid_model.enable_roboflow(True)
    with torch.no_grad():
        hybrid_output = hybrid_model(input_tensor)
    
    return base_output, hybrid_output