import torch
import numpy as np

def calculate_iou(pred, target, num_classes):
    """
    Calculate mean Intersection over Union (IoU) for segmentation.
    
    Args:
        pred: Tensor of predictions (B, H, W)
        target: Tensor of ground truth (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Mean IoU across all classes (excluding background if ignore_background=True)
    """
    iou_sum = 0.0
    valid_classes = 0
    
    # Loop over classes
    for cls in range(num_classes):
        # Skip background class (0)
        if cls == 0:
            continue
            
        # Create binary masks for this class
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        # Calculate intersection and union
        intersection = torch.logical_and(pred_mask, target_mask).sum().float()
        union = torch.logical_or(pred_mask, target_mask).sum().float()
        
        # Calculate IoU for this class
        if union > 0:
            iou = intersection / union
            iou_sum += iou.item()
            valid_classes += 1
    
    # Return mean IoU
    return iou_sum / valid_classes if valid_classes > 0 else 0.0

def calculate_dice_score(pred, target, num_classes):
    """
    Calculate mean Dice score for segmentation.
    
    Args:
        pred: Tensor of predictions (B, H, W)
        target: Tensor of ground truth (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Mean Dice score across all classes (excluding background)
    """
    dice_sum = 0.0
    valid_classes = 0
    
    # Loop over classes
    for cls in range(num_classes):
        # Skip background class (0)
        if cls == 0:
            continue
            
        # Create binary masks for this class
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        # Calculate dice numerator and denominator
        intersection = torch.logical_and(pred_mask, target_mask).sum().float() * 2.0
        total = pred_mask.sum().float() + target_mask.sum().float()
        
        # Calculate dice for this class
        if total > 0:
            dice = intersection / total
            dice_sum += dice.item()
            valid_classes += 1
    
    # Return mean dice
    return dice_sum / valid_classes if valid_classes > 0 else 0.0

def calculate_levels(mask, num_classes):
    """
    Calculate water, silt, and debris levels based on segmentation mask.
    
    Args:
        mask: Tensor of segmentation mask
        num_classes: Number of classes in the model
        
    Returns:
        Dictionary with water, silt, and debris levels
    """
    # Convert tensor to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Total pixels
    total_pixels = mask.size
    
    # Class indices for different conditions
    # These should be adjusted based on your specific classes
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
    water_level = min(5.0, (water_percent / 20))
    silt_level = min(5.0, (silt_percent / 20))
    debris_level = min(5.0, (debris_percent / 20))
    
    return {
        'water_level': water_level,
        'water_percentage': water_percent,
        'silt_level': silt_level,
        'silt_percentage': silt_percent,
        'debris_level': debris_level,
        'debris_percentage': debris_percent
    }

def calculate_class_weights(dataset):
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        dataset: Dataset object
        
    Returns:
        Array of weights for each class
    """
    # Count pixels of each class
    class_counts = np.zeros(dataset.num_classes)
    
    for i in range(len(dataset)):
        _, mask = dataset[i]
        for cls in range(dataset.num_classes):
            class_counts[cls] += torch.sum(mask == cls).item()
    
    # Calculate weights (inverse frequency)
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (class_counts * dataset.num_classes)
    
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * dataset.num_classes
    
    return class_weights