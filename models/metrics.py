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
    
    Water Level: 1-Dry, 2-Low, 3-Normal, 4-High, 5-Overflow
    Silt Level: 2-Light, 3-Normal, 4-Dirty, 5-Heavily Silted (No level 1)
    Debris Level: 2-Light, 3-Normal, 4-Heavy, 5-Blocked (No level 1)
    
    Default all levels to "Normal" (3) if no indicators present.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    total_pixels = mask.size
    
    # Class indices
    water_surface_idx = 2  # Water_Surface
    canal_bank_idx = 7     # Canal_Bank
    side_slope_idx = 3     # Side_Slope
    dry_canal_idx = 9      # Dry_Canal_Bed
    silt_deposit_idx = 6   # Silt_Deposit
    vegetation_idx = 8     # Vegetation
    water_discolor_idx = 5 # Water_Discoloration
    floating_debris_idx = 4 # Floating_Debris (fixed index from 7 to an appropriate value)
    
    # Calculate pixel counts
    water_surface = np.sum(mask == water_surface_idx)
    canal_bank = np.sum(mask == canal_bank_idx)
    side_slope = np.sum(mask == side_slope_idx)
    dry_canal = np.sum(mask == dry_canal_idx)
    silt_deposit = np.sum(mask == silt_deposit_idx)
    vegetation = np.sum(mask == vegetation_idx)
    water_discolor = np.sum(mask == water_discolor_idx)
    floating_debris = np.sum(mask == floating_debris_idx)
    
    # Calculate percentages for reporting
    water_percent = (water_surface / total_pixels) * 100 if total_pixels > 0 else 0
    
    # ----- WATER LEVEL CALCULATION (1-5) -----
    # Default to Normal
    water_level = 3
    
    # Check for dry canal first
    if dry_canal > 0:
        water_level = 1  # Dry
    else:
        # Calculate bank/slope areas for comparison
        bank_slope_area = canal_bank + side_slope
        
        if bank_slope_area > 0:
            # Calculate the ratio of water to bank+slope
            water_ratio = water_surface / bank_slope_area
            
            # Determine water level based on the ratio
            if water_surface == 0:
                water_level = 1  # Dry if no water detected
            elif water_ratio < 0.15:
                water_level = 2  # Low
            elif water_ratio < 0.40:
                water_level = 3  # Normal
            elif water_ratio < 0.70:
                water_level = 4  # High
            else:
                water_level = 5  # Overflow
    
    # ----- SILT LEVEL CALCULATION (2-5) -----
    # Default to Normal (3)
    silt_level = 3
    
    # Consider multiple sources as indicators for silt
    veg_as_silt = vegetation * 0.3  # 30% of vegetation considered as silt indicator
    silt_total = silt_deposit + veg_as_silt + water_discolor
    silt_percent = (silt_total / total_pixels) * 100 if total_pixels > 0 else 0
    
    # If no silt indicators are present, keep as Normal (3)
    # If silt is present, determine the level based on ratio
    if silt_total == 0:
        silt_level = 3  # Normal - when there's none
    else:
        silt_ratio = silt_total / total_pixels
        if silt_ratio < 0.10:
            silt_level = 2  # Light - when there's little
        elif silt_ratio < 0.25:
            silt_level = 4  # Dirty - when there's many
        else:
            silt_level = 5  # Heavily Silted - when there's too many
    
    # ----- DEBRIS LEVEL CALCULATION (2-5) -----
    # Default to Normal (3)
    debris_level = 3
    
    # Consider floating debris and part of vegetation as debris
    veg_as_debris = vegetation * 0.7  # 70% of vegetation considered as debris indicator
    debris_total = floating_debris + veg_as_debris
    debris_percent = (debris_total / total_pixels) * 100 if total_pixels > 0 else 0
    
    # If no debris indicators are present, keep as Normal (3)
    # If debris is present, determine the level based on ratio
    if debris_total == 0:
        debris_level = 3  # Normal - when there's none
    else:
        debris_ratio = debris_total / total_pixels
        if debris_ratio < 0.10:
            debris_level = 2  # Light - when there's little
        elif debris_ratio < 0.25:
            debris_level = 4  # Heavy - when there's many
        else:
            debris_level = 5  # Blocked - when there's too many
    
    return {
        'water_level': int(water_level),
        'water_percentage': round(water_percent),
        'silt_level': int(silt_level),
        'silt_percentage': round(silt_percent),
        'debris_level': int(debris_level),
        'debris_percentage': round(debris_percent)
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