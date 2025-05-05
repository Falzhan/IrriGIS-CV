import torch
import numpy as np

def calculate_levels(mask, num_classes):
    """
    Calculate water, silt, and debris levels from segmentation mask.
    
    Args:
        mask (torch.Tensor): Segmentation mask with class predictions
        num_classes (int): Number of classes in the segmentation model
    
    Returns:
        dict: Dictionary containing level assessments
    """
    # Convert mask to numpy array if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask_np = mask.numpy()
    else:
        mask_np = np.array(mask)
    
    # Count pixels for each class
    # Assuming classes: 0=background, 1=water, 2=silt, 3=debris
    total_pixels = float(mask_np.size)  # Convert to float to ensure float division
    
    # Count pixels for each class
    water_pixels = np.sum(mask_np == 1)
    silt_pixels = np.sum(mask_np == 2)
    debris_pixels = np.sum(mask_np == 3)
    
    # Calculate levels on a scale of 1-5
    # 1 = very low, 5 = very high
    water_level = 1 + (water_pixels / total_pixels) * 4 if total_pixels > 0 else 1
    silt_level = 1 + (silt_pixels / total_pixels) * 4 if total_pixels > 0 else 1
    debris_level = 1 + (debris_pixels / total_pixels) * 4 if total_pixels > 0 else 1
    
    return {
        'water_level': float(water_level),
        'silt_level': float(silt_level),
        'debris_level': float(debris_level),
        'water_percentage': float(water_pixels / total_pixels * 100) if total_pixels > 0 else 0,
        'silt_percentage': float(silt_pixels / total_pixels * 100) if total_pixels > 0 else 0,
        'debris_percentage': float(debris_pixels / total_pixels * 100) if total_pixels > 0 else 0
    }

def calculate_iou(pred_mask, gt_mask, num_classes):
    """
    Calculate IoU (Intersection over Union) for each class.
    
    Args:
        pred_mask (torch.Tensor): Predicted segmentation mask
        gt_mask (torch.Tensor): Ground truth segmentation mask
        num_classes (int): Number of classes
        
    Returns:
        torch.Tensor: IoU score for each class
    """
    iou_scores = []
    
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.numpy()
    
    for cls in range(num_classes):
        # Create binary masks for current class
        pred_binary = (pred_mask == cls).astype(np.int32)
        gt_binary = (gt_mask == cls).astype(np.int32)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    
    return iou_scores