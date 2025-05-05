import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from models.detector import CanalMonitorNet
from utils.data_loader import get_dataloader
from models.metrics import calculate_iou, calculate_dice_score

def train():
    # Load config
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError("Config file is empty or invalid YAML")

    # Create directories for model checkpoints and logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"models/checkpoints_{timestamp}")
    log_dir = Path(f"logs/run_{timestamp}")
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Save config for reference
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Set device
    device = torch.device(config.get('training', {}).get('device', 'cpu'))
    print(f"Using device: {device}")
    
    # Get number of classes from config
    num_classes = config['model']['num_classes']
    print(f"Training with {num_classes} classes")
    
    # Initialize model
    model = CanalMonitorNet(num_classes=num_classes)
    model = model.to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config['training'].get('resume_from_checkpoint'):
        checkpoint_path = config['training']['resume_from_checkpoint']
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Create data loaders
    train_loader = get_dataloader(
        config['data']['train_path'],
        config['data']['annotations_path'],
        batch_size=config['training']['batch_size']
    )
    
    # Create validation loader if validation path is provided
    val_loader = None
    if config['data'].get('val_path') and config['data'].get('val_annotations_path'):
        val_loader = get_dataloader(
            config['data']['val_path'],
            config['data']['val_annotations_path'],
            batch_size=config['training']['batch_size']
        )
        print(f"Created validation loader with {len(val_loader)} batches")
    
    # Define loss function
    if config['training'].get('weighted_loss', False) and config['training'].get('class_weights'):
        # Use weighted cross entropy loss
        class_weights = torch.tensor(config['training']['class_weights']).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    # Define optimizer
    learning_rate = config['training']['learning_rate']
    if config['training'].get('optimizer', 'adam').lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif config['training'].get('optimizer').lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate,
            momentum=config['training'].get('momentum', 0.9),
            weight_decay=config['training'].get('weight_decay', 0.0001)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = None
    if config['training'].get('lr_scheduler', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
    
    # Initialize metrics tracking
    best_val_loss = float('inf')
    best_val_iou = 0.0
    train_losses = []
    val_losses = []
    val_ious = []
    
    # Training loop
    epochs = config['training']['epochs']
    print(f"Starting training for {epochs} epochs")
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Log training loss
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            running_val_iou = 0.0
            running_val_dice = 0.0
            val_batch_count = 0
            
            # Use tqdm for progress bar
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            with torch.no_grad():
                for images, masks in val_pbar:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    val_loss = criterion(outputs, masks)
                    
                    # Calculate metrics
                    preds = torch.argmax(outputs, dim=1)
                    iou = calculate_iou(preds, masks, num_classes)
                    dice = calculate_dice_score(preds, masks, num_classes)
                    
                    # Update running values
                    running_val_loss += val_loss.item()
                    running_val_iou += iou
                    running_val_dice += dice
                    val_batch_count += 1
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        "loss": f"{val_loss.item():.4f}", 
                        "IoU": f"{iou:.4f}"
                    })
            
            # Calculate average validation metrics
            avg_val_loss = running_val_loss / val_batch_count if val_batch_count > 0 else 0
            avg_val_iou = running_val_iou / val_batch_count if val_batch_count > 0 else 0
            avg_val_dice = running_val_dice / val_batch_count if val_batch_count > 0 else 0
            
            val_losses.append(avg_val_loss)
            val_ious.append(avg_val_iou)
            
            # Log validation metrics
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('IoU/val', avg_val_iou, epoch)
            writer.add_scalar('Dice/val', avg_val_dice, epoch)
            
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, "
                  f"Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}")
            
            # Update learning rate scheduler if using
            if scheduler:
                scheduler.step(avg_val_loss)
            
            # Save best model based on IoU
            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_iou': avg_val_iou,
                    'best_metric': best_val_iou,
                }, checkpoint_dir / 'best_model.pth')
                print(f"Saved new best model with IoU: {best_val_iou:.4f}")
            
            # Save best model based on loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_iou': avg_val_iou,
                    'best_metric': best_val_loss,
                }, checkpoint_dir / 'best_loss_model.pth')
                print(f"Saved new best model with loss: {best_val_loss:.4f}")
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if val_loader else None,
                'val_iou': avg_val_iou if val_loader else None,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1] if val_losses else None,
        'val_iou': val_ious[-1] if val_ious else None,
    }, checkpoint_dir / 'final_model.pth')
    
    # Plot and save training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if val_ious:
        plt.subplot(1, 2, 2)
        plt.plot(val_ious, label='Val IoU')
        plt.title('IoU History')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(log_dir / 'training_history.png')
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Training complete! Models saved to {checkpoint_dir}")
    print(f"Logs saved to {log_dir}")

if __name__ == '__main__':
    train()