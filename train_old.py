import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from pathlib import Path
from models.detector import CanalMonitorNet
from utils.data_loader_old import get_dataloader

def train():
    # Load config
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError("Config file is empty or invalid YAML")

    # Set device
    device = torch.device(config.get('training', {}).get('device', 'cpu'))
    print(f"Using device: {device}")
    
    # Initialize model
    model = CanalMonitorNet(num_classes=config['model']['num_classes'])
    model = model.to(device)
    
    # Create data loaders
    train_loader = get_dataloader(
        config['data']['train_path'],
        config['data']['annotations_path'],
        batch_size=config['training']['batch_size']
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
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
            
            running_loss += loss.item()
            
        # Print statistics
        print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, '
              f'Loss: {running_loss/len(train_loader):.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, f'models/checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()