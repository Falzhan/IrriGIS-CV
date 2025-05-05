import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models._utils import IntermediateLayerGetter

class CanalMonitorNet(nn.Module):
    def __init__(self, num_classes=12):
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