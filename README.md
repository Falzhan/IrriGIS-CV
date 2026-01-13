# IrriGIS - Canal Monitor System

![IrriGIS Logo](logo.png)

**AI-powered canal monitoring system for detecting water levels, silt buildup, and debris accumulation**

## Overview

The IrriGIS Canal Monitor System is a comprehensive AI-based application designed to analyze canal images and detect critical conditions related to water management. The system uses deep learning models trained on labeled canal images to provide accurate assessments of:

- **Water Levels** (Dry to Overflow conditions)
- **Silt Buildup** (Light sediment to Critical siltation)
- **Debris Accumulation** (Minor obstructions to Complete blockages)

This tool helps maintenance teams prioritize their work and make data-driven decisions for canal maintenance and irrigation management.

## Key Features

### Inference Mode
- **Single Image Analysis**: Upload individual canal images for immediate assessment
- **Batch Processing**: Analyze entire directories of canal images at once
- **Adjustable Detection Threshold**: Fine-tune sensitivity for different canal conditions
- **Visual Results**: Interactive result cards with color-coded severity indicators
- **Detailed Visualization**: Comprehensive analysis views with segmentation masks and metrics

### Training Mode
- **Custom Model Training**: Train models on your own canal image datasets
- **Flexible Configuration**: Adjust epochs, learning rate, batch size, and device (CPU/GPU)
- **Progress Monitoring**: Real-time training progress with detailed metrics
- **Model Management**: Save, load, and manage trained models

### Advanced Features
- **Hybrid AI Models**: Combine standard deep learning with Roboflow API for enhanced water detection
- **Data Preparation Tools**: Built-in utilities for dataset organization and splitting
- **Comprehensive Metrics**: IoU, Dice scores, and detailed performance tracking
- **Debug Console**: Full logging and error tracking for troubleshooting

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended for GPU acceleration)
- Basic understanding of canal monitoring concepts

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/IrriGIS-CanalMonitor.git
cd IrriGIS-CanalMonitor

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Launch the main application
python mainUI.py
```

## 📦 Project Structure

```
IrriGIS-CanalMonitor/
├── mainUI.py                  # Main application interface
├── inference.py               # AI prediction and analysis engine
├── train.py                   # Model training functionality
├── config.yaml                # Configuration settings
├── requirements.txt           # Python dependencies
├── logo.png                   # Application logo
├── LICENSE                    # License information
├── .gitignore                 # Git ignore patterns
├── 
├── build/                     # Build artifacts and executables
├── config/                    # Configuration files
├── data/                      # Dataset storage
│   ├── annotations/           # COCO format annotations
│   ├── processed/             # Processed data
│   └── raw/                   # Raw images and datasets
├── logs/                      # Training logs and metrics
├── models/                    # AI model definitions
│   ├── detector.py            # Main detection model
│   ├── metrics.py             # Performance metrics
│   └── __init__.py
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit tests
├── utils/                     # Utility functions
│   ├── data_loader.py         # Data loading utilities
│   ├── descriptor.py          # Condition descriptors
│   ├── visualization.py       # Visualization tools
│   └── ...
└── README.md                  # This file
```

## Configuration

The system uses a YAML-based configuration file (`config.yaml`) for easy customization:

```yaml
training:
  batch_size: 4
  device: cpu  # or 'cuda' for GPU
  epochs: 50
  learning_rate: 0.0001
  lr_scheduler: true
  optimizer: adam
```

## Detection Levels and Interpretation

### Water Level Scale (1-5)
- **1 - Dry**: Canal is dry, critical condition requiring immediate attention
- **2 - Low**: Below operational requirements, low water flow
- **3 - Normal**: Optimal water flow conditions
- **4 - High**: Near overflow conditions
- **5 - Overflow**: Critical overflow, immediate action required

### Silt Level Scale (2-5)
- **2 - Light**: Minor silt accumulation, light sediment buildup
- **3 - Normal**: Typical silt conditions
- **4 - Dirty**: Significant silt buildup, maintenance needed
- **5 - Heavily Silted**: Critical siltation, urgent dredging required

### Debris Level Scale (2-5)
- **2 - Light**: Minor debris present, limited obstruction
- **3 - Normal**: Manageable debris level
- **4 - Heavy**: Heavy debris accumulation affecting flow
- **5 - Blocked**: Critical blockage, urgent debris removal needed

### Color Indicators
- **🟢 Green (Levels 1-2)**: Normal or minor issues
- **🟡 Yellow/Orange (Level 3)**: Moderate concern
- **🔴 Red (Levels 4-5)**: Critical attention required

## Usage Guide

### Inference Tab

1. **Select Model**: Load a trained `.pth` model file
2. **Adjust Threshold**: Set detection sensitivity (default 50% provides optimal balance)
3. **Analyze Images**: 
   - Single images: Click "Select Images for Analysis"
   - Batch processing: Click "Batch Process Directory"
4. **Review Results**: Interactive cards show water, silt, and debris levels with visual indicators

### Training Tab

1. **Prepare Data**: 
   - Select training data folder
   - Choose annotation file (COCO format)
   - Set save location for trained models
2. **Configure Training**:
   - Set epochs, learning rate, batch size
   - Choose device (CPU/GPU)
3. **Start Training**: Monitor progress in real-time
4. **Save Models**: Automatically saves best models based on performance metrics

## Technical Details

### AI Architecture

- **Base Model**: DeepLabV3+ with ResNet backbone
- **Input Size**: 224x224 RGB images
- **Output**: 10-class segmentation (including background)
- **Training**: Cross-entropy loss with optional weighted classes
- **Optimization**: Adam or SGD with learning rate scheduling

### Supported Classes

1. Background
2. Water Surface
3. Water Line
4. Dry Canal Bed
5. Silt Deposit
6. Water Discoloration
7. Floating Debris
8. Vegetation
9. Canal Bank
10. Side Slope

### Performance Metrics

- **IoU (Intersection over Union)**: Measures segmentation accuracy
- **Dice Score**: Alternative segmentation metric
- **Training/Validation Loss**: Monitors model learning progress

## Model Accuracy and Limitations

### Accuracy Factors

- Image quality and lighting conditions significantly impact results
- Models perform best on canal types similar to training data
- Detection threshold affects sensitivity vs. specificity trade-off

### Limitations

- May not detect unusual or extreme conditions not in training data
- Requires clear, unobstructed canal images for optimal performance
- AI predictions should be validated by human experts

## Advanced Usage

### Custom Model Training

```python
from train import train
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Customize training parameters
config['training']['epochs'] = 100
config['training']['batch_size'] = 8

# Start training
train(config, save_dir='custom_training_output')
```

### Batch Prediction

```python
from inference import CanalPredictor, batch_predict

# Initialize predictor
predictor = CanalPredictor(
    checkpoint_path='models/best_model.pth',
    max_percentage=60  # Adjust threshold as needed
)

# Process directory
batch_predict(predictor, 'data/test_images', 'results/batch_output')
```

## 📦 Dependencies

Key Python packages required:

- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `PySide6>=6.9.0` - Qt-based GUI framework
- `opencv-python>=4.7.0` - Image processing
- `albumentations>=1.3.0` - Image augmentations
- `pycocotools>=2.0.6` - COCO dataset utilities
- `matplotlib>=3.5.0` - Visualization
- `numpy>=1.21.0` - Numerical computing

## Testing

Run unit tests to verify functionality:

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## Contact

For questions or support, please contact:

- **Developer**: Falzhan
- **Email**: [montefalconzhander@gmail.com](mailto:montefalconzhander@gmail.com)

## References

- PyTorch Documentation: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- COCO Dataset Format: [https://cocodataset.org/](https://cocodataset.org/)
- DeepLabV3+ Architecture: [https://arxiv.org/abs/1802.02611](https://arxiv.org/abs/1802.02611)

##  Future Enhancements

- Real-time video analysis for continuous monitoring
- Mobile application for field use
- Cloud-based API for remote processing
- Integration with IoT sensors for comprehensive monitoring
- Automated alert system for critical conditions

---

**IrriGIS - Canal Monitor System** © 2025 | Falzhan
