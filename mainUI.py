## Made by KiritoZhander45
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QFileDialog, QProgressBar, QSlider, QScrollArea,
                             QGridLayout, QMessageBox, QSpinBox, QStyle,
                             QSplitter, QTextEdit, QComboBox, QDoubleSpinBox,
                             QGroupBox, QFormLayout, QCheckBox, QStyleFactory,
                             QFrame, QRadioButton)
from PySide6.QtCore import Qt, Signal, QThread, QSize, QPropertyAnimation, QEasingCurve, QSettings
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor, QLinearGradient, QBrush, QPainter
import sys
import os
import torch
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from models.detector import CanalMonitorNet
from utils.descriptor import CanalDescriptor
from train import train
from inference import CanalPredictor, batch_predict
from utils.add_datasetUI import add_dataset
from utils.split_dataset import split_dataset  # Add this import at the top

# Theme colors
THEME = {
    'moss_green': "#A0C878",
    'pastel_green': "#DDEB9D",
    'creamy_beige': "#FAF6E9",
    'off_white': "#FFFDF6",
    'text_dark': "#444444",
    'text_light': "#FFFFFF",
    'border': "#85A665",
    'highlight': "#B8D78C"
}

class CollapsibleConsole(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.is_expanded = False
        self.collapsed_height = 30
        self.expanded_height = 200
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with toggle button
        header_widget = QWidget()
        header_widget.setStyleSheet(f"background-color: {THEME['moss_green']}; border-top: 1px solid {THEME['border']};")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 4, 10, 4)
        
        self.toggle_btn = QPushButton("▼ Debug Console")
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {THEME['text_light']};
                border: none;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: {THEME['off_white']};
            }}
        """)
        self.toggle_btn.clicked.connect(self.toggle_console)
        header_layout.addWidget(self.toggle_btn)
        header_layout.addStretch()
        
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['pastel_green']};
                color: {THEME['text_dark']};
                border: none;
                border-radius: 4px;
                padding: 3px 8px;
            }}
            QPushButton:hover {{
                background-color: {THEME['highlight']};
            }}
        """)
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self.clear_console)
        header_layout.addWidget(clear_btn)
        
        layout.addWidget(header_widget)
        
        # Console content
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(f"""
            QTextEdit {{
                background-color: {THEME['off_white']};
                color: {THEME['text_dark']};
                border: none;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
            }}
        """)
        self.console.setVisible(False)
        layout.addWidget(self.console)
        
        self.animation = QPropertyAnimation(self, b"minimumHeight")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        
    def toggle_console(self):
        if self.is_expanded:
            # Collapse
            self.animation.setStartValue(self.expanded_height)
            self.animation.setEndValue(self.collapsed_height)
            self.toggle_btn.setText("▼ Debug Console")
            # Disconnect any existing connections
            try:
                self.animation.finished.disconnect()
            except:
                pass
            self.animation.finished.connect(lambda: self.console.setVisible(False))
        else:
            # Expand
            # Disconnect any existing connections
            try:
                self.animation.finished.disconnect()
            except:
                pass
            self.console.setVisible(True)
            self.animation.setStartValue(self.collapsed_height)
            self.animation.setEndValue(self.expanded_height)
            self.toggle_btn.setText("▲ Debug Console")
            
        self.animation.start()
        self.is_expanded = not self.is_expanded
    
    def log(self, message):
        self.console.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def clear_console(self):
        self.console.clear()

class StyledSlider(QWidget):
    valueChanged = Signal(int)
    
    def __init__(self, min_val=0, max_val=100, default_val=50, label="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if (label):
            header = QLabel(label)
            header.setStyleSheet(f"color: {THEME['text_dark']}; font-weight: bold;")
            layout.addWidget(header)
        
        slider_layout = QHBoxLayout()
        
        # Create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default_val)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 8px;
                background: {THEME['creamy_beige']};
                border-radius: 4px;
            }}
            
            QSlider::handle:horizontal {{
                background: {THEME['moss_green']};
                border: 2px solid {THEME['border']};
                width: 18px;
                margin: -6px 0;
                border-radius: 10px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: {THEME['highlight']};
            }}
            
            QSlider::add-page:horizontal {{
                background: {THEME['creamy_beige']};
                border-radius: 4px;
            }}
            
            QSlider::sub-page:horizontal {{
                background: {THEME['moss_green']};
                border-radius: 4px;
            }}
        """)
        
        # Create value display
        self.value_label = QLabel(f"{default_val}%")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setMinimumWidth(40)
        self.value_label.setStyleSheet(f"color: {THEME['text_dark']};")
        
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.value_label)
        
        layout.addLayout(slider_layout)
        
        # Connect signals
        self.slider.valueChanged.connect(self.handle_value_change)
    
    def handle_value_change(self, value):
        self.value_label.setText(f"{value}%")
        self.valueChanged.emit(value)
    
    def value(self):
        return self.slider.value()
    
    def setValue(self, value):
        self.slider.setValue(value)

class StyledButton(QPushButton):
    def __init__(self, text, icon=None, primary=True, parent=None):
        super().__init__(text, parent)
        if icon:
            self.setIcon(icon)
        
        if primary:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME['moss_green']};
                    color: {THEME['text_light']};
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {THEME['highlight']};
                }}
                QPushButton:pressed {{
                    background-color: {THEME['border']};
                }}
                QPushButton:disabled {{
                    background-color: #CCCCCC;
                    color: #888888;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME['pastel_green']};
                    color: {THEME['text_dark']};
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background-color: {THEME['highlight']};
                }}
                QPushButton:pressed {{
                    background-color: {THEME['border']};
                }}
                QPushButton:disabled {{
                    background-color: #CCCCCC;
                    color: #888888;
                }}
            """)

class StyledGroupBox(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(f"""
            QGroupBox {{
                background-color: {THEME['creamy_beige']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
                margin-top: 1.5ex;
                font-weight: bold;
                color: {THEME['text_dark']};
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: {THEME['moss_green']};
                color: {THEME['text_light']};
                border-radius: 3px;
            }}
        """)

class TrainingWorker(QThread):
    progress = Signal(str, int)  # message, progress percentage
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, config, save_dir):
        super().__init__()
        self.config = config
        self.save_dir = save_dir
        
    def run(self):
        try:
            # Run training with periodic progress updates
            self.progress.emit("Initializing training...", 0)
            train(self.config, self.save_dir, progress_callback=self.update_progress)
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Error during training: {str(e)}")
    
    def update_progress(self, message, percentage):
        self.progress.emit(message, percentage)

class ResultWidget(QWidget):
    def __init__(self, image_path, result, predictor):
        super().__init__()
        self.full_result = result
        self.predictor = predictor
        self.descriptor = CanalDescriptor()  # Initialize descriptor
        
        # Make widget clickable
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME['creamy_beige']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
            }}
            QLabel {{
                color: {THEME['text_dark']};
            }}
        """)
        
        layout = QVBoxLayout()
        
        # Display original image
        pixmap = QPixmap(image_path)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(pixmap.scaled(280, 280, Qt.KeepAspectRatio))
        layout.addWidget(image_label)
        
        # Create indicator bars for water, silt, and debris levels
        levels = result['levels']
        
        # File name
        file_name = Path(image_path).name
        name_label = QLabel(file_name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet(f"font-weight: bold; color: {THEME['text_dark']};")
        layout.addWidget(name_label)
        
        # Add indicator bars with descriptions
        self.add_level_indicator(layout, "Water", levels['water_level'], 
            self.descriptor.water_remarks[int(levels['water_level'])][0])
        self.add_level_indicator(layout, "Silt", levels['silt_level'],
            self.descriptor.silt_remarks.get(int(levels['silt_level']), ["Normal silt conditions."])[0])
        self.add_level_indicator(layout, "Debris", levels['debris_level'],
            self.descriptor.debris_remarks.get(int(levels['debris_level']), ["Normal debris conditions."])[0])
        
        # Add short summary
        summary = self.descriptor.get_short_remark(
            levels['water_level'],
            levels['silt_level'],
            levels['debris_level']
        )
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        summary_label.setAlignment(Qt.AlignCenter)
        summary_label.setStyleSheet(f"""
            font-style: italic;
            color: {THEME['text_dark']};
            padding: 5px;
            font-size: 11px;
        """)
        layout.addWidget(summary_label)
        
        self.setLayout(layout)
        
    def add_level_indicator(self, layout, name, level, description):
        level_layout = QVBoxLayout()
        
        # Label with name and level
        label = QLabel(f"{name}: {level}/5")
        label.setStyleSheet("font-weight: bold;")
        level_layout.addWidget(label)
        
        # Progress bar as indicator
        indicator = QProgressBar()
        indicator.setMinimum(0)
        indicator.setMaximum(5)
        indicator.setValue(level)
        indicator.setTextVisible(False)
        indicator.setFixedHeight(8)
        
        # Color based on level
        color = self.get_level_color(level)
        indicator.setStyleSheet(f"""
            QProgressBar {{
                background-color: {THEME['off_white']};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)
        
        level_layout.addWidget(indicator)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 10px; font-style: italic;")
        level_layout.addWidget(desc_label)
        
        layout.addLayout(level_layout)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.show_full_details()
    
    def enterEvent(self, event):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME['highlight']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
            }}
            QLabel {{
                color: {THEME['text_dark']};
            }}
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME['creamy_beige']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
            }}
            QLabel {{
                color: {THEME['text_dark']};
            }}
        """)
        super().leaveEvent(event)
        
    def get_level_color(self, level):
        if level <= 1:
            return "#4CAF50"  # Green
        elif level <= 3:
            return "#FFC107"  # Yellow/Orange
        else:
            return "#F44336"  # Red
        
    def show_full_details(self):
        # Create new window with full visualization
        self.predictor.visualize(self.full_result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('IrriGIS', 'CanalMonitor')
        self.setWindowTitle("IrriGIS - Canal Monitor System")
        self.setMinimumSize(900, 700)
        
        # Apply theme to the entire application
        self.apply_theme()
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {THEME['border']};
                background-color: {THEME['creamy_beige']};
                border-radius: 5px;
            }}
            
            QTabBar::tab {{
                background-color: {THEME['pastel_green']};
                color: {THEME['text_dark']};
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 8px 12px;
                margin-right: 2px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {THEME['moss_green']};
                color: {THEME['text_light']};
                font-weight: bold;
            }}
            
            QTabBar::tab:hover {{
                background-color: {THEME['highlight']};
            }}
        """)
        
        # Create tabs in desired order - Inference first
        self.tabs.addTab(self.create_inference_tab(), "Inference")
        self.tabs.addTab(self.create_training_tab(), "Training")
        self.tabs.addTab(self.create_help_tab(), "Help")
        
        main_layout.addWidget(self.tabs)
        
        # Add debug console (collapsed by default)
        self.debug_console = CollapsibleConsole()
        main_layout.addWidget(self.debug_console)
        
        self.setCentralWidget(main_widget)
        
        # Initialize with a welcome message
        self.debug_console.log("Welcome to IrriGIS - Canal Monitor System")
        self.debug_console.log("System ready. Select a tab to begin.")
    
    def apply_theme(self):
        # Set global application style
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        
        # Create a palette with our theme colors
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(THEME['off_white']))
        palette.setColor(QPalette.WindowText, QColor(THEME['text_dark']))
        palette.setColor(QPalette.Base, QColor(THEME['creamy_beige']))
        palette.setColor(QPalette.AlternateBase, QColor(THEME['pastel_green']))
        palette.setColor(QPalette.ToolTipBase, QColor(THEME['text_dark']))
        palette.setColor(QPalette.ToolTipText, QColor(THEME['text_light']))
        palette.setColor(QPalette.Text, QColor(THEME['text_dark']))
        palette.setColor(QPalette.Button, QColor(THEME['moss_green']))
        palette.setColor(QPalette.ButtonText, QColor(THEME['text_light']))
        palette.setColor(QPalette.Highlight, QColor(THEME['moss_green']))
        palette.setColor(QPalette.HighlightedText, QColor(THEME['text_light']))
        
        QApplication.setPalette(palette)
    
    def create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Data selection group
        data_group = StyledGroupBox("Data Selection")
        data_layout = QVBoxLayout()
        
        # Training data folder
        train_layout = QHBoxLayout()
        self.train_path_label = QLabel("No folder selected")
        self.train_path_label.setStyleSheet(f"""
            background-color: {THEME['off_white']};
            padding: 5px;
            border-radius: 3px;
            border: 1px solid {THEME['border']};
        """)
        train_btn = StyledButton("Select Training Data", primary=False)
        train_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        train_btn.clicked.connect(self.select_train_folder)
        train_layout.addWidget(QLabel("Training Data:"))
        train_layout.addWidget(self.train_path_label, 1)
        train_layout.addWidget(train_btn)
        data_layout.addLayout(train_layout)
        
        # Annotation file
        anno_layout = QHBoxLayout()
        self.anno_path_label = QLabel("No file selected")
        self.anno_path_label.setStyleSheet(f"""
            background-color: {THEME['off_white']};
            padding: 5px;
            border-radius: 3px;
            border: 1px solid {THEME['border']};
        """)
        anno_btn = StyledButton("Select Annotations", primary=False)
        anno_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        anno_btn.clicked.connect(self.select_annotation)
        anno_layout.addWidget(QLabel("Annotations:"))
        anno_layout.addWidget(self.anno_path_label, 1)
        anno_layout.addWidget(anno_btn)
        data_layout.addLayout(anno_layout)
        
        # Save location
        save_layout = QHBoxLayout()
        self.save_path_label = QLabel("No location selected")
        saved_save_path = self.settings.value('save_location', "No location selected")
        self.save_path_label.setText(saved_save_path)
        self.save_path_label.setStyleSheet(f"""
            background-color: {THEME['off_white']};
            padding: 5px;
            border-radius: 3px;
            border: 1px solid {THEME['border']};
        """)
        save_btn = StyledButton("Select Save Location", primary=False)
        save_btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        save_btn.clicked.connect(self.select_save_location)
        save_layout.addWidget(QLabel("Save Location:"))
        save_layout.addWidget(self.save_path_label, 1)
        save_layout.addWidget(save_btn)
        data_layout.addLayout(save_layout)

        # Add Confirm Data button
        confirm_btn = StyledButton("Confirm Data", primary=True)
        confirm_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        confirm_btn.clicked.connect(self.confirm_data)
        data_layout.addWidget(confirm_btn)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Training parameters group
        params_group = StyledGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {THEME['off_white']};
                border: 1px solid {THEME['border']};
                border-radius: 3px;
                padding: 3px;
            }}
        """)
        params_layout.addRow("Epochs:", self.epochs_spin)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {THEME['off_white']};
                border: 1px solid {THEME['border']};
                border-radius: 3px;
                padding: 3px;
            }}
        """)
        params_layout.addRow("Learning Rate:", self.lr_spin)
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(4)
        self.batch_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {THEME['off_white']};
                border: 1px solid {THEME['border']};
                border-radius: 3px;
                padding: 3px;
            }}
        """)
        params_layout.addRow("Batch Size:", self.batch_spin)
        
        # Device selection
        device_layout = QHBoxLayout()
        self.device_cpu = QRadioButton("CPU")
        self.device_gpu = QRadioButton("GPU (CUDA)")
        self.device_cpu.setChecked(True)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.device_gpu.setEnabled(True)
            cuda_info = f"(detected {torch.cuda.get_device_name(0)})"
            self.device_gpu.setText(f"GPU {cuda_info}")
        else:
            self.device_gpu.setEnabled(False)
            self.device_gpu.setText("GPU (not available)")
        
        device_layout.addWidget(self.device_cpu)
        device_layout.addWidget(self.device_gpu)
        device_layout.addStretch()
        params_layout.addRow("Device:", device_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress section
        progress_group = StyledGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v/%m)")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {THEME['off_white']};
                border: 1px solid {THEME['border']};
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {THEME['moss_green']};
                border-radius: 5px;
            }}
        """)
        
        self.progress_label = QLabel("Ready to train")
        self.progress_label.setStyleSheet(f"color: {THEME['text_dark']};")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        
        # Training controls
        buttons_layout = QHBoxLayout()
        self.train_btn = StyledButton("Start Training", primary=True)
        self.train_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.train_btn)
        buttons_layout.addStretch()
        
        progress_layout.addLayout(buttons_layout)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_inference_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Model selection group
        model_group = StyledGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        
        self.model_path_label = QLabel("No model selected")
        saved_model_path = self.settings.value('model_path', "No model selected")
        self.model_path_label.setText(saved_model_path)
        self.model_path_label.setStyleSheet(f"""
            background-color: {THEME['off_white']};
            padding: 5px;
            border-radius: 3px;
            border: 1px solid {THEME['border']};
        """)
        
        model_btn = StyledButton("Select Model File (.pth)", primary=False)
        model_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        model_btn.clicked.connect(self.select_model)
        
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_path_label, 1)
        model_layout.addWidget(model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Detection settings group
        settings_group = StyledGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        
        # Threshold control using custom slider
        self.threshold_slider = StyledSlider(0, 100, 50, "Detection Threshold:")
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        settings_layout.addWidget(self.threshold_slider)
        
        # Explanation
        threshold_info = QLabel("Higher threshold means more confident detections but may miss some issues. Lower threshold detects more potential issues but may include false positives.")
        threshold_info.setWordWrap(True)
        threshold_info.setStyleSheet(f"font-style: italic; color: {THEME['text_dark']};")
        settings_layout.addWidget(threshold_info)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Image selection and processing
        image_group = StyledGroupBox("Image Analysis")
        image_layout = QVBoxLayout()
        
        buttons_layout = QHBoxLayout()
        image_btn = StyledButton("Select Images for Analysis", primary=True)
        image_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        image_btn.clicked.connect(self.select_images)
        
        batch_btn = StyledButton("Batch Process Directory", primary=False)
        batch_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        batch_btn.clicked.connect(self.batch_process_directory)
        
        buttons_layout.addWidget(image_btn)
        buttons_layout.addWidget(batch_btn)
        image_layout.addLayout(buttons_layout)
        
        # Results area with better styling
        results_label = QLabel("Analysis Results:")
        results_label.setStyleSheet(f"font-weight: bold; color: {THEME['text_dark']};")
        image_layout.addWidget(results_label)
        
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {THEME['off_white']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
            }}
        """)
        
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setContentsMargins(10, 10, 10, 10)
        self.results_layout.setSpacing(15)
        
        self.results_area.setWidget(self.results_widget)
        image_layout.addWidget(self.results_area)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        widget.setLayout(layout)
        return widget
    
    def create_help_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create scrollable area for help content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        help_content = QWidget()
        help_layout = QVBoxLayout(help_content)
        
        # Application title and description
        title_label = QLabel("IrrGIS - Canal Monitor - User Guide")
        title_label.setStyleSheet(f"""
            color: {THEME['text_dark']};
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        """)
        help_layout.addWidget(title_label)
        
        # Add overview
        overview_group = StyledGroupBox("Overview")
        overview_layout = QVBoxLayout()
        
        overview_text = QLabel(
            "The IrriGIS - Canal Monitor System is designed to analyze images of canals to detect "
            "issues related to water level, silt buildup, and debris accumulation. "
            "The system uses AI models trained on labeled images to provide accurate "
            "assessments and help maintenance teams prioritize their work."
        )
        overview_text.setWordWrap(True)
        overview_layout.addWidget(overview_text)
        
        overview_group.setLayout(overview_layout)
        help_layout.addWidget(overview_group)
        
        # Add inference section
        inference_group = StyledGroupBox("Using the Inference Tab")
        inference_layout = QVBoxLayout()
        
        inference_steps = QLabel(
            "<b>1. Select a Model:</b> Click 'Select Model File' to choose a trained .pth model file.<br><br>"
            "<b>2. Adjust Detection Threshold:</b> Use the slider to set the detection threshold percentage. "
            "The default value of 50% provides a good balance between detection sensitivity and accuracy. "
            "Lower values will detect more potential issues but may include false positives. "
            "Higher values ensure more confident detections but may miss some issues.<br><br>"
            "<b>3. Select Images:</b> Click 'Select Images for Analysis' to choose canal images for analysis, "
            "or use 'Batch Process Directory' to analyze multiple images at once.<br><br>"
            "<b>4. Review Results:</b> Results will appear in the grid below, showing assessment for water level, "
            "silt level, and debris level on a scale of 1-5. Click on any result card to see detailed "
            "visualizations and analysis."
        )
        inference_steps.setWordWrap(True)
        inference_steps.setTextFormat(Qt.RichText)
        inference_layout.addWidget(inference_steps)
        
        inference_group.setLayout(inference_layout)
        help_layout.addWidget(inference_group)
        
        # Add training section
        training_group = StyledGroupBox("Using the Training Tab")
        training_layout = QVBoxLayout()
        
        training_steps = QLabel(
            "<b>1. Select Training Data:</b> Click 'Select Training Data' to choose a folder containing training images.<br><br>"
            "<b>2. Select Annotations:</b> Choose the JSON annotation file that contains labels for your training images.<br><br>"
            "<b>3. Select Save Location:</b> Choose where to save the trained model and associated files.<br><br>"
            "<b>4. Configure Training Parameters:</b><br>"
            "&nbsp;&nbsp;• <b>Epochs:</b> Number of full training cycles (default: 50)<br>"
            "&nbsp;&nbsp;• <b>Learning Rate:</b> Controls how quickly the model adapts (default: 0.001)<br>"
            "&nbsp;&nbsp;• <b>Batch Size:</b> Number of images processed at once (default: 4)<br>"
            "&nbsp;&nbsp;• <b>Device:</b> Choose CPU or GPU (if available) for training<br><br>"
            "<b>5. Start Training:</b> Click 'Start Training' to begin the training process. The progress bar "
            "will show the current status, and messages will appear in the debug console."
        )
        training_steps.setWordWrap(True)
        training_steps.setTextFormat(Qt.RichText)
        training_layout.addWidget(training_steps)
        
        training_tips = QLabel(
            "<b>Tips:</b><br>"
            "• You can modify the config.yaml file for advanced settings by clicking 'Edit Config'<br>"
            "• Training on GPU is much faster than CPU if you have compatible hardware<br>"
            "• Higher epoch counts generally improve model accuracy but take longer to train"
        )
        training_tips.setWordWrap(True)
        training_tips.setTextFormat(Qt.RichText)
        training_layout.addWidget(training_tips)
        
        training_group.setLayout(training_layout)
        help_layout.addWidget(training_group)
        
        # Add debug console section
        debug_group = StyledGroupBox("Using the Debug Console")
        debug_layout = QVBoxLayout()
        
        debug_text = QLabel(
            "The debug console at the bottom of the application provides detailed information about "
            "operations and any errors that might occur. Click on the '▼ Debug Console' header to "
            "expand the console and view the logs. Click 'Clear' to remove all messages."
        )
        debug_text.setWordWrap(True)
        debug_layout.addWidget(debug_text)
        
        debug_group.setLayout(debug_layout)
        help_layout.addWidget(debug_group)
        
        # Add disclaimer
        disclaimer_group = StyledGroupBox("Disclaimer")
        disclaimer_layout = QVBoxLayout()
        
        disclaimer_text = QLabel(
            "<b>Model Accuracy Disclaimer:</b><br><br>"
            "The AI models used in this application are trained on specific datasets and may not be 100% accurate "
            "in all situations. The default detection threshold is set to 50% as it provides an optimal balance "
            "between sensitivity and specificity for most canal monitoring scenarios.<br><br>"
            "Factors that may affect model accuracy include:<br>"
            "• Image quality and lighting conditions<br>"
            "• Canal types different from those in the training data<br>"
            "• Unusual or extreme conditions not represented in the training data<br><br>"
            "Always use professional judgment when interpreting the results. The system is designed as a "
            "decision support tool, not as a replacement for expert assessment."
        )
        disclaimer_text.setWordWrap(True)
        disclaimer_text.setTextFormat(Qt.RichText)
        disclaimer_layout.addWidget(disclaimer_text)
        
        disclaimer_group.setLayout(disclaimer_layout)
        help_layout.addWidget(disclaimer_group)
        
        # Version information
        version_label = QLabel("IrriGIS - Canal Monitor System v0.00.")
        version_label.setAlignment(Qt.AlignRight)
        version_label.setStyleSheet("color: gray; font-style: italic;")
        help_layout.addWidget(version_label)
        
        scroll.setWidget(help_content)
        layout.addWidget(scroll)
        
        widget.setLayout(layout)
        return widget
    
    def select_train_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Training Data Folder")
        if folder:
            self.train_path_label.setText(folder)
            self.debug_console.log(f"Selected training data folder: {folder}")
            self.update_train_button()
    
    def select_annotation(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Annotation File", "", "JSON Files (*.json)")
        if file:
            self.anno_path_label.setText(file)
            self.debug_console.log(f"Selected annotation file: {file}")
            self.update_train_button()
    
    def select_save_location(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Location")
        if folder:
            self.save_path_label.setText(folder)
            self.settings.setValue('save_location', folder)
            self.debug_console.log(f"Selected save location: {folder}")
            self.update_train_button()
    
    def select_config_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Config File", "", "YAML Files (*.yaml *.yml)")
        if file:
            self.config_path_label.setText(file)
            self.debug_console.log(f"Selected config file: {file}")
            
            # Try to load config values
            try:
                with open(file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Update UI fields from config
                if 'training' in config:
                    if 'epochs' in config['training']:
                        self.epochs_spin.setValue(config['training']['epochs'])
                    if 'learning_rate' in config['training']:
                        self.lr_spin.setValue(config['training']['learning_rate'])
                    if 'batch_size' in config['training']:
                        self.batch_spin.setValue(config['training']['batch_size'])
                    if 'device' in config['training']:
                        if config['training']['device'].lower() == 'cuda':
                            self.device_gpu.setChecked(True)
                        else:
                            self.device_cpu.setChecked(True)
                
                self.debug_console.log("Successfully loaded configuration values")
            except Exception as e:
                self.debug_console.log(f"Error loading config file: {str(e)}")
    
    def open_config_in_editor(self):
        config_path = self.config_path_label.text()
        if config_path == "Default config":
            # Use existing config file
            config_path = Path(__file__).parent / 'config' / 'config.yaml'
            self.config_path_label.setText(str(config_path))
            self.debug_console.log(f"Using default config at {config_path}")

        # Open in system editor
        try:
            if sys.platform == 'win32':
                os.startfile(config_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', config_path])
            else:  # Linux
                subprocess.Popen(['xdg-open', config_path])
            
            self.debug_console.log(f"Opened config file in editor: {config_path}")
        except Exception as e:
            self.debug_console.log(f"Error opening config file: {str(e)}")
            QMessageBox.warning(self, "Error", f"Could not open config file: {str(e)}")
    
    def update_train_button(self):
        self.train_btn.setEnabled(
            self.train_path_label.text() != "No folder selected" and
            self.anno_path_label.text() != "No file selected" and
            self.save_path_label.text() != "No location selected"
        )
    
    def start_training(self):
        # Use existing config file as base
        config_path = Path(__file__).parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update config with UI values
        config['data']['train_path'] = self.train_path_label.text()
        config['data']['annotations_path'] = self.anno_path_label.text()
        config['training'].update({
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'device': 'cuda' if self.device_gpu.isChecked() else 'cpu'
        })
        
        # Log config
        self.debug_console.log("Starting training with configuration:")
        self.debug_console.log(f"- Epochs: {config['training']['epochs']}")
        self.debug_console.log(f"- Learning Rate: {config['training']['learning_rate']}")
        self.debug_console.log(f"- Batch Size: {config['training']['batch_size']}")
        self.debug_console.log(f"- Device: {config['training']['device']}")
        
        # Update UI
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Initializing training...")
        
        # Start training in separate thread
        self.training_thread = TrainingWorker(config, self.save_path_label.text())
        self.training_thread.progress.connect(self.update_training_progress)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.error.connect(self.training_error)
        self.training_thread.start()
    
    def update_training_progress(self, message, percentage):
        self.progress_label.setText(message)
        self.progress_bar.setValue(percentage)
        self.debug_console.log(message)
    
    def training_finished(self):
        self.progress_label.setText("Training completed successfully!")
        self.progress_bar.setValue(100)
        self.train_btn.setEnabled(True)
        self.debug_console.log("Training completed successfully")
        
        # Show success message
        QMessageBox.information(self, "Training Complete", 
                              "Model training completed successfully!\n\n"
                              f"Model saved to: {self.save_path_label.text()}")
    
    def training_error(self, error_message):
        self.progress_label.setText("Training failed")
        self.train_btn.setEnabled(True)
        self.debug_console.log(f"ERROR: {error_message}")
        
        # Show error message
        QMessageBox.critical(self, "Training Error", 
                           f"An error occurred during training:\n\n{error_message}")
    
    def select_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth)")
        if file:
            # Verify model compatibility
            try:
                checkpoint = torch.load(file, map_location='cpu')
                num_classes = None
                for key, value in checkpoint['model_state_dict'].items():
                    if 'classifier.3.weight' in key:
                        num_classes = value.shape[0]
                        break
                
                if num_classes == 10:  # 9 classes + background
                    self.model_path_label.setText(file)
                    self.settings.setValue('model_path', file)
                    self.debug_console.log(f"Selected valid model: {file}")
                else:
                    QMessageBox.warning(self, "Incompatible Model",
                                      f"Model has {num_classes} classes, expected 10")
                    self.debug_console.log(f"Incompatible model selected: {file} (has {num_classes} classes)")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load model: {str(e)}")
                self.debug_console.log(f"Error loading model {file}: {str(e)}")
    
    def update_threshold(self, value):
        self.debug_console.log(f"Detection threshold updated to {value}%")
    
    def select_images(self):
        if self.model_path_label.text() == "No model selected":
            QMessageBox.warning(self, "Error", "Please select a model first")
            self.debug_console.log("Cannot select images: No model selected")
            return
        
        # Allow selecting multiple images
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg)"
        )
        
        if files:
            self.debug_console.log(f"Selected {len(files)} images for analysis")
            self.process_images(files)
    
    def batch_process_directory(self):
        if self.model_path_label.text() == "No model selected":
            QMessageBox.warning(self, "Error", "Please select a model first")
            self.debug_console.log("Cannot batch process: No model selected")
            return
        
        # Select directory
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Images")
        if not directory:
            return
        
        # Find all images in directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(Path(directory).glob(ext)))
        
        if not image_files:
            QMessageBox.information(self, "No Images", "No image files found in the selected directory")
            self.debug_console.log(f"No images found in directory: {directory}")
            return
        
        self.debug_console.log(f"Batch processing {len(image_files)} images from {directory}")
        self.process_images([str(f) for f in image_files])
    
    def process_images(self, files):
        # Clear previous results
        for i in reversed(range(self.results_layout.count())): 
            self.results_layout.itemAt(i).widget().setParent(None)
        
        # Show processing message
        processing_label = QLabel("Processing images... Please wait.")
        processing_label.setAlignment(Qt.AlignCenter)
        processing_label.setStyleSheet(f"color: {THEME['text_dark']}; font-style: italic;")
        self.results_layout.addWidget(processing_label, 0, 0, 1, 3)
        QApplication.processEvents()
        
        try:
            # Initialize predictor with current threshold
            predictor = CanalPredictor(
                self.model_path_label.text(),
                max_percentage=self.threshold_slider.value()
            )
            
            # Remove processing message
            processing_label.setParent(None)
            
            # Process each image
            for i, file in enumerate(files):
                try:
                    self.debug_console.log(f"Processing image: {os.path.basename(file)}")
                    result = predictor.predict(file)
                    widget = ResultWidget(file, result, predictor)
                    row = i // 3
                    col = i % 3
                    self.results_layout.addWidget(widget, row, col)
                    QApplication.processEvents()  # Keep UI responsive
                except Exception as e:
                    self.debug_console.log(f"Error processing {file}: {str(e)}")
                    error_widget = QLabel(f"Error: {os.path.basename(file)}\n{str(e)}")
                    error_widget.setStyleSheet("color: red; background-color: #FFEEEE; padding: 10px; border-radius: 5px;")
                    error_widget.setAlignment(Qt.AlignCenter)
                    row = i // 3
                    col = i % 3
                    self.results_layout.addWidget(error_widget, row, col)
            
            self.debug_console.log(f"Processed {len(files)} images successfully")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error initializing predictor: {str(e)}")
            self.debug_console.log(f"Error initializing predictor: {str(e)}")

    def confirm_data(self):
        if self.train_path_label.text() == "No folder selected" or \
           self.anno_path_label.text() == "No file selected":
            QMessageBox.warning(self, "Error", "Please select both training data and annotations first")
            return
            
        try:
            # Step 1: Prepare initial dataset
            self.debug_console.log("Step 1/2: Moving files to main directory...")
            add_dataset(
                new_dataset_path=self.train_path_label.text(),
                new_annotation_file=self.anno_path_label.text(),
                main_dir='data/raw/main',
                annotations_file='data/annotations/instances_default.json',
                update_splits=True
            )
            self.debug_console.log("Files moved successfully!")

            # Step 2: Split dataset
            self.debug_console.log("Step 2/2: Splitting dataset into train and validation sets...")
            split_dataset(
                main_dir='data/raw/main',
                train_dir='data/raw/train',
                val_dir='data/raw/val',
                annotations_file='data/annotations/instances_default.json',
                val_size=0.2
            )
            self.debug_console.log("Dataset split completed!")

            # Show success message with details
            QMessageBox.information(
                self, 
                "Data Preparation Complete", 
                "Data preparation completed successfully!\n\n"
                "Directory structure created:\n"
                "- data/raw/main (all images)\n"
                "- data/raw/train (training set)\n"
                "- data/raw/val (validation set)\n\n"
                "Annotation files created:\n"
                "- data/annotations/instances_default.json\n"
                "- data/annotations/instances_train.json\n"
                "- data/annotations/instances_val.json"
            )
            
            self.update_train_button()
            
        except Exception as e:
            error_msg = f"Error during data preparation: {str(e)}"
            self.debug_console.log(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application icon
    # app.setWindowIcon(QIcon("path/to/icon.png"))  # Uncomment and set path to icon
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())