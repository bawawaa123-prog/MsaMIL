import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Setup logging configuration
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    
def load_checkpoint(filepath, model=None, optimizer=None, device='cpu'):
    """
    Load model checkpoint
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def setup_seed(seed=42):
    """
    Setup random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dirs(*dirs):
    """
    Create directories if they don't exist
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def count_parameters(model):
    """
    Count trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalize_coordinates(coords, image_width, image_height):
    """
    Normalize coordinates to [0, 1] range
    
    Args:
        coords: [N, 2] - (x, y) coordinates
        image_width: original image width
        image_height: original image height
    
    Returns:
        normalized_coords: [N, 2] - normalized coordinates
    """
    normalized_coords = coords.copy()
    normalized_coords[:, 0] = coords[:, 0] / image_width   # x normalization
    normalized_coords[:, 1] = coords[:, 1] / image_height  # y normalization
    
    # Clamp to [0, 1] range
    normalized_coords = np.clip(normalized_coords, 0.0, 1.0)
    
    return normalized_coords

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None, save_path=None):
    """
    Plot training curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves (if provided)
    if train_accs is not None and val_accs is not None:
        axes[1].plot(train_accs, label='Train Acc', color='blue')
        axes[1].plot(val_accs, label='Val Acc', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Accuracy (No Data)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compute_class_weights(labels, num_classes=None):
    """
    Compute class weights for imbalanced datasets
    
    Args:
        labels: list or array of labels
        num_classes: number of classes (optional)
    
    Returns:
        class_weights: tensor of class weights
    """
    if num_classes is None:
        num_classes = len(np.unique(labels))
    
    label_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Compute weights: total_samples / (num_classes * class_count)
    class_weights = total_samples / (num_classes * label_counts)
    
    # Handle zero counts
    class_weights = np.where(label_counts == 0, 0, class_weights)
    
    return torch.FloatTensor(class_weights)

def print_model_summary(model, input_size=None):
    """
    Print model summary
    """
    print("Model Summary:")
    print("=" * 50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    if input_size:
        print(f"Input size: {input_size}")
    
    print("=" * 50)

class EarlyStopping:
    """
    Early stopping implementation
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        self.is_better = self._get_is_better_func()
        
    def _get_is_better_func(self):
        if self.mode == 'min':
            return lambda score, best: score < best - self.min_delta
        else:  # mode == 'max'
            return lambda score, best: score > best + self.min_delta
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def get_device(prefer_gpu=True):
    """
    Get the best available device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def format_time(seconds):
    """
    Format time in seconds to human readable format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def save_config(config_dict, save_path):
    """
    Save configuration to JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config_dict.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except TypeError:
            serializable_config[key] = str(value)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_config, f, indent=2, ensure_ascii=False)

def load_config(config_path):
    """
    Load configuration from JSON file
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config