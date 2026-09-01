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
import cv2
from PIL import Image
import openslide
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    设置日志记录
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # 创建logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置文件
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换不可序列化的对象
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, torch.device):
            serializable_config[key] = str(value)
        elif callable(value):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   best_metric: float,
                   save_path: str,
                   additional_info: Optional[Dict] = None):
    """
    保存模型检查点
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'timestamp': datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)

def load_checkpoint(model: torch.nn.Module,
                   checkpoint_path: str,
                   device: torch.device,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict:
    """
    加载模型检查点
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器参数
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器参数
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def plot_training_history(history: Dict[str, List], save_path: str):
    """
    绘制训练历史曲线
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU曲线
    if 'train_iou' in history:
        axes[0, 1].plot(history['train_iou'], label='Train IoU', color='blue', linewidth=2)
        if 'val_iou' in history:
            axes[0, 1].plot(history['val_iou'], label='Val IoU', color='red', linewidth=2)
        axes[0, 1].set_title('IoU Curve')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 准确率曲线
    if 'train_acc' in history:
        axes[1, 0].plot(history['train_acc'], label='Train Acc', color='blue', linewidth=2)
        if 'val_acc' in history:
            axes[1, 0].plot(history['val_acc'], label='Val Acc', color='red', linewidth=2)
        axes[1, 0].set_title('Accuracy Curve')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], color='green', linewidth=2)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int],
                         class_names: List[str],
                         save_path: str,
                         normalize: bool = True):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(y_true: List[int], 
                     y_pred: List[int],
                     y_scores: Optional[List[float]] = None,
                     class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    计算分类指标
    """
    metrics = {}
    
    # 基本指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # 每类指标
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    if class_names:
        for i, class_name in enumerate(class_names):
            metrics[f'{class_name}_precision'] = precision_per_class[i]
            metrics[f'{class_name}_recall'] = recall_per_class[i]
            metrics[f'{class_name}_f1'] = f1_per_class[i]
    
    # AUC (如果有预测概率)
    if y_scores is not None:
        try:
            if len(set(y_true)) == 2:  # 二分类
                metrics['auc'] = roc_auc_score(y_true, y_scores)
            else:  # 多分类
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=list(range(len(set(y_true)))))
                metrics['auc'] = roc_auc_score(y_true_bin, y_scores, multi_class='ovr')
        except:
            pass
    
    return metrics

def extract_all_patches_from_wsi(wsi_path: str, 
                                patch_size: int = 512,
                                stride: int = 256,
                                level: int = 0,
                                max_patches: int = 10000) -> List[np.ndarray]:
    """
    从WSI中提取所有patches (用于端到端训练)
    
    Args:
        wsi_path: WSI文件路径
        patch_size: patch大小
        stride: 滑动步长
        level: 缩放级别
        max_patches: 最大patch数量
    
    Returns:
        patches: patch列表
    """
    patches = []
    
    try:
        # 打开WSI
        slide = openslide.OpenSlide(wsi_path)
        
        # 获取缩略图用于背景检测
        thumbnail = slide.get_thumbnail((1024, 1024))
        thumbnail_gray = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2GRAY)
        
        # 简单的背景检测(阈值化)
        _, tissue_mask = cv2.threshold(thumbnail_gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 计算缩放比例
        scale_x = slide.dimensions[0] / 1024
        scale_y = slide.dimensions[1] / 1024
        
        # 遍历tissue区域提取patches
        patch_count = 0
        for y in range(0, slide.dimensions[1] - patch_size, stride):
            for x in range(0, slide.dimensions[0] - patch_size, stride):
                if patch_count >= max_patches:
                    break
                
                # 检查是否在tissue区域内
                thumb_x = int(x / scale_x)
                thumb_y = int(y / scale_y)
                
                if (thumb_x < tissue_mask.shape[1] and thumb_y < tissue_mask.shape[0] and
                    tissue_mask[thumb_y, thumb_x] > 0):
                    
                    # 提取patch
                    patch = slide.read_region((x, y), level, (patch_size, patch_size))
                    patch = np.array(patch.convert('RGB'))
                    
                    # 简单的质量检查
                    if not is_background_patch(patch):
                        patches.append(patch)
                        patch_count += 1
            
            if patch_count >= max_patches:
                break
        
        slide.close()
        
    except Exception as e:
        print(f"Error extracting patches from {wsi_path}: {e}")
    
    return patches

def is_background_patch(patch: np.ndarray, 
                       white_threshold: int = 220,
                       white_ratio_threshold: float = 0.8) -> bool:
    """
    判断patch是否为背景
    """
    # 转为灰度图
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    
    # 计算白色像素比例
    white_pixels = np.sum(gray > white_threshold)
    total_pixels = gray.size
    white_ratio = white_pixels / total_pixels
    
    return white_ratio > white_ratio_threshold

def normalize_coordinates(coords: List[Tuple[int, int]], 
                         wsi_dimensions: Tuple[int, int]) -> torch.Tensor:
    """
    归一化坐标到[0, 1]范围
    """
    coords_array = np.array(coords, dtype=np.float32)
    coords_array[:, 0] /= wsi_dimensions[0]  # x坐标归一化
    coords_array[:, 1] /= wsi_dimensions[1]  # y坐标归一化
    
    return torch.from_numpy(coords_array)

def create_directory_structure(base_dir: str):
    """
    创建项目目录结构
    """
    directories = [
        'checkpoints/unet_pretrain',
        'checkpoints/end_to_end', 
        'checkpoints/mil_finetune',
        'logs',
        'data/pathology_seg/train/images',
        'data/pathology_seg/train/masks',
        'data/pathology_seg/val/images',
        'data/pathology_seg/val/masks',
        'data/wsi_classification/train',
        'data/wsi_classification/val',
        'data/features/train',
        'data/features/val'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
    
    print(f"? 目录结构创建完成: {base_dir}")

def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]):
    """
    打印模型参数统计
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"模型总参数量: {total_params:,}")
    print(f"输入尺寸: {input_size}")
    
    # 估算模型大小(MB)
    param_size = total_params * 4 / (1024 * 1024)  # 假设float32
    print(f"估算模型大小: {param_size:.2f} MB")

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

if __name__ == "__main__":
    # 测试辅助函数
    print("Testing helper functions...")
    
    # 创建目录结构
    create_directory_structure("test_project")
    
    # 测试日志
    logger = setup_logging("test_project/logs", "test_experiment")
    logger.info("This is a test log message")
    
    print("? Helper functions test completed")