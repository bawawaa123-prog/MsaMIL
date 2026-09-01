import os
import sys
import traceback
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.segmentation_dataset import PathologySegmentationDataset
from models.SFFM import UNetPlusPlus, CombinedLoss
import numpy as np
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = len(train_loader)
    
    epoch_start = datetime.now()
    
    with tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}") as pbar:
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss, ce_loss, dice_loss = criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                pred_masks = torch.argmax(outputs, dim=1)
                iou = calculate_iou(pred_masks, masks)
                dice = calculate_dice(pred_masks, masks)
            
            total_loss += loss.item()
            total_iou += iou
            total_dice += dice
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
    
    epoch_time = (datetime.now() - epoch_start).total_seconds()
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    
    print(f"Train - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}, Time: {epoch_time:.1f}s")
    
    return avg_loss, avg_iou, avg_dice, epoch_time

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss, _, _ = criterion(outputs, masks)
            
            pred_masks = torch.argmax(outputs, dim=1)
            iou = calculate_iou(pred_masks, masks)
            dice = calculate_dice(pred_masks, masks)
            
            total_loss += loss.item()
            total_iou += iou
            total_dice += dice
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    
    print(f"Val   - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")
    
    return avg_loss, avg_iou, avg_dice

def calculate_iou(pred, target):
    """计算IoU"""
    intersection = torch.logical_and(pred == 1, target == 1).sum().float()
    union = torch.logical_or(pred == 1, target == 1).sum().float()
    return (intersection / (union + 1e-8)).item()

def calculate_dice(pred, target):
    """计算Dice系数"""
    intersection = torch.logical_and(pred == 1, target == 1).sum().float()
    total = (pred == 1).sum().float() + (target == 1).sum().float()
    return (2.0 * intersection / (total + 1e-8)).item()

def main():
    """主训练函数"""
    
    # 配置参数 - 修改：改为1024分辨率
    config = {
        'train_image_dir': r'data/pathology_seg/train/images',
        'train_mask_dir': r'data/pathology_seg/train/masks',
        'val_image_dir': r'data/pathology_seg/val/images', 
        'val_mask_dir': r'data/pathology_seg/val/masks',
        
        'batch_size': 2,  # 1024分辨率需要更小的batch size
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'image_size': 1024,  # 修改：改为1024，符合论文要求
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        'save_dir': 'checkpoints/unet_pretrain',
        'log_dir': 'logs/unet_pretrain',
        
        'ce_weight': 0.4,
        'dice_weight': 0.6,
        'class_weights': [1.0, 2.0],
        
        'save_interval': 10,
        'val_interval': 1,
        'early_stopping_patience': 15,
    }
    
    print("=" * 60)
    print("UNet++ Pretraining Configuration (1024x1024)")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:25}: {value}")
    print("=" * 60)
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 检查数据路径
    for path_key in ['train_image_dir', 'train_mask_dir', 'val_image_dir', 'val_mask_dir']:
        if not os.path.exists(config[path_key]):
            print(f"? Path not found: {config[path_key]}")
            print("\n请按以下格式准备数据集：")
            print("data/pathology_seg/")
            print("├── train/")
            print("│   ├── images/  <- 训练图像(1024x1024)")
            print("│   └── masks/   <- RGB掩码（红色通道=病灶）")
            print("└── val/")
            print("    ├── images/  <- 验证图像(1024x1024)")
            print("    └── masks/   <- RGB掩码")
            return
    
    # 创建数据集 - 修改：使用1024分辨率
    train_dataset = PathologySegmentationDataset(
        image_dir=config['train_image_dir'],
        mask_dir=config['train_mask_dir'],
        image_size=config['image_size'],  # 1024
        is_training=True,
        mask_format='rgb'
    )
    
    val_dataset = PathologySegmentationDataset(
        image_dir=config['val_image_dir'], 
        mask_dir=config['val_mask_dir'],
        image_size=config['image_size'],  # 1024
        is_training=False,
        mask_format='rgb'
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("? 数据集为空，请检查数据路径和格式")
        return
    
    print(f"? 训练样本数量: {len(train_dataset)}")
    print(f"? 验证样本数量: {len(val_dataset)}")
    
    # 创建数据加载器 - 修改：减少worker数量
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=1,  # 1024分辨率减少worker
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=1,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    model = UNetPlusPlus(in_channels=3, out_channels=2, features=[32, 64, 128, 256])
    model.to(device)
    
    print(f"? Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = CombinedLoss(
        ce_weight=config['ce_weight'], 
        dice_weight=config['dice_weight'],
        class_weights=config['class_weights']
    )
    criterion.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=8, 
        factor=0.5,
        min_lr=1e-7,
        verbose=True
    )
    
    # 训练历史
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': [],
        'epoch_time': []
    }
    
    best_val_iou = 0.0
    early_stopping_counter = 0
    
    print("\n" + "=" * 60)
    print("开始训练UNet++分割模型 (1024x1024)")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        for epoch in range(1, config['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{config['num_epochs']}")
            print("-" * 50)
            
            # 训练
            train_loss, train_iou, train_dice, epoch_time = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, config['num_epochs']
            )
            
            # 验证
            if epoch % config['val_interval'] == 0:
                val_loss, val_iou, val_dice = validate_epoch(model, val_loader, criterion, device)
                
                # 学习率调度
                scheduler.step(val_iou)
                
                # 记录历史
                history['train_loss'].append(train_loss)
                history['train_iou'].append(train_iou)
                history['train_dice'].append(train_dice)
                history['val_loss'].append(val_loss)
                history['val_iou'].append(val_iou)
                history['val_dice'].append(val_dice)
                history['epoch_time'].append(epoch_time)
                
                # 保存最佳模型
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    early_stopping_counter = 0
                    
                    # 保存最佳模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_iou': val_iou,
                        'val_dice': val_dice,
                        'config': config
                    }, os.path.join(config['save_dir'], 'best_unet_iou.pth'))
                    
                    print(f"? Best model saved! Val IoU: {val_iou:.4f}")
                else:
                    early_stopping_counter += 1
                
                # 早停检查
                if early_stopping_counter >= config['early_stopping_patience']:
                    print(f"Early stopping triggered! Best Val IoU: {best_val_iou:.4f}")
                    break
                
                # 定期保存
                if epoch % config['save_interval'] == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_iou': val_iou,
                        'history': history
                    }, os.path.join(config['save_dir'], f'unet_epoch_{epoch}.pth'))
    
    except KeyboardInterrupt:
        print("\n?? Training interrupted by user")
        
    except Exception as e:
        print(f"\n? Training error: {e}")
        traceback.print_exc()
    
    finally:
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n? Training completed! Total time: {total_time/3600:.2f} hours")
        print(f"? Best validation IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    main()