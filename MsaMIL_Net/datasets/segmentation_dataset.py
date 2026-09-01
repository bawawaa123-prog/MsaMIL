import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PathologySegmentationDataset(Dataset):
    """
    病理分割数据集
    用于UNet++预训练，处理1024×1024病理图像和RGB掩码
    """
    
    def __init__(self, 
                 image_dir: str,
                 mask_dir: str,
                 image_size: int = 1024,
                 is_training: bool = True,
                 mask_format: str = 'rgb'):
        """
        Args:
            image_dir: 图像文件夹路径
            mask_dir: 掩码文件夹路径
            image_size: 图像尺寸 (论文要求1024×1024)
            is_training: 是否为训练模式
            mask_format: 掩码格式 ('rgb' 或 'grayscale')
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.is_training = is_training
        self.mask_format = mask_format
        
        # 获取图像文件列表
        self.image_files = []
        self.mask_files = []
        
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            image_files = sorted([f for f in os.listdir(image_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                
                # 尝试多种掩码文件扩展名
                mask_name = os.path.splitext(img_file)[0]
                mask_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
                
                mask_path = None
                for ext in mask_extensions:
                    potential_mask = os.path.join(mask_dir, mask_name + ext)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break
                
                if mask_path and os.path.exists(img_path):
                    self.image_files.append(img_path)
                    self.mask_files.append(mask_path)
        
        print(f"Found {len(self.image_files)} image-mask pairs in {image_dir}")
        
        # 数据增强策略
        if self.is_training:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.GaussianBlur(blur_limit=(1, 3), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        Returns:
            image: [3, H, W] 预处理后的图像
            mask: [H, W] 二值掩码 (0=背景, 1=病灶)
        """
        # 读取图像
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        try:
            # 读取图像 (RGB)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 读取掩码
            if self.mask_format == 'rgb':
                # RGB掩码：红色通道表示病灶区域
                mask_rgb = cv2.imread(mask_path)
                mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
                
                # 提取红色通道作为病灶掩码
                red_channel = mask_rgb[:, :, 0]
                mask = (red_channel > 128).astype(np.uint8)  # 二值化
                
            else:
                # 灰度掩码
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 128).astype(np.uint8)
            
            # 确保图像和掩码尺寸一致
            if image.shape[:2] != mask.shape:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # 应用数据增强
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed['image']  # [3, H, W]
            mask_tensor = torch.from_numpy(transformed['mask']).long()  # [H, W]
            
            return image_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # 返回零张量作为fallback
            return torch.zeros(3, self.image_size, self.image_size), \
                   torch.zeros(self.image_size, self.image_size, dtype=torch.long)
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重用于损失函数
        Returns:
            class_weights: [background_weight, lesion_weight]
        """
        total_pixels = 0
        lesion_pixels = 0
        
        print("计算类别权重...")
        
        for i in range(min(100, len(self))):  # 采样部分数据计算
            _, mask = self[i]
            total_pixels += mask.numel()
            lesion_pixels += (mask == 1).sum().item()
        
        background_pixels = total_pixels - lesion_pixels
        
        if lesion_pixels == 0:
            return torch.tensor([1.0, 2.0])
        
        # 反比例权重
        background_weight = total_pixels / (2 * background_pixels)
        lesion_weight = total_pixels / (2 * lesion_pixels)
        
        # 归一化
        total_weight = background_weight + lesion_weight
        background_weight /= total_weight
        lesion_weight /= total_weight
        
        # 增强病灶类别权重
        lesion_weight *= 2.0
        
        weights = torch.tensor([background_weight, lesion_weight])
        print(f"类别权重 - 背景: {background_weight:.3f}, 病灶: {lesion_weight:.3f}")
        
        return weights


class WSIClassificationDataset(Dataset):
    """
    WSI分类数据集
    用于端到端训练和MIL微调
    """
    
    def __init__(self, 
                 data_dir: str,
                 mode: str = 'train',
                 max_patches_per_wsi: int = 200):
        """
        Args:
            data_dir: 数据根目录
            mode: 'train' 或 'val'
            max_patches_per_wsi: 每个WSI最大patch数量
        """
        self.data_dir = data_dir
        self.mode = mode
        self.max_patches = max_patches_per_wsi
        
        # 构建数据路径
        mode_dir = os.path.join(data_dir, mode)
        
        self.wsi_paths = []
        self.labels = []
        self.class_names = []
        
        if os.path.exists(mode_dir):
            # 遍历类别文件夹
            class_dirs = sorted([d for d in os.listdir(mode_dir) 
                               if os.path.isdir(os.path.join(mode_dir, d))])
            
            for class_idx, class_name in enumerate(class_dirs):
                self.class_names.append(class_name)
                class_dir = os.path.join(mode_dir, class_name)
                
                # 获取WSI文件
                wsi_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.svs', '.tif', '.tiff', '.ndpi', '.mrxs'))]
                
                for wsi_file in wsi_files:
                    wsi_path = os.path.join(class_dir, wsi_file)
                    self.wsi_paths.append(wsi_path)
                    self.labels.append(class_idx)
        
        self.num_classes = len(self.class_names)
        print(f"Found {len(self.wsi_paths)} WSI files in {self.num_classes} classes")
        print(f"Classes: {self.class_names}")
    
    def __len__(self) -> int:
        return len(self.wsi_paths)
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """
        返回WSI路径和标签
        """
        return self.wsi_paths[idx], self.labels[idx]


def create_dataloaders(config: dict):
    """
    创建数据加载器
    """
    # UNet++预训练数据加载器
    if config.get('task') == 'segmentation':
        train_dataset = PathologySegmentationDataset(
            image_dir=config['train_image_dir'],
            mask_dir=config['train_mask_dir'],
            image_size=config['image_size'],
            is_training=True,
            mask_format=config.get('mask_format', 'rgb')
        )
        
        val_dataset = PathologySegmentationDataset(
            image_dir=config['val_image_dir'],
            mask_dir=config['val_mask_dir'],
            image_size=config['image_size'],
            is_training=False,
            mask_format=config.get('mask_format', 'rgb')
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )
        
        return train_loader, val_loader, train_dataset.get_class_weights()
    
    # WSI分类数据加载器
    elif config.get('task') == 'classification':
        train_dataset = WSIClassificationDataset(
            data_dir=config['wsi_data_dir'],
            mode='train',
            max_patches_per_wsi=config.get('max_patches_per_wsi', 200)
        )
        
        val_dataset = WSIClassificationDataset(
            data_dir=config['wsi_data_dir'],
            mode='val',
            max_patches_per_wsi=config.get('max_patches_per_wsi', 200)
        )
        
        return train_dataset, val_dataset
    
    else:
        raise ValueError(f"Unknown task: {config.get('task')}")

if __name__ == "__main__":
    # 测试分割数据集
    config = {
        'train_image_dir': 'data/pathology_seg/train/images',
        'train_mask_dir': 'data/pathology_seg/train/masks',
        'val_image_dir': 'data/pathology_seg/val/images',
        'val_mask_dir': 'data/pathology_seg/val/masks',
        'image_size': 1024,
        'batch_size': 2,
        'task': 'segmentation'
    }
    
    try:
        train_loader, val_loader, class_weights = create_dataloaders(config)
        print(f"? 数据加载器创建成功")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        print(f"类别权重: {class_weights}")
        
        # 测试一个批次
        for images, masks in train_loader:
            print(f"图像shape: {images.shape}")
            print(f"掩码shape: {masks.shape}")
            print(f"掩码值范围: {masks.min()}-{masks.max()}")
            break
            
    except Exception as e:
        print(f"? 数据加载器测试失败: {e}")