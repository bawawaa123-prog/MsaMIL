import os
import sys
import time
import math
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("segmentation_models_pytorch is required. Install via pip install segmentation-models-pytorch")

# 进度条（可选）
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class SegDataset(Dataset):
    """
    语义分割数据集：
    - 图像为RGB，大小为1024x1024（或任意，可自动resize到目标尺寸）
    - 掩码为RGB三通道，红色通道为病灶，需二值化
    目录结构示例：
    data_root/
      train/
        images/  (*.png/*.jpg)
        masks/   (与images同名的掩码)
      val/
        images/
        masks/
    若没有images/masks子目录，则尝试在train/或val/下直接按同名文件匹配（或带mask后缀）。
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def __init__(self,
                 root: str,
                 split: str = "train",
                 images_subdir: str = "images",
                 masks_subdir: str = "masks",
                 mask_suffix: str = "",
                 img_size: int = 1024,
                 red_threshold: int = 127,
                 normalize: bool = True,
                 augment: bool = False):
        super().__init__()
        self.root = root
        self.split = split
        self.images_subdir = images_subdir
        self.masks_subdir = masks_subdir
        self.mask_suffix = mask_suffix
        self.img_size = img_size
        self.red_threshold = red_threshold
        self.normalize = normalize
        self.augment = augment and (split == "train")

        split_dir = os.path.join(root, split)
        img_dir = os.path.join(split_dir, images_subdir)
        mask_dir = os.path.join(split_dir, masks_subdir)

        pairs = []
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            # 标准 images/ 和 masks/ 结构
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith(self.IMG_EXTS):
                    img_path = os.path.join(img_dir, fn)
                    mask_name = os.path.splitext(fn)[0] + self.mask_suffix + os.path.splitext(fn)[1]
                    mask_path = os.path.join(mask_dir, mask_name)
                    if os.path.exists(mask_path):
                        pairs.append((img_path, mask_path))
        else:
            # 回退：在split目录下直接配对
            fns = [f for f in os.listdir(split_dir) if f.lower().endswith(self.IMG_EXTS)]
            for fn in sorted(fns):
                img_path = os.path.join(split_dir, fn)
                stem, ext = os.path.splitext(fn)
                candidates = [
                    os.path.join(split_dir, stem + self.mask_suffix + ext),
                    os.path.join(split_dir, stem + "_mask" + ext),
                    os.path.join(split_dir, stem + "-mask" + ext),
                ]
                mask_path = None
                for c in candidates:
                    if os.path.exists(c):
                        mask_path = c
                        break
                # 若有单独masks目录也尝试找
                if mask_path is None and os.path.isdir(mask_dir):
                    c = os.path.join(mask_dir, stem + self.mask_suffix + ext)
                    if os.path.exists(c):
                        mask_path = c
                if mask_path is not None:
                    pairs.append((img_path, mask_path))

        if len(pairs) == 0:
            raise FileNotFoundError(f"No image-mask pairs found under {split_dir}. Please ensure images/masks subdirs or matching filenames exist.")

        self.pairs = pairs
        print(f"✓ {split}: found {len(self.pairs)} pairs")

        # 预定义归一化（ImageNet，适配EfficientNet-B3）
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def _resize(self, img: np.ndarray, size: int) -> np.ndarray:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    def _maybe_augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return img, mask
        # 简单增强：水平/垂直翻转 + 随机90度旋转 + 亮度对比度
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        # 旋转 0/90/180/270
        k = np.random.randint(0, 4)
        if k:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        # 亮度/对比度
        if np.random.rand() < 0.5:
            alpha = 0.8 + 0.4 * np.random.rand()  # 0.8~1.2
            beta = np.random.randint(-20, 21)     # -20~20
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        # 读取RGB
        img = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        # 提取红色通道并二值化（红色为病灶）
        red = mask_rgb[:, :, 0]
        mask_bin = (red > self.red_threshold).astype(np.uint8)

        # resize到目标尺寸
        if (img.shape[0] != self.img_size) or (img.shape[1] != self.img_size):
            img = self._resize(img, self.img_size)
            mask_bin = self._resize(mask_bin, self.img_size)
            mask_bin = (mask_bin > 127).astype(np.uint8)

        # 增强
        img, mask_bin = self._maybe_augment(img, mask_bin)

        # 归一化
        img = img.astype(np.float32) / 255.0
        if self.normalize:
            img = (img - self.mean) / self.std

        # To tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # [3,H,W]
        mask = torch.from_numpy(mask_bin).long()                # [H,W], 0/1

        return img, mask


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits: [B,2,H,W]
    targets: [B,H,W] in {0,1}
    只对前景通道计算Dice Loss。
    """
    probs = torch.softmax(logits, dim=1)[:, 1, ...]  # foreground prob [B,H,W]
    targets_f = targets.float()
    inter = (probs * targets_f).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2)) + eps
    dice = (2 * inter + eps) / union
    return 1.0 - dice.mean()


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """preds/targets: [B,H,W] in {0,1}"""
    inter = (preds & targets).sum().item()
    union = (preds | targets).sum().item() + eps
    return inter / union


def compute_train_fg_ratio(pairs, img_size: int, red_threshold: int) -> float:
    """快速统计训练集前景像素占比（不做增强/归一化）。"""
    total_fg = 0
    total_px = 0
    for img_path, mask_path in pairs:
        try:
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            red = mask_rgb[:, :, 0]
            if mask_rgb.shape[0] != img_size or mask_rgb.shape[1] != img_size:
                red = cv2.resize(red, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            fg = (red > red_threshold).astype(np.uint8)
            total_fg += int(fg.sum())
            total_px += int(fg.size)
        except Exception:
            continue
    if total_px == 0:
        return 0.0
    return total_fg / total_px


def train_one_epoch(model, loader, optimizer, device, ce_loss, dice_weight: float, scaler: GradScaler, use_amp: bool, show_pbar: bool = True, grad_clip: float = 0.0):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    iterator = loader if (tqdm is None or not show_pbar) else tqdm(loader, desc="Train", leave=False, ncols=100)
    for imgs, masks in iterator:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast(device_type='cuda' if str(device).startswith('cuda') else 'cpu'):
                logits = model(imgs)
                loss_ce = ce_loss(logits, masks)
                loss_dice = dice_loss_from_logits(logits, masks)
                loss = loss_ce + dice_weight * loss_dice
        else:
            logits = model(imgs)
            loss_ce = ce_loss(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = loss_ce + dice_weight * loss_dice

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1).bool()
            iou = compute_iou(preds, masks.bool())

        total_loss += loss.item()
        total_iou += iou
        n_batches += 1

    return total_loss / max(1, n_batches), total_iou / max(1, n_batches)


@torch.no_grad()
def validate(model, loader, device, ce_loss, dice_weight: float, show_pbar: bool = True):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    iterator = loader if (tqdm is None or not show_pbar) else tqdm(loader, desc="Val", leave=False, ncols=100)
    for imgs, masks in iterator:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss_ce = ce_loss(logits, masks)
        loss_dice = dice_loss_from_logits(logits, masks)
        loss = loss_ce + dice_weight * loss_dice

        preds = torch.argmax(logits, dim=1).bool()
        iou = compute_iou(preds, masks.bool())

        total_loss += loss.item()
        total_iou += iou
        n_batches += 1

    return total_loss / max(1, n_batches), total_iou / max(1, n_batches)


@torch.no_grad()
def save_random_visualizations(model,
                               dataset: Dataset,
                               device,
                               mean: np.ndarray,
                               std: np.ndarray,
                               save_dir: str,
                               epoch: int,
                               num_samples: int = 4,
                               img_size: int = 1024,
                               use_amp: bool = False):
    """随机从dataset中抽取若干样本，保存 原图-预测-标注 的拼图。"""
    if num_samples <= 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    H = W = img_size

    # 随机选择索引
    indices = list(range(len(dataset)))
    if len(indices) == 0:
        return
    if len(indices) <= num_samples:
        sel = indices
    else:
        sel = random.sample(indices, num_samples)

    model.eval()
    to_uint8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)

    for i, idx in enumerate(sel):
        img_t, mask_t = dataset[idx]  # img:[3,H,W] (norm), mask:[H,W]
        img_in = img_t.unsqueeze(0).to(device)

        # 推理
        if use_amp:
            with autocast(device_type='cuda' if str(device).startswith('cuda') else 'cpu'):
                logits = model(img_in)
        else:
            logits = model(img_in)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)  # [H,W]

        # 反归一化到原图显示
        img_np = img_t.cpu().numpy().transpose(1, 2, 0)  # [H,W,3]
        img_np = (img_np * std[None, None, :]) + mean[None, None, :]
        img_np = to_uint8(img_np * 255.0)

        # GT
        gt_np = mask_t.cpu().numpy().astype(np.uint8)

        # 构造红色mask图（预测/GT）
        pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        pred_rgb[:, :, 0] = pred * 255
        gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        gt_rgb[:, :, 0] = gt_np * 255

        # 拼接：原图 | 预测 | GT
        panel = np.concatenate([img_np, pred_rgb, gt_rgb], axis=1)

        out_path = os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{i+1}.png")
        Image.fromarray(panel).save(out_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train SMP Unet++ (EfficientNet-B3) for lesion segmentation (red channel masks)")
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录（包含train/与val/子目录）')
    parser.add_argument('--save_dir', type=str, default='checkpoints/unet_smp_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--images_subdir', type=str, default='images')
    parser.add_argument('--masks_subdir', type=str, default='masks')
    parser.add_argument('--mask_suffix', type=str, default='', help='掩码文件名与图像同名时的后缀（若有）')
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--red_threshold', type=int, default=127, help='红通道二值化阈值(0-255)')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_fg', type=float, default=5.0, help='CE损失中前景类别权重')
    parser.add_argument('--weight_fg_auto', action='store_true', help='自动根据训练集前景像素占比设置前景权重')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Dice损失的权重')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='梯度裁剪阈值（0关闭）')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心（基于Val IoU）')
    parser.add_argument('--amp', action='store_true', help='启用混合精度')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'plateau'], help='学习率调度器')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Cosine最小学习率')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='SMP编码器预训练权重（例如 imagenet 或 noisy-student，如可用）')
    parser.add_argument('--vis_samples', type=int, default=4, help='每个epoch随机保存的可视化样本数')
    parser.add_argument('--vis_split', type=str, default='val', choices=['train', 'val'], help='从哪个数据集抽样可视化')
    parser.add_argument('--no_pbar', action='store_true', help='禁用进度条')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets & Loaders
    train_set = SegDataset(
        root=args.data_root, split='train',
        images_subdir=args.images_subdir, masks_subdir=args.masks_subdir,
        mask_suffix=args.mask_suffix, img_size=args.img_size,
        red_threshold=args.red_threshold, normalize=True, augment=True,
    )
    val_set = SegDataset(
        root=args.data_root, split='val',
        images_subdir=args.images_subdir, masks_subdir=args.masks_subdir,
        mask_suffix=args.mask_suffix, img_size=args.img_size,
        red_threshold=args.red_threshold, normalize=True, augment=False,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size//2), shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Model (与SFFM一致：classes=2)
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b3',
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=2,
        activation=None
    ).to(device)

    # Loss & Optimizer
    # 自动前景权重（按像素占比反比设权重，clip到[1, 20]）
    if args.weight_fg_auto:
        fg_ratio = compute_train_fg_ratio(train_set.pairs, args.img_size, args.red_threshold)
        # 防止除零/极端情况
        eps = 1e-6
        auto_w = float((1.0 - fg_ratio + eps) / (fg_ratio + eps))
        auto_w = float(np.clip(auto_w, 1.0, 20.0))
        print(f"Auto foreground weight computed from train masks: ratio={fg_ratio:.6f}, weight_fg={auto_w:.3f}")
        weight_fg = auto_w
    else:
        weight_fg = args.weight_fg

    class_weights = torch.tensor([1.0, weight_fg], dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=args.min_lr, verbose=True)
    else:
        scheduler = None

    scaler = GradScaler(enabled=args.amp)

    best_val_iou = 0.0
    patience_cnt = 0
    best_ckpt_path = os.path.join(args.save_dir, 'best_smp_unet_iou.pth')

    print("Start training...")
    vis_dir = os.path.join(args.save_dir, 'vis')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, device, ce_loss,
            args.dice_weight, scaler, args.amp, show_pbar=not args.no_pbar, grad_clip=args.grad_clip
        )
        val_loss, val_iou = validate(
            model, val_loader, device, ce_loss,
            args.dice_weight, show_pbar=not args.no_pbar
        )
        dt = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs} | {dt:.1f}s\n"
              f"  Train  - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}\n"
              f"  Val    - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

        # 每个epoch后保存可视化样本
        vis_dataset = train_set if args.vis_split == 'train' else val_set
        save_random_visualizations(
            model=model,
            dataset=vis_dataset,
            device=device,
            mean=val_set.mean,  # 与训练时一致
            std=val_set.std,
            save_dir=vis_dir,
            epoch=epoch + 1,
            num_samples=args.vis_samples,
            img_size=args.img_size,
            use_amp=args.amp,
        )

        # 调度器步进
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_iou)
            else:
                scheduler.step()

        # 早停 & 保存最优
        if val_iou > best_val_iou + 1e-4:
            best_val_iou = val_iou
            patience_cnt = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'iou': float(best_val_iou)}
            }, best_ckpt_path)
            print(f"  ✓ New best IoU: {best_val_iou:.4f}. Saved to {best_ckpt_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val IoU: {best_val_iou:.4f}")
                break

    print(f"Training done. Best Val IoU: {best_val_iou:.4f}")
    print(f"Best checkpoint saved at: {best_ckpt_path}")


if __name__ == '__main__':
    main()
