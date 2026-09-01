import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class PathologySegDataset(Dataset):
    def __init__(self, images: list[Path], masks: list[Path], image_size: int = 1024,
                 augment: bool = False, strong_aug: bool = False, seed: int = 42):
        assert len(images) == len(masks), "Number of images and masks must match"
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.augment = augment
        self.strong_aug = strong_aug
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)
        random.seed(seed)

    @staticmethod
    def _load_image(path: Path, size: int) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.BILINEAR)
        return img

    @staticmethod
    def _load_mask_red(path: Path, size: int) -> Image.Image:
        m = Image.open(path).convert("RGB")
        if size is not None:
            m = m.resize((size, size), Image.NEAREST)
        m_np = np.array(m)
        R = m_np[:, :, 0].astype(np.uint8)
        fg = (R > 127).astype(np.uint8)
        return Image.fromarray(fg * 255)

    @staticmethod
    def _rand_k90(img: Image.Image, mask: Image.Image):
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k, expand=True)
            mask = mask.rotate(90 * k, expand=True)
        return img, mask

    @staticmethod
    def _hflip(img: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

    @staticmethod
    def _vflip(img: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

    @staticmethod
    def _random_affine(img: Image.Image, mask: Image.Image, angle_range: float = 10.0, scale_min: float = 0.95,
                       scale_max: float = 1.05, out_size: int = 1024):
        angle = random.uniform(-angle_range, angle_range)
        scale = random.uniform(scale_min, scale_max)
        img_t = TF.affine(img, angle=angle, translate=(0, 0), scale=scale, shear=0,
                          interpolation=transforms.InterpolationMode.BILINEAR)
        mask_t = TF.affine(mask, angle=angle, translate=(0, 0), scale=scale, shear=0,
                           interpolation=transforms.InterpolationMode.NEAREST)
        img_t = img_t.resize((out_size, out_size), Image.BILINEAR)
        mask_t = mask_t.resize((out_size, out_size), Image.NEAREST)
        return img_t, mask_t

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        img = self._load_image(img_path, self.image_size)
        mask = self._load_mask_red(mask_path, self.image_size)

        if self.augment:
            img, mask = self._rand_k90(img, mask)
            img, mask = self._hflip(img, mask)
            img, mask = self._vflip(img, mask)
            if self.strong_aug:
                img, mask = self._random_affine(img, mask, out_size=self.image_size)
                img = self.color_jitter(img)

        img_t = self.to_tensor(img)
        mask_np = (np.array(mask) > 127).astype(np.float32)  
        mask_t = torch.from_numpy(mask_np)  
        return img_t, mask_t


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, dice_eps: float = 1e-6,
                  pos_weight: float | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pos_weight is not None and pos_weight > 0 and pos_weight != 1.0:
        pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), targets, pos_weight=pw)
    else:
        bce = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), targets)
    probs = torch.sigmoid(logits).squeeze(1)
    inter = torch.sum(probs * targets)
    union = torch.sum(probs) + torch.sum(targets)
    dice = (2 * inter + dice_eps) / (union + dice_eps)
    dice_loss = 1.0 - dice
    return bce + dice_loss, bce, dice_loss


@torch.no_grad()
def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits).squeeze(1)
    preds = (probs > thresh).float()
    inter = torch.sum(preds * targets).item()
    union = torch.sum(preds).item() + torch.sum(targets).item() - inter
    if union == 0:
        return 1.0
    return (inter + eps) / (union + eps)


def build_file_lists(root: Path, split: str | None, val_ratio: float, seed: int):
    if split is not None:
        images_dir = root / split / "images"
        masks_dir = root / split / "masks"
        images = sorted([p for p in images_dir.iterdir() if _is_image(p)])
        def map_mask(p: Path):
            # Common same-stem extensions
            for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
                cand = masks_dir / f"{p.stem}{ext}"
                if cand.exists():
                    return cand
            # Compatibility: *_mask naming
            for suf in ["_mask", "-mask", "_label", "-label", ".mask"]:
                cand = masks_dir / f"{p.stem}{suf}.png"
                if cand.exists():
                    return cand
            raise FileNotFoundError(f"Missing mask for: {p.stem}.* in {masks_dir}")
        masks = [map_mask(p) for p in images]
        return images, masks

    images_dir = root / "images"
    masks_dir = root / "masks"
    files = sorted([p for p in images_dir.iterdir() if _is_image(p)])
    rnd = random.Random(seed)
    rnd.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_imgs = files[:n_val]
    train_imgs = files[n_val:]
    def map_mask(p: Path):
        for ext in [".png", ".jpg", ".tif", ".tiff", ".bmp"]:
            cand = masks_dir / f"{p.stem}{ext}"
            if cand.exists():
                return cand

        for suf in ["_mask", "-mask", "_label", "-label", ".mask"]:
            cand = masks_dir / f"{p.stem}{suf}.png"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Missing mask for: {p.stem}.*")
    train_masks = [map_mask(p) for p in train_imgs]
    val_masks = [map_mask(p) for p in val_imgs]
    return (train_imgs, train_masks), (val_imgs, val_masks)


def run_epoch(model, loader, optimizer, device, scaler=None, train: bool = True,
              dropout_layer: nn.Module | None = None, pos_weight: float | None = None):
    model.train(train)
    epoch_loss = 0.0
    epoch_iou = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="Train" if train else "Val", ncols=100)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outer = (lambda: torch.enable_grad()) if train else torch.inference_mode
        with outer():
            amp_enabled = scaler is not None and device.type == 'cuda'
            with torch.amp.autocast(device_type=str(device.type), enabled=amp_enabled):
                logits = model(imgs)  # [B,1,H,W]
                if dropout_layer is not None and train:
                    logits = dropout_layer(logits)
                loss, bce, dsc = bce_dice_loss(logits, masks, pos_weight=pos_weight)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        epoch_loss += loss.item()
        epoch_iou += iou_from_logits(logits, masks)
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "IoU": f"{(epoch_iou/n_batches):.4f}"})
    return epoch_loss / max(n_batches, 1), epoch_iou / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, device, thresholds: list[float], pos_weight: float | None = None,
                   tta: bool = False):
    model.eval()
    epoch_loss = 0.0
    n_batches = 0
    iou_sums = {float(t): 0.0 for t in thresholds}
    pbar = tqdm(loader, desc="Val", ncols=100)
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=str(device.type), enabled=(device.type == 'cuda')):
            if not tta:
                logits = model(imgs)
            else:
                logits1 = model(imgs)
                imgs_hf = torch.flip(imgs, dims=[3])
                logits2 = model(imgs_hf)
                logits2 = torch.flip(logits2, dims=[3])
                logits = (logits1 + logits2) / 2
            loss, _, _ = bce_dice_loss(logits, targets, pos_weight=pos_weight)
        probs = torch.sigmoid(logits).squeeze(1)  # [B,H,W]
        for t in thresholds:
            preds = (probs > t).float()
            inter = torch.sum(preds * targets)
            union = torch.sum(preds) + torch.sum(targets) - inter
            batch_iou = 1.0 if union.item() == 0 else ((inter + 1e-6) / (union + 1e-6)).item()
            iou_sums[float(t)] += batch_iou
        epoch_loss += loss.item()
        n_batches += 1
        iou_05 = None
        if 0.5 in iou_sums:
            iou_05 = iou_sums[0.5] / n_batches
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "IoU@0.5": f"{(iou_05 if iou_05 is not None else 0):.4f}"})

    avg_loss = epoch_loss / max(n_batches, 1)
    iou_avg = {t: (s / max(n_batches, 1)) for t, s in iou_sums.items()}
    best_thr = max(iou_avg, key=iou_avg.get)
    best_iou = iou_avg[best_thr]
    return avg_loss, iou_avg, float(best_thr), float(best_iou)


@torch.no_grad()
def save_epoch_visuals(model, dataset, device, save_dir: Path, epoch: int, num_samples: int = 5,
                       seed: int = 42, split: str = "val", bin_thresh: float = 0.5):
    model.eval()
    if len(dataset) == 0 or num_samples <= 0:
        return
    rng = random.Random(seed * 1000 + epoch * 13 + (0 if split == 'train' else 1))
    idxs = [rng.randrange(0, len(dataset)) for _ in range(min(num_samples, len(dataset)))]

    vis_dir = save_dir / f"vis_{split}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for k, idx in enumerate(idxs):
        img_t, mask_t = dataset[idx]
        stem = None
        try:
            stem = Path(dataset.images[idx]).stem
        except Exception:
            pass
        img = img_t.unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(img)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        img_np = (img_t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        gt_np = mask_t.numpy().astype(np.uint8) * 255
        pred_np = (probs > bin_thresh).astype(np.uint8) * 255

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(gt_np, cmap="Reds")
        axs[1].set_title("GT")
        axs[1].axis("off")

        axs[2].imshow(pred_np, cmap="Reds")
        axs[2].set_title("Pred")
        axs[2].axis("off")

        name_part = stem if stem is not None else f"sample_{k+1}"
        out_path = vis_dir / f"epoch_{epoch:03d}_{split}_thr{bin_thresh:.2f}_{name_part}.png"
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train SMP Unet++ (binary, single-channel) for SFFM pre-segmentation")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Dataset root containing images/ and masks/, or train/ and val/ subfolders")
    parser.add_argument("--save_dir", type=str, default=f"checkpoints/new_unet_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--encoder", type=str, default="efficientnet-b3")
    parser.add_argument(
        "--encoder_weights",
        type=str,
        default="imagenet",
        help="Encoder pretrained weights. Use 'imagenet' (default) or 'none' to disable downloads.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.12, help="If no val/ folder, split validation by this ratio")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dropout2d", type=float, default=0.2, help="Apply Dropout2d on logits to reduce overfitting (0 disables)")
    parser.add_argument("--early_stop_patience", type=int, default=30, help="Early stopping patience (0 disables)")
    parser.add_argument("--strong_aug", action="store_true", help="Use stronger augmentations (affine + color jitter)")
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Positive class weight for BCE; use >1.0 for imbalance")
    parser.add_argument("--thr_min", type=float, default=0.30, help="Min threshold for validation scan")
    parser.add_argument("--thr_max", type=float, default=0.70, help="Max threshold for validation scan")
    parser.add_argument("--thr_step", type=float, default=0.05, help="Step size for validation threshold scan")
    parser.add_argument("--val_tta", action="store_true", help="Use simple TTA (horizontal flip) during validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    if (root / "train").exists() and (root / "val").exists():
        train_imgs, train_masks = build_file_lists(root, "train", args.val_ratio, args.seed)
        val_imgs, val_masks = build_file_lists(root, "val", args.val_ratio, args.seed)
    else:
        (train_imgs, train_masks), (val_imgs, val_masks) = build_file_lists(root, None, args.val_ratio, args.seed)

    train_ds = PathologySegDataset(train_imgs, train_masks, image_size=args.image_size, augment=True, strong_aug=args.strong_aug)
    val_ds = PathologySegDataset(val_imgs, val_masks, image_size=args.image_size, augment=False)
    def _seed_worker(worker_id: int) -> None:
        # Ensure each worker has a different, deterministic RNG stream.
        base = int(args.seed)
        s = base + int(worker_id) + 1337
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker if args.num_workers and args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker if args.num_workers and args.num_workers > 0 else None,
    )

    # Model: Unet++ + optional ImageNet encoder pretraining; binary segmentation (classes=1)
    enc_w = str(args.encoder_weights).strip().lower()
    encoder_weights = None if enc_w in {"none", "null", ""} else args.encoder_weights
    model = smp.UnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == 'cuda'))

    best_iou_so_far = 0.0
    history = []
    dropout_layer = nn.Dropout2d(p=args.dropout2d) if args.dropout2d and args.dropout2d > 0 else None
    # Early stopping counter
    epochs_no_improve = 0
    # Pre-build threshold list
    thresholds = np.arange(args.thr_min, args.thr_max + 1e-8, args.thr_step).tolist()
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss, train_iou = run_epoch(model, train_loader, optimizer, device, scaler=scaler, train=True,
                                          dropout_layer=dropout_layer, pos_weight=args.pos_weight)
        val_loss, iou_map, val_best_thr, val_best_iou = evaluate_epoch(model, val_loader, device, thresholds,
                                                                       pos_weight=args.pos_weight, tta=args.val_tta)
        scheduler.step(val_best_iou)

        print(f"Train  - Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        val_iou_05 = iou_map.get(0.5, None)
        if val_iou_05 is None:
            # Convenience display for IoU@0.5 (does not affect scheduler)
            val_iou_05 = list(iou_map.values())[len(iou_map)//2]
        print(f"Val    - Loss: {val_loss:.4f} | IoU@0.5: {val_iou_05:.4f} | BestIoU: {val_best_iou:.4f} @thr={val_best_thr:.2f}")
        print(f"LR     - {optimizer.param_groups[0]['lr']:.6f}")

        # Save per-epoch visualizations
        try:
            save_epoch_visuals(model, train_ds, device, save_dir, epoch=epoch, num_samples=5, split='train', bin_thresh=0.5)
            save_epoch_visuals(model, val_ds, device, save_dir, epoch=epoch, num_samples=5, split='val', bin_thresh=val_best_thr)
        except Exception as e:
            print(f"Failed to save visualizations: {e}")

        # Save best checkpoint (based on best threshold IoU for this epoch)
        if val_best_iou > best_iou_so_far:
            best_iou_so_far = val_best_iou
            ckpt = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'metrics': {
                    'iou_best': float(best_iou_so_far),
                    'best_threshold': float(val_best_thr),
                    'iou_map': {f"{k:.2f}": float(v) for k, v in iou_map.items()},
                },
                'encoder': args.encoder,
                'image_size': args.image_size,
            }
            torch.save(ckpt, save_dir / 'best_model.pth')
            print(f"Saved best model to {save_dir / 'best_model.pth'} (Val BestIoU={best_iou_so_far:.4f} @thr={val_best_thr:.2f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping: val IoU did not improve for {epochs_no_improve} epochs (patience={args.early_stop_patience})")
                break

        # Append history
        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_iou': float(train_iou),
            'val_loss': float(val_loss),
            'val_iou_05': float(val_iou_05),
            'val_iou_best': float(val_best_iou),
            'val_best_thr': float(val_best_thr),
            'lr': float(optimizer.param_groups[0]['lr'])
        })

        # Note: early stopping is optional; training can be stopped by patience or epochs.

    # Save training history
    try:
        import json
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Failed to save training history: {e}")

    print(f"Training finished. Best validation IoU: {best_iou_so_far:.4f}")
    print(f"Checkpoint: {save_dir / 'best_model.pth'}\nUse this path as SFFM(unet_model_path=...) to load.")


if __name__ == "__main__":
    main()
