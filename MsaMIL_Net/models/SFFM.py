import os
import csv
import json
import math
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import openslide  # type: ignore
except Exception:
    openslide = None


try:
    import segmentation_models_pytorch as smp
    print("✓ SFFM: SMP库导入成功")
except ImportError:
    print("❌ SFFM: 需要安装segmentation_models_pytorch库：pip install segmentation-models-pytorch")
    raise ImportError("segmentation_models_pytorch is required for SFFM")



os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "0"
Image.MAX_IMAGE_PIXELS = None


class SFFM:

    def __init__(
        self,
        unet_model_path: Optional[str] = None,
        low_res_size: int = 1024,
        lesion_threshold: float = 0.7,
        prob_threshold: float = 0.1,
        stride_overlap: float = 0.0,
        enable_fallback: bool = True,
        fallback_topk: int = 12,
        fallback_min_ratio: float = 0.1,
        check_quality: bool = True,
        min_mean: float = 50.0,
        min_std: float = 10.0,
        patch_sizes: Optional[Dict[str, int]] = None,
        device: str = 'cuda',
        encoder_name: str = 'efficientnet-b3',
        encoder_weights: str = 'imagenet',
        precomputed_images_dir: str = os.environ.get("MSAMIL_SFFM_PRECOMPUTED_DIR", "data/images_1024"),
        output_patches_dir: str = os.environ.get("MSAMIL_SFFM_OUTPUT_DIR", "results/sffm_patches"),
        force_full_mask: bool = False,
        apply_color_filter: bool = False,
        color_filter_mode: str = 'red_strict',  # red_strict | red_margin | none
        red_r_min: int = 150,
        red_g_max: int = 100,
        red_b_max: int = 100,
        red_margin: int = 40,

        enable_random_fill: bool = False,
        target_coverage: float = 0.30,
        random_seed: int = 42,
    ) -> None:

        self.low_res_size = int(low_res_size)
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.precomputed_images_dir = Path(precomputed_images_dir)
        self.output_patches_dir = Path(output_patches_dir)
        self.force_full_mask = bool(force_full_mask)


        self.apply_color_filter = bool(apply_color_filter)
        self.color_filter_mode = color_filter_mode
        self.red_r_min = int(red_r_min)
        self.red_g_max = int(red_g_max)
        self.red_b_max = int(red_b_max)
        self.red_margin = int(red_margin)


        self.prob_threshold = float(prob_threshold)
        self.lesion_threshold = float(lesion_threshold)
        self.stride_overlap = float(stride_overlap)
        self.enable_fallback = bool(enable_fallback)
        self.fallback_topk = int(fallback_topk)
        self.fallback_min_ratio = float(fallback_min_ratio)
        self.check_quality = bool(check_quality)
        self.min_mean = float(min_mean)
        self.min_std = float(min_std)


        self.enable_random_fill = bool(enable_random_fill)
        self.target_coverage = float(target_coverage)
        self.random_seed = int(random_seed)


        if patch_sizes is None:
            self.patch_sizes = {
                '20x': 512,
                '10x': 1024,
                '5x': 2048,
            }
        else:
            self.patch_sizes = patch_sizes


        print(f"创建SMP UNet++模型用于SFFM...")
        print(f"编码器: {self.encoder_name}")
        print(f"预训练权重: {self.encoder_weights}")

        self.unet = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,        # EfficientNet-B3
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        ).to(self.device)


        if unet_model_path and os.path.exists(unet_model_path):
            try:
                print(f"Loading SMP UNet++ checkpoint from {unet_model_path}")
                checkpoint = torch.load(unet_model_path, map_location=self.device, weights_only=True)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.unet.load_state_dict(checkpoint['model_state_dict'])
                    val_iou = checkpoint.get('metrics', {}).get('iou_best', None)
                    if isinstance(val_iou, float):
                        print(f"✓ Loaded SMP UNet++ (best IoU: {val_iou:.4f})")
                    else:
                        print("✓ Loaded SMP UNet++")
                else:
                    self.unet.load_state_dict(checkpoint)
                    print("✓ Loaded SMP UNet++ weights (raw state_dict)")
                print("SMP UNet++ pretrained weights loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}. Will fallback to random weights or full mask mode.")
        else:
            if unet_model_path:
                print(f"⚠️  Checkpoint not found at {unet_model_path}. Using random weights or full mask mode.")
            else:
                print("⚠️  No pretrained SMP UNet++ weights path provided. Using random weights or full mask mode.")

        if self.force_full_mask:
            print("🛈 force_full_mask=True -> 将使用全1掩码，进行纯网格提取 (不依赖分割模型输出)")

        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False


        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"✓ SMP UNet++模型初始化完成")
        print(f"  总参数: {total_params:,}")
        print(f"  取样参数: prob_thr={self.prob_threshold}, lesion_ratio_thr={self.lesion_threshold}, overlap={self.stride_overlap}")
        if self.enable_fallback:
            print(f"  回退策略: topK={self.fallback_topk}, min_ratio={self.fallback_min_ratio}")
        if self.check_quality:
            print(f"  质量阈值: min_mean={self.min_mean}, min_std={self.min_std}")
        if self.enable_random_fill:
            print(f"  覆盖率: 目标 per-scale coverage={self.target_coverage:.2f} (非重叠随机填充启用)")

        try:
            self.output_patches_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            fallback = Path("results/sffm_patches").resolve()
            print(
                f"[SFFM][WARN] output_patches_dir not writable: {self.output_patches_dir} ({e}). "
                f"Falling back to {fallback}"
            )
            self.output_patches_dir = fallback
            self.output_patches_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            fallback = Path("results/sffm_patches").resolve()
            print(
                f"[SFFM][WARN] failed to create output_patches_dir: {self.output_patches_dir} ({e}). "
                f"Falling back to {fallback}"
            )
            self.output_patches_dir = fallback
            self.output_patches_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------

    # ------------------------
    def load_precomputed_thumbnail(self, wsi_path: str) -> Tuple[np.ndarray, Tuple[float, float], str]:
        wsi_name = Path(wsi_path).stem
        precomputed_path = self.precomputed_images_dir / f"{wsi_name}.png"
        if not precomputed_path.exists():
            raise FileNotFoundError(f"预处理图像不存在: {precomputed_path}")

        with Image.open(precomputed_path) as im:
            thumbnail = np.array(im.convert('RGB'))


        suffix = Path(wsi_path).suffix.lower()
        if openslide is not None and suffix in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']:
            slide = openslide.OpenSlide(wsi_path)
            original_width, original_height = slide.dimensions
            slide.close()
        else:
            with Image.open(wsi_path) as im:
                original_width, original_height = im.size

        s1 = original_width / self.low_res_size
        s2 = original_height / self.low_res_size
        return thumbnail, (s1, s2), wsi_name

    def generate_thumbnail(self, wsi_path: str) -> Tuple[np.ndarray, Tuple[float, float]]:
        if openslide is None:
            raise RuntimeError("generate_thumbnail 需要 openslide 支持")
        slide = openslide.OpenSlide(wsi_path)
        original_width, original_height = slide.dimensions
        s1 = original_width / self.low_res_size
        s2 = original_height / self.low_res_size
        thumbnail = slide.get_thumbnail((self.low_res_size, self.low_res_size))
        thumbnail = np.array(thumbnail.convert('RGB'))
        thumbnail = cv2.resize(thumbnail, (self.low_res_size, self.low_res_size))
        slide.close()
        return thumbnail, (s1, s2)

    # ------------------------

    # ------------------------
    def segment_lesions(self, thumbnail: np.ndarray) -> np.ndarray:
        if self.force_full_mask:
            h, w = thumbnail.shape[:2]
            rgb_mask = np.zeros((h, w, 3), dtype=np.float32)
            rgb_mask[:, :, 0] = 1.0
            return rgb_mask

        thumbnail_tensor = torch.from_numpy(thumbnail).permute(2, 0, 1).float() / 255.0
        thumbnail_tensor = thumbnail_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.unet(thumbnail_tensor)  # [1,1,H,W]
            tumor_prob = torch.sigmoid(logits)    # [1,1,H,W]
            mask = tumor_prob.squeeze(0).cpu().numpy()  # [1,H,W]
            mask = np.transpose(mask, (1, 2, 0))  # [H,W,1]
            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
            rgb_mask[:, :, 0] = mask[:, :, 0]
        return rgb_mask

    @staticmethod
    def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        return arr

    def _build_color_mask(self, thumbnail: np.ndarray) -> np.ndarray:
        if not self.apply_color_filter or self.color_filter_mode == 'none':
            return np.ones((thumbnail.shape[0], thumbnail.shape[1]), dtype=np.uint8)
        R = thumbnail[:, :, 0].astype(np.int16)
        G = thumbnail[:, :, 1].astype(np.int16)
        B = thumbnail[:, :, 2].astype(np.int16)
        if self.color_filter_mode == 'red_strict':
            cm = ((R >= self.red_r_min) & (G <= self.red_g_max) & (B <= self.red_b_max)).astype(np.uint8)
        elif self.color_filter_mode == 'red_margin':
            cm = ((R > 127) & (R - np.maximum(G, B) > self.red_margin)).astype(np.uint8)
        else:
            cm = np.ones_like(R, dtype=np.uint8)
        if cm.sum() == 0 and self.apply_color_filter:
            print("[ColorFilterWarn] 颜色过滤产生空掩码，已回退为全通过。请检查阈值设置。")
            cm[:] = 1
        return cm

    def save_patches(self, patches: List[np.ndarray], coords: List[Tuple[int, int]], wsi_name: str, scale: str) -> List[str]:
        wsi_dir = self.output_patches_dir / wsi_name
        scale_dir = wsi_dir / scale
        scale_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for i, (patch, (x, y)) in enumerate(zip(patches, coords)):
            filename = f"{wsi_name}_{scale}_{x}_{y}_{i:04d}.png"
            save_path = scale_dir / filename
            Image.fromarray(patch.astype(np.uint8)).save(save_path)
            saved_paths.append(str(save_path))
        print(f"✓ 保存 {len(patches)} 个 {scale} patches 到 {scale_dir}")
        return saved_paths

    @staticmethod
    def _rect_from_center(cx: int, cy: int, dk: int, dim_w: int, dim_h: int) -> Optional[Tuple[int, int, int, int]]:
        left = max(0, cx - dk // 2)
        top = max(0, cy - dk // 2)
        right = left + dk
        bottom = top + dk
        if right > dim_w or bottom > dim_h:

            left = max(0, dim_w - dk)
            top = max(0, dim_h - dk)
            right = left + dk
            bottom = top + dk
        if left < 0 or top < 0 or right > dim_w or bottom > dim_h or (right - left != dk) or (bottom - top != dk):
            return None
        return (left, top, right, bottom)

    @staticmethod
    def _overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:

        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def process_wsi(self, wsi_path: str, save_patches: bool = True) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[Tuple[int, int]]], Dict[str, List[str]]]:
        print(f"Processing WSI: {wsi_path}")
        if not os.path.exists(wsi_path):
            print(f"❌ WSI file not found: {wsi_path}")
            empty_result = {scale: [] for scale in (self.patch_sizes.keys())}
            return empty_result, empty_result, empty_result

        try:

            wsi_name = Path(wsi_path).stem
            try:
                thumbnail, (s1, s2), wsi_name = self.load_precomputed_thumbnail(wsi_path)
                print(f"Loaded precomputed thumbnail: {wsi_name} with scale factors: s1={s1:.2f}, s2={s2:.2f}")
            except FileNotFoundError as e:
                print(
                    f"[SFFM][WARN] precomputed thumbnail not found under {self.precomputed_images_dir} ({e}). "
                    "Will generate thumbnail on the fly."
                )
                suffix = Path(wsi_path).suffix.lower()
                use_openslide = openslide is not None and suffix in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']
                if use_openslide:
                    thumbnail, (s1, s2) = self.generate_thumbnail(wsi_path)
                else:
                    with Image.open(wsi_path) as im:
                        original_width, original_height = im.size
                        s1 = original_width / float(self.low_res_size)
                        s2 = original_height / float(self.low_res_size)
                        thumb = im.convert('RGB')
                        thumb = thumb.resize((self.low_res_size, self.low_res_size), Image.BILINEAR)
                        thumbnail = np.array(thumb)
                print(f"Generated thumbnail: {wsi_name} with scale factors: s1={s1:.2f}, s2={s2:.2f}")


            mask = self.segment_lesions(thumbnail)
            raw_prob = mask[:, :, 0].copy()
            if self.apply_color_filter:
                color_mask = self._build_color_mask(thumbnail)
                before_pos = float((raw_prob > 0.5).mean())
                mask[:, :, 0] *= color_mask
                after_pos = float((mask[:, :, 0] > 0.5).mean())
                print(f"Color filter applied ({self.color_filter_mode}): bin>0.5 比例 {before_pos:.4f} -> {after_pos:.4f}")


            wsi_dir = self.output_patches_dir / wsi_name
            wsi_dir.mkdir(parents=True, exist_ok=True)


            overall_lesion_ratio = None
            bin_path = None
            try:
                raw_prob_img = (raw_prob * 255).clip(0, 255).astype(np.uint8)
                raw_prob_rgb = self._to_rgb_uint8(raw_prob_img)
                raw_path = wsi_dir / f"{wsi_name}_mask_prob_raw.png"
                Image.fromarray(raw_prob_rgb).save(raw_path)

                filt_prob_img = (mask[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
                filt_prob_rgb = self._to_rgb_uint8(filt_prob_img)
                filt_path = wsi_dir / (f"{wsi_name}_mask_prob_filtered.png" if self.apply_color_filter else f"{wsi_name}_mask_prob.png")
                Image.fromarray(filt_prob_rgb).save(filt_path)

                bin_mask = (mask[:, :, 0] > self.prob_threshold).astype(np.uint8)
                overall_lesion_ratio = float(bin_mask.mean())
                bin_rgb = self._to_rgb_uint8((bin_mask * 255).astype(np.uint8))
                bin_path = wsi_dir / f"{wsi_name}_mask_bin_thr{self.prob_threshold:.2f}.png"
                Image.fromarray(bin_rgb).save(bin_path)
                print(f"✓ Saved masks to {wsi_dir} | lesion_ratio={overall_lesion_ratio:.4f}")
            except Exception as e:
                print(f"⚠️  Failed to save probability/binary masks: {e}")


            suffix = Path(wsi_path).suffix.lower()
            use_openslide = openslide is not None and suffix in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']
            slide = None
            orig_img = None
            if use_openslide:
                slide = openslide.OpenSlide(wsi_path)
                dim_w, dim_h = slide.dimensions
            else:
                orig_img = Image.open(wsi_path)
                dim_w, dim_h = orig_img.size

            filtered_patches: Dict[str, List[np.ndarray]] = {scale: [] for scale in self.patch_sizes.keys()}
            patch_coords: Dict[str, List[Tuple[int, int]]] = {scale: [] for scale in self.patch_sizes.keys()}
            saved_paths: Dict[str, List[str]] = {scale: [] for scale in self.patch_sizes.keys()}
            csv_rows: List[List] = []

            coverage_by_scale: Dict[str, float] = {}
            random_added_by_scale: Dict[str, int] = {}

            rng = np.random.default_rng(self.random_seed)

            total_scales = len(self.patch_sizes)
            for si, (scale, dk) in enumerate(self.patch_sizes.items(), start=1):
                print(f"Processing scale {scale} with patch size dk={dk}px")
                win_w = max(1, int(round(dk / s1)))
                win_h = max(1, int(round(dk / s2)))
                step_x = max(1, int(round(win_w * (1.0 - self.stride_overlap))))
                step_y = max(1, int(round(win_h * (1.0 - self.stride_overlap))))


                window_infos = []  # (lesion_ratio, x, y, center_x, center_y)
                for y in range(0, max(1, self.low_res_size - win_h + 1), step_y):
                    for x in range(0, max(1, self.low_res_size - win_w + 1), step_x):
                        mask_region = mask[y:y + win_h, x:x + win_w, 0]
                        lesion_ratio = float(np.sum(mask_region >= self.prob_threshold)) / float(win_h * win_w)
                        center_x = int((x + win_w // 2) * s1)
                        center_y = int((y + win_h // 2) * s2)
                        window_infos.append((lesion_ratio, x, y, center_x, center_y))


                selected_infos = [w for w in window_infos if w[0] > self.lesion_threshold]


                if self.enable_fallback and len(selected_infos) == 0:
                    window_infos_sorted = sorted(window_infos, key=lambda t: t[0], reverse=True)
                    selected_infos = [w for w in window_infos_sorted if w[0] >= self.fallback_min_ratio][: self.fallback_topk]
                    if len(selected_infos) > 0:
                        print(f"  [Fallback] scale={scale}: 选择 top{len(selected_infos)} (max_ratio={window_infos_sorted[0][0]:.3f})")


                scale_patches: List[np.ndarray] = []
                scale_coords: List[Tuple[int, int]] = []
                selected_rects: List[Tuple[int, int, int, int]] = []
                for lesion_ratio, _x, _y, center_x, center_y in selected_infos:
                    rect = self._rect_from_center(center_x, center_y, dk, dim_w, dim_h)
                    if rect is None:
                        continue
                    left, top, right, bottom = rect
                    try:
                        if use_openslide:
                            patch_rgba = slide.read_region((left, top), 0, (dk, dk))
                            patch = np.array(patch_rgba.convert('RGB'))
                        else:
                            patch = np.array(orig_img.crop((left, top, right, bottom)).convert('RGB'))
                        if (not self.check_quality) or (np.mean(patch) > self.min_mean and np.std(patch) > self.min_std):
                            selected_rects.append(rect)
                            scale_patches.append(patch)
                            scale_coords.append((center_x, center_y))
                            filename = f"{wsi_name}_{scale}_{center_x}_{center_y}_{len(scale_patches)-1:04d}.png"
                            patch_path = (self.output_patches_dir / wsi_name / scale) / filename
                            csv_rows.append([scale, center_x, center_y, str(patch_path) if save_patches else filename, float(lesion_ratio)])
                    except Exception as e:
                        print(f"⚠️  Error extracting patch at ({left}, {top}): {e}")


                image_area = float(dim_w * dim_h)
                patch_area = float(dk * dk)
                coverage = (len(scale_patches) * patch_area) / image_area if image_area > 0 else 0.0


                added_count = 0
                if self.enable_random_fill and coverage < self.target_coverage:

                    selected_set = set((cx, cy) for (_, _, _, cx, cy) in selected_infos)
                    candidates = [(lr, x, y, cx, cy) for (lr, x, y, cx, cy) in window_infos if (cx, cy) not in selected_set]


                    if len(scale_coords) > 0:
                        def min_dist_to_selected(cxy: Tuple[int, int]) -> float:
                            cx, cy = cxy
                            return min(((cx - sx) ** 2 + (cy - sy) ** 2) for (sx, sy) in scale_coords)
                        candidates.sort(key=lambda t: min_dist_to_selected((t[3], t[4])))
                    else:
                        candidates.sort(key=lambda t: t[0], reverse=True)


                    added_rects: List[Tuple[int, int, int, int]] = []
                    for lesion_ratio, _x, _y, cx, cy in candidates:
                        if coverage >= self.target_coverage:
                            break
                        rect = self._rect_from_center(cx, cy, dk, dim_w, dim_h)
                        if rect is None:
                            continue

                        conflict = False
                        for r in selected_rects:
                            if self._overlap(rect, r):
                                conflict = True
                                break
                        if conflict:
                            continue
                        for r in added_rects:
                            if self._overlap(rect, r):
                                conflict = True
                                break
                        if conflict:
                            continue


                        left, top, right, bottom = rect
                        try:
                            if use_openslide:
                                patch_rgba = slide.read_region((left, top), 0, (dk, dk))
                                patch = np.array(patch_rgba.convert('RGB'))
                            else:
                                patch = np.array(orig_img.crop((left, top, right, bottom)).convert('RGB'))
                        except Exception:
                            continue

                        if self.check_quality and not (np.mean(patch) > self.min_mean and np.std(patch) > self.min_std):
                            continue


                        added_rects.append(rect)
                        selected_rects.append(rect)
                        scale_patches.append(patch)
                        scale_coords.append((cx, cy))
                        added_count += 1
                        filename = f"{wsi_name}_{scale}_{cx}_{cy}_{len(scale_patches)-1:04d}.png"
                        patch_path = (self.output_patches_dir / wsi_name / scale) / filename
                        csv_rows.append([scale, cx, cy, str(patch_path) if save_patches else filename, float(lesion_ratio)])

                        coverage = (len(scale_patches) * patch_area) / image_area if image_area > 0 else 0.0

                    if coverage < self.target_coverage:
                        print(f"  [Fill] scale={scale}: 无法完全达到目标覆盖率 {self.target_coverage:.2f}，当前 {coverage:.3f}，已新增 {added_count} 个")
                    else:
                        print(f"  [Fill] scale={scale}: 覆盖率达到 {coverage:.3f}，新增 {added_count} 个")


                filtered_patches[scale] = scale_patches
                patch_coords[scale] = scale_coords
                random_added_by_scale[scale] = int(added_count)
                coverage_by_scale[scale] = float(coverage)

                if save_patches and scale_patches:
                    saved_paths[scale] = self.save_patches(scale_patches, scale_coords, wsi_name, scale)


                if window_infos:
                    max_ratio = max(w[0] for w in window_infos)
                    print(f"  窗口统计: total={len(window_infos)}, 通过阈值={len(selected_infos)}, max_ratio={max_ratio:.3f}, 覆盖率={coverage:.3f}")


                try:
                    curr_total = sum(len(filtered_patches[k]) for k in filtered_patches.keys())
                except Exception:
                    curr_total = len(scale_patches)
                print(f"[完成] {wsi_name} - 尺度 {scale} 切割完成: {len(scale_patches)} 个patch | 已完成尺度 {si}/{total_scales} | 累计patch {curr_total}")

            if slide is not None:
                slide.close()
            if orig_img is not None:
                try:
                    orig_img.close()
                except Exception:
                    pass


            total_patches = sum(len(filtered_patches[k]) for k in filtered_patches.keys())
            print("Extraction completed:")
            for scale in self.patch_sizes.keys():
                print(f"  - {scale}: {len(filtered_patches[scale])} patches (coverage={coverage_by_scale.get(scale, 0.0):.3f}, added={random_added_by_scale.get(scale, 0)})")
            print(f"  - Total: {total_patches} patches")


            try:
                csv_path = self.output_patches_dir / wsi_name / f"{wsi_name}_patches.csv"
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["scale", "center_x", "center_y", "patch_path", "lesion_ratio"])
                    writer.writerows(csv_rows)
                print(f"✓ Saved patch metadata CSV: {csv_path} (rows={len(csv_rows)})")
            except Exception as e:
                print(f"⚠️  Failed to save CSV metadata: {e}")


            try:
                summary = {
                    "wsi_name": wsi_name,
                    "prob_threshold": float(self.prob_threshold),
                    "lesion_ratio": float(overall_lesion_ratio) if overall_lesion_ratio is not None else None,
                    "paths": {
                        "prob_raw": str(raw_path) if 'raw_path' in locals() else None,
                        "prob_filtered": str(filt_path) if 'filt_path' in locals() else None,
                        "mask_binary": str(bin_path) if bin_path is not None else None,
                        "patch_csv": str(csv_path) if 'csv_path' in locals() else None,
                    },
                    "patch_counts": {k: int(len(v)) for k, v in filtered_patches.items()},
                    "total_patches": int(total_patches),
                    "coverage": {
                        "target": float(self.target_coverage),
                        "by_scale": coverage_by_scale,
                        "random_fill_added": random_added_by_scale,
                    },
                }
                with open(self.output_patches_dir / wsi_name / f"{wsi_name}_summary.json", "w", encoding="utf-8") as jf:
                    json.dump(summary, jf, ensure_ascii=False, indent=2)
                print(f"✓ Saved summary JSON: {self.output_patches_dir / wsi_name / f'{wsi_name}_summary.json'}")
            except Exception as e:
                print(f"⚠️  Failed to save summary JSON: {e}")

            if save_patches:
                print(f"✓ All patches saved to: {self.output_patches_dir / wsi_name}")

            return filtered_patches, patch_coords, saved_paths

        except Exception as e:
            print(f"❌ Error processing WSI {wsi_path}: {e}")
            traceback.print_exc()
            empty_result = {scale: [] for scale in self.patch_sizes.keys()}
            return empty_result, empty_result, empty_result



if __name__ == "__main__":
    sffm = SFFM(
        unet_model_path="checkpoints/unet_smp_reg_20250928_151727/best_model.pth",
        low_res_size=1024,
        lesion_threshold=0.7,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        precomputed_images_dir="/mnt/nas/ljh/MsaMIL_Net_Data/images_1024",
        output_patches_dir="/mnt/nas/ljh/MsaMIL_Net_Data/patches",
        force_full_mask=False,
        enable_random_fill=True,
        target_coverage=0.30,
    )

