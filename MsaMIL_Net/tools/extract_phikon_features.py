#!/usr/bin/env python3
"""Batch extract Phikon-v2 features for multi-scale WSI patches.

The script scans patch tiles that follow the naming pattern
`<slide_id>_<x>_<y>.png` (top-left coordinates in pixels) inside
per-scale folders such as `/mnt/.../patches_20x/<slide_id>/...`.
For every WSI (slide) listed in `all_data.csv`, we:
    1. Collect tiles from every requested magnification.
    2. Estimate the full-resolution width/height of the slide by
       taking the maximum `(x + patch_size)` and `(y + patch_size)`.
    3. Sort all tiles by `(x, y, scale_order)` so downstream MIL
       consumers receive deterministic bags.
     4. Optionally resize each tile to the size expected by the
         Phikon-v2 processor (with configurable multithreaded I/O),
         then run the encoder and capture the CLS token (1024-D)
         per tile.
     5. Save `features_dir/<slide_id>.pt` (torch tensor [N, 1024])
         and `features_dir/<slide_id>_coords.npy` (float32 [N, 2])
         where coords are normalized centers within the inferred
         width/height of the slide, plus
         `features_dir/<slide_id>_scales.npy` (int64 [N]) storing the
         encoded magnification per patch.

Example:
    python tools/extract_phikon_features.py \
        --csv data/all_data.csv \
        --scale 20x:512:/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_20x \
        --scale 10x:1024:/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_10x \
        --scale 5x:2048:/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_5x \
        --output-dir data/features \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import PngImagePlugin
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

@dataclass(frozen=True)
class ScaleConfig:
    name: str
    patch_size: int
    root: Path
    order: int


@dataclass
class PatchRecord:
    path: Path
    x: int
    y: int
    scale: ScaleConfig


SCALE_ENCODING = {
    "5x": 2,
    "10x": 1,
    "20x": 0,
}

#解析命令行参数中的尺度配置
def parse_scale_arg(arg: str, index: int) -> ScaleConfig:
    """Parse `name:patch_size:/abs/path` definitions from CLI."""
    try:
        name, patch_str, dir_str = arg.split(":", 2)
        patch_size = int(patch_str)
    except ValueError as exc:  # pragma: no cover - user error
        raise argparse.ArgumentTypeError(
            f"--scale expects name:patch_size:/abs/path, got '{arg}'"
        ) from exc

    root = Path(dir_str).expanduser().resolve()
    return ScaleConfig(name=name, patch_size=patch_size, root=root, order=index)

#将序列分成固定大小的块
def chunked(seq: Sequence[PatchRecord], size: int) -> Iterable[Sequence[PatchRecord]]:
    for idx in range(0, len(seq), size):  # 使用range生成器，步长为size，遍历整个序列
        yield seq[idx : idx + size]  # 生成当前索引到索引+size的切片

#从图像处理器推断需要的图像尺寸
def infer_resize_edge(processor) -> int | None:
    size_attr = getattr(processor, "size", None)# 获取处理器的size属性，如果不存在则返回None
    if isinstance(size_attr, dict):
        if "shortest_edge" in size_attr:
            return int(size_attr["shortest_edge"])
        if {"height", "width"}.issubset(size_attr) and size_attr["height"] == size_attr["width"]:
            return int(size_attr["height"])
    elif isinstance(size_attr, int):
        return size_attr
    return None

#收集某个slide的所有图像块
def collect_patches(slide_id: str, scales: Sequence[ScaleConfig]) -> List[PatchRecord]:
    records: List[PatchRecord] = []
    for scale in scales:
        slide_dir = scale.root / slide_id
        if not slide_dir.exists():
            continue
        for img_path in slide_dir.glob("*.png"):
            stem_parts = img_path.stem.split("_")
            if len(stem_parts) < 3:
                continue
            try:
                x = int(stem_parts[-2])
                y = int(stem_parts[-1])
            except ValueError:
                continue
            records.append(PatchRecord(path=img_path, x=x, y=y, scale=scale))
    return records

#收集某个slide的所有图像块
def compute_extents(patches: Sequence[PatchRecord]) -> Tuple[int, int]:
    max_w = max(rec.x + rec.scale.patch_size for rec in patches)
    max_h = max(rec.y + rec.scale.patch_size for rec in patches)
    return max_w, max_h

#归一化坐标到[0,1]范围
def normalize_coords(patches: Sequence[PatchRecord], max_w: int, max_h: int) -> np.ndarray:
    coords = np.zeros((len(patches), 2), dtype=np.float32)
    for idx, rec in enumerate(patches):
        cx = rec.x + rec.scale.patch_size / 2.0
        cy = rec.y + rec.scale.patch_size / 2.0
        coords[idx, 0] = cx / max_w if max_w else 0.0
        coords[idx, 1] = cy / max_h if max_h else 0.0
    return coords

#使用Phikon-v2模型提取特征
def extract_features(
    patches: Sequence[PatchRecord],
    processor: AutoImageProcessor,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    manual_resize: bool,
    resize_edge: int | None,
    num_workers: int,
    skip_broken: bool = False,
) -> tuple[torch.Tensor, List[int]]:
    outputs: List[torch.Tensor] = []

    def load_and_prepare(rec: PatchRecord) -> Image.Image | None:
        """Load a patch image and perform optional resize."""
        try:
            with Image.open(rec.path) as img:
                rgb = img.convert("RGB")
        except Exception as exc:
            # Workaround: Pillow may raise ValueError when png text chunks
            # (e.g., embedded ICC profile) exceed MAX_TEXT_CHUNK.
            # Try to fallback to OpenCV if available, otherwise re-raise
            # with a helpful message guiding user to either install opencv
            # or strip the iCCP chunk using pngcrush/pngquant.
            msg = str(exc)
            if "Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK" in msg:
                try:
                    import cv2
                    import numpy as _np

                    data = _np.fromfile(str(rec.path), dtype=_np.uint8)
                    arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if arr is None:
                        raise RuntimeError("cv2 failed to decode PNG")
                    # Convert BGR -> RGB
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    rgb = Image.fromarray(arr)
                except Exception:
                    raise RuntimeError(
                        "Failed to read PNG via fallback (OpenCV).\n" \
                        "Consider installing opencv-python (pip install opencv-python) " \
                        "or strip the iCCP chunk with: pngcrush -ow -rem allb file.png"
                    ) from exc
            else:
                # Try a generic fallback: read with OpenCV if available
                try:
                    import cv2
                    import numpy as _np

                    data = _np.fromfile(str(rec.path), dtype=_np.uint8)
                    arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if arr is None:
                        raise RuntimeError("cv2 failed to decode PNG")
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    rgb = Image.fromarray(arr)
                except Exception:
                    if skip_broken:
                        print(f"[WARN] Skipping corrupted patch: {rec.path} (error: {exc})")
                        return None
                    raise
        if manual_resize and resize_edge:
            rgb = rgb.resize((resize_edge, resize_edge), resample=Image.BILINEAR)
        return rgb

    executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    try:
        processed = 0
        kept_global_indices: List[int] = []
        for batch in tqdm(chunked(patches, batch_size), total=math.ceil(len(patches) / batch_size), leave=False):
            batch_start = processed
            processed += len(batch)
            if executor is not None:
                images_local = list(executor.map(load_and_prepare, batch))
            else:
                images_local = [load_and_prepare(rec) for rec in batch]

            # Filter out any None images produced by the loader (skipped due to corruption)
            images: List[Image.Image] = []
            local_kept_idxs: List[int] = []
            for i, img in enumerate(images_local):
                if img is None:
                    continue
                images.append(img)
                local_kept_idxs.append(batch_start + i)

            if len(images) == 0:
                # Nothing to process for this batch
                continue

            # Process this batch and append the CLS tokens
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                result = model(**inputs)
                cls = result.last_hidden_state[:, 0, :].cpu()
            outputs.append(cls)
            kept_global_indices.extend(local_kept_idxs)
    finally:
        if executor is not None:
            executor.shutdown()
    if len(outputs) == 0:
        # When no patches are provided, return an empty feature tensor [0, D]
        # D is the model hidden size (e.g., 1024 for Phikon-v2).
        try:
            d = int(getattr(model.config, "hidden_size", 0))
        except Exception:
            d = 0
        return torch.empty((0, d), dtype=torch.float32), []
    return torch.cat(outputs, dim=0), kept_global_indices

#主函数，协调整个处理流程
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Phikon-v2 features for all slides listed in a CSV.")
    parser.add_argument("--csv", type=Path, default=Path("data/all_data.csv"), help="CSV with at least a slide_id column")
    parser.add_argument(
        "--scale",
        action="append",
        default=[
            "20x:512:/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_20x",
            "10x:1024:/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_10x",
            "5x:2048:/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_5x",
        ],
        help="Register a scale as name:patch_size:/abs/path (can repeat).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/features_wsi"))
    parser.add_argument("--model-name", default="owkin/phikon-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Force device (cpu / cuda / cuda:1). Auto-detect if omitted.")
    parser.add_argument(
        "--manual-resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize tiles to processor.size before encoding (default: True).",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip slides whose .pt and _coords.npy already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of slides to process for smoke tests.")
    parser.add_argument("--num-workers", type=int, default=6, help="Thread workers for patch loading (set 0/1 to disable multithreading).")
    parser.add_argument(
        "--sort-order",
        choices=["xy", "yx"],
        default="xy",
        help="Order patches by 'x then y' (xy) or 'y then x' (yx). Default: xy",
    )
    parser.add_argument(
        "--png-text-chunk-limit",
        type=int,
        default=None,
        help="Override Pillow's PNG text chunk limit in bytes (default: unlimited).",
    )
    parser.add_argument(
        "--skip-broken",
        action="store_true",
        default=False,
        help="If set, skip corrupted/unreadable patches instead of stopping the run.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="Optional path to save global mean/std (infer format from suffix .json/.npz/.pt).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="Only load Hugging Face weights/processors from local cache; never reach huggingface.co.",
    )
    args = parser.parse_args()

    scales = [parse_scale_arg(defn, idx) for idx, defn in enumerate(args.scale)]
    # 去重：避免用户在命令行中重复传入与默认重复的scale定义
    unique = []
    seen_roots = set()
    for s in scales:
        if s.root in seen_roots:
            print(f"[WARN] Duplicate scale root ignored: {s.root}")
            continue
        seen_roots.add(s.root)
        unique.append(s)
    scales = unique
    if not scales:
        raise ValueError("No scales configured.")

    df = pd.read_csv(args.csv)
    if "slide_id" not in df.columns:
        raise ValueError("CSV must contain a 'slide_id' column.")
    slide_ids = df["slide_id"].dropna().astype(str).tolist()
    if args.limit:
        slide_ids = slide_ids[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Allow reading large PNG text chunks (some WSI tools embed huge ICC profiles).
    limit = args.png_text_chunk_limit if args.png_text_chunk_limit and args.png_text_chunk_limit > 0 else sys.maxsize
    try:
        PngImagePlugin.MAX_TEXT_CHUNK = limit
        if hasattr(PngImagePlugin, "MAX_TEXT_MEMORY"):
            PngImagePlugin.MAX_TEXT_MEMORY = limit
    except Exception:
        # ignore if these attributes are missing (older Pillow)
        pass

    model_repo_exists = Path(args.model_name).expanduser().exists()
    local_only_flag = args.local_files_only or model_repo_exists
    if local_only_flag and not model_repo_exists:
        print(f"[WARN] --local-files-only requested but path '{args.model_name}' not found; falling back to online download.")
        local_only_flag = False

    model = AutoModel.from_pretrained(args.model_name, local_files_only=local_only_flag)
    processor = AutoImageProcessor.from_pretrained(args.model_name, local_files_only=local_only_flag)
    resize_edge = infer_resize_edge(processor)
    hidden_size = None
    # Quick check: show model hidden size to confirm CLS token dimension
    try:
        hidden_size = int(getattr(model.config, "hidden_size"))
        print(f"Loaded model: {args.model_name}, hidden_size: {hidden_size}")
    except Exception:
        hidden_size = None

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    model.to(device)
    model.eval()

    stats_enabled = args.stats_out is not None
    stats_count = 0
    stats_sum: torch.Tensor | None = None
    stats_sumsq: torch.Tensor | None = None

    def accumulate_stats(batch: torch.Tensor):
        nonlocal stats_sum, stats_sumsq, stats_count
        if not stats_enabled or batch.numel() == 0:
            return
        feats = batch.to(torch.float64)
        if stats_sum is None or stats_sumsq is None:
            stats_sum = torch.zeros(feats.shape[1], dtype=torch.float64)
            stats_sumsq = torch.zeros_like(stats_sum)
        stats_sum += feats.sum(dim=0)
        stats_sumsq += (feats * feats).sum(dim=0)
        stats_count += feats.shape[0]

    print(f"Using device: {device_str}")
    print(f"Patch sort order: {args.sort_order}")
    print(f"Scale encoding: {SCALE_ENCODING}")
    print(f"Slides to process: {len(slide_ids)}")
    print("Resize edge (manual flag):", resize_edge if args.manual_resize else "processor default")

    for slide_id in tqdm(slide_ids, desc="Slides"):
        out_feat = args.output_dir / f"{slide_id}.pt"
        out_coord = args.output_dir / f"{slide_id}_coords.npy"
        out_scale = args.output_dir / f"{slide_id}_scales.npy"
        if args.skip_existing and out_feat.exists() and out_coord.exists() and out_scale.exists():
            if stats_enabled:
                # Use weights_only=True when available (safer for torch.load picking behavior)
                try:
                    existing_features = None
                    try:
                        existing_features = torch.load(out_feat, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
                    except TypeError:
                        # Older PyTorch versions don't accept weights_only; fallback
                        existing_features = torch.load(out_feat, map_location="cpu")
                    if isinstance(existing_features, dict) and 'features' in existing_features:
                        existing_features = existing_features['features']
                    accumulate_stats(existing_features)
                except Exception as exc:
                    print(f"[WARN] Failed to load existing features for stats ({out_feat}): {exc}")
            continue

        patches = collect_patches(slide_id, scales)
        if not patches:
            print(f"[WARN] No patches found for {slide_id}, skipping.")
            continue

        # 打印当前 slide_id 与 per-scale 统计信息，帮助诊断为何会有大量 batch
        print(f"Processing slide: {slide_id}")
        scale_counts = Counter([rec.scale.name for rec in patches])
        total_patches = len(patches)
        expected_batches = math.ceil(total_patches / max(1, args.batch_size))
        print(f"   patches per scale: {dict(scale_counts)}")
        # Diagnostic: unique stems per scale and union_count to detect duplicates
        per_scale_unique = {}
        for s in scales:
            stems = set([rec.path.stem for rec in patches if rec.scale == s])
            per_scale_unique[s.name] = len(stems)
        union_stems = set([rec.path.stem + "@" + rec.scale.name for rec in patches])
        print(f"   unique stems per scale: {per_scale_unique}, union_count: {len(union_stems)}")
        print(f"   total_patches: {total_patches}, batch_size: {args.batch_size}, expected_batches: {expected_batches}")

        # Use patch center (x + patch_size/2, y + patch_size/2) to sort spatially.
        # Support both 'xy' (x primary) and 'yx' (y primary) orders.
        cx = lambda rec: rec.x + rec.scale.patch_size / 2.0
        cy = lambda rec: rec.y + rec.scale.patch_size / 2.0
        if args.sort_order == "xy":
            patches.sort(key=lambda rec: (
                cx(rec),
                cy(rec),
                SCALE_ENCODING.get(rec.scale.name, rec.scale.order)
            ))
        else:
            patches.sort(key=lambda rec: (
                cy(rec),
                cx(rec),
                SCALE_ENCODING.get(rec.scale.name, rec.scale.order)
            ))
        max_w, max_h = compute_extents(patches)
        coords = normalize_coords(patches, max_w, max_h)
        scale_ids = np.zeros(len(patches), dtype=np.int64)
        for idx, rec in enumerate(patches):
            if rec.scale.name in SCALE_ENCODING:
                scale_ids[idx] = SCALE_ENCODING[rec.scale.name]
            else:
                scale_ids[idx] = rec.scale.order
        features, kept_indices = extract_features(
            patches,
            processor,
            model,
            device,
            batch_size=max(1, args.batch_size),
            manual_resize=args.manual_resize,
            resize_edge=resize_edge,
            num_workers=max(1, args.num_workers or 1),
            skip_broken=args.skip_broken,
        )

        # Sanity: extracted feature dimension matches model.hidden_size
        try:
            if hidden_size is not None and features.shape[1] != hidden_size:
                print(f"[WARN] Extracted feature dimension {features.shape[1]} != model hidden_size {hidden_size} for slide {slide_id}")
        except Exception:
            pass

        # If the image loader skipped broken patches, drop their coords/scales
        if args.skip_broken and kept_indices:
            # kept_indices are indices in the original patches list to keep
            kept_indices_arr = np.array(kept_indices, dtype=np.int64)
            coords = coords[kept_indices_arr]
            scale_ids = scale_ids[kept_indices_arr]

        # Sanity: features length should match coords length
        if features.shape[0] != coords.shape[0]:
            print(f"[WARN] Feature count {features.shape[0]} != coords count {coords.shape[0]} for slide {slide_id}")

        torch.save(features, out_feat)
        np.save(out_coord, coords)
        np.save(out_scale, scale_ids)

        accumulate_stats(features)

    if stats_enabled:
        if stats_count == 0 or stats_sum is None or stats_sumsq is None:
            print("[WARN] stats-out specified but no features were processed; skipping stats export.")
        else:
            mean = stats_sum / stats_count
            var = torch.clamp(stats_sumsq / stats_count - mean * mean, min=0.0)
            std = torch.sqrt(var + 1e-8)
            stats_data = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "count": int(stats_count),
            }
            stats_path = args.stats_out.expanduser()
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            suffix = stats_path.suffix.lower()
            if suffix in {".pt", ".pth"}:
                torch.save(stats_data, stats_path)
            elif suffix == ".npz":
                np.savez(stats_path, mean=np.array(stats_data["mean"], dtype=np.float32), std=np.array(stats_data["std"], dtype=np.float32), count=stats_count)
            else:
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats_data, f)
            print(f"[INFO] Saved global feature stats to {stats_path} (count={stats_count}).")

    print("Done.")


if __name__ == "__main__":
    main()
