#!/usr/bin/env python3
"""Extract EfficientNet-B3 features for multi-scale WSI patches (quan layout).

This script mirrors tools/extract_efficientnet_features.py but reads patches from
per-slide/per-scale folders such as:
  <root>/<slide_id>/<scale>/<slide_id>_<scale>_<x>_<y>_<idx>.png
Example: Z:\\ljh\\MsaMIL_Net_Data\\patches_quan\\4\\5x\\4_5x_1024_1024_0000.png

Outputs match datasets.feature_dataset.PreExtractedFeatureDataset:
  - features_dir/<slide_id>.pt   -> torch.Tensor [N, output_dim]
  - features_dir/<slide_id>_coords.npy -> float32 [N, 2]
  - features_dir/<slide_id>_scales.npy -> int64 [N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, PngImagePlugin
from tqdm import tqdm
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


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


class EfficientNetFeatureExtractor(nn.Module):
    """Thin wrapper that exposes EfficientNet-B3 as a 1024-D encoder."""

    def __init__(self, output_dim: int = 1024):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        backbone = efficientnet_b3(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features
        self.project = nn.Linear(in_features, output_dim)
        nn.init.normal_(self.project.weight, std=0.02)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.project(x)
        return x


def parse_scale_arg(arg: str, index: int) -> ScaleConfig:
    try:
        name, patch_str, dir_str = arg.split(":", 2)
        patch_size = int(patch_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--scale expects name:patch_size:/abs/path, got '{arg}'"
        ) from exc
    root = Path(dir_str).expanduser().resolve()
    return ScaleConfig(name=name, patch_size=patch_size, root=root, order=index)


def chunked(seq: Sequence[PatchRecord], size: int) -> Iterable[Sequence[PatchRecord]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _parse_coords_from_stem(stem: str, patch_size: int) -> tuple[int, int] | None:
    """Try several heuristics to extract (x,y) from a filename stem.

    Heuristics (in order):
    1) If stem tokens look like 'slide_scale_x_y_idx' (>=5 parts), take parts[2], parts[3].
    2) If stem ends with '..._<x>_<y>' use the last two parts.
    3) Extract integer tokens and take the last two large-enough integers (drop trailing small index tokens).
    4) Fall back to None if parsing fails.
    """
    parts = stem.split("_")
    # Heuristic 1
    if len(parts) >= 5:
        try:
            x = int(parts[2])
            y = int(parts[3])
            return x, y
        except Exception:
            pass
    # Heuristic 2
    if len(parts) >= 3:
        try:
            x = int(parts[-2])
            y = int(parts[-1])
            return x, y
        except Exception:
            pass
    # Heuristic 3: extract ints, drop trailing tiny indices (like 0, 1) if possible
    ints = [int(m) for m in re.findall(r"\d+", stem)]
    if len(ints) < 2:
        return None
    # Drop trailing very small integers that are likely sequence indices (e.g., 0,1,2)
    while len(ints) >= 3 and ints[-1] < 10:
        ints.pop()
    if len(ints) >= 2:
        x, y = ints[-2], ints[-1]
        return x, y
    return None


def collect_patches(slide_id: str, scales: Sequence[ScaleConfig], layout: str = "auto") -> List[PatchRecord]:
    records: List[PatchRecord] = []
    for scale in scales:
        # Try to detect a layout or obey provided layout instruction
        # Candidate layouts:
        #  - slide-scale: root/slide_id/scale/*.png (default assumed by `patches_quan`)
        #  - scale-slide: root/scale/slide_id/*.png
        #  - flat: root/slide_id/*.png
        slide_scale_dir = scale.root / slide_id / scale.name
        scale_slide_dir = scale.root / scale.name / slide_id
        slide_dir = scale.root / slide_id
        chosen_dir = None
        resolved_layout = None
        if layout == "slide-scale" and slide_scale_dir.exists():
            chosen_dir = slide_scale_dir
            resolved_layout = "slide-scale"
        elif layout == "scale-slide" and scale_slide_dir.exists():
            chosen_dir = scale_slide_dir
            resolved_layout = "scale-slide"
        elif layout == "flat" and slide_dir.exists():
            chosen_dir = slide_dir
            resolved_layout = "flat"
        else:
            # auto-detect
            if slide_scale_dir.exists():
                chosen_dir = slide_scale_dir
                resolved_layout = "slide-scale"
            elif scale_slide_dir.exists():
                chosen_dir = scale_slide_dir
                resolved_layout = "scale-slide"
            elif slide_dir.exists():
                chosen_dir = slide_dir
                resolved_layout = "flat"
        if chosen_dir is None:
            print(
                f"[WARN] No patch directory found for slide='{slide_id}', scale='{scale.name}'. "
                f"Tried slide-scale={slide_scale_dir}, scale-slide={scale_slide_dir}, flat={slide_dir}"
            )
            continue
        else:
            # Inform user which directory layout we detected for this scale
            print(f"[INFO] Detected layout '{resolved_layout}' for scale '{scale.name}' (root={scale.root}) -> using {chosen_dir}")
        # Enumerate images
        per_scale_count = 0
        for img_path in chosen_dir.glob("*.png"):
            stem = img_path.stem
            coords = _parse_coords_from_stem(stem, scale.patch_size)
            if coords is None:
                # Could not parse coords robustly; skip this file
                continue
            x, y = coords
            records.append(PatchRecord(path=img_path, x=x, y=y, scale=scale))
            per_scale_count += 1
        if per_scale_count == 0:
            print(f"[WARN] Found dir {chosen_dir} for scale '{scale.name}' but no valid PNGs parsed; check filenames.")
    return records


def compute_extents(patches: Sequence[PatchRecord]) -> Tuple[int, int]:
    max_w = max(rec.x + rec.scale.patch_size for rec in patches)
    max_h = max(rec.y + rec.scale.patch_size for rec in patches)
    return max_w, max_h


def normalize_coords(patches: Sequence[PatchRecord], max_w: int, max_h: int) -> np.ndarray:
    coords = np.zeros((len(patches), 2), dtype=np.float32)
    for idx, rec in enumerate(patches):
        cx = rec.x + rec.scale.patch_size / 2.0
        cy = rec.y + rec.scale.patch_size / 2.0
        coords[idx, 0] = cx / max_w if max_w else 0.0
        coords[idx, 1] = cy / max_h if max_h else 0.0
    return coords


def build_transform(input_size: int) -> T.Compose:
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    mean = weights.meta.get("mean") if weights.meta else None
    std = weights.meta.get("std") if weights.meta else None
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_and_prepare(rec: PatchRecord, transform: T.Compose, skip_broken: bool) -> torch.Tensor | None:
    try:
        with Image.open(rec.path) as img:
            rgb = img.convert("RGB")
    except Exception as exc:
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
    return transform(rgb)


def extract_features(
    patches: Sequence[PatchRecord],
    model: EfficientNetFeatureExtractor,
    device: torch.device,
    batch_size: int,
    transform: T.Compose,
    num_workers: int,
    skip_broken: bool,
    amp: bool,
) -> tuple[torch.Tensor, List[int]]:
    outputs: List[torch.Tensor] = []
    executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    try:
        processed = 0
        kept_indices: List[int] = []
        for batch in tqdm(chunked(patches, batch_size), total=math.ceil(len(patches) / batch_size), leave=False):
            batch_start = processed
            processed += len(batch)
            if executor is not None:
                tensors_local = list(executor.map(lambda rec: load_and_prepare(rec, transform, skip_broken), batch))
            else:
                tensors_local = [load_and_prepare(rec, transform, skip_broken) for rec in batch]

            tensors: List[torch.Tensor] = []
            local_kept: List[int] = []
            for idx, tensor in enumerate(tensors_local):
                if tensor is None:
                    continue
                tensors.append(tensor)
                local_kept.append(batch_start + idx)

            if not tensors:
                continue

            inputs = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            with torch.no_grad():
                if amp and device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        feats = model(inputs)
                else:
                    feats = model(inputs)
            outputs.append(feats.cpu())
            kept_indices.extend(local_kept)
    finally:
        if executor is not None:
            executor.shutdown()

    if not outputs:
        return torch.empty((0, model.project.out_features), dtype=torch.float32), []
    return torch.cat(outputs, dim=0).float(), kept_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EfficientNet-B3 features for WSI patches (quan layout).")
    parser.add_argument("--csv", type=Path, default=Path("data/all_data.csv"))
    parser.add_argument(
        "--scale",
        action="append",
        default=None,
        help="Register a scale as name:patch_size:/abs/path (can repeat). Root should be the per-slide folder containing scale subfolders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/features_efficientnet_quan"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:1; auto if omitted")
    parser.add_argument("--input-size", type=int, default=512, help="Square resize for patches before EfficientNet.")
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=6, help="Thread workers for patch loading (0/1 disables pool).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip slides whose feature files already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of slides for smoke tests.")
    parser.add_argument("--sort-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--skip-broken", action="store_true", help="Skip corrupted patches instead of aborting.")
    parser.add_argument("--png-text-chunk-limit", type=int, default=None)
    parser.add_argument("--stats-out", type=Path, default=None, help="Optional global mean/std output path.")
    parser.add_argument(
        "--projector-out",
        type=Path,
        default=None,
        help="Optional path to save the EfficientNet projector weights for reproducibility.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use autocast(bfloat16) on CUDA.")
    parser.add_argument(
        "--layout",
        choices=["auto", "slide-scale", "scale-slide", "flat"],
        default="auto",
        help="Patch directory layout: 'slide-scale' means <root>/<slide>/<scale> (default 'auto'). 'scale-slide' means <root>/<scale>/<slide>. 'flat' means <root>/<slide>/*.png.",
    )
    args = parser.parse_args()

    scale_args = args.scale or [
        "20x:512:/mnt/nas/ljh/MsaMIL_Net_Data/patches_quan",
        "10x:1024:/mnt/nas/ljh/MsaMIL_Net_Data/patches_quan",
        "5x:2048:/mnt/nas/ljh/MsaMIL_Net_Data/patches_quan",
    ]

    scales = [parse_scale_arg(defn, idx) for idx, defn in enumerate(scale_args)]
    # Allow multiple scales sharing the same root (e.g., patches_quan has 5x/10x/20x under one root)
    unique: List[ScaleConfig] = []
    seen = set()
    for cfg in scales:
        key = (cfg.name, cfg.root)
        if key in seen:
            print(f"[WARN] Duplicate scale definition ignored: name={cfg.name}, root={cfg.root}")
            continue
        seen.add(key)
        unique.append(cfg)
    scales = unique
    if not scales:
        raise ValueError("No scales configured.")

    # Layout: optional override; auto-detect by default
    layout = getattr(args, "layout", "auto") if hasattr(args, "layout") else "auto"

    df = pd.read_csv(args.csv)
    id_col = None
    if "slide_id" in df.columns:
        id_col = "slide_id"
    elif "image_id" in df.columns:
        id_col = "image_id"
    else:
        raise ValueError("CSV must contain 'slide_id' or 'image_id'.")
    slide_ids = df[id_col].dropna().astype(str).tolist()
    if args.limit:
        slide_ids = slide_ids[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    limit = args.png_text_chunk_limit if args.png_text_chunk_limit and args.png_text_chunk_limit > 0 else sys.maxsize
    try:
        PngImagePlugin.MAX_TEXT_CHUNK = limit
        if hasattr(PngImagePlugin, "MAX_TEXT_MEMORY"):
            PngImagePlugin.MAX_TEXT_MEMORY = limit
    except Exception:
        pass

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model = EfficientNetFeatureExtractor(output_dim=args.output_dim).to(device)
    model.eval()

    if args.projector_out is not None:
        projector_path = args.projector_out.expanduser()
        # If a directory is provided, drop a default filename inside it
        if projector_path.is_dir() or projector_path.suffix == "":
            projector_path = projector_path / "projector_weights.pth"
        projector_path.parent.mkdir(parents=True, exist_ok=True)
        projector_payload = {
            "state_dict": model.project.state_dict(),
            "in_features": model.project.in_features,
            "out_features": model.project.out_features,
            "output_dim": args.output_dim,
            "input_size": args.input_size,
            "backbone": "efficientnet_b3",
            "weights": EfficientNet_B3_Weights.IMAGENET1K_V1.value,
        }
        torch.save(projector_payload, projector_path)
        print(f"[INFO] Saved projector weights to {projector_path}")

    transform = build_transform(args.input_size)

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

    print(f"Slides to process: {len(slide_ids)} | Scale encoding: {SCALE_ENCODING}")
    for slide_id in tqdm(slide_ids, desc="Slides"):
        out_feat = args.output_dir / f"{slide_id}.pt"
        out_coord = args.output_dir / f"{slide_id}_coords.npy"
        out_scale = args.output_dir / f"{slide_id}_scales.npy"
        if args.skip_existing and out_feat.exists() and out_coord.exists() and out_scale.exists():
            if stats_enabled:
                try:
                    feats = torch.load(out_feat, map_location="cpu")
                    accumulate_stats(feats)
                except Exception as exc:
                    print(f"[WARN] Failed to load existing features for stats ({out_feat}): {exc}")
            continue

        patches = collect_patches(slide_id, scales, layout=layout)
        if not patches:
            print(f"[WARN] No patches found for {slide_id}, skipping.")
            continue

        print(f"Processing slide: {slide_id}")
        scale_counts = Counter([rec.scale.name for rec in patches])
        total_patches = len(patches)
        expected_batches = math.ceil(total_patches / max(1, args.batch_size))
        print(f"  patches per scale: {dict(scale_counts)}")
        print(f"  total_patches={total_patches}, batch_size={args.batch_size}, expected_batches={expected_batches}")

        cx = lambda rec: rec.x + rec.scale.patch_size / 2.0
        cy = lambda rec: rec.y + rec.scale.patch_size / 2.0
        if args.sort_order == "xy":
            patches.sort(key=lambda rec: (cx(rec), cy(rec), SCALE_ENCODING.get(rec.scale.name, rec.scale.order)))
        else:
            patches.sort(key=lambda rec: (cy(rec), cx(rec), SCALE_ENCODING.get(rec.scale.name, rec.scale.order)))

        max_w, max_h = compute_extents(patches)
        coords = normalize_coords(patches, max_w, max_h)
        scale_ids = np.zeros(len(patches), dtype=np.int64)
        for idx, rec in enumerate(patches):
            scale_ids[idx] = SCALE_ENCODING.get(rec.scale.name, rec.scale.order)

        features, kept_indices = extract_features(
            patches,
            model,
            device,
            batch_size=max(1, args.batch_size),
            transform=transform,
            num_workers=max(1, args.num_workers or 1),
            skip_broken=args.skip_broken,
            amp=args.amp,
        )

        num_skipped = total_patches - features.shape[0]
        if num_skipped > 0:
            print(f"[WARN] Skipped {num_skipped}/{total_patches} patches for slide {slide_id} (corrupted/unreadable)")

        if args.skip_broken:
            if kept_indices:
                kept_arr = np.array(kept_indices, dtype=np.int64)
                coords = coords[kept_arr]
                scale_ids = scale_ids[kept_arr]
            else:
                coords = coords[:0]
                scale_ids = scale_ids[:0]

        if features.shape[0] != coords.shape[0]:
            print(f"[WARN] Feature count {features.shape[0]} != coords count {coords.shape[0]} for slide {slide_id}")
        if features.shape[0] == 0:
            print(f"[WARN] No valid features for {slide_id}. Skipping save to {args.output_dir}.")
            continue

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
                np.savez(
                    stats_path,
                    mean=np.array(stats_data["mean"], dtype=np.float32),
                    std=np.array(stats_data["std"], dtype=np.float32),
                    count=stats_count,
                )
            else:
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats_data, f)
            print(f"[INFO] Saved global feature stats to {stats_path} (count={stats_count}).")

    print("Done.")


if __name__ == "__main__":
    main()
