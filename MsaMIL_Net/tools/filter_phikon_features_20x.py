#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter Phikon features by scale and save to a new directory.

Default: keep 20x (scale code 0) only.

Expected input file layout in input_dir:
- <wsi_id>.pt              (torch Tensor or dict with key 'features')
- <wsi_id>_coords.npy      (float32, shape [N,2])
- <wsi_id>_scales.npy      (int64, shape [N])

Output layout in output_dir mirrors input names.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

SCALE_NAME_TO_CODE = {
    "20x": 0,
    "10x": 1,
    "5x": 2,
}


def _load_features(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "features" in obj:
        obj = obj["features"]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Feature file must be Tensor or dict(features=...), got {type(obj)}")
    return obj


def _save_features(path: Path, feats: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)


def _filter_one(
    feat_path: Path,
    coords_path: Path,
    scales_path: Path,
    out_dir: Path,
    keep_code: int,
    min_keep: int,
) -> Tuple[int, int]:
    feats = _load_features(feat_path).float()
    coords = np.load(coords_path).astype(np.float32)
    scales = np.load(scales_path).astype(np.int64)

    n_total = int(feats.shape[0])
    if coords.shape[0] != n_total or scales.shape[0] != n_total:
        raise ValueError(f"Length mismatch for {feat_path.stem}")

    mask = scales == int(keep_code)
    keep_idx = np.where(mask)[0]
    n_keep = int(keep_idx.shape[0])

    if n_keep < int(min_keep):
        return n_total, 0

    feats_k = feats[keep_idx]
    coords_k = coords[keep_idx]
    scales_k = scales[keep_idx]

    out_feat = out_dir / feat_path.name
    out_coords = out_dir / coords_path.name
    out_scales = out_dir / scales_path.name

    _save_features(out_feat, feats_k)
    np.save(out_coords, coords_k)
    np.save(out_scales, scales_k)

    return n_total, n_keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Phikon features by scale code")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input features dir (e.g., data/features_phikon_Yi)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: <input>_20x)",
    )
    parser.add_argument(
        "--scale-name",
        type=str,
        default="20x",
        choices=list(SCALE_NAME_TO_CODE.keys()),
        help="Which scale to keep (20x/10x/5x)",
    )
    parser.add_argument(
        "--scale-code",
        type=int,
        default=None,
        help="Override scale code directly (if set, ignores scale-name)",
    )
    parser.add_argument(
        "--min-keep",
        type=int,
        default=1,
        help="Minimum kept patches required to save this WSI (default: 1)",
    )

    args = parser.parse_args()
    in_dir: Path = args.input_dir.expanduser().resolve()
    if args.output_dir is None:
        out_dir = in_dir.parent / f"{in_dir.name}_20x"
    else:
        out_dir = args.output_dir.expanduser().resolve()

    keep_code = int(args.scale_code) if args.scale_code is not None else int(SCALE_NAME_TO_CODE[args.scale_name])

    if not in_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {in_dir}")

    pt_files = sorted(in_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {in_dir}")

    total_wsis = 0
    kept_wsis = 0
    total_patches = 0
    kept_patches = 0

    for feat_path in pt_files:
        stem = feat_path.stem
        coords_path = in_dir / f"{stem}_coords.npy"
        scales_path = in_dir / f"{stem}_scales.npy"
        if not coords_path.exists() or not scales_path.exists():
            print(f"[Skip] Missing coords/scales for {stem}")
            continue

        total_wsis += 1
        n_total, n_keep = _filter_one(
            feat_path=feat_path,
            coords_path=coords_path,
            scales_path=scales_path,
            out_dir=out_dir,
            keep_code=keep_code,
            min_keep=args.min_keep,
        )
        total_patches += n_total
        kept_patches += n_keep
        if n_keep > 0:
            kept_wsis += 1

    print(f"Done. WSIs processed: {total_wsis}, kept: {kept_wsis}")
    print(f"Patches total: {total_patches}, kept: {kept_patches}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
