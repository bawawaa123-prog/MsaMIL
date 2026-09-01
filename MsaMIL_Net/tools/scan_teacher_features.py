#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan teacher feature files (.pt + _coords.npy + _scales.npy) for corruption.

What it checks per slide_id:
- features tensor: NaN/Inf counts, shape, dtype, max_abs, mean L2 norm
- coords: NaN/Inf, shape, min/max, out-of-range counts (default [0,1])
- scales: NaN/Inf, dtype, unique values, invalid value counts (default {0,1,2})
- length consistency among features/coords/scales

Outputs a JSON report for easy grep & follow-up.

Example:
  python tools/scan_teacher_features.py \
    --features-dir data/features_phikon_Yi \
    --out results/scan_reports/phikon_Yi_scan.json

Tip:
  Add --strict-coords to enforce coords within [0,1] strictly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SlideScanResult:
    slide_id: str
    feat_path: str
    coords_path: str
    scales_path: str

    ok: bool
    issues: List[str]

    # features
    feat_shape: List[int]
    feat_dtype: str
    feat_numel: int
    feat_nonfinite: int
    feat_nan: int
    feat_inf: int
    feat_max_abs: float
    feat_mean_l2: float

    # coords
    coords_shape: List[int]
    coords_dtype: str
    coords_nonfinite: int
    coords_min: List[float]
    coords_max: List[float]
    coords_oor: int

    # scales
    scales_shape: List[int]
    scales_dtype: str
    scales_nonfinite: int
    scales_unique: List[int]
    scales_invalid: int


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load_feature_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict):
        # common patterns
        for k in ("features", "feat", "x"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                break
        else:
            raise TypeError(f"Unsupported dict payload keys={list(obj.keys())[:10]}")
    else:
        raise TypeError(f"Unsupported payload type: {type(obj)}")

    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"Expected [N,D] features tensor, got shape={tuple(t.shape)}")
    return t


def _count_nonfinite_torch(t: torch.Tensor) -> Tuple[int, int, int]:
    # returns (nonfinite, nan, inf)
    nan = torch.isnan(t)
    inf = torch.isinf(t)
    nonfinite = nan | inf
    return int(nonfinite.sum().item()), int(nan.sum().item()), int(inf.sum().item())


def _count_nonfinite_np(a: np.ndarray) -> Tuple[int, int, int]:
    nan = np.isnan(a)
    inf = np.isinf(a)
    nonfinite = nan | inf
    return int(nonfinite.sum()), int(nan.sum()), int(inf.sum())


def scan_one(
    *,
    feat_path: Path,
    coords_path: Path,
    scales_path: Path,
    valid_scales: Tuple[int, ...],
    coords_tol: float,
    strict_coords: bool,
) -> SlideScanResult:
    slide_id = feat_path.stem
    issues: List[str] = []

    # ---------------- features ----------------
    try:
        feats = _load_feature_tensor(feat_path)
    except Exception as e:
        return SlideScanResult(
            slide_id=slide_id,
            feat_path=str(feat_path),
            coords_path=str(coords_path),
            scales_path=str(scales_path),
            ok=False,
            issues=[f"features_load_error: {type(e).__name__}: {e}"],
            feat_shape=[],
            feat_dtype="",
            feat_numel=0,
            feat_nonfinite=0,
            feat_nan=0,
            feat_inf=0,
            feat_max_abs=float("nan"),
            feat_mean_l2=float("nan"),
            coords_shape=[],
            coords_dtype="",
            coords_nonfinite=0,
            coords_min=[float("nan"), float("nan")],
            coords_max=[float("nan"), float("nan")],
            coords_oor=0,
            scales_shape=[],
            scales_dtype="",
            scales_nonfinite=0,
            scales_unique=[],
            scales_invalid=0,
        )

    feats_f = feats.detach()
    if feats_f.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        issues.append(f"features_unexpected_dtype: {str(feats_f.dtype)}")
        feats_f = feats_f.float()

    feat_nonfinite, feat_nan, feat_inf = _count_nonfinite_torch(feats_f)
    if feat_nonfinite > 0:
        issues.append(f"features_nonfinite: {feat_nonfinite} (nan={feat_nan}, inf={feat_inf})")

    try:
        feat_max_abs = float(feats_f.abs().max().item())
    except Exception:
        feat_max_abs = float("nan")

    # mean l2 norm over rows (avoid huge memory)
    try:
        # [N, D] -> [N]
        mean_l2 = float(torch.linalg.vector_norm(feats_f.float(), ord=2, dim=1).mean().item())
    except Exception:
        mean_l2 = float("nan")

    # ---------------- coords ----------------
    coords_shape: List[int] = []
    coords_dtype: str = ""
    coords_nonfinite = 0
    coords_min = [float("nan"), float("nan")]
    coords_max = [float("nan"), float("nan")]
    coords_oor = 0

    if not coords_path.exists():
        issues.append("coords_missing")
    else:
        try:
            coords = np.load(coords_path)
            coords_shape = list(coords.shape)
            coords_dtype = str(coords.dtype)
            if coords.ndim != 2 or coords.shape[1] != 2:
                issues.append(f"coords_bad_shape: {coords.shape}")
            else:
                nf, n_nan, n_inf = _count_nonfinite_np(coords)
                coords_nonfinite = nf
                if nf > 0:
                    issues.append(f"coords_nonfinite: {nf} (nan={n_nan}, inf={n_inf})")

                coords_min = [
                    _as_float(np.min(coords[:, 0])),
                    _as_float(np.min(coords[:, 1])),
                ]
                coords_max = [
                    _as_float(np.max(coords[:, 0])),
                    _as_float(np.max(coords[:, 1])),
                ]

                tol = 0.0 if strict_coords else float(coords_tol)
                oor = (coords < (0.0 - tol)) | (coords > (1.0 + tol))
                coords_oor = int(oor.sum())
                if coords_oor > 0:
                    issues.append(f"coords_out_of_range: {coords_oor} (tol={tol})")
        except Exception as e:
            issues.append(f"coords_load_error: {type(e).__name__}: {e}")

    # ---------------- scales ----------------
    scales_shape: List[int] = []
    scales_dtype: str = ""
    scales_nonfinite = 0
    scales_unique: List[int] = []
    scales_invalid = 0

    if not scales_path.exists():
        issues.append("scales_missing")
    else:
        try:
            scales = np.load(scales_path)
            scales_shape = list(scales.shape)
            scales_dtype = str(scales.dtype)
            if scales.ndim != 1:
                issues.append(f"scales_bad_shape: {scales.shape}")
            else:
                if np.issubdtype(scales.dtype, np.floating):
                    nf, n_nan, n_inf = _count_nonfinite_np(scales)
                    scales_nonfinite = nf
                    if nf > 0:
                        issues.append(f"scales_nonfinite: {nf} (nan={n_nan}, inf={n_inf})")

                # convert to int for unique/valid checks (preserve NaN already handled)
                try:
                    scales_i = scales.astype(np.int64, copy=False)
                except Exception:
                    scales_i = scales.astype(np.int64)

                uniq = np.unique(scales_i)
                scales_unique = [int(x) for x in uniq[:20]]
                invalid_mask = ~np.isin(scales_i, np.array(valid_scales, dtype=np.int64))
                scales_invalid = int(invalid_mask.sum())
                if scales_invalid > 0:
                    issues.append(f"scales_invalid: {scales_invalid} valid={valid_scales}")
        except Exception as e:
            issues.append(f"scales_load_error: {type(e).__name__}: {e}")

    # ---------------- length consistency ----------------
    n_feat = int(feats.shape[0])
    if coords_path.exists() and coords_shape:
        if len(coords_shape) >= 1 and coords_shape[0] != n_feat:
            issues.append(f"len_mismatch_feats_vs_coords: feats={n_feat} coords={coords_shape[0]}")
    if scales_path.exists() and scales_shape:
        if len(scales_shape) >= 1 and scales_shape[0] != n_feat:
            issues.append(f"len_mismatch_feats_vs_scales: feats={n_feat} scales={scales_shape[0]}")

    ok = len(issues) == 0

    return SlideScanResult(
        slide_id=slide_id,
        feat_path=str(feat_path),
        coords_path=str(coords_path),
        scales_path=str(scales_path),
        ok=ok,
        issues=issues,
        feat_shape=list(feats.shape),
        feat_dtype=str(feats.dtype),
        feat_numel=int(feats.numel()),
        feat_nonfinite=int(feat_nonfinite),
        feat_nan=int(feat_nan),
        feat_inf=int(feat_inf),
        feat_max_abs=float(feat_max_abs),
        feat_mean_l2=float(mean_l2),
        coords_shape=coords_shape,
        coords_dtype=coords_dtype,
        coords_nonfinite=int(coords_nonfinite),
        coords_min=[float(coords_min[0]), float(coords_min[1])],
        coords_max=[float(coords_max[0]), float(coords_max[1])],
        coords_oor=int(coords_oor),
        scales_shape=scales_shape,
        scales_dtype=scales_dtype,
        scales_nonfinite=int(scales_nonfinite),
        scales_unique=scales_unique,
        scales_invalid=int(scales_invalid),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan teacher feature files for NaN/Inf and inconsistencies")
    parser.add_argument("--features-dir", type=str, required=True, help="Directory containing <id>.pt and *_coords.npy")
    parser.add_argument("--out", type=str, default="", help="Output JSON path (default: results/scan_reports/<timestamp>.json)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit number of .pt files to scan")
    parser.add_argument("--valid-scales", type=str, default="0,1,2", help="Comma-separated valid scale encodings")
    parser.add_argument("--coords-tol", type=float, default=1e-3, help="Allow coords within [-tol,1+tol] if not strict")
    parser.add_argument("--strict-coords", action="store_true", help="Require coords within [0,1] strictly")
    parser.add_argument("--progress-every", type=int, default=50)

    args = parser.parse_args()
    feat_dir = Path(args.features_dir).expanduser().resolve()
    if not feat_dir.exists():
        raise FileNotFoundError(f"features-dir not found: {feat_dir}")

    valid_scales = tuple(int(x) for x in str(args.valid_scales).split(",") if str(x).strip() != "")

    pt_files = sorted([p for p in feat_dir.glob("*.pt") if p.is_file()])
    if args.limit and args.limit > 0:
        pt_files = pt_files[: int(args.limit)]

    out_path: Path
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = (Path("results") / "scan_reports" / f"scan_{feat_dir.name}_{ts}.json").resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[SlideScanResult] = []
    bad: List[SlideScanResult] = []

    for idx, feat_path in enumerate(pt_files, start=1):
        slide_id = feat_path.stem
        coords_path = feat_dir / f"{slide_id}_coords.npy"
        scales_path = feat_dir / f"{slide_id}_scales.npy"

        r = scan_one(
            feat_path=feat_path,
            coords_path=coords_path,
            scales_path=scales_path,
            valid_scales=valid_scales,
            coords_tol=float(args.coords_tol),
            strict_coords=bool(args.strict_coords),
        )
        results.append(r)
        if not r.ok:
            bad.append(r)

        if args.progress_every > 0 and (idx % int(args.progress_every) == 0):
            print(f"[{idx}/{len(pt_files)}] scanned. bad={len(bad)}")

    # summary
    total = len(results)
    bad_n = len(bad)
    nonfinite_feats = sum(1 for r in results if r.feat_nonfinite > 0)
    nonfinite_coords = sum(1 for r in results if r.coords_nonfinite > 0)
    coords_oor = sum(1 for r in results if r.coords_oor > 0)
    scales_invalid = sum(1 for r in results if r.scales_invalid > 0)
    len_mismatch = sum(1 for r in results if any(s.startswith("len_mismatch") for s in r.issues))

    report: Dict[str, Any] = {
        "features_dir": str(feat_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "total": total,
            "bad": bad_n,
            "nonfinite_feats_files": nonfinite_feats,
            "nonfinite_coords_files": nonfinite_coords,
            "coords_out_of_range_files": coords_oor,
            "scales_invalid_files": scales_invalid,
            "len_mismatch_files": len_mismatch,
        },
        "bad_samples": [asdict(r) for r in bad],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Scan Summary ===")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"Report saved to: {out_path}")

    if bad_n:
        print("\nTop bad samples (up to 10):")
        for r in bad[:10]:
            print(f"- {r.slide_id}: {r.issues}")


if __name__ == "__main__":
    main()
