#!/usr/bin/env python3
"""Print summary and sample values of a WSI coords .npy file.

Usage:
    python tools/print_coords.py data/features/64629_coords.npy --limit 20

Outputs:
    - file exists
    - shape, dtype
    - min/max per column
    - summary statistics (mean/std)
    - first N and last N rows
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description='Print a features coords numpy file')
    p.add_argument('path', type=Path, help='Path to the coords .npy file')
    p.add_argument('--limit', type=int, default=100, help='How many rows to show from head/tail')
    p.add_argument('--stats', action='store_true', help='Show mean/std statistics')
    args = p.parse_args()

    if not args.path.exists():
        print(f"File not found: {args.path}")
        return

    coords = np.load(args.path)
    print(f"Loaded: {args.path}")
    print(f"  shape: {coords.shape}")
    print(f"  dtype: {coords.dtype}")

    if coords.size == 0:
        print("  (empty array)")
        return

    minv = coords.min(axis=0)
    maxv = coords.max(axis=0)
    print(f"  min: {minv}")
    print(f"  max: {maxv}")

    if args.stats:
        meanv = coords.mean(axis=0)
        stdv = coords.std(axis=0)
        print(f"  mean: {meanv}")
        print(f"  std:  {stdv}")

    n = coords.shape[0]
    print(f"  showing first {min(args.limit, n)} rows:")
    print(coords[:args.limit])
    if n > args.limit:
        print(f"  showing last {min(args.limit, n)} rows:")
        print(coords[-args.limit:])


if __name__ == '__main__':
    main()
