#!/usr/bin/env python3
"""Generate stable train/val/test splits (optionally k-fold) and save to CSV.

Why:
- Keep dataset split consistent across stages (offline feature training vs end-to-end training).
- Avoid accidental leakage where a WSI is train in one stage and test in another.

Output CSV schema:
- slide_id: str
- label: original label string (from label_file)
- fold: int (0..k-1). If k<=1, fold==0.
- split: one of {train,val,test}

If k>1:
- First create a single stratified test split (constant across folds).
- Remaining samples are split into k folds using StratifiedKFold.
- Test samples are duplicated for every fold so that selecting fold=i includes the same test set.

Usage examples:
- Single split (auto-random seed each run):
    python MsaMIL_Net/tools/make_splits.py --label-file data/all_data.csv --out-csv data/splits.csv --test-ratio 0.15 --val-ratio 0.15

- Single split (deterministic):
    python MsaMIL_Net/tools/make_splits.py --label-file data/all_data.csv --out-csv data/splits.csv --test-ratio 0.15 --val-ratio 0.15 --seed 42

- K-fold with fixed test (deterministic):
    python MsaMIL_Net/tools/make_splits.py --label-file data/all_data.csv --out-csv data/splits_k5.csv --kfold 5 --test-ratio 0.15 --seed 42

- Force random seed explicitly:
    python MsaMIL_Net/tools/make_splits.py --label-file data/all_data.csv --out-csv data/splits.csv --seed -1

- Repeated full experiments (each fold has its own train/val/test with the same global ratios):
    python MsaMIL_Net/tools/make_splits.py --label-file data/all_data.csv --out-csv splits/repeats5.csv --repeats 5 --test-ratio 0.1 --val-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import secrets

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
except Exception as e:  # pragma: no cover
    raise RuntimeError("make_splits.py requires scikit-learn") from e


SPLIT_VALUES = {"train", "val", "test"}


def _infer_id_column(df: pd.DataFrame) -> str:
    for cand in ("slide_id", "image_id", "wsi_id"):
        if cand in df.columns:
            return cand
    raise ValueError("label_file must contain one of columns: slide_id / image_id / wsi_id")


def _filter_by_features(df: pd.DataFrame, features_dir: Path, id_col: str) -> pd.DataFrame:
    features_dir = features_dir.expanduser().resolve()
    if not features_dir.exists():
        raise FileNotFoundError(f"features_dir not found: {features_dir}")

    keep_mask = []
    for sid in df[id_col].astype(str).tolist():
        feat = features_dir / f"{sid}.pt"
        coord = features_dir / f"{sid}_coords.npy"
        keep_mask.append(feat.exists() and coord.exists())
    kept = int(sum(keep_mask))
    print(f"[make_splits] Filter by features_dir: keep {kept}/{len(df)}")
    return df.loc[keep_mask].reset_index(drop=True)


def _filter_by_patch_root(df: pd.DataFrame, patch_root: Path, id_col: str) -> pd.DataFrame:
    patch_root = patch_root.expanduser().resolve()
    if not patch_root.exists():
        raise FileNotFoundError(f"patch_root not found: {patch_root}")

    scale_dirs = [patch_root / "patches_20x", patch_root / "patches_10x", patch_root / "patches_5x"]

    keep_mask = []
    for sid in df[id_col].astype(str).tolist():
        exists_any = any((sd / sid).exists() for sd in scale_dirs)
        keep_mask.append(exists_any)
    kept = int(sum(keep_mask))
    print(f"[make_splits] Filter by patch_root: keep {kept}/{len(df)}")
    return df.loc[keep_mask].reset_index(drop=True)


@dataclass
class SplitMeta:
    label_file: str
    out_csv: str
    seed: int
    test_ratio: float
    val_ratio: float
    kfold: int
    repeats: int
    mode: str
    num_samples: int
    num_classes: int
    class_counts: Dict[str, int]


def _make_one_full_split(df: pd.DataFrame, *, seed: int, test_ratio: float, val_ratio: float) -> pd.DataFrame:
    """生成一次完整的 train/val/test 划分（val/test 都按总占比）。"""
    id_col = _infer_id_column(df)
    ids = df[id_col].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    trainval_idx, test_idx = _stratified_split_train_test(ids, labels, test_ratio=test_ratio, seed=seed)
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 将“占总样本的val_ratio”换算到 trainval 内部
    val_ratio_in_trainval = 0.0
    if val_ratio > 0:
        denom = max(1e-12, (1.0 - float(test_ratio)))
        val_ratio_in_trainval = float(val_ratio) / denom
        if val_ratio_in_trainval >= 1.0:
            raise ValueError(
                f"val_ratio too large after adjusting for test split: val_ratio_in_trainval={val_ratio_in_trainval:.4f}"
            )

    train_idx2, val_idx2 = _stratified_split_train_val(
        trainval_df[id_col].tolist(),
        trainval_df["label"].astype(str).tolist(),
        val_ratio=val_ratio_in_trainval,
        seed=seed + 1000003,
    )
    train_df = trainval_df.iloc[train_idx2].copy()
    val_df = trainval_df.iloc[val_idx2].copy()

    out = []
    for split_name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        if part.empty:
            continue
        tmp = part[[id_col, "label"]].copy()
        tmp.rename(columns={id_col: "slide_id"}, inplace=True)
        tmp["split"] = split_name
        out.append(tmp)
    return pd.concat(out, axis=0, ignore_index=True)


def _stratified_split_train_test(
    ids: List[str],
    labels: List[str],
    *,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if test_ratio <= 0:
        idx_all = np.arange(len(ids))
        return idx_all, np.array([], dtype=np.int64)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_ratio), random_state=int(seed))
    (train_idx, test_idx) = next(sss.split(np.zeros(len(ids)), labels))
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def _stratified_split_train_val(
    ids: List[str],
    labels: List[str],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if val_ratio <= 0:
        idx_all = np.arange(len(ids))
        return idx_all, np.array([], dtype=np.int64)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(seed))
    (train_idx, val_idx) = next(sss.split(np.zeros(len(ids)), labels))
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def build_splits(df: pd.DataFrame, *, seed: int, test_ratio: float, val_ratio: float, kfold: int) -> pd.DataFrame:
    # 语义约定：test_ratio / val_ratio 都是“占总样本的比例”。
    # 由于我们先划分 test，再在剩余 trainval 上划分 val，
    # 所以需要把 val_ratio 换算为 trainval 内部的比例：val_ratio / (1 - test_ratio)。
    if test_ratio < 0 or val_ratio < 0:
        raise ValueError(f"test_ratio/val_ratio must be >=0, got test_ratio={test_ratio}, val_ratio={val_ratio}")
    if test_ratio >= 1.0:
        raise ValueError(f"test_ratio must be < 1, got {test_ratio}")
    if (test_ratio + val_ratio) >= 1.0:
        raise ValueError(
            f"test_ratio + val_ratio must be < 1 (both are fractions of total). got {test_ratio}+{val_ratio}"
        )

    if "label" not in df.columns:
        raise ValueError("label_file must contain 'label' column")

    id_col = _infer_id_column(df)
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    ids = df[id_col].tolist()
    labels = df["label"].astype(str).tolist()

    if len(ids) == 0:
        raise ValueError("No samples found in label_file (after filtering)")

    # de-duplicate by id (keep first)
    if len(set(ids)) != len(ids):
        before = len(df)
        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
        ids = df[id_col].tolist()
        labels = df["label"].astype(str).tolist()
        print(f"[make_splits] De-duplicated IDs: {before} -> {len(df)}")

    # trainval/test first (test constant across folds)
    trainval_idx, test_idx = _stratified_split_train_test(ids, labels, test_ratio=test_ratio, seed=seed)

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # single split (k<=1): make train/val on trainval
    if kfold <= 1:
        # 将“占总样本的val_ratio”换算到 trainval 内部
        val_ratio_in_trainval = 0.0
        if val_ratio > 0:
            denom = max(1e-12, (1.0 - float(test_ratio)))
            val_ratio_in_trainval = float(val_ratio) / denom
            if val_ratio_in_trainval >= 1.0:
                raise ValueError(
                    f"val_ratio too large after adjusting for test split: val_ratio_in_trainval={val_ratio_in_trainval:.4f}"
                )
        train_idx2, val_idx2 = _stratified_split_train_val(
            trainval_df[id_col].tolist(),
            trainval_df["label"].astype(str).tolist(),
            val_ratio=val_ratio_in_trainval,
            seed=seed,
        )
        train_df = trainval_df.iloc[train_idx2].copy()
        val_df = trainval_df.iloc[val_idx2].copy()

        out = []
        for split_name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
            if part.empty:
                continue
            tmp = part[[id_col, "label"]].copy()
            tmp.rename(columns={id_col: "slide_id"}, inplace=True)
            tmp["fold"] = 0
            tmp["split"] = split_name
            out.append(tmp)
        result = pd.concat(out, axis=0, ignore_index=True)
        return result

    # k-fold on trainval
    skf = StratifiedKFold(n_splits=int(kfold), shuffle=True, random_state=int(seed))
    y_tv = trainval_df["label"].astype(str).to_numpy()

    out_parts = []
    for fold_idx, (train_i, val_i) in enumerate(skf.split(np.zeros(len(trainval_df)), y_tv)):
        train_df = trainval_df.iloc[train_i].copy()
        val_df = trainval_df.iloc[val_i].copy()

        for split_name, part in (("train", train_df), ("val", val_df)):
            tmp = part[[id_col, "label"]].copy()
            tmp.rename(columns={id_col: "slide_id"}, inplace=True)
            tmp["fold"] = int(fold_idx)
            tmp["split"] = split_name
            out_parts.append(tmp)

        if not test_df.empty:
            tmp_t = test_df[[id_col, "label"]].copy()
            tmp_t.rename(columns={id_col: "slide_id"}, inplace=True)
            tmp_t["fold"] = int(fold_idx)
            tmp_t["split"] = "test"
            out_parts.append(tmp_t)

    result = pd.concat(out_parts, axis=0, ignore_index=True)
    return result


def build_repeated_splits(df: pd.DataFrame, *, seed: int, test_ratio: float, val_ratio: float, repeats: int) -> pd.DataFrame:
    """重复生成 K 份完整划分，每一折都是一次独立实验：train/val/test 均存在且比例基于总样本。

    这不是传统k折交叉验证（val≈1/k），而是 repeated stratified split。
    """
    if repeats <= 0:
        raise ValueError(f"repeats must be > 0, got {repeats}")

    id_col = _infer_id_column(df)
    df = df.copy()
    df[id_col] = df[id_col].astype(str)

    # de-duplicate by id (keep first)
    if df[id_col].duplicated().any():
        before = len(df)
        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
        print(f"[make_splits] De-duplicated IDs: {before} -> {len(df)}")

    parts = []
    for fold_idx in range(int(repeats)):
        seed_i = int(seed) + int(fold_idx)
        one = _make_one_full_split(df, seed=seed_i, test_ratio=float(test_ratio), val_ratio=float(val_ratio))
        one["fold"] = int(fold_idx)
        parts.append(one)

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out[["slide_id", "label", "fold", "split"]]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label-file", required=True, type=str)
    p.add_argument("--out-csv", required=True, type=str)
    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="random seed; omit for auto-random each run; use -1 to force random",
    )
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--kfold", type=int, default=0, help="k-fold count; <=1 means single split")
    p.add_argument("--repeats", type=int, default=0, help="repeat full train/val/test splits; each repeat is one fold")
    p.add_argument("--features-dir", type=str, default=None, help="optional: filter IDs that have .pt and _coords.npy")
    p.add_argument("--patch-root", type=str, default=None, help="optional: filter IDs that have patch folders")
    args = p.parse_args()

    # 默认每次运行随机划分：不给seed则自动生成；seed=-1 也视为随机
    if args.seed is None or int(args.seed) < 0:
        args.seed = int(secrets.randbelow(2**31 - 1))
        print(f"[make_splits] Auto-random seed={args.seed}")

    label_file = Path(args.label_file).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(label_file)
    id_col = _infer_id_column(df)

    if args.features_dir:
        df = _filter_by_features(df, Path(args.features_dir), id_col)
    if args.patch_root:
        df = _filter_by_patch_root(df, Path(args.patch_root), id_col)

    mode = "cv"
    if int(args.repeats) > 0:
        mode = "repeats"
        split_df = build_repeated_splits(
            df,
            seed=int(args.seed),
            test_ratio=float(args.test_ratio),
            val_ratio=float(args.val_ratio),
            repeats=int(args.repeats),
        )
    else:
        split_df = build_splits(
            df,
            seed=int(args.seed),
            test_ratio=float(args.test_ratio),
            val_ratio=float(args.val_ratio),
            kfold=int(args.kfold),
        )

    # basic validation
    if not set(split_df["split"].unique().tolist()).issubset(SPLIT_VALUES):
        raise RuntimeError(f"Invalid split values in output: {split_df['split'].unique().tolist()}")

    split_df.to_csv(out_csv, index=False)
    print(f"[make_splits] Wrote: {out_csv} (rows={len(split_df)})")

    # meta json
    class_counts = df["label"].astype(str).value_counts().to_dict()
    meta = SplitMeta(
        label_file=str(label_file),
        out_csv=str(out_csv),
        seed=int(args.seed),
        test_ratio=float(args.test_ratio),
        val_ratio=float(args.val_ratio),
        kfold=int(args.kfold),
        repeats=int(args.repeats),
        mode=str(mode),
        num_samples=int(len(df)),
        num_classes=int(len(class_counts)),
        class_counts={str(k): int(v) for k, v in class_counts.items()},
    )
    meta_path = out_csv.with_suffix(out_csv.suffix + ".meta.json")
    meta_path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[make_splits] Wrote: {meta_path}")


if __name__ == "__main__":
    main()
