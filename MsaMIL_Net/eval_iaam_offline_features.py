#!/usr/bin/env python3
"""Validate a frozen IAAM checkpoint directly on offline EfficientNet features.

This script bypasses NMFEM by loading the `.pt/.npy` feature bundles produced by
`tools/extract_efficientnet_features.py`, feeding them straight into IAAM, and
reporting accuracy/AUC/per-class stats.  It is meant to be a lightweight sanity
check showing that IAAM still performs well whenever the feature space matches
what it saw during offline training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.feature_dataset import PreExtractedFeatureDataset, collate_fn
from models.IAAM import IAAM
from train_iaam_from_features import set_seed


def _build_dataloader(
    features_dir: Path,
    label_file: Path,
    split: str,
    max_patches: int | None,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[PreExtractedFeatureDataset, DataLoader]:
    dataset = PreExtractedFeatureDataset(
        features_dir=str(features_dir),
        label_file=str(label_file),
        split=split,
        max_patches_per_wsi=max_patches,
        test_size=0.0,
        val_size=0.2,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return dataset, loader


def _load_iaam(checkpoint_path: Path, num_classes: int, device: torch.device) -> IAAM:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    input_dim = state.get("input_proj.weight", torch.empty(0)).shape[1] if "input_proj.weight" in state else 1024
    model = IAAM(
        d_model=512,
        input_dim=input_dim,
        mhe_layers=2,
        num_heads=8,
        low_rank=64,
        num_queries=10,
        num_classes=num_classes,
        dropout=0.1,
    )
    model.sort_order = ckpt.get("config", {}).get("sort_order", "xy")
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    model.eval()
    model.to(device)
    return model


def _evaluate(
    model: IAAM,
    loader: DataLoader,
    device: torch.device,
    label_names: List[str],
    save_records: Path | None = None,
) -> Dict[str, float]:
    num_classes = len(label_names)
    total = 0
    total_correct = 0
    probs_list: List[np.ndarray] = []
    labels_list: List[int] = []
    per_class_total = [0 for _ in range(num_classes)]
    per_class_correct = [0 for _ in range(num_classes)]
    csv_records: List[Dict[str, object]] = []

    with torch.no_grad():
        for features, coords, scales, label, wsi_id in loader:
            features = features.to(device)
            coords = coords.to(device)
            scales = scales.to(device)
            label = label.to(device)

            logits, _ = model(features.squeeze(0), scales.squeeze(0), coords.squeeze(0))
            probs = F.softmax(logits, dim=-1)

            pred = int(torch.argmax(probs).item())
            label_idx = int(label.item())
            total += 1
            total_correct += int(pred == label_idx)
            per_class_total[label_idx] += 1
            if pred == label_idx:
                per_class_correct[label_idx] += 1

            probs_list.append(probs.cpu().numpy())
            labels_list.append(label_idx)

            if save_records is not None:
                csv_records.append({
                    "wsi_id": wsi_id,
                    "label_idx": label_idx,
                    "label_name": label_names[label_idx],
                    "pred_idx": pred,
                    "pred_name": label_names[pred],
                    "logits": [float(v) for v in logits.cpu().tolist()],
                    "probs": [float(v) for v in probs.cpu().tolist()],
                })

    accuracy = total_correct / max(1, total)
    per_class_acc = {
        label_names[i]: (per_class_correct[i] / per_class_total[i]) if per_class_total[i] else float("nan")
        for i in range(num_classes)
    }

    metrics: Dict[str, float] = {"accuracy": accuracy}
    for name, acc in per_class_acc.items():
        metrics[f"acc_{name}"] = acc

    if probs_list and len(set(labels_list)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            y_true = np.array(labels_list)
            y_prob = np.stack(probs_list)
            metrics["auc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            pass

    if save_records is not None and csv_records:
        save_records.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "wsi_id",
            "label_idx",
            "label_name",
            "pred_idx",
            "pred_name",
        ]
        logit_headers = [f"logit_{name}" for name in label_names]
        prob_headers = [f"prob_{name}" for name in label_names]
        fieldnames.extend(logit_headers)
        fieldnames.extend(prob_headers)

        import csv

        with open(save_records, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in csv_records:
                row = {
                    "wsi_id": rec["wsi_id"],
                    "label_idx": rec["label_idx"],
                    "label_name": rec["label_name"],
                    "pred_idx": rec["pred_idx"],
                    "pred_name": rec["pred_name"],
                }
                for i, name in enumerate(label_names):
                    row[f"logit_{name}"] = rec["logits"][i]
                    row[f"prob_{name}"] = rec["probs"][i]
                writer.writerow(row)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate IAAM on offline EfficientNet features")
    parser.add_argument("--features-dir", default="data/features_efficientnet", type=Path)
    parser.add_argument("--label-file", default="data/all_data.csv", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val",
                        help="Which split to evaluate (or 'all' for train+val+test)")
    parser.add_argument("--max-patches", type=int, default=512, help="Match IAAM training bag size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional CSV path for per-WSI predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    # Load model once; dataset.num_classes should be consistent across splits
    # We will build a temporary dataset for num_classes only if needed
    # Load training dataset to read num_classes first when evaluating 'all'
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]

    # Load model using any available split to get num_classes (train/val/test)
    found = False
    for try_split in ['train', 'val', 'test']:
        try:
            sample_ds, _ = _build_dataloader(
                features_dir=args.features_dir,
                label_file=args.label_file,
                split=try_split,
                max_patches=args.max_patches,
                num_workers=1,
                pin_memory=False,
            )
            if len(sample_ds) > 0:
                model = _load_iaam(args.checkpoint, sample_ds.num_classes, device)
                found = True
                break
        except Exception:
            continue
    if not found:
        # Final fallback: try loading using default num_classes=2
        model = _load_iaam(args.checkpoint, 2, device)

    all_metrics = {}
    for sp in splits:
        dataset, loader = _build_dataloader(
            features_dir=args.features_dir,
            label_file=args.label_file,
            split=sp,
            max_patches=args.max_patches,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        save_path = None
        if args.save_csv is not None:
            # create one per-split CSV: e.g. val_predictions.csv -> train_predictions.csv
            base = str(args.save_csv)
            if base.endswith('.csv'):
                base = base[:-4]
            save_path = Path(f"{base}_{sp}.csv")
        metrics = _evaluate(
            model,
            loader,
            device,
            dataset.label_names,
            save_records=save_path,
        )
        all_metrics[sp] = metrics

    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
