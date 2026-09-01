#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project imports work when running from repo root.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.feature_dataset import PreExtractedFeatureDataset, collate_fn
from models.IAAM import IAAM


def _resolve_path(p: str | None) -> str | None:
    if p is None:
        return None
    p = str(p).strip()
    if not p:
        return None
    path = Path(p).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def _load_ckpt(path: str) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(obj)}")
    return obj


def _build_model_from_ckpt(ckpt: Dict[str, Any], device: torch.device) -> IAAM:
    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    label_names = ckpt.get("label_names")
    num_classes = int(len(label_names)) if isinstance(label_names, list) and len(label_names) > 0 else int(cfg.get("num_classes", 2))

    model = IAAM(
        d_model=int(cfg.get("d_model", 512)),
        input_dim=int(cfg.get("input_dim", 1024)),
        mhe_layers=int(cfg.get("mhe_layers", 1)),
        num_heads=int(cfg.get("num_heads", 8)),
        low_rank=int(cfg.get("low_rank", 64)),
        num_queries=int(cfg.get("num_queries", 10)),
        num_classes=num_classes,
        dropout=float(cfg.get("dropout", 0.1)),
    )

    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid model_state_dict type: {type(state)}")

    # Compatibility for legacy checkpoints:
    # - old: mhe.layers.*.self_attn.W_V.weight
    # - new: mhe.layers.*.self_attn.W_V_low.weight
    # - old may contain self_attn.out_proj.bias while current code uses bias=False.
    adapted_state: Dict[str, Any] = dict(state)
    remapped = 0
    for k in list(adapted_state.keys()):
        if k.endswith(".self_attn.W_V.weight"):
            k_new = k.replace(".self_attn.W_V.weight", ".self_attn.W_V_low.weight")
            if k_new not in adapted_state:
                adapted_state[k_new] = adapted_state[k]
                remapped += 1

    incompatible = model.load_state_dict(adapted_state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    allowed_unexpected = {
        k for k in unexpected if k.endswith(".self_attn.W_V.weight") or k.endswith(".self_attn.out_proj.bias")
    }
    allowed_missing = {
        k for k in missing if k.endswith(".self_attn.W_V_low.weight")
    }
    real_unexpected = [k for k in unexpected if k not in allowed_unexpected]
    real_missing = [k for k in missing if k not in allowed_missing]
    if real_missing or real_unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch after compatibility mapping. "
            f"missing={real_missing}, unexpected={real_unexpected}"
        )
    if remapped > 0:
        print(f"[Compat] remapped legacy self_attn W_V -> W_V_low for {remapped} tensor(s)")

    model.to(device)
    model.eval()
    return model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> None:
    ckpt_path = _resolve_path(args.ckpt)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = _load_ckpt(ckpt_path)
    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}

    features_dir = _resolve_path(args.features_dir) or _resolve_path(cfg.get("features_dir"))
    label_file = _resolve_path(args.label_file) or _resolve_path(cfg.get("label_file"))
    split_csv = _resolve_path(args.split_csv) or _resolve_path(cfg.get("split_csv"))
    fold = int(args.fold if args.fold is not None else cfg.get("fold", 0))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    if features_dir is None:
        raise ValueError("features_dir is missing (neither arg nor checkpoint config provides it)")
    if label_file is None:
        raise ValueError("label_file is missing (neither arg nor checkpoint config provides it)")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    _set_seed(seed)

    model = _build_model_from_ckpt(ckpt, device)

    dataset = PreExtractedFeatureDataset(
        features_dir=features_dir,
        label_file=label_file,
        split="test",
        test_size=float(cfg.get("test_size", 0.15)),
        val_size=float(cfg.get("val_size", 0.15)),
        random_state=seed,
        max_patches_per_wsi=int(cfg.get("max_patches_per_wsi", 512)),
        deterministic_eval_subsample=bool(cfg.get("deterministic_eval_subsample", False)),
        split_csv=split_csv,
        fold=fold,
        skip_broken=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=collate_fn,
    )

    label_names = ckpt.get("label_names")
    if not isinstance(label_names, list) or len(label_names) == 0:
        label_names = list(dataset.label_names)

    out_csv = _resolve_path(args.output_csv)
    if out_csv is None:
        ts = Path(ckpt_path).stem
        out_csv = str((_PROJECT_ROOT / f"results/YiYuan/features_phikon_queries_10/per_wsi_predictions/test_predictions_from_{ts}.csv").resolve())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    num_classes = len(label_names)
    logit_headers = [f"logit_{label_names[i]}" for i in range(num_classes)]
    prob_headers = [f"prob_{label_names[i]}" for i in range(num_classes)]
    fieldnames = ["wsi_id", "label_idx", "label_name", "pred_idx", "pred_name"] + logit_headers + prob_headers

    rows: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            features, coords, scales, labels, wsi_id = batch
            features = features.to(device)
            coords = coords.to(device)
            scales = scales.to(device)
            labels = labels.to(device)

            logits, _ = model(features, scales, coords)
            probs = F.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs).item())
            label_idx = int(labels.item())

            total += 1
            correct += int(pred_idx == label_idx)

            probs_np = probs.detach().cpu().numpy().tolist()
            logits_np = logits.detach().cpu().numpy().tolist()

            row: Dict[str, Any] = {
                "wsi_id": str(wsi_id),
                "label_idx": label_idx,
                "label_name": label_names[label_idx] if 0 <= label_idx < num_classes else "",
                "pred_idx": pred_idx,
                "pred_name": label_names[pred_idx] if 0 <= pred_idx < num_classes else "",
            }
            for i in range(num_classes):
                row[logit_headers[i]] = float(logits_np[i]) if i < len(logits_np) else ""
                row[prob_headers[i]] = float(probs_np[i]) if i < len(probs_np) else ""
            rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    summary = {
        "checkpoint": ckpt_path,
        "features_dir": features_dir,
        "label_file": label_file,
        "split_csv": split_csv,
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "test_samples": total,
        "test_acc": (float(correct) / float(total)) if total > 0 else None,
        "output_csv": out_csv,
    }
    summary_path = str(Path(out_csv).with_suffix(".summary.json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Done] test predictions csv: {out_csv}")
    print(f"[Done] summary json: {summary_path}")
    print(f"[Stats] test_samples={total}, test_acc={(float(correct) / float(total)) if total > 0 else float('nan'):.4f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load IAAM best checkpoint, recover training config, run TEST split prediction, and export per-WSI scores."
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="/private/ljh-data/shared/MsaMIL/MsaMIL_Net/results/YiYuan/features_phikon_queries_10/best_model_20260107_115422.pth",
        help="Path to best_model checkpoint",
    )
    p.add_argument("--features-dir", type=str, default=None, help="Override features_dir")
    p.add_argument("--label-file", type=str, default=None, help="Override label_file")
    p.add_argument("--split-csv", type=str, default=None, help="Override split_csv")
    p.add_argument("--fold", type=int, default=None, help="Override fold")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    p.add_argument(
        "--output-csv",
        type=str,
        default="/private/ljh-data/shared/MsaMIL/MsaMIL_Net/results/YiYuan/features_phikon_queries_10/per_wsi_predictions/test_predictions_best_20260107_115422.csv",
        help="Output CSV path",
    )
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run(args)
