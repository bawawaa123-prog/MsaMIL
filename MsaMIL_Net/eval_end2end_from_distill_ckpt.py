#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end test for (NMFEM + IAAM) using a checkpoint from train_NMFEM_distill_from_phikon.py.

Key goals:
- Load a distillation checkpoint that contains both `NMFEM` and `iaam` weights.
- Use the SAME patch reading and preprocessing policy as train_NMFEM_distill_from_phikon.py:
  - multi-scale patch discovery from folders (20x/10x/5x)
  - sort order (xy/yx) matches extractor
  - robust PIL + optional OpenCV fallback for broken PNGs
  - Resize -> ToTensor -> ImageNet Normalize
  - bag sampling can be deterministic per-WSI or random
- Evaluate on split="test" from a split CSV (optionally joining with a dataset CSV to get labels).
- Save overall and per-class metrics (loss/acc/auc etc) to a JSON file.

Usage example:
  python eval_end2end_from_distill_ckpt.py \
    --ckpt results/NMFEM_distill_phikon/run_xxx/epoch_030.pth \
    --split-csv splits/YiYuan/splits_phikon_03.csv --fold 0 \
    --patches-20x /private/ljh-data/shared/data/patches_20x \
    --patches-10x /private/ljh-data/shared/data/patches_10x \
    --patches-5x  /private/ljh-data/shared/data/patches_5x \
    --bag-size 512 --input-size 512 \
    --deterministic-eval true \
    --out results/eval_end2end/test_run.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL import PngImagePlugin


_DEFAULT_MSAMIL_PYTHON = "/home/ljh/anaconda3/envs/msamil/bin/python"


def _maybe_reexec_into_msamil() -> None:
    if os.environ.get("MSAMIL_AUTO_SWITCH", "1") == "0":
        return
    if os.environ.get("_MSAMIL_REEXECED", "0") == "1":
        return
    target = Path(os.environ.get("MSAMIL_PYTHON", _DEFAULT_MSAMIL_PYTHON)).expanduser()
    try:
        target = target.resolve()
    except Exception:
        pass
    try:
        cur = Path(sys.executable).resolve()
    except Exception:
        cur = Path(sys.executable)
    if target.exists() and cur != target:
        os.environ["_MSAMIL_REEXECED"] = "1"
        print(f"[Env] Re-exec into msamil python: {target}")
        os.execv(str(target), [str(target), *sys.argv])


try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    _maybe_reexec_into_msamil()
    raise

if "msamil" not in str(sys.executable):
    _maybe_reexec_into_msamil()

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.IAAM import IAAM
from models.NMFEM import NMFEM


# Some patch PNGs may contain huge ICC/text chunks; relax Pillow limits.
PngImagePlugin.MAX_TEXT_CHUNK = max(PngImagePlugin.MAX_TEXT_CHUNK, sys.maxsize)


SCALE_ENCODING = {
    "20x": 0,
    "10x": 1,
    "5x": 2,
}


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    patch_size: int
    root: Path


@dataclass(frozen=True)
class PatchRecord:
    path: Path
    x: int
    y: int
    scale: ScaleConfig


def _infer_id_column(df) -> str:
    for cand in ("slide_id", "image_id", "wsi_id"):
        if cand in df.columns:
            return cand
    raise ValueError("CSV must contain one of columns: slide_id / image_id / wsi_id")


def _load_split_csv_with_optional_join(
    *,
    split_csv: str,
    fold: int,
    split: str,
    dataset_csv: str | None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load samples for a given (fold, split).

    Supports two formats:
    1) split_csv already contains: id_col, label, split, fold
    2) split_csv contains: id_col, split, fold; dataset_csv provides id_col + label
    """
    import pandas as pd

    sdf = pd.read_csv(split_csv)
    id_col = _infer_id_column(sdf)
    if "split" not in sdf.columns or "fold" not in sdf.columns:
        raise ValueError("split_csv must contain 'split' and 'fold' columns")

    sdf = sdf.copy()
    sdf[id_col] = sdf[id_col].astype(str)
    sdf["split"] = sdf["split"].astype(str)
    sdf["fold"] = sdf["fold"].fillna(0).astype(int)

    if "label" not in sdf.columns:
        if not dataset_csv:
            raise ValueError(
                "split_csv has no 'label' column; please provide --dataset-csv containing labels"
            )
        ddf = pd.read_csv(dataset_csv)
        did_col = _infer_id_column(ddf)
        if "label" not in ddf.columns:
            raise ValueError("dataset_csv must contain 'label' column")
        ddf = ddf.copy()
        ddf[did_col] = ddf[did_col].astype(str)
        ddf["label"] = ddf["label"].astype(str)
        # Join on inferred ID columns.
        sdf = sdf.merge(ddf[[did_col, "label"]], left_on=id_col, right_on=did_col, how="left")
        sdf.drop(columns=[did_col], inplace=True, errors="ignore")
        if sdf["label"].isna().any():
            missing = sdf[sdf["label"].isna()][id_col].astype(str).head(5).tolist()
            raise ValueError(f"Some IDs in split_csv have no label in dataset_csv (e.g. {missing})")

    sdf["label"] = sdf["label"].astype(str)

    subset = sdf[(sdf["fold"] == int(fold)) & (sdf["split"] == str(split))]
    if subset.empty:
        raise ValueError(f"No samples found in split_csv for fold={fold}, split='{split}'")

    label_names = sorted(sdf["label"].astype(str).unique().tolist())
    out: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        out.append({"wsi_id": str(row[id_col]), "label": str(row["label"])})
    return out, label_names


def collect_patch_records(wsi_id: str, scales: Sequence[ScaleConfig]) -> List[PatchRecord]:
    records: List[PatchRecord] = []
    for scale in scales:
        slide_dir = scale.root / wsi_id
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


def sort_patch_records(records: List[PatchRecord], sort_order: str) -> None:
    def cx(rec: PatchRecord) -> float:
        return rec.x + rec.scale.patch_size / 2.0

    def cy(rec: PatchRecord) -> float:
        return rec.y + rec.scale.patch_size / 2.0

    def scale_key(rec: PatchRecord) -> int:
        return SCALE_ENCODING.get(rec.scale.name, 0)

    if sort_order == "xy":
        records.sort(key=lambda rec: (cx(rec), cy(rec), scale_key(rec)))
    elif sort_order == "yx":
        records.sort(key=lambda rec: (cy(rec), cx(rec), scale_key(rec)))
    else:
        raise ValueError("sort_order must be 'xy' or 'yx'")


class End2EndWSIDataset(Dataset):
    """WSI-level dataset for end-to-end evaluation.

    This intentionally mirrors patch reading and preprocessing from
    train_NMFEM_distill_from_phikon.py, but does NOT require teacher features.

    It computes per-patch:
    - `scales`: int codes in {0,1,2}
    - `coords`: normalized (center_x/max_w, center_y/max_h)
    from patch filenames and the same extents rule as the extractor.
    """

    def __init__(
        self,
        *,
        split_csv: str,
        fold: int,
        split: str,
        dataset_csv: str | None,
        label_names_override: List[str] | None,
        scales: Sequence[ScaleConfig],
        sort_order: str,
        bag_size: int,
        input_size: int,
        random_seed: int,
        deterministic_eval: bool,
        skip_missing: bool,
    ) -> None:
        super().__init__()
        self.split = str(split)
        self.scales = list(scales)
        self.sort_order = str(sort_order)
        self.bag_size = int(bag_size)
        self.input_size = int(input_size)
        self.random_seed = int(random_seed)
        self.deterministic_eval = bool(deterministic_eval)
        self.skip_missing = bool(skip_missing)

        samples, label_names = _load_split_csv_with_optional_join(
            split_csv=split_csv,
            fold=fold,
            split=split,
            dataset_csv=dataset_csv,
        )
        if label_names_override is not None:
            override = list(label_names_override)
            if not set(label_names).issubset(set(override)):
                raise ValueError(
                    "label_names from split_csv must be a subset of label_names_override. "
                    f"split_csv={label_names}, override={override}"
                )
            self.label_names = override
        else:
            self.label_names = label_names
        self.label2idx = {name: idx for idx, name in enumerate(self.label_names)}

        kept: List[Dict[str, Any]] = []
        missing: List[str] = []
        for s in samples:
            wsi_id = str(s["wsi_id"])
            # We consider a slide "present" if any scale folder has pngs.
            has_any = False
            for sc in self.scales:
                d = sc.root / wsi_id
                if d.exists() and any(d.glob("*.png")):
                    has_any = True
                    break
            if has_any:
                kept.append({"wsi_id": wsi_id, "label": str(s["label"])})
            else:
                missing.append(wsi_id)

        if missing and not self.skip_missing:
            raise FileNotFoundError(
                f"Missing patch PNGs for {len(missing)} slides (example: {missing[0]}). "
                "Check patches_20x/10x/5x roots."
            )
        if missing:
            print(f"[End2EndDataset] Skipping {len(missing)} slides without patch files.")

        self.samples = kept

        try:
            resize = transforms.Resize(
                (self.input_size, self.input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        except TypeError:
            resize = transforms.Resize(
                (self.input_size, self.input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )

        self.transform = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._patch_cache: Dict[str, List[PatchRecord]] = {}
        self._warned_read_fail: set[str] = set()

        print(
            f"[End2EndDataset] split={split} samples={len(self.samples)} bag_size={self.bag_size} "
            f"sort_order={self.sort_order} input={self.input_size} deterministic_eval={self.deterministic_eval}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _try_load_rgb(path: Path) -> Image.Image | None:
        """Best-effort image load (PIL first; OpenCV fallback) like the distill script."""
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            return rgb
        except Exception as exc:
            try:
                import cv2
                import numpy as _np

                data = _np.fromfile(str(path), dtype=_np.uint8)
                arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if arr is None:
                    return None
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(arr)
            except Exception:
                _ = exc
                return None

    def _get_patch_records(self, wsi_id: str) -> List[PatchRecord]:
        cached = self._patch_cache.get(wsi_id)
        if cached is not None:
            return cached
        records = collect_patch_records(wsi_id, self.scales)
        if not records:
            raise FileNotFoundError(
                f"No patch PNGs found for {wsi_id}. Checked: "
                + ", ".join(str(s.root / wsi_id) for s in self.scales)
            )
        sort_patch_records(records, self.sort_order)
        self._patch_cache[wsi_id] = records
        return records

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        wsi_id = str(sample["wsi_id"])
        label_name = str(sample["label"])
        label = int(self.label2idx[label_name])

        records = self._get_patch_records(wsi_id)
        # Compute extents exactly like tools/extract_phikon_features.py (based on patch records).
        max_w = max(int(rec.x + rec.scale.patch_size) for rec in records)
        max_h = max(int(rec.y + rec.scale.patch_size) for rec in records)

        n_avail = int(len(records))
        k = min(int(self.bag_size), n_avail)
        if k <= 0:
            raise RuntimeError(f"No patches available for {wsi_id}")

        if self.split == "train":
            rng = np.random.RandomState(self.random_seed + idx + random.randint(0, 10_000_000))
            chosen = rng.choice(n_avail, size=k, replace=False)
        else:
            if self.deterministic_eval:
                rng = np.random.RandomState(self.random_seed + idx)
                chosen = rng.choice(n_avail, size=k, replace=False)
            else:
                chosen = np.random.choice(n_avail, size=k, replace=False)

        chosen = np.asarray(chosen, dtype=np.int64)

        chosen_list = chosen.tolist()
        chosen_set = set(chosen_list)
        fallback_pool = [i for i in range(n_avail) if i not in chosen_set]
        if self.split == "train":
            rng.shuffle(fallback_pool)
        else:
            if self.deterministic_eval:
                rng.shuffle(fallback_pool)

        candidate_indices = chosen_list + fallback_pool

        images: List[torch.Tensor] = []
        coords_list: List[List[float]] = []
        scales_list: List[int] = []

        for j in candidate_indices:
            rec = records[j]
            rgb = self._try_load_rgb(rec.path)
            if rgb is None:
                if wsi_id not in self._warned_read_fail:
                    print(f"[End2EndDataset] WARN failed to read patch PNG (skipping): {rec.path}")
                    self._warned_read_fail.add(wsi_id)
                continue

            images.append(self.transform(rgb))

            # Compute normalized center coords, matching extractor logic.
            cx = float(rec.x) + float(rec.scale.patch_size) / 2.0
            cy = float(rec.y) + float(rec.scale.patch_size) / 2.0
            coords_list.append([cx / float(max_w), cy / float(max_h)])
            scales_list.append(int(SCALE_ENCODING.get(rec.scale.name, 0)))

            if len(images) >= k:
                break

        if len(images) == 0:
            raise RuntimeError(f"All sampled patches unreadable for {wsi_id} (n_avail={n_avail}).")

        patch_batch = torch.stack(images, dim=0)  # [K',3,input,input]
        coords = torch.tensor(coords_list, dtype=torch.float32)
        scales = torch.tensor(scales_list, dtype=torch.long)

        return {
            "wsi_id": wsi_id,
            "label": label,
            "patches": patch_batch,
            "coords": coords,
            "scales": scales,
        }


def encode_patches_in_chunks(*, NMFEM: NMFEM, patches: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk = int(chunk_size)
    if chunk <= 0:
        return NMFEM(patches)
    if patches.ndim != 4:
        raise ValueError(f"patches must be [K,3,H,W], got shape={tuple(patches.shape)}")
    outs: List[torch.Tensor] = []
    for i in range(0, int(patches.shape[0]), chunk):
        outs.append(NMFEM(patches[i : i + chunk]))
    return torch.cat(outs, dim=0)


def _json_safe(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def _compute_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return None
        if num_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr"))
    except Exception:
        return None


def _compute_per_class_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> Dict[int, float | None]:
    out: Dict[int, float | None] = {}
    try:
        from sklearn.metrics import roc_auc_score

        for c in range(int(num_classes)):
            yt = (y_true == c).astype(np.int64)
            if len(np.unique(yt)) < 2:
                out[c] = None
                continue
            out[c] = float(roc_auc_score(yt, probs[:, c]))
        return out
    except Exception:
        for c in range(int(num_classes)):
            out[c] = None
        return out


def _safe_div(n: float, d: float) -> float | None:
    if d <= 0:
        return None
    return float(n / d)


def _compute_f1_from_confusion(cm: List[List[int]]) -> Tuple[Dict[int, Dict[str, float | None]], Dict[str, float | None]]:
    num_classes = int(len(cm))
    if num_classes == 0:
        return {}, {"f1_macro": None, "f1_weighted": None, "f1_micro": None}

    supports = [int(sum(row)) for row in cm]
    total = int(sum(supports))

    per_class: Dict[int, Dict[str, float | None]] = {}
    for c in range(num_classes):
        tp = int(cm[c][c])
        fp = int(sum(cm[r][c] for r in range(num_classes)) - tp)
        fn = int(sum(cm[c][k] for k in range(num_classes)) - tp)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = float(2.0 * precision * recall / (precision + recall))

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    f1_vals = [per_class[c]["f1"] for c in range(num_classes) if supports[c] > 0]
    f1_vals_finite = [float(x) for x in f1_vals if x is not None and math.isfinite(float(x))]
    f1_macro = float(np.mean(f1_vals_finite)) if f1_vals_finite else None

    if total > 0:
        f1_weighted_sum = 0.0
        weight_sum = 0
        for c in range(num_classes):
            if supports[c] <= 0:
                continue
            f1c = per_class[c]["f1"]
            if f1c is None or not math.isfinite(float(f1c)):
                continue
            f1_weighted_sum += float(f1c) * float(supports[c])
            weight_sum += int(supports[c])
        f1_weighted = float(f1_weighted_sum / float(weight_sum)) if weight_sum > 0 else None
    else:
        f1_weighted = None

    total_tp = int(sum(cm[c][c] for c in range(num_classes)))
    f1_micro = _safe_div(total_tp, total)

    overall = {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
    }
    return per_class, overall


@dataclass(frozen=True)
class EvalConfig:
    ckpt: str
    split_csv: str
    fold: int
    dataset_csv: str | None

    patches_20x: str
    patches_10x: str
    patches_5x: str

    sort_order: str = "xy"
    input_size: int = 512
    bag_size: int = 512
    patch_batch_size: int = 64

    deterministic_eval: bool = True
    seed: int = 42

    device: str = "auto"  # auto/cpu/cuda/cuda:0
    amp: bool = True
    val_amp: bool = False  # keep fp32 by default

    # Where to write metrics
    out: str = ""
    num_workers: int = 6
    # If True, also write per-WSI predictions into the JSON.
    save_per_sample: bool = True


# ------------------------
# EDIT CONFIG HERE
# ------------------------
CFG = EvalConfig(
    # Distill checkpoint from train_NMFEM_distill_from_phikon.py
    ckpt="/private/ljh-data/shared/MsaMIL/MsaMIL_Net/results/NMFEM_distill_phikon_q10_20x/run_20260130_155322/best_NMFEM.pth",
    # Split definition (must contain fold/split; label can be in this csv or in dataset_csv)
    split_csv="splits/YiYuan/splits_phikon_02.csv",
    fold=0,
    # Optional: if split_csv has no 'label' column, provide dataset_csv with labels
    dataset_csv=None,
    # Patch roots (must mirror your training patch roots)
    patches_20x="/private/ljh-data/shared/data/patches_20x",
    patches_10x="/private/ljh-data/shared/data/patches_10x",
    patches_5x="/private/ljh-data/shared/data/patches_5x",
    # Sorting and preprocessing
    sort_order="xy",
    input_size=512,
    bag_size=512,
    patch_batch_size=64,
    # Test sampling policy: True => fixed per-WSI; False => random each access
    deterministic_eval=False,
    seed=90,
    # Runtime
    device="auto",
    amp=True,
    val_amp=False,
    num_workers=6,
    # Output path: empty => results/eval_end2end/run_YYYYmmdd_HHMMSS.json
    out="results/eval_end2end/run_queries06_xin.json",
    save_per_sample=True,
)


def _device_from_arg(s: str) -> torch.device:
    s = str(s).strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("cuda"):
        return torch.device(s)
    raise ValueError(f"Unknown --device: {s}")


def _infer_iaam_hparams_from_state(sd: Dict[str, Any]) -> Dict[str, Any]:
    # Defaults aligned with training.
    out: Dict[str, Any] = {
        "d_model": 512,
        "input_dim": 1024,
        "mhe_layers": 2,
        "num_heads": 8,
        "low_rank": 64,
        "num_queries": 10,
        "dropout": 0.01,
        "num_classes": None,
    }

    w = sd.get("input_proj.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape")) == 2:
        out["d_model"] = int(w.shape[0])
        out["input_dim"] = int(w.shape[1])

    # Classifier heads in this repo can be classifier.weight or classifier.3.weight.
    for k in ("classifier.3.weight", "classifier.weight"):
        cw = sd.get(k)
        if hasattr(cw, "shape") and len(getattr(cw, "shape")) == 2:
            out["num_classes"] = int(cw.shape[0])
            break

    # Infer queries from embedding weight if present.
    for k in ("dmq.learnable_queries", "dmq.query_embed.weight"):
        qw = sd.get(k)
        if hasattr(qw, "shape") and len(getattr(qw, "shape")) == 2:
            out["num_queries"] = int(qw.shape[0])
            break

    # Infer MHE layers by counting submodules in state keys.
    # e.g., mhe.layers.0.self_attn.W_Q_low.weight
    layer_ids = set()
    for k in sd.keys():
        if k.startswith("mhe.layers."):
            parts = k.split(".")
            if len(parts) > 2:
                try:
                    layer_ids.add(int(parts[2]))
                except Exception:
                    pass
    if layer_ids:
        out["mhe_layers"] = int(max(layer_ids) + 1)

    # Infer low_rank from low-rank attention weights if possible.
    # W_Q_low: [low_rank*num_heads, d_model]
    # out_proj: [d_model, low_rank*num_heads]
    num_heads = int(out.get("num_heads", 0) or 0)
    if num_heads > 0:
        wq = sd.get("mhe.layers.0.self_attn.W_Q_low.weight")
        if hasattr(wq, "shape") and len(getattr(wq, "shape")) == 2:
            r_times_h = int(wq.shape[0])
            if r_times_h % num_heads == 0:
                out["low_rank"] = int(r_times_h // num_heads)
        else:
            op = sd.get("mhe.layers.0.self_attn.out_proj.weight")
            if hasattr(op, "shape") and len(getattr(op, "shape")) == 2:
                r_times_h = int(op.shape[1])
                if r_times_h % num_heads == 0:
                    out["low_rank"] = int(r_times_h // num_heads)

    return out


def _infer_NMFEM_hparams_from_state(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Infer NMFEM hyperparameters from its state_dict.

    NOTE: num_heads is not inferable from weights; caller should provide a fallback.
    """
    out: Dict[str, Any] = {
        "output_dim": 1024,
        "num_layers": 2,
        "input_patch_size": 512,
    }

    w = sd.get("final_proj.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape")) == 2:
        out["output_dim"] = int(w.shape[0])

    # Infer input_patch_size from position embedding length (seq_len = (H/32)^2).
    pe = sd.get("position_embedding")
    if hasattr(pe, "shape") and len(getattr(pe, "shape")) == 3:
        seq_len = int(pe.shape[1])
        grid = int(round(math.sqrt(seq_len)))
        if grid > 0 and grid * grid == seq_len:
            out["input_patch_size"] = int(grid * 32)

    # Infer transformer depth by counting layer ids.
    layer_ids = set()
    for k in sd.keys():
        if k.startswith("transformer_encoder.layers."):
            parts = k.split(".")
            if len(parts) > 3:
                try:
                    layer_ids.add(int(parts[2]))
                except Exception:
                    pass
    if layer_ids:
        out["num_layers"] = int(max(layer_ids) + 1)

    return out


def evaluate_end2end(*, NMFEM: NMFEM, iaam: IAAM, loader: DataLoader, device: torch.device, cfg: EvalConfig, label_names: List[str]) -> Dict[str, Any]:
    NMFEM.eval()
    iaam.eval()

    losses: List[float] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []
    all_pred: List[int] = []
    per_sample: List[Dict[str, Any]] = []

    skipped_nonfinite = 0
    skipped_total = 0

    num_classes = len(label_names)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Test", leave=False):
            wsi_id = str(batch.get("wsi_id", ""))
            patches = batch["patches"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            scales = batch["scales"].to(device, non_blocking=True)
            label = torch.tensor([int(batch["label"])], device=device, dtype=torch.long)

            use_amp = bool(cfg.amp) and device.type == "cuda" and bool(cfg.val_amp)
            with torch.autocast(device_type=str(device.type), enabled=use_amp):
                feats = encode_patches_in_chunks(
                    NMFEM=NMFEM,
                    patches=patches,
                    chunk_size=int(cfg.patch_batch_size),
                )
                logits, _ = iaam(feats, scales, coords)

                if not (torch.isfinite(feats).all() and torch.isfinite(logits).all()):
                    skipped_nonfinite += 1
                    skipped_total += 1
                    if wsi_id:
                        print(f"[Test][WARN] non-finite tensors detected; skipping wsi_id={wsi_id}")
                    continue

                loss = F.cross_entropy(logits.unsqueeze(0), label)
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    skipped_total += 1
                    if wsi_id:
                        print(f"[Test][WARN] non-finite loss; skipping wsi_id={wsi_id}")
                    continue

            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            if not np.isfinite(probs).all():
                skipped_nonfinite += 1
                skipped_total += 1
                if wsi_id:
                    print(f"[Test][WARN] non-finite probs; skipping wsi_id={wsi_id}")
                continue

            y = int(label.item())
            p = int(probs.argmax(axis=-1))

            losses.append(float(loss.item()))
            all_labels.append(y)
            all_probs.append(probs)
            all_pred.append(p)
            if bool(getattr(cfg, "save_per_sample", True)):
                per_sample.append(
                    {
                        "wsi_id": wsi_id,
                        "y": y,
                        "y_name": label_names[y] if 0 <= y < len(label_names) else str(y),
                        "pred": p,
                        "pred_name": label_names[p] if 0 <= p < len(label_names) else str(p),
                        "probs": probs.tolist(),
                        "loss": float(loss.item()),
                    }
                )
            skipped_total += 1

    probs_np = np.stack(all_probs, axis=0) if all_probs else np.zeros((0, num_classes), dtype=np.float32)
    labels_np = np.asarray(all_labels, dtype=np.int64) if all_labels else np.zeros((0,), dtype=np.int64)
    pred_np = np.asarray(all_pred, dtype=np.int64) if all_pred else np.zeros((0,), dtype=np.int64)

    overall: Dict[str, Any] = {
        "n": int(labels_np.shape[0]),
        "skipped_nonfinite": int(skipped_nonfinite),
        "skipped_total": int(skipped_total),
        "loss": float(np.mean([x for x in losses if math.isfinite(x)])) if losses else None,
        "acc": float((pred_np == labels_np).mean()) if labels_np.size > 0 else None,
        "auc": _compute_auc(labels_np, probs_np, num_classes),
    }

    # Confusion matrix
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, p in zip(labels_np.tolist(), pred_np.tolist()):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            cm[y][p] += 1

    per_class_auc = _compute_per_class_auc(labels_np, probs_np, num_classes)

    per_class_prf, overall_prf = _compute_f1_from_confusion(cm)

    per_class: Dict[str, Any] = {}
    for c in range(num_classes):
        idxs = np.where(labels_np == c)[0]
        supp = int(idxs.size)
        if supp == 0:
            per_class[label_names[c]] = {
                "support": 0,
                "loss": None,
                "acc": None,
                "auc": per_class_auc.get(c),
            }
            continue
        cls_losses = [losses[i] for i in idxs.tolist() if i < len(losses)]
        cls_acc = float((pred_np[idxs] == labels_np[idxs]).mean())
        prf = per_class_prf.get(c, {})
        per_class[label_names[c]] = {
            "support": supp,
            "loss": float(np.mean(cls_losses)) if cls_losses else None,
            "acc": cls_acc,
            "auc": per_class_auc.get(c),
            "precision": prf.get("precision"),
            "recall": prf.get("recall"),
            "f1": prf.get("f1"),
        }

    overall.update(overall_prf)

    out = {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(label_names),
            "matrix": cm,
        },
    }
    if bool(getattr(cfg, "save_per_sample", True)):
        out["per_sample"] = per_sample
    return out



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end NMFEM+IAAM from a distill checkpoint (config is in-code CFG)."
    )
    parser.add_argument("--print-config", action="store_true", help="Print the current CFG and exit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate paths / label mapping, do not run evaluation",
    )
    args = parser.parse_args()

    cfg = CFG
    if args.print_config:
        print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
        return

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = _device_from_arg(cfg.device)

    ckpt_path = Path(cfg.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a dict containing at least 'NMFEM' and 'iaam'.")
    if "NMFEM" not in payload or "iaam" not in payload:
        raise ValueError("Checkpoint missing required keys: 'NMFEM' and/or 'iaam'.")

    label_names_override = payload.get("label_names")
    if label_names_override is not None and not isinstance(label_names_override, list):
        label_names_override = None

    scales = [
        ScaleConfig(name="20x", patch_size=512, root=Path(cfg.patches_20x).expanduser().resolve()),
        ScaleConfig(name="10x", patch_size=1024, root=Path(cfg.patches_10x).expanduser().resolve()),
        ScaleConfig(name="5x", patch_size=2048, root=Path(cfg.patches_5x).expanduser().resolve()),
    ]

    test_ds = End2EndWSIDataset(
        split_csv=cfg.split_csv,
        fold=cfg.fold,
        split="test",
        dataset_csv=cfg.dataset_csv,
        label_names_override=label_names_override,
        scales=scales,
        sort_order=cfg.sort_order,
        bag_size=cfg.bag_size,
        input_size=cfg.input_size,
        random_seed=cfg.seed,
        deterministic_eval=cfg.deterministic_eval,
        skip_missing=True,
    )

    def _collate(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch_list) == 1
        return batch_list[0]

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(getattr(cfg, "num_workers", 6)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        persistent_workers=int(getattr(cfg, "num_workers", 6)) > 0,
    )

    # Build NMFEM from its own state dict (prefer weights-derived shapes).
    train_cfg = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
    NMFEM_state = payload.get("NMFEM")
    if not isinstance(NMFEM_state, dict):
        raise TypeError(f"payload['NMFEM'] must be a state_dict dict, got {type(NMFEM_state)}")

    NMFEM_hp = _infer_NMFEM_hparams_from_state(NMFEM_state)
    NMFEM_heads = int(train_cfg.get("NMFEM_heads", 8))

    NMFEM = NMFEM(
        output_dim=int(NMFEM_hp["output_dim"]),
        num_heads=NMFEM_heads,
        num_layers=int(NMFEM_hp["num_layers"]),
        pretrained=False,
        freeze_backbone=False,
        unfreeze_backbone_blocks=0,
        input_patch_size=int(NMFEM_hp["input_patch_size"]),
        use_checkpoint=False,
    )

    # Build IAAM from its own state dict (and label_names length).
    iaam_state = payload.get("iaam")
    if not isinstance(iaam_state, dict):
        raise TypeError(f"payload['iaam'] must be a state_dict dict, got {type(iaam_state)}")

    iaam_hp = _infer_iaam_hparams_from_state(iaam_state)
    num_classes = len(test_ds.label_names)
    if iaam_hp.get("num_classes") is not None and int(iaam_hp["num_classes"]) != int(num_classes):
        raise ValueError(
            f"[IAAM] num_classes mismatch: dataset={num_classes}, iaam_state={iaam_hp['num_classes']}. "
            "Please ensure label set/order matches the checkpoint."
        )

    iaam = IAAM(
        d_model=int(iaam_hp["d_model"]),
        input_dim=int(iaam_hp["input_dim"]),
        mhe_layers=int(iaam_hp["mhe_layers"]),
        num_heads=int(iaam_hp["num_heads"]),
        low_rank=int(iaam_hp["low_rank"]),
        num_queries=int(iaam_hp["num_queries"]),
        num_classes=int(num_classes),
        dropout=float(iaam_hp["dropout"]),
    )

    # Load weights
    NMFEM.load_state_dict(NMFEM_state, strict=True)
    iaam.load_state_dict(iaam_state, strict=True)

    NMFEM.to(device)
    iaam.to(device)

    if args.dry_run:
        print("[DryRun] Dataset/model/checkpoint look OK; skip evaluation.")
        print(f"[DryRun] test_samples={len(test_ds)} num_classes={len(test_ds.label_names)}")
        return

    # Evaluate
    metrics = evaluate_end2end(
        NMFEM=NMFEM,
        iaam=iaam,
        loader=test_loader,
        device=device,
        cfg=cfg,
        label_names=test_ds.label_names,
    )

    out_dir = Path("results/eval_end2end")
    if cfg.out.strip():
        out_path = Path(cfg.out).expanduser()
        out_dir = out_path.parent
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"run_{ts}.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "when": datetime.now().isoformat(),
        "ckpt": str(ckpt_path),
        "eval_config": asdict(cfg),
        "label_names": list(test_ds.label_names),
        "metrics": metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_safe)

    overall = metrics.get("overall", {})
    print(
        "[Done] "
        f"n={overall.get('n')} loss={overall.get('loss')} acc={overall.get('acc')} auc={overall.get('auc')} "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
