from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL import PngImagePlugin
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# Mitigate CUDA memory fragmentation (helps avoid sudden OOM after long runs).
# Must be set before importing torch.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
	# PyTorch recommends expandable segments to reduce fragmentation.
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


_DEFAULT_MSAMIL_PYTHON = "/home/ljh/anaconda3/envs/msamil/bin/python"


def _maybe_reexec_into_msamil() -> None:
	"""Best-effort re-exec into the msamil conda environment Python.

	This avoids common failures when the user runs `python ...` from a shell
	that is not in the msamil env (e.g., missing torch).

	Control via env vars:
	- MSAMIL_AUTO_SWITCH=0 to disable
	- MSAMIL_PYTHON=/path/to/python to override target
	"""
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
	# If torch is missing, try to re-exec into msamil.
	_maybe_reexec_into_msamil()
	raise


if "msamil" not in str(sys.executable):
	# Torch may exist in other envs, but user wants msamil for consistency.
	_maybe_reexec_into_msamil()

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.IAAM import IAAM
from models.MSFEM import MSFEM


# Some patch PNGs may contain huge ICC/text chunks; relax Pillow limits.
PngImagePlugin.MAX_TEXT_CHUNK = max(PngImagePlugin.MAX_TEXT_CHUNK, sys.maxsize)


def _auto_find_latest_iaam_ckpt() -> str | None:
	"""Best-effort find a recent IAAM checkpoint under results/.

	This is meant to support a no-args training run.
	Priority:
	- results/YiYuan/**/best_model_*.pth
	- results/**/best_model_*.pth
	"""
	root = Path("results")
	if not root.exists():
		return None
	patterns = [
		(root / "YiYuan").glob("**/best_model_*.pth"),
		root.glob("**/best_model_*.pth"),
	]
	best: Path | None = None
	best_mtime = -1.0
	for it in patterns:
		for p in it:
			try:
				mt = p.stat().st_mtime
			except Exception:
				continue
			if mt > best_mtime:
				best = p
				best_mtime = mt
	return str(best) if best is not None else None


def _count_trainable_params(module: torch.nn.Module) -> tuple[int, int]:
	trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
	total = sum(p.numel() for p in module.parameters())
	return int(trainable), int(total)


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


def load_split_csv(split_csv: str, fold: int, split: str) -> Tuple[List[Dict[str, Any]], List[str]]:
	import pandas as pd

	sdf = pd.read_csv(split_csv)
	id_col = _infer_id_column(sdf)
	if "label" not in sdf.columns:
		raise ValueError("split_csv must contain 'label' column")
	if "split" not in sdf.columns:
		raise ValueError("split_csv must contain 'split' column")
	if "fold" not in sdf.columns:
		raise ValueError("split_csv must contain 'fold' column")

	sdf = sdf.copy()
	sdf[id_col] = sdf[id_col].astype(str)
	sdf["split"] = sdf["split"].astype(str)
	sdf["fold"] = sdf["fold"].fillna(0).astype(int)

	subset = sdf[(sdf["fold"] == int(fold)) & (sdf["split"] == split)]
	if subset.empty:
		raise ValueError(f"No samples found in split_csv for fold={fold}, split='{split}'")

	# Keep label order stable within the provided split file.
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


class PhikonDistillWSIDataset(Dataset):
	"""WSI-level dataset: returns a sampled bag of patch images + aligned teacher features.

	IMPORTANT: Distillation correctness depends on the patch ordering matching
	tools/extract_phikon_features.py ordering (center coord + scale encoding).
	"""

	def __init__(
		self,
		*,
		split_csv: str,
		fold: int,
		split: str,
		label_names_override: List[str] | None,
		teacher_features_dir: str,
		scales: Sequence[ScaleConfig],
		sort_order: str,
		bag_size: int,
		input_size: int,
		random_seed: int,
		deterministic_eval: bool,
		skip_missing: bool,
	):
		super().__init__()
		self.split = split
		self.scales = list(scales)
		self.sort_order = sort_order
		self.teacher_features_dir = Path(teacher_features_dir).expanduser().resolve()
		self.bag_size = int(bag_size)
		self.input_size = int(input_size)
		self.random_seed = int(random_seed)
		self.deterministic_eval = bool(deterministic_eval)
		self.skip_missing = bool(skip_missing)

		samples, label_names = load_split_csv(split_csv, fold, split)
		if label_names_override is not None:
			override = list(label_names_override)
			# Allow split_csv to contain a subset of classes (e.g., smoke tests or
			# highly imbalanced folds). But never allow unknown labels.
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
			wsi_id = s["wsi_id"]
			feat = self.teacher_features_dir / f"{wsi_id}.pt"
			coord = self.teacher_features_dir / f"{wsi_id}_coords.npy"
			scale = self.teacher_features_dir / f"{wsi_id}_scales.npy"
			if feat.exists() and coord.exists() and scale.exists():
				kept.append(
					{
						"wsi_id": wsi_id,
						"label": s["label"],
						"feat": str(feat),
						"coord": str(coord),
						"scale": str(scale),
					}
				)
			else:
				missing.append(wsi_id)

		if missing and not self.skip_missing:
			raise FileNotFoundError(
				f"Missing teacher features for {len(missing)} slides (example: {missing[0]}). "
				f"Check teacher_features_dir={self.teacher_features_dir}"
			)
		if missing:
			print(f"[DistillDataset] Skipping {len(missing)} slides without teacher files.")

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

		self._path_cache: Dict[str, List[Path]] = {}
		# Cache aligned (patch_paths, teacher_indices) per slide.
		# teacher_indices are indices into teacher_feats/coords/scales.
		self._align_cache: Dict[str, Tuple[List[Path], np.ndarray]] = {}
		self._warned_mismatch: set[str] = set()
		self._warned_align_low: set[str] = set()
		self._warned_read_fail: set[str] = set()

		print(
			f"[DistillDataset] split={split} samples={len(self.samples)} bag_size={self.bag_size} "
			f"sort_order={self.sort_order} input={self.input_size}"
		)

	def __len__(self) -> int:
		return len(self.samples)

	@staticmethod
	def _try_load_rgb(path: Path) -> Image.Image | None:
		"""Best-effort image load similar to teacher extractor.

		Returns RGB PIL image on success; None on failure.
		"""
		try:
			with Image.open(path) as img:
				rgb = img.convert("RGB")
			return rgb
		except Exception as exc:
			# Try OpenCV fallback (optional). This mirrors tools/extract_phikon_features.py.
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

	@classmethod
	def _is_png_readable(cls, path: Path) -> bool:
		return cls._try_load_rgb(path) is not None

	def _align_student_patches_to_teacher(
		self,
		*,
		wsi_id: str,
		teacher_coords: np.ndarray,
		teacher_scales: np.ndarray,
	) -> Tuple[List[Path], np.ndarray]:
		"""Align student patch files to teacher entries.

		Teacher extractor saves (features, coords, scales) for the subset of patches it
		managed to decode. The student side may see extra patches (or different readable
		set). We align by reconstructing (scale, x, y) keys from teacher coords/scales
		using the same slide extents normalization as the extractor.

		Returns:
			(paths, teacher_indices)
		where paths[j] corresponds to teacher_indices[j].
		"""
		cached = self._align_cache.get(wsi_id)
		if cached is not None:
			return cached

		records = collect_patch_records(wsi_id, self.scales)
		if not records:
			raise FileNotFoundError(
				f"No patch PNGs found for {wsi_id}. Checked: "
				+ ", ".join(str(s.root / wsi_id) for s in self.scales)
			)
		sort_patch_records(records, self.sort_order)

		# Compute extents exactly like tools/extract_phikon_features.py (based on patch records).
		max_w = max(int(rec.x + rec.scale.patch_size) for rec in records)
		max_h = max(int(rec.y + rec.scale.patch_size) for rec in records)

		# Map scale_code -> patch_size (needed to reconstruct top-left from center coords).
		size_by_code: Dict[int, int] = {}
		for sc in self.scales:
			code = int(SCALE_ENCODING.get(sc.name, 0))
			size_by_code[code] = int(sc.patch_size)

		# Build teacher key -> teacher index map.
		# Key: (scale_code, x_top_left, y_top_left)
		teacher_map: Dict[Tuple[int, int, int], int] = {}
		for i in range(int(teacher_coords.shape[0])):
			code = int(teacher_scales[i])
			ps = size_by_code.get(code)
			if ps is None:
				continue
			cx = int(round(float(teacher_coords[i, 0]) * float(max_w)))
			cy = int(round(float(teacher_coords[i, 1]) * float(max_h)))
			x0 = int(round(cx - float(ps) / 2.0))
			y0 = int(round(cy - float(ps) / 2.0))
			# Small tolerance for float rounding during normalization/denormalization:
			# register a 3x3 neighborhood so later exact lookup by (x,y) can succeed.
			for dx in (0, -1, 1):
				for dy in (0, -1, 1):
					k = (code, x0 + dx, y0 + dy)
					# Keep first assignment to avoid accidental overwrites.
					if k not in teacher_map:
						teacher_map[k] = int(i)

		aligned_paths: List[Path] = []
		aligned_teacher_idx: List[int] = []
		for rec in records:
			code = int(SCALE_ENCODING.get(rec.scale.name, 0))
			key = (code, int(rec.x), int(rec.y))
			t_idx = teacher_map.get(key)
			if t_idx is None:
				continue
			aligned_paths.append(rec.path)
			aligned_teacher_idx.append(int(t_idx))

		# If alignment is unexpectedly low, fall back to the previous "prefix truncate" behavior.
		# This keeps training running even if teacher/student were produced from different patch roots.
		expected_n = int(teacher_coords.shape[0])
		if len(aligned_paths) == 0 or (expected_n > 0 and len(aligned_paths) < max(1, int(0.6 * min(expected_n, len(records))))):
			if wsi_id not in self._warned_align_low:
				print(
					f"[DistillDataset] WARN low teacher/patch alignment for {wsi_id}: "
					f"aligned={len(aligned_paths)} teacher={expected_n} patches={len(records)}. "
					"Falling back to prefix truncation; recommend re-extract teacher features with the same patch set."
				)
				self._warned_align_low.add(wsi_id)
			paths_all = [r.path for r in records]
			if len(paths_all) >= expected_n:
				paths_all = paths_all[:expected_n]
				teacher_idx = np.arange(expected_n, dtype=np.int64)
			else:
				teacher_idx = np.arange(len(paths_all), dtype=np.int64)
			aligned_paths = paths_all
			aligned_teacher_idx = teacher_idx.tolist()

		out = (aligned_paths, np.asarray(aligned_teacher_idx, dtype=np.int64))
		self._align_cache[wsi_id] = out
		return out

	def _get_sorted_patch_paths(self, wsi_id: str, expected_n: int | None = None) -> List[Path]:
		cached = self._path_cache.get(wsi_id)
		if cached is not None:
			if expected_n is None:
				return cached
			# If caller expects a specific length, return a best-effort slice.
			# We avoid raising hard errors to keep training resilient.
			if len(cached) >= int(expected_n):
				return cached[: int(expected_n)]
			return cached

		records = collect_patch_records(wsi_id, self.scales)
		if not records:
			raise FileNotFoundError(
				f"No patch PNGs found for {wsi_id}. Checked: "
				+ ", ".join(str(s.root / wsi_id) for s in self.scales)
			)
		sort_patch_records(records, self.sort_order)
		paths = [rec.path for rec in records]

		if expected_n is not None and len(paths) != int(expected_n):
			expected_n = int(expected_n)
			# Common case: teacher extractor was run with --skip-broken and skipped a few unreadable PNGs.
			# If patches > expected_n, try to drop unreadable files and then truncate extras.
			if len(paths) > expected_n:
				# IMPORTANT: do not scan/read all files if not needed.
				# We only need *expected_n* readable PNGs aligned to teacher length.
				readable: List[Path] = []
				for p in paths:
					if self._is_png_readable(p):
						readable.append(p)
						if len(readable) >= expected_n:
							break
				paths = readable
				if len(paths) == expected_n:
					if wsi_id not in self._warned_mismatch:
						print(
							f"[DistillDataset] WARN mismatch fixed for {wsi_id}: "
							f"using {expected_n} patches aligned to teacher (auto-skip broken / truncate extras)."
						)
						self._warned_mismatch.add(wsi_id)
					self._path_cache[wsi_id] = paths
					return paths
			# If patches < expected_n, we cannot reliably align; return what we have and truncate teacher later.
			if wsi_id not in self._warned_mismatch:
				print(
					f"[DistillDataset] WARN patch/teacher count mismatch for {wsi_id}: "
					f"patches={len(paths)} teacher={expected_n}. "
					"Will truncate to min length; recommend re-extract teacher features if this repeats."
				)
				self._warned_mismatch.add(wsi_id)

		self._path_cache[wsi_id] = paths
		return paths

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		sample = self.samples[idx]
		wsi_id = sample["wsi_id"]
		label_name = sample["label"]
		label = int(self.label2idx[label_name])

		# Teacher features are expected to be plain tensors saved by our extractor.
		# Use weights_only=True to avoid unsafe pickle execution paths.
		try:
			teacher_feats = torch.load(sample["feat"], map_location="cpu", weights_only=True)
		except TypeError:
			# Older torch doesn't support weights_only.
			teacher_feats = torch.load(sample["feat"], map_location="cpu")
		if isinstance(teacher_feats, dict) and "features" in teacher_feats:
			teacher_feats = teacher_feats["features"]
		if not isinstance(teacher_feats, torch.Tensor):
			raise TypeError(
				f"Teacher feature file must be tensor or dict(features=...), got {type(teacher_feats)}"
			)
		teacher_feats = teacher_feats.float()  # [N,1024]

		coords = np.load(sample["coord"]).astype(np.float32)  # [N,2]
		scales = np.load(sample["scale"]).astype(np.int64)  # [N]

		n_total = int(teacher_feats.shape[0])
		if coords.shape[0] != n_total or scales.shape[0] != n_total:
			raise ValueError(f"Teacher files length mismatch for {wsi_id}")

		# Align patch files to teacher entries by (scale,x,y) keys; this is stricter than
		# count-based truncation and avoids "shift" misalignment when a middle patch is missing.
		paths, teacher_index = self._align_student_patches_to_teacher(
			wsi_id=wsi_id,
			teacher_coords=coords,
			teacher_scales=scales,
		)
		n_avail = int(len(paths))
		if n_avail <= 0:
			raise RuntimeError(
				f"No aligned patches available for {wsi_id} (teacher_n={n_total}, patches_found={len(self._path_cache.get(wsi_id, []))})"
			)

		k = min(self.bag_size, n_avail)
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

		# Robust reading: if an image is broken at training-time, skip it and pull another index.
		# This avoids hard crashes even if earlier verify/filter missed it.
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
		kept_teacher_indices: List[int] = []
		for j in candidate_indices:
			p = paths[j]
			rgb = self._try_load_rgb(p)
			if rgb is None:
				if wsi_id not in self._warned_read_fail:
					print(f"[DistillDataset] WARN failed to read patch PNG (skipping): {p}")
					self._warned_read_fail.add(wsi_id)
				continue
			images.append(self.transform(rgb))
			kept_teacher_indices.append(int(teacher_index[j]))
			if len(images) >= k:
				break
		if len(images) == 0:
			raise RuntimeError(f"All sampled patches unreadable for {wsi_id} (n_avail={n_avail}).")
		patch_batch = torch.stack(images, dim=0)  # [K',3,input,input]

		idx_t = torch.tensor(kept_teacher_indices, dtype=torch.long)
		t_feats = teacher_feats[idx_t]
		t_coords = torch.from_numpy(coords[idx_t.cpu().numpy()]).float()
		t_scales = torch.from_numpy(scales[idx_t.cpu().numpy()]).long()

		return {
			"wsi_id": wsi_id,
			"label": label,
			"patches": patch_batch,
			"teacher_feats": t_feats,
			"coords": t_coords,
			"scales": t_scales,
		}


def _rng_state_dict() -> Dict[str, Any]:
	state: Dict[str, Any] = {
		"python": random.getstate(),
		"numpy": np.random.get_state(),
		"torch": torch.get_rng_state(),
	}
	if torch.cuda.is_available():
		try:
			state["torch_cuda"] = torch.cuda.get_rng_state_all()
		except Exception:
			state["torch_cuda"] = None
	return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
	try:
		if "python" in state and state["python"] is not None:
			random.setstate(state["python"])
	except Exception:
		pass
	try:
		if "numpy" in state and state["numpy"] is not None:
			np.random.set_state(state["numpy"])
	except Exception:
		pass
	try:
		if "torch" in state and state["torch"] is not None:
			torch.set_rng_state(state["torch"])
	except Exception:
		pass
	if torch.cuda.is_available():
		try:
			cuda_state = state.get("torch_cuda")
			if cuda_state is not None:
				torch.cuda.set_rng_state_all(cuda_state)
		except Exception:
			pass


def _save_checkpoint(
	*,
	path: Path,
	msfem: MSFEM,
	iaam: IAAM,
	optimizer: torch.optim.Optimizer,
	scaler: torch.cuda.amp.GradScaler,
	epoch: int,
	best_val_auc: float,
	cfg: TrainConfig,
	label_names: List[str],
) -> None:
	payload: Dict[str, Any] = {
		"msfem": msfem.state_dict(),
		"iaam": iaam.state_dict(),
		"optimizer": optimizer.state_dict(),
		"scaler": scaler.state_dict(),
		"epoch": int(epoch),
		"best_val_auc": float(best_val_auc),
		"config": asdict(cfg),
		"label_names": list(label_names),
		"rng": _rng_state_dict(),
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, path)


def cosine_distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
	student_n = F.normalize(student, dim=-1)
	teacher_n = F.normalize(teacher, dim=-1)
	return (1.0 - (student_n * teacher_n).sum(dim=-1)).mean()


def kl_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
	t = float(temperature)
	s = student_logits / t
	q = teacher_logits / t
	return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(q, dim=-1), reduction="batchmean") * (t * t)


def evaluate_student(
	*,
	msfem: MSFEM,
	iaam: IAAM,
	loader: DataLoader,
	device: torch.device,
	amp: bool,
	num_classes: int,
	epoch: int,
	cfg: "TrainConfig",
) -> Dict[str, Any]:
	msfem.eval()
	iaam.eval()

	losses: List[float] = []
	all_labels: List[int] = []
	all_probs: List[np.ndarray] = []
	skipped_nonfinite = 0
	skipped_total = 0

	try:
		from sklearn.metrics import roc_auc_score

		sk_ok = True
	except Exception:
		sk_ok = False

	with torch.inference_mode():
		for batch in tqdm(loader, desc="Val", leave=False):
			wsi_id = str(batch.get("wsi_id", ""))
			patches = batch["patches"].to(device, non_blocking=True)
			teacher_feats_cpu = batch["teacher_feats"]
			coords = batch["coords"].to(device, non_blocking=True)
			scales = batch["scales"].to(device, non_blocking=True)
			label = torch.tensor([int(batch["label"])], device=device, dtype=torch.long)

			# Per-epoch weights (mirror training schedule)
			lambda_feat_now = _lambda_feat_for_epoch(cfg=cfg, epoch=int(epoch))
			w_ce = _ce_weight_for_epoch(cfg=cfg, epoch=int(epoch))
			# Logit-KD ramps in after kd_start_epoch; disabled when IAAM is trainable by default.
			ramp = int(getattr(cfg, "ramp_epochs", 5))
			def _ramp(now: int, start: int) -> float:
				if now < start:
					return 0.0
				if ramp <= 0:
					return 1.0
				return float(min(1.0, (now - start + 1) / float(ramp)))
			w_kd = float(getattr(cfg, "lambda_kd", 0.0)) * _ramp(int(epoch), int(getattr(cfg, "kd_start_epoch", int(cfg.warmup_epochs) + 1)))
			iaam_trainable = bool(int(getattr(cfg, "iaam_train_start_epoch", -1)) >= 1 and int(epoch) >= int(cfg.iaam_train_start_epoch))
			disable_kd = bool(getattr(cfg, "disable_logit_kd_when_training_iaam", True)) and iaam_trainable
			need_teacher_feats = (float(lambda_feat_now) > 0.0) or ((not disable_kd) and float(w_kd) > 0.0)
			teacher_feats = teacher_feats_cpu.to(device, non_blocking=True) if need_teacher_feats else None

			# Validation is much more sensitive to AMP overflow. Default to fp32 unless cfg.val_amp=True.
			use_amp = bool(amp) and device.type == "cuda" and bool(getattr(cfg, "val_amp", False))
			with torch.autocast(device_type=str(device.type), enabled=use_amp):
				student_feats = encode_patches_in_chunks(
					msfem=msfem,
					patches=patches,
					chunk_size=_effective_patch_batch_size(cfg=cfg, epoch=int(epoch), is_train=False),
				)
				s_logits, _ = iaam(student_feats, scales, coords)
				# Guard: if model outputs go non-finite, skip this sample to keep val stable.
				bad = (not torch.isfinite(student_feats).all()) or (not torch.isfinite(s_logits).all())
				if (teacher_feats is not None) and (not torch.isfinite(teacher_feats).all()):
					bad = True
				if bad:
					skipped_nonfinite += 1
					skipped_total += 1
					if wsi_id:
						print(f"[Val][WARN] non-finite tensors detected; skipping wsi_id={wsi_id}")
					else:
						print("[Val][WARN] non-finite tensors detected; skipping one sample")
					continue
				loss = student_feats.new_tensor(0.0)
				if float(lambda_feat_now) > 0.0:
					assert teacher_feats is not None
					loss_feat = cosine_distill_loss(student_feats, teacher_feats)
					loss = loss + float(lambda_feat_now) * loss_feat
				if float(w_ce) > 0.0:
					loss_ce = F.cross_entropy(s_logits.unsqueeze(0), label)
					loss = loss + float(w_ce) * loss_ce
				if (not disable_kd) and float(w_kd) > 0.0:
					assert teacher_feats is not None
					with torch.no_grad():
						t_logits, _ = iaam(teacher_feats, scales, coords)
					loss_kd = kl_kd_loss(s_logits.unsqueeze(0), t_logits.unsqueeze(0), float(cfg.temperature))
					loss = loss + float(w_kd) * loss_kd
				if not torch.isfinite(loss):
					skipped_nonfinite += 1
					skipped_total += 1
					if wsi_id:
						print(f"[Val][WARN] non-finite loss; skipping wsi_id={wsi_id}")
					else:
						print("[Val][WARN] non-finite loss; skipping one sample")
					continue
				losses.append(float(loss.item()))

				probs = F.softmax(s_logits, dim=-1).detach().cpu().numpy()
				if not np.isfinite(probs).all():
					skipped_nonfinite += 1
					skipped_total += 1
					if wsi_id:
						print(f"[Val][WARN] non-finite probs; skipping wsi_id={wsi_id}")
					else:
						print("[Val][WARN] non-finite probs; skipping one sample")
					continue
				all_labels.append(int(label.item()))
				all_probs.append(probs)
				skipped_total += 1

	if skipped_nonfinite > 0:
		print(f"[Val] skipped_nonfinite={skipped_nonfinite} (epoch={epoch})")

	if not all_labels or not losses:
		return {"loss": float("nan"), "acc": float("nan"), "auc": float("nan")}

	probs_np = np.stack(all_probs, axis=0)
	labels_np = np.asarray(all_labels, dtype=np.int64)
	unique = np.unique(labels_np)
	pred = probs_np.argmax(axis=1)
	acc = float((pred == labels_np).mean())

	auc = float("nan")
	if sk_ok and unique.size >= 2:
		try:
			if num_classes == 2:
				auc = float(roc_auc_score(labels_np, probs_np[:, 1]))
			else:
				auc = float(roc_auc_score(labels_np, probs_np, multi_class="ovr"))
		except Exception:
			auc = float("nan")
	elif unique.size < 2:
		# Common case: fold/val split contains a single class; AUC is undefined.
		print(f"[Val][WARN] AUC undefined (only one class in val): classes={unique.tolist()}")

	finite_losses = [x for x in losses if math.isfinite(x)]
	val_loss = float(np.mean(finite_losses)) if finite_losses else float("nan")
	return {"loss": val_loss, "acc": acc, "auc": auc}


def _lambda_feat_for_epoch(*, cfg: Any, epoch: int) -> float:
	"""Get feature-distillation weight for a given epoch.

	Defaults to constant cfg.lambda_feat, unless decay is enabled.

	Config knobs:
	- lambda_feat_decay_start_epoch: int (0 => auto: iaam_train_start_epoch if enabled)
	- lambda_feat_decay_epochs: int (<=0 disables)
	- lambda_feat_final: float
	"""
	base = float(getattr(cfg, "lambda_feat", 1.0))
	final = float(getattr(cfg, "lambda_feat_final", base))
	decay_epochs = int(getattr(cfg, "lambda_feat_decay_epochs", 0))
	if decay_epochs <= 0:
		return base
	start = int(getattr(cfg, "lambda_feat_decay_start_epoch", 0))
	if start <= 0:
		auto = int(getattr(cfg, "iaam_train_start_epoch", -1))
		if auto >= 1:
			start = auto
		else:
			return base
	if epoch < start:
		return base
	# Linear decay from base -> final over decay_epochs epochs (inclusive).
	# epoch==start => progress=1/decay_epochs (i.e., start decaying right after start)
	# epoch==start+decay_epochs-1 => progress=1.0 (reaches final)
	progress = float(epoch - start + 1) / float(max(1, decay_epochs))
	progress = float(min(1.0, max(0.0, progress)))
	return base * (1.0 - progress) + final * progress


def _cosine_ease01(t: float) -> float:
	"""Smooth 0->1 easing (cosine)."""
	t = float(min(1.0, max(0.0, t)))
	return 0.5 * (1.0 - math.cos(math.pi * t))


def _ce_weight_for_epoch(*, cfg: Any, epoch: int) -> float:
	"""Compute CE weight for a given epoch.

	Base behavior:
	- CE ramps in from 0 to `lambda_ce` starting at `ce_start_epoch` over `ramp_epochs`.

	Optional late-stage boost (synchronized with feature-distill decay by default):
	- Increase CE max weight from `lambda_ce` -> `lambda_ce_final`.
	- Start/length default to the same window as `lambda_feat` decay.

	Config knobs:
	- lambda_ce_final: float (default= lambda_ce)
	- lambda_ce_boost_start_epoch: int (0 => auto)
	- lambda_ce_boost_epochs: int (0 => auto)
	"""
	base_max = float(getattr(cfg, "lambda_ce", 0.0))
	if base_max <= 0.0:
		return 0.0
	# ramp factor
	ramp_epochs = int(getattr(cfg, "ramp_epochs", 0))
	ce_start = int(getattr(cfg, "ce_start_epoch", 0))
	if epoch < ce_start:
		return 0.0
	if ramp_epochs <= 0:
		ramp_factor = 1.0
	else:
		ramp_factor = float(min(1.0, (epoch - ce_start + 1) / float(ramp_epochs)))

	final_max = float(getattr(cfg, "lambda_ce_final", base_max))
	boost_epochs = int(getattr(cfg, "lambda_ce_boost_epochs", 0))
	boost_start = int(getattr(cfg, "lambda_ce_boost_start_epoch", 0))
	if boost_start <= 0:
		# Prefer syncing with feature-distill decay window.
		auto = int(getattr(cfg, "lambda_feat_decay_start_epoch", 0))
		if auto <= 0:
			auto = int(getattr(cfg, "iaam_train_start_epoch", -1))
		boost_start = auto
	if boost_epochs <= 0:
		boost_epochs = int(getattr(cfg, "lambda_feat_decay_epochs", 0))

	max_now = base_max
	if (
		final_max != base_max
		and boost_start >= 1
		and boost_epochs > 0
		and epoch >= boost_start
	):
		if boost_epochs <= 1:
			max_now = final_max
		else:
			# Inclusive schedule: epoch==boost_start -> t=0, epoch==boost_start+boost_epochs-1 -> t=1
			t = float(epoch - boost_start) / float(max(1, boost_epochs - 1))
			t = _cosine_ease01(t)
			max_now = base_max * (1.0 - t) + final_max * t

	return float(max_now) * float(ramp_factor)


def _effective_patch_batch_size(*, cfg: Any, epoch: int, is_train: bool) -> int:
	"""Choose MSFEM micro-batch size (chunk size) for the current epoch.

	This is a targeted mitigation for CUDA OOM when IAAM fine-tuning starts,
	as parameter gradients increase memory pressure.

	Config knobs:
	- patch_batch_size (base)
	- patch_batch_size_after_iaam_train (if >0 and epoch>=iaam_train_start_epoch)
	- val_patch_batch_size (if >0 and is_train=False)
	"""
	base = int(getattr(cfg, "patch_batch_size", 0))
	if not is_train:
		v = int(getattr(cfg, "val_patch_batch_size", 0))
		if v > 0:
			return v
	after = int(getattr(cfg, "patch_batch_size_after_iaam_train", 0))
	start = int(getattr(cfg, "iaam_train_start_epoch", -1))
	if after > 0 and start >= 1 and epoch >= start:
		return after
	return base


def encode_patches_in_chunks(
	*,
	msfem: MSFEM,
	patches: torch.Tensor,
	chunk_size: int,
	use_checkpoint: bool = False,
) -> torch.Tensor:
	"""Encode a bag of K patches by slicing into micro-batches.

	Notes:
	- Chunking reduces *per-forward* peak memory.
	- When MSFEM is trainable, autograd retains activations for every chunk until
	  backward, so chunking alone may still OOM.
	- use_checkpoint=True trades extra compute for significantly lower VRAM by
	  recomputing MSFEM activations during backward.
	"""

	# NOTE: torch.utils.checkpoint requires at least one input with requires_grad=True;
	# otherwise the returned tensor will not have a grad_fn and backward() will fail.
	# We do NOT want to set the large image tensor requires_grad (would allocate huge input grads).
	# Instead, pass a tiny dummy scalar that requires grad and is connected with +0.
	dummy_grad = torch.zeros((), device=patches.device, dtype=patches.dtype, requires_grad=True)
	msfem_trainable = any(p.requires_grad for p in msfem.parameters())

	def _ckpt_forward(x: torch.Tensor, dummy: torch.Tensor) -> torch.Tensor:
		y = msfem(x)
		return y + dummy * 0.0
	chunk = int(chunk_size)
	if chunk <= 0:
		return msfem(patches)
	if patches.ndim != 4:
		raise ValueError(f"patches must be [K,3,H,W], got shape={tuple(patches.shape)}")
	outs: List[torch.Tensor] = []
	for i in range(0, int(patches.shape[0]), chunk):
		micro = patches[i : i + chunk]
		if use_checkpoint:
			# Prefer non-reentrant checkpointing when available (better AMP compatibility).
			try:
				out = grad_checkpoint(_ckpt_forward, micro, dummy_grad, use_reentrant=False)  # type: ignore[call-arg]
			except TypeError:
				out = grad_checkpoint(_ckpt_forward, micro, dummy_grad)
			# Safety: if checkpoint produced a non-differentiable tensor, fall back to normal forward.
			if msfem_trainable and (not getattr(out, "requires_grad", False)):
				if not getattr(encode_patches_in_chunks, "_warned_ckpt_no_grad", False):
					print("[WARN] checkpoint_msfem_chunks produced non-grad output; falling back to normal forward for stability.")
					setattr(encode_patches_in_chunks, "_warned_ckpt_no_grad", True)
				out = msfem(micro)
			outs.append(out)
		else:
			outs.append(msfem(micro))
	return torch.cat(outs, dim=0)


def _set_batchnorm_eval(module: torch.nn.Module) -> None:
	"""Force all BatchNorm layers into eval mode.

	This mitigates instability when using very small micro-batches (patch_batch_size)
	while the EfficientNet backbone is trainable.
	"""
	for m in module.modules():
		if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
			m.eval()


@dataclass
class TrainConfig:
	teacher_features_dir: str
	split_csv: str
	fold: int
	sort_order: str

	patches_20x: str
	patches_10x: str
	patches_5x: str

	iaam_ckpt: str

	input_size: int = 256
	bag_size: int = 256
	patch_batch_size: int = 16
	# Optional: reduce MSFEM micro-batch once IAAM fine-tuning starts (OOM mitigation).
	# 0 disables.
	patch_batch_size_after_iaam_train: int = 0
	# Optional: freeze backbone BatchNorm running stats when micro-batch is small.
	# 0 disables. Recommended threshold: 32.
	freeze_backbone_bn_patch_batch_leq: int = 0
	# Optional: override MSFEM micro-batch for validation only (0 => use effective train value).
	val_patch_batch_size: int = 0
	msfem_layers: int = 2
	msfem_heads: int = 8
	freeze_backbone: bool = True
	unfreeze_backbone_blocks: int = 2
	# Optional: progressively unfreeze more backbone blocks after warmup.
	# 0 disables (keeps unfreeze_backbone_blocks for all epochs).
	unfreeze_backbone_blocks_after_warmup: int = 0
	checkpoint_iaam: bool = False
	# Memory: checkpoint MSFEM chunk forwards (recompute on backward) to reduce VRAM.
	checkpoint_msfem_chunks: bool = True
	# IAAM fine-tuning schedule
	# -1: never train IAAM (default distillation)
	# >=1: unfreeze + train IAAM starting from that epoch (inclusive)
	iaam_train_start_epoch: int = -1
	iaam_lr_mult: float = 0.1
	# When IAAM becomes trainable, the "teacher logits" (computed via the same IAAM)
	# are no longer a fixed target. By default we disable logit-KD in that phase.
	disable_logit_kd_when_training_iaam: bool = True

	epochs: int = 30
	warmup_epochs: int = 5
	lr: float = 1e-4
	weight_decay: float = 1e-4
	amp: bool = True
	# Default: run validation in fp32 for numerical stability.
	val_amp: bool = False
	seed: int = 42

	lambda_feat: float = 1.0
	# Optional: decay feature-distillation weight in late-stage end-to-end fine-tuning.
	# 0 disables (keeps lambda_feat constant).
	lambda_feat_decay_start_epoch: int = 0
	lambda_feat_decay_epochs: int = 0
	lambda_feat_final: float = 0.0
	lambda_kd: float = 0.5
	lambda_ce: float = 0.5
	# Optional: late-stage CE boost (max weight), typically synchronized with lambda_feat decay.
	# Example: keep lambda_ce=0.2 for most of training, then boost to 1.0 while turning off distill.
	lambda_ce_final: float = 1.0
	# 0 => auto (lambda_feat_decay_start_epoch if set, else iaam_train_start_epoch)
	lambda_ce_boost_start_epoch: int = 0
	# 0 => auto (lambda_feat_decay_epochs)
	lambda_ce_boost_epochs: int = 0
	temperature: float = 2.0

	# Stabilization knobs (recommended)
	# Loss ramping: instead of a hard switch after warmup
	ce_start_epoch: int = 0  # 0 => auto (warmup+1)
	kd_start_epoch: int = 0  # 0 => auto (warmup+1)
	ramp_epochs: int = 5
	# Reduce LR at stage switch to mitigate gradient spike
	lr_after_warmup_mult: float = 0.5
	# Optional: further reduce LR for the final CE-only fine-tuning stage.
	# 1.0 disables.
	lr_after_distill_mult: float = 1.0
	# 0 => auto (lambda_feat_decay_start_epoch + lambda_feat_decay_epochs - 1)
	lr_after_distill_start_epoch: int = 0
	# Gradient clipping (0 disables)
	grad_clip_norm: float = 1.0

	# Run controls (kept in-code, no CLI)
	debug: bool = False
	# Robustness: when a batch produces NaN/Inf, skip this update instead of
	# contaminating optimizer/model state. The wsi_id will be logged for diagnosis.
	skip_nonfinite_batches: bool = True
	# 0 => unlimited skips per epoch. If >0 and exceeded, raise to stop training.
	max_nonfinite_skips_per_epoch: int = 0
	# Robustness: if a WSI triggers CUDA OOM (common around IAAM fine-tune), optionally
	# skip that batch instead of crashing the whole run.
	skip_oom_batches: bool = True
	# 0 => unlimited OOM skips per epoch. If >0 and exceeded, raise.
	max_oom_skips_per_epoch: int = 0
	# Optional: clear CUDA cache each epoch to reduce fragmentation.
	empty_cache_each_epoch: bool = True
	resume_path: str | None = None
	# Resume-time validation behavior for the loaded epoch:
	# - "auto": if metrics.jsonl has no line for that epoch, run val once and append it
	# - "always": always run val once for that epoch (even if metrics already exists)
	# - "never": never run val for the loaded epoch (default behavior: start from next epoch)
	resume_val_policy: str = "auto"
	save_every_epoch: bool = True
	# Validation performance knobs
	# 0 => use bag_size (same as train). Otherwise use this bag size for val only.
	val_bag_size: int = 0

	# LR split for MSFEM parts
	msfem_backbone_lr_mult: float = 0.2
	# Optional: use a different IAAM LR multiplier in the final stage.
	# 0 disables (uses iaam_lr_mult for all epochs).
	iaam_lr_mult_after_distill: float = 0.0
	# 0 => auto (same as lr_after_distill_start_epoch)
	iaam_lr_mult_after_distill_start_epoch: int = 0

	num_workers: int = 6
	deterministic_eval: bool = True
	skip_missing: bool = True

	save_dir: str = "results/msfem_distill_phikon"


# ------------------------
# EDIT CONFIG HERE
# ------------------------
# 说明：按你的要求，这里集中写死所有配置，不再依赖命令行参数。
# A30 24GB 建议：
# - patch_batch_size 可提高到 64（更稳的BN统计、更快）
# - bag_size 可先用 256，验证稳定后再升到 384/512
# - warmup 拉长，CE/KD 采用 ramp，避免你之前第6个epoch那种“硬切换崩掉”
CFG = TrainConfig(
	teacher_features_dir="data/features_phikon_Yi",
	split_csv="splits/YiYuan/splits_phikon_03.csv",
	fold=0,
	sort_order="xy",
	patches_20x="/private/ljh-data/shared/data/patches_20x",
	patches_10x="/private/ljh-data/shared/data/patches_10x",
	patches_5x="/private/ljh-data/shared/data/patches_5x",
	iaam_ckpt="/private/ljh-data/shared/MsaMIL/MsaMIL_Net/results/YiYuan/features_phikon_queries_10/best_model_20260107_115422.pth",
	# data/patch resize size (必须能被32整除；与MSFEM实现一致)
	input_size=512,
	# bag size: 每个WSI抽样patch数量
	bag_size=512,
	# val bag size: 0 表示与训练相同（=bag_size）；你希望保持 512
	val_bag_size=0,
	# micro-batch: 每次送入MSFEM的patch数量（A30可提高）
	patch_batch_size=64,
	# OOM fix: when IAAM becomes trainable (epoch>=24), reduce micro-batch to lower peak VRAM.
	patch_batch_size_after_iaam_train=0,
	# Prefer B: freeze BN when micro-batch <=32 to reduce BN-stat noise/drift.
	freeze_backbone_bn_patch_batch_leq=32,
	val_patch_batch_size=0,
	msfem_layers=2,
	msfem_heads=8,
	# backbone训练策略：先冻结更稳；如果想更强对齐，可解冻最后1~2个block
	freeze_backbone=True,
	unfreeze_backbone_blocks=1,
	# warmup后适当多解冻一些backbone blocks，提升“从图像到Phikon空间”的可学习性
	unfreeze_backbone_blocks_after_warmup=2,
	checkpoint_iaam=False,
	# IAAM 微调：更稳健的默认做法是“最后若干个epoch小学习率端到端对齐”。
	# -1 表示全程冻结；你也可以改成 0（自动=warmup+1）或某个具体epoch（如 21/24）。
	# 默认：蒸馏阶段保持 IAAM 冻结（更符合“用已训 IAAM 当老师/聚合器”的设定）
	iaam_train_start_epoch=11,
	iaam_lr_mult=0.05,
	# Late-stage CE-only fine-tune: smaller LR, but slightly larger IAAM LR to adapt to MSFEM's distribution.
	lr_after_distill_mult=0.5,
	lr_after_distill_start_epoch=0,
	iaam_lr_mult_after_distill=0.4,
	iaam_lr_mult_after_distill_start_epoch=0,
	disable_logit_kd_when_training_iaam=True,
	# epoch设计（更稳的三阶段）
	epochs=15,
	warmup_epochs=10,
	# 主要学习率（MSFEM非backbone部分）
	lr=1e-4,
	weight_decay=1e-4,
	amp=True,
	seed=42,
	# loss权重：特征对齐为主，KD/CE 慢慢加
	lambda_feat=1.0,
	# Late-stage: after IAAM fine-tune begins, gradually turn off feature distillation.
	# 在端到端阶段（epoch>=11）逐步把特征蒸馏衰减到 0（与总 epochs=15 对齐）。
	lambda_feat_decay_start_epoch=11,
	lambda_feat_decay_epochs=5,
	lambda_feat_final=0.0,
	lambda_kd=0.3,
	lambda_ce=0.2,
	# Late-stage: synchronize CE boost with feat-distill decay window (auto start/len).
	# This yields roughly: epoch11..15 => CE max 0.2 -> 1.0 (cosine eased), while lambda_feat decays to 0.
	lambda_ce_final=1.0,
	lambda_ce_boost_start_epoch=0,
	lambda_ce_boost_epochs=0,
	temperature=2.0,
	# KD从warmup结束就开始ramp；CE稍晚开始（避免早期监督过强）
	kd_start_epoch=5,
	ce_start_epoch=8,
	ramp_epochs=6,
	# warmup结束时降低LR，抑制目标切换导致的梯度冲击
	lr_after_warmup_mult=0.5,
	grad_clip_norm=1.0,
	# backbone lr更小
	msfem_backbone_lr_mult=0.2,
	num_workers=4,
	deterministic_eval=True,
	skip_missing=True,
	save_dir="results/msfem_distill_phikon_q10_Xin",
	debug=False,
	empty_cache_each_epoch=True,
	# Resume: 继续上一次 run 的 last_msfem.pth（完成 epoch_030 后将从 epoch_031 继续）。
	# 若你想从某个固定 epoch 继续，也可以把这里改成 epoch_XXX.pth。
	# 新 run 建议先不要 resume，避免误把旧 run 的 IAAM/MSFEM 状态带进来。
	resume_path=None,
	resume_val_policy="auto",
	save_every_epoch=True,
)


def main() -> None:
	cfg = CFG

	# Convenience: allow iaam_train_start_epoch=0 meaning "auto" -> start right after warmup.
	if int(cfg.iaam_train_start_epoch) == 0:
		cfg = TrainConfig(**{**asdict(cfg), "iaam_train_start_epoch": int(cfg.warmup_epochs) + 1})
	# Convenience: allow ce/kd start epoch 0 meaning "auto" -> start right after warmup.
	if int(cfg.ce_start_epoch) == 0:
		cfg = TrainConfig(**{**asdict(cfg), "ce_start_epoch": int(cfg.warmup_epochs) + 1})
	if int(cfg.kd_start_epoch) == 0:
		cfg = TrainConfig(**{**asdict(cfg), "kd_start_epoch": int(cfg.warmup_epochs) + 1})

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	debug = bool(cfg.debug)
	if debug:
		print(f"[Debug] python={sys.executable}")
		print(f"[Debug] torch={torch.__version__}")
		print(f"[Debug] device={device}")
		print(f"[Debug] cfg={json.dumps(asdict(cfg), ensure_ascii=False)}")

	# Resolve IAAM checkpoint (supports passing --iaam-ckpt auto; also falls back if default path is missing).
	iaam_ckpt_path = Path(str(cfg.iaam_ckpt)).expanduser()
	if str(cfg.iaam_ckpt).strip().lower() == "auto" or not iaam_ckpt_path.exists():
		found = _auto_find_latest_iaam_ckpt()
		if found is None:
			raise FileNotFoundError(
				f"IAAM checkpoint not found: {cfg.iaam_ckpt}. "
				"Pass --iaam-ckpt /path/to/best_model_xxx.pth or ensure results/**/best_model_*.pth exists."
			)
		cfg = TrainConfig(**{**asdict(cfg), "iaam_ckpt": found})
		print(f"[IAAM] Using checkpoint: {cfg.iaam_ckpt}")

	# Load IAAM checkpoint first so we can enforce consistent class ordering
	try:
		ckpt = torch.load(cfg.iaam_ckpt, map_location="cpu", weights_only=True)
	except TypeError:
		ckpt = torch.load(cfg.iaam_ckpt, map_location="cpu")
	except Exception:
		ckpt = torch.load(cfg.iaam_ckpt, map_location="cpu")
	ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
	label_names_override = ckpt.get("label_names") if isinstance(ckpt, dict) else None
	if label_names_override is not None and not isinstance(label_names_override, list):
		label_names_override = None
	state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
	if isinstance(state, dict):
		keys = list(state.keys())
		if keys and str(keys[0]).startswith("module."):
			state = {k[len("module."):]: v for k, v in state.items()}
	# Infer num_classes from state dict when possible
	def _infer_num_classes_from_state(sd: Dict[str, Any]) -> int | None:
		for k in ("classifier.3.weight", "classifier.weight"):
			w = sd.get(k)
			if hasattr(w, "shape") and len(getattr(w, "shape")) == 2:
				return int(w.shape[0])
		return None

	def _infer_num_queries_from_state(sd: Dict[str, Any]) -> int | None:
		t = sd.get("dmq.learnable_queries")
		if hasattr(t, "shape") and len(getattr(t, "shape")) == 2:
			return int(t.shape[0])
		return None

	def _infer_heads_times_low_rank_from_state(sd: Dict[str, Any]) -> int | None:
		# MLA: W_Q_low weight shape is [heads*low_rank, d_model]
		wq = sd.get("mhe.layers.0.self_attn.W_Q_low.weight")
		if hasattr(wq, "shape") and len(getattr(wq, "shape")) == 2:
			return int(wq.shape[0])
		return None
	num_classes_ckpt = _infer_num_classes_from_state(state) if isinstance(state, dict) else None
	# Infer num_queries / (heads*low_rank) when ckpt config is missing
	q_infer = _infer_num_queries_from_state(state) if isinstance(state, dict) else None
	hxlr_infer = _infer_heads_times_low_rank_from_state(state) if isinstance(state, dict) else None

	# Fill missing low_rank if needed (requires num_heads from config)
	num_heads_cfg = int(ckpt_cfg.get("num_heads", 8))
	low_rank_cfg = ckpt_cfg.get("low_rank")
	if (low_rank_cfg is None or low_rank_cfg == "") and hxlr_infer is not None and num_heads_cfg > 0:
		try:
			ckpt_cfg = dict(ckpt_cfg)
			ckpt_cfg["low_rank"] = int(hxlr_infer // num_heads_cfg)
		except Exception:
			pass

	scales = [
		ScaleConfig(name="20x", patch_size=512, root=Path(cfg.patches_20x).expanduser().resolve()),
		ScaleConfig(name="10x", patch_size=1024, root=Path(cfg.patches_10x).expanduser().resolve()),
		ScaleConfig(name="5x", patch_size=2048, root=Path(cfg.patches_5x).expanduser().resolve()),
	]

	train_ds = PhikonDistillWSIDataset(
		split_csv=cfg.split_csv,
		fold=cfg.fold,
		split="train",
		label_names_override=label_names_override,
		teacher_features_dir=cfg.teacher_features_dir,
		scales=scales,
		sort_order=cfg.sort_order,
		bag_size=cfg.bag_size,
		input_size=cfg.input_size,
		random_seed=cfg.seed,
		deterministic_eval=cfg.deterministic_eval,
		skip_missing=cfg.skip_missing,
	)
	val_ds = PhikonDistillWSIDataset(
		split_csv=cfg.split_csv,
		fold=cfg.fold,
		split="val",
		label_names_override=label_names_override,
		teacher_features_dir=cfg.teacher_features_dir,
		scales=scales,
		sort_order=cfg.sort_order,
		bag_size=(int(cfg.val_bag_size) if int(cfg.val_bag_size) > 0 else cfg.bag_size),
		input_size=cfg.input_size,
		random_seed=cfg.seed,
		deterministic_eval=cfg.deterministic_eval,
		skip_missing=cfg.skip_missing,
	)

	def _collate(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
		assert len(batch_list) == 1
		return batch_list[0]

	train_loader = DataLoader(
		train_ds,
		batch_size=1,
		shuffle=True,
		num_workers=cfg.num_workers,
		pin_memory=torch.cuda.is_available(),
		collate_fn=_collate,
		persistent_workers=cfg.num_workers > 0,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=1,
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=torch.cuda.is_available(),
		collate_fn=_collate,
		persistent_workers=cfg.num_workers > 0,
	)

	num_classes = len(train_ds.label_names)
	if num_classes_ckpt is not None and int(num_classes) != int(num_classes_ckpt):
		raise ValueError(
			"[IAAM] num_classes mismatch: "
			f"split_csv num_classes={num_classes}, ckpt num_classes={num_classes_ckpt}. "
			"请确保 splits_phikon_03.csv 的label集合/顺序与 IAAM checkpoint 的 label_names 一致。"
		)
	if label_names_override is not None and len(label_names_override) != num_classes:
		raise ValueError(
			"[IAAM] label_names mismatch: "
			f"ckpt label_names len={len(label_names_override)}, dataset num_classes={num_classes}."
		)
	if debug:
		print(f"[Debug] label_names={train_ds.label_names} (num_classes={num_classes})")
		print(f"[Debug] train_samples={len(train_ds)} val_samples={len(val_ds)}")
		# Alignment sanity check on the first sample (teacher count == patch count)
		if len(train_ds.samples) > 0:
			try:
				s0 = train_ds.samples[0]
				wsi_id0 = str(s0["wsi_id"])
				feat0 = torch.load(s0["feat"], map_location="cpu", weights_only=True)
				if isinstance(feat0, dict) and "features" in feat0:
					feat0 = feat0["features"]
				n0 = int(feat0.shape[0]) if isinstance(feat0, torch.Tensor) else -1
				_ = train_ds._get_sorted_patch_paths(wsi_id0, expected_n=n0)
				print(f"[Debug] alignment_ok wsi_id={wsi_id0} n_teacher={n0} bag_size={cfg.bag_size}")
			except Exception as exc:
				print(f"[Debug][WARN] alignment check failed: {exc}")

	iaam = IAAM(
		d_model=int(ckpt_cfg.get("d_model", 512)),
		input_dim=int(ckpt_cfg.get("input_dim", 1024)),
		mhe_layers=int(ckpt_cfg.get("mhe_layers", 1)),
		num_heads=int(ckpt_cfg.get("num_heads", 8)),
		low_rank=int(ckpt_cfg.get("low_rank", 64)),
		num_queries=int(ckpt_cfg.get("num_queries", q_infer if q_infer is not None else 10)),
		num_classes=int(num_classes),
		dropout=float(ckpt_cfg.get("dropout", 0.01)),
	)
	# 按你的要求：严格匹配checkpoint结构
	if not isinstance(state, dict):
		raise TypeError(f"IAAM checkpoint state_dict must be dict, got {type(state)}")
	iaam.load_state_dict(state, strict=True)
	iaam.to(device)

	def _set_iaam_trainable(trainable: bool) -> None:
		iaam.train(mode=bool(trainable))
		for p in iaam.parameters():
			p.requires_grad = bool(trainable)
		# If frozen, keep IAAM in eval for deterministic behavior.
		if not trainable:
			iaam.eval()

	# Default: keep IAAM frozen unless user explicitly enables fine-tuning.
	_set_iaam_trainable(False)

	msfem = MSFEM(
		output_dim=1024,
		num_heads=cfg.msfem_heads,
		num_layers=cfg.msfem_layers,
		pretrained=True,
		freeze_backbone=cfg.freeze_backbone,
		unfreeze_backbone_blocks=cfg.unfreeze_backbone_blocks,
		input_patch_size=cfg.input_size,
		use_checkpoint=False,
	)
	msfem.to(device)

	def _apply_backbone_unfreeze(unfreeze_blocks: int) -> None:
		"""Freeze entire backbone then unfreeze last N blocks (EfficientNet features modules)."""
		if not bool(cfg.freeze_backbone):
			# Fully trainable backbone.
			for p in msfem.backbone.parameters():
				p.requires_grad = True
			return
		# Freeze all first.
		for p in msfem.backbone.parameters():
			p.requires_grad = False
		n = max(0, int(unfreeze_blocks))
		if n <= 0:
			return
		blocks = list(msfem.backbone.children()) if isinstance(msfem.backbone, torch.nn.Sequential) else []
		if not blocks:
			return
		for m in blocks[-n:]:
			for p in m.parameters():
				p.requires_grad = True

	# Ensure the initial freeze/unfreeze policy matches cfg even if MSFEM init logic changes.
	_apply_backbone_unfreeze(int(cfg.unfreeze_backbone_blocks))
	if debug:
		tr, tot = _count_trainable_params(msfem)
		try:
			tr_bb, tot_bb = _count_trainable_params(msfem.backbone)
		except Exception:
			tr_bb, tot_bb = -1, -1
		print(f"[Debug] MSFEM trainable={tr}/{tot} | backbone trainable={tr_bb}/{tot_bb}")
		if cfg.freeze_backbone:
			print(f"[Debug] freeze_backbone=True, unfreeze_backbone_blocks={cfg.unfreeze_backbone_blocks}")

	# Optimizer: split MSFEM backbone vs head/transformer for more stable fine-tuning.
	# IMPORTANT: include ALL backbone params so later unfreezing works without rebuilding the optimizer.
	backbone_params = list(msfem.backbone.parameters())
	head_params: List[torch.nn.Parameter] = []
	for name, p in msfem.named_parameters():
		if not p.requires_grad:
			continue
		if name.startswith("backbone."):
			continue
		head_params.append(p)
	# IAAM params (kept in optimizer so fine-tuning can be enabled by schedule)
	iaam_params = list(iaam.parameters())
	param_groups = [
		{
			"name": "msfem_head",
			"params": head_params,
			"lr": float(cfg.lr),
			"weight_decay": float(cfg.weight_decay),
		},
		{
			"name": "msfem_backbone",
			"params": backbone_params,
			"lr": float(cfg.lr) * float(cfg.msfem_backbone_lr_mult),
			"weight_decay": float(cfg.weight_decay),
		},
		{
			"name": "iaam",
			"params": iaam_params,
			"lr": float(cfg.lr) * float(cfg.iaam_lr_mult),
			"weight_decay": float(cfg.weight_decay),
		},
	]
	optimizer = torch.optim.AdamW(param_groups)

	# Prefer the new API when available (PyTorch >= 2.1), fallback for older versions.
	try:
		scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")  # type: ignore[attr-defined]
	except Exception:
		scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

	start_epoch = 1
	best_auc = -1.0
	resumed_epoch: int | None = None

	def _metrics_has_epoch(path: Path, epoch_num: int) -> bool:
		if not path.exists():
			return False
		try:
			text = path.read_text(encoding="utf-8", errors="replace")
		except Exception:
			return False
		for line in text.splitlines():
			s = line.strip()
			if not s:
				continue
			try:
				obj = json.loads(s)
			except Exception:
				continue
			try:
				ep = int(obj.get("epoch", -1))
			except Exception:
				continue
			if ep == int(epoch_num):
				return True
		return False

	def _append_metrics(run_dir_: Path, summary: Dict[str, Any]) -> None:
		path = run_dir_ / "metrics.jsonl"
		with open(path, "a", encoding="utf-8") as f:
			def _json_safe(v: Any) -> Any:
				if isinstance(v, float):
					if math.isnan(v) or math.isinf(v):
						return None
				return v
			safe_summary = {k: _json_safe(v) for k, v in summary.items()}
			f.write(json.dumps(safe_summary, ensure_ascii=False, allow_nan=False) + "\n")

	if cfg.resume_path:
		resume_path = Path(str(cfg.resume_path)).expanduser().resolve()
		if not resume_path.exists():
			raise FileNotFoundError(f"resume_path not found: {resume_path}")
		resume_payload = torch.load(resume_path, map_location="cpu")
		if not isinstance(resume_payload, dict) or "msfem" not in resume_payload:
			raise ValueError("Resume checkpoint must be a dict containing at least key 'msfem'.")
		msfem.load_state_dict(resume_payload["msfem"], strict=True)
		if "iaam" in resume_payload and resume_payload["iaam"] is not None:
			try:
				iaam.load_state_dict(resume_payload["iaam"], strict=False)
			except Exception as exc:
				print(f"[Resume][WARN] Failed to load IAAM state: {exc}")
		if "optimizer" in resume_payload and resume_payload["optimizer"] is not None:
			try:
				optimizer.load_state_dict(resume_payload["optimizer"])
			except Exception as exc:
				print(f"[Resume][WARN] Failed to load optimizer state: {exc}")
		if "scaler" in resume_payload and resume_payload["scaler"] is not None:
			try:
				scaler.load_state_dict(resume_payload["scaler"])
			except Exception as exc:
				print(f"[Resume][WARN] Failed to load scaler state: {exc}")
		if "rng" in resume_payload and isinstance(resume_payload["rng"], dict):
			_restore_rng_state(resume_payload["rng"])
		prev_epoch = int(resume_payload.get("epoch", 0))
		resumed_epoch = int(prev_epoch) if int(prev_epoch) > 0 else None
		start_epoch = max(1, prev_epoch + 1)
		best_auc = float(resume_payload.get("best_val_auc", -1.0))
		run_dir = resume_path.parent
		print(f"[Resume] Loaded {resume_path}")
		print(f"[Resume] start_epoch={start_epoch}, best_val_auc={best_auc}")
		try:
			with open(run_dir / "resume_note.json", "w", encoding="utf-8") as f:
				json.dump(
					{
						"resumed_at": datetime.now().isoformat(),
						"resume_path": str(resume_path),
						"start_epoch": start_epoch,
						"best_val_auc": best_auc,
					},
					f,
					ensure_ascii=False,
					indent=2,
				)
		except Exception:
			pass
	else:
		run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		run_dir = Path(cfg.save_dir) / f"run_{run_ts}"
		run_dir.mkdir(parents=True, exist_ok=True)
		with open(run_dir / "config.json", "w", encoding="utf-8") as f:
			json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

	best_path = run_dir / "best_msfem.pth"
	last_path = run_dir / "last_msfem.pth"
	# Optional: after resuming from epoch_{k}.pth, you may want to run validation for epoch k
	# (e.g., when the job was killed during val). We support this by checking metrics.jsonl.
	if resumed_epoch is not None:
		policy = str(getattr(cfg, "resume_val_policy", "auto")).strip().lower()
		if policy not in {"auto", "always", "never"}:
			print(f"[Resume][WARN] Unknown resume_val_policy='{policy}', fallback to 'auto'.")
			policy = "auto"
		metrics_path = run_dir / "metrics.jsonl"
		has_metrics = _metrics_has_epoch(metrics_path, int(resumed_epoch))
		do_val = (policy == "always") or (policy == "auto" and (not has_metrics))
		if do_val:
			print(
				f"[Resume] Running val for resumed epoch={resumed_epoch} "
				f"(policy={policy}, metrics_has_epoch={has_metrics})"
			)
			val_metrics = evaluate_student(
				msfem=msfem,
				iaam=iaam,
				loader=val_loader,
				device=device,
				amp=cfg.amp,
				num_classes=num_classes,
				epoch=int(resumed_epoch),
				cfg=cfg,
			)
			summary = {
				"epoch": int(resumed_epoch),
				"train_loss": None,
				"train_cos": None,
				"train_ce": None,
				"val_loss": val_metrics.get("loss"),
				"val_acc": val_metrics.get("acc"),
				"val_auc": val_metrics.get("auc"),
				"note": "resume_val_only",
			}
			_append_metrics(run_dir, summary)
			print(
				f"[Resume] Val done for epoch {resumed_epoch:03d} | "
				f"val_loss={float(val_metrics.get('loss', float('nan'))):.4f} "
				f"acc={float(val_metrics.get('acc', float('nan'))):.4f} auc={val_metrics.get('auc')}"
			)
			auc = val_metrics.get("auc")
			if isinstance(auc, (int, float)) and not (isinstance(auc, float) and math.isnan(auc)) and float(auc) > best_auc:
				best_auc = float(auc)
				_save_checkpoint(
					path=best_path,
					msfem=msfem,
					iaam=iaam,
					optimizer=optimizer,
					scaler=scaler,
					epoch=int(resumed_epoch),
					best_val_auc=best_auc,
					cfg=cfg,
					label_names=train_ds.label_names,
				)
				print(f"[Best] Updated best_msfem.pth from resume-val (val_auc={best_auc:.4f})")
		else:
			print(
				f"[Resume] Skip resume-epoch val (policy={policy}, metrics_has_epoch={has_metrics}). "
				f"Continue training from epoch {start_epoch:03d}."
			)

	if cfg.iaam_train_start_epoch >= 1:
		print(
			f"[IAAM] Fine-tuning enabled: start_epoch={cfg.iaam_train_start_epoch} "
			f"(iaam_lr={float(cfg.lr) * float(cfg.iaam_lr_mult):.2e})"
		)
	else:
		print("[IAAM] Fine-tuning disabled (IAAM frozen throughout).")

	# One-time trainability summary before epoch loop.
	iaam_tr, iaam_tot = _count_trainable_params(iaam)
	msfem_tr, msfem_tot = _count_trainable_params(msfem)
	iaam_group = next((g for g in optimizer.param_groups if str(g.get("name")) == "iaam"), None)
	if iaam_group is not None:
		gp = iaam_group.get("params", [])
		req = sum(1 for p in gp if getattr(p, "requires_grad", False))
		print(
			f"[Debug] Pre-train trainability: IAAM={iaam_tr}/{iaam_tot} trainable (mode={'train' if iaam.training else 'eval'}) "
			f"| MSFEM={msfem_tr}/{msfem_tot} trainable | iaam_group params={len(gp)} req_grad={req} lr={float(iaam_group.get('lr', 0.0)):.2e}"
		)
	else:
		print(
			f"[Debug] Pre-train trainability: IAAM={iaam_tr}/{iaam_tot} trainable (mode={'train' if iaam.training else 'eval'}) "
			f"| MSFEM={msfem_tr}/{msfem_tot} trainable | iaam_group=NOT_FOUND"
		)

	def _backbone_trainable() -> bool:
		try:
			return any(p.requires_grad for p in msfem.backbone.parameters())
		except Exception:
			return any(p.requires_grad for p in msfem.parameters())

	def _should_freeze_backbone_bn() -> bool:
		# When backbone is trainable and per-forward micro-batch is small, BatchNorm
		# running stats can become noisy. Freezing BN avoids chunk-size-dependent drift.
		thr = int(getattr(cfg, "freeze_backbone_bn_patch_batch_leq", 0))
		if thr <= 0:
			return False
		micro = int(_effective_patch_batch_size(cfg=cfg, epoch=int(epoch), is_train=True))
		return _backbone_trainable() and micro > 0 and micro <= thr

	def _auto_distill_end_epoch() -> int:
		start = int(getattr(cfg, "lambda_feat_decay_start_epoch", 0) or 0)
		decay = int(getattr(cfg, "lambda_feat_decay_epochs", 0) or 0)
		if start > 0 and decay > 0:
			return int(start + decay - 1)
		return 0

	def _apply_epoch_lrs(epoch_num: int) -> None:
		"""Deterministically set param-group LRs from cfg for this epoch.

		This makes resume runs reproducible and ensures cfg changes (e.g., IAAM LR tweaks)
		take effect without relying on optimizer state from an older run.
		"""
		base_lr = float(cfg.lr)
		# Stage switch (warmup -> joint objectives)
		if int(epoch_num) >= int(cfg.warmup_epochs) + 1 and float(cfg.lr_after_warmup_mult) > 0.0:
			base_lr *= float(cfg.lr_after_warmup_mult)
		# Final CE-only fine-tune stage
		distill_end = int(getattr(cfg, "lr_after_distill_start_epoch", 0) or 0)
		if distill_end <= 0:
			distill_end = _auto_distill_end_epoch()
		if distill_end > 0 and int(epoch_num) >= int(distill_end) and float(getattr(cfg, "lr_after_distill_mult", 1.0)) > 0.0:
			base_lr *= float(getattr(cfg, "lr_after_distill_mult", 1.0))

		iaam_mult = float(cfg.iaam_lr_mult)
		iaam_mult2 = float(getattr(cfg, "iaam_lr_mult_after_distill", 0.0) or 0.0)
		iaam_start2 = int(getattr(cfg, "iaam_lr_mult_after_distill_start_epoch", 0) or 0)
		if iaam_start2 <= 0:
			iaam_start2 = distill_end
		if iaam_mult2 > 0.0 and iaam_start2 > 0 and int(epoch_num) >= int(iaam_start2):
			iaam_mult = iaam_mult2

		lr_head = base_lr
		lr_bb = base_lr * float(cfg.msfem_backbone_lr_mult)
		lr_iaam = base_lr * iaam_mult
		for g in optimizer.param_groups:
			name = str(g.get("name", ""))
			if name == "msfem_head":
				g["lr"] = float(lr_head)
			elif name == "msfem_backbone":
				g["lr"] = float(lr_bb)
			elif name == "iaam":
				g["lr"] = float(lr_iaam)
		if bool(getattr(cfg, "debug", False)) or int(epoch_num) == int(start_epoch) or int(epoch_num) in {int(cfg.warmup_epochs) + 1, distill_end}:
			print(f"[LR] epoch={epoch_num:03d} head={lr_head:.2e} bb={lr_bb:.2e} iaam={lr_iaam:.2e} (iaam_mult={iaam_mult:.3f})")

	for epoch in range(start_epoch, cfg.epochs + 1):
		# Reduce memory fragmentation/pressure across long runs.
		if device.type == "cuda" and bool(getattr(cfg, "empty_cache_each_epoch", True)):
			try:
				torch.cuda.empty_cache()
				gc.collect()
			except Exception:
				pass
		# Optional: after warmup, unfreeze more backbone blocks to improve feature distillation.
		if (
			int(epoch) == int(cfg.warmup_epochs) + 1
			and int(getattr(cfg, "unfreeze_backbone_blocks_after_warmup", 0)) > 0
			and bool(cfg.freeze_backbone)
		):
			_apply_backbone_unfreeze(int(cfg.unfreeze_backbone_blocks_after_warmup))
			if bool(cfg.debug):
				tr, tot = _count_trainable_params(msfem)
				tr_bb, tot_bb = _count_trainable_params(msfem.backbone)
				print(
					f"[Debug] backbone unfreeze after warmup: blocks={cfg.unfreeze_backbone_blocks_after_warmup} "
					f"| MSFEM trainable={tr}/{tot} backbone={tr_bb}/{tot_bb}"
				)
		# Set learning rates for this epoch (reproducible across resume).
		_apply_epoch_lrs(int(epoch))
		msfem.train()
		# Unfreeze IAAM according to schedule.
		iaam_trainable = bool(cfg.iaam_train_start_epoch >= 1 and epoch >= int(cfg.iaam_train_start_epoch))
		_set_iaam_trainable(iaam_trainable)
		# IMPORTANT: if IAAM is frozen, keep it in eval mode to disable dropout.
		# Otherwise the logged acc/ce can look artificially low/noisy even when
		# feature alignment is improving.
		if iaam_trainable:
			iaam.train()
		else:
			iaam.eval()
		# Debug: confirm IAAM is actually trainable + in optimizer at the switch.
		if epoch == start_epoch or epoch == int(getattr(cfg, "iaam_train_start_epoch", -1)):
			iaam_tr, iaam_tot = _count_trainable_params(iaam)
			iaam_group = next((g for g in optimizer.param_groups if str(g.get("name")) == "iaam"), None)
			if iaam_group is not None:
				gp = iaam_group.get("params", [])
				req = sum(1 for p in gp if getattr(p, "requires_grad", False))
				print(
					f"[Debug] Epoch {epoch:03d} IAAM switch check: iaam_trainable={iaam_trainable} "
					f"| IAAM trainable_params={iaam_tr}/{iaam_tot} (mode={'train' if iaam.training else 'eval'}) "
					f"| iaam_group req_grad={req}/{len(gp)} lr={float(iaam_group.get('lr', 0.0)):.2e}"
				)
			else:
				print(
					f"[Debug] Epoch {epoch:03d} IAAM switch check: iaam_trainable={iaam_trainable} "
					f"| IAAM trainable_params={iaam_tr}/{iaam_tot} (mode={'train' if iaam.training else 'eval'}) "
					"| iaam_group=NOT_FOUND"
				)
		freeze_bn = _should_freeze_backbone_bn()
		if freeze_bn:
			_set_batchnorm_eval(msfem.backbone)
			if bool(cfg.debug):
				print(
					f"[Debug] Epoch {epoch:03d}: forcing backbone BatchNorm eval "
					f"(patch_batch_size={_effective_patch_batch_size(cfg=cfg, epoch=int(epoch), is_train=True)})"
				)
		msfem_any_trainable = any(p.requires_grad for p in msfem.parameters())
		epoch_losses: List[float] = []
		epoch_cos: List[float] = []
		epoch_ce_mon: List[float] = []
		epoch_skipped_nonfinite = 0
		epoch_skipped_oom = 0
		nonfinite_wsi_ids: List[str] = []
		oom_wsi_ids: List[str] = []
		label_names = list(getattr(train_ds, "label_names", []))
		epoch_correct = 0
		epoch_total = 0

		pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", dynamic_ncols=True, mininterval=0.5)
		for step_idx, batch in enumerate(pbar, start=1):
			wsi_raw = batch.get("wsi_id", "")
			if isinstance(wsi_raw, (list, tuple)):
				wsi_id = str(wsi_raw[0]) if len(wsi_raw) == 1 else ",".join(str(x) for x in wsi_raw)
			else:
				wsi_id = str(wsi_raw)

			patches = batch["patches"].to(device, non_blocking=True)
			teacher_feats_cpu = batch["teacher_feats"]
			coords = batch["coords"].to(device, non_blocking=True)
			scales_t = batch["scales"].to(device, non_blocking=True)
			label = torch.tensor([int(batch["label"])], device=device, dtype=torch.long)

			optimizer.zero_grad(set_to_none=True)

			use_amp = cfg.amp and device.type == "cuda"
			nonfinite_reason: str | None = None
			s_logits: torch.Tensor | None = None
			cos_mean = float("nan")
			ce_mon = float("nan")
			loss: torch.Tensor | None = None
			chunk_size = _effective_patch_batch_size(cfg=cfg, epoch=int(epoch), is_train=True)
			use_msfem_ckpt = bool(getattr(cfg, "checkpoint_msfem_chunks", False)) and msfem_any_trainable
			with torch.autocast(device_type=str(device.type), enabled=use_amp):
				try:
					student_feats = encode_patches_in_chunks(
						msfem=msfem,
						patches=patches,
						chunk_size=chunk_size,
						use_checkpoint=use_msfem_ckpt,
					)
				except torch.OutOfMemoryError:
					# Common around IAAM fine-tuning: end-to-end gradients + large bag_size.
					epoch_skipped_oom += 1
					oom_wsi_ids.append(wsi_id)
					print(
						f"[OOM][train] epoch={epoch:03d} step={step_idx} wsi_id={wsi_id} "
						f"chunk_size={int(chunk_size)} checkpoint_msfem={bool(use_msfem_ckpt)}"
					)
					optimizer.zero_grad(set_to_none=True)
					if device.type == "cuda":
						try:
							torch.cuda.empty_cache()
							gc.collect()
						except Exception:
							pass
					max_oom = int(getattr(cfg, "max_oom_skips_per_epoch", 0) or 0)
					if max_oom > 0 and epoch_skipped_oom > max_oom:
						raise RuntimeError(
							f"Too many OOM batches in epoch {epoch}: {epoch_skipped_oom} > {max_oom}. "
							"Stopping to avoid wasting compute."
						)
					if bool(getattr(cfg, "skip_oom_batches", True)):
						continue
					raise
				# Monitor how well student aligns to teacher.
				lambda_feat_now = _lambda_feat_for_epoch(cfg=cfg, epoch=int(epoch))
				# Use ramped weights to avoid hard objective switch.
				warmup = int(cfg.warmup_epochs)
				ramp = int(getattr(cfg, "ramp_epochs", 5))
				def _ramp(now: int, start: int) -> float:
					if now < start:
						return 0.0
					if ramp <= 0:
						return 1.0
					return float(min(1.0, (now - start + 1) / float(ramp)))
				w_ce = _ce_weight_for_epoch(cfg=cfg, epoch=int(epoch))
				w_kd = float(cfg.lambda_kd) * _ramp(int(epoch), int(cfg.kd_start_epoch))
				# If IAAM is trainable, stop logit-KD by default (IAAM is no longer a fixed teacher).
				iaam_trainable = bool(cfg.iaam_train_start_epoch >= 1 and epoch >= int(cfg.iaam_train_start_epoch))
				disable_kd = bool(cfg.disable_logit_kd_when_training_iaam) and iaam_trainable
				need_teacher_feats = (float(lambda_feat_now) > 0.0) or ((not disable_kd) and float(w_kd) > 0.0)
				teacher_feats = teacher_feats_cpu.to(device, non_blocking=True) if need_teacher_feats else None

				if bool(getattr(cfg, "skip_nonfinite_batches", True)):
					# NOTE: teacher tensors are on CPU originally and are expected to be finite.
					if not torch.isfinite(student_feats).all():
						nonfinite_reason = "student_feats_nonfinite"
					elif need_teacher_feats and torch.is_floating_point(teacher_feats_cpu) and (not torch.isfinite(teacher_feats_cpu).all()):
						nonfinite_reason = "teacher_feats_nonfinite"
					elif torch.is_floating_point(coords) and (not torch.isfinite(coords).all()):
						nonfinite_reason = "coords_nonfinite"
					elif teacher_feats is not None and (not torch.isfinite(teacher_feats).all()):
						nonfinite_reason = "teacher_feats_nonfinite"

				if nonfinite_reason is None:
					if teacher_feats is not None:
						# Cosine monitor is always no_grad.
						with torch.no_grad():
							student_n = F.normalize(student_feats.float(), dim=-1)
							teacher_n = F.normalize(teacher_feats.float(), dim=-1)
							cos_mean = float((student_n * teacher_n).sum(dim=-1).mean().item())
						# IMPORTANT: when lambda_feat is 0, skip building distill loss graph completely.
						if float(lambda_feat_now) > 0.0:
							loss_feat = cosine_distill_loss(student_feats, teacher_feats)
						else:
							loss_feat = student_feats.new_tensor(0.0)
					else:
						# No teacher tensors on device in CE-only stage.
						cos_mean = float("nan")
						loss_feat = student_feats.new_tensor(0.0)
				else:
					loss_feat = None
				if bool(cfg.debug) and (int(epoch) == int(start_epoch) or int(epoch) == int(getattr(cfg, "lambda_feat_decay_start_epoch", 0) or -999)):
					print(
						f"[Debug] weights@epoch{epoch:03d}: lambda_feat={float(lambda_feat_now):.4f} w_ce={float(w_ce):.4f} w_kd={float(w_kd):.4f} "
						f"(ce_max={float(getattr(cfg,'lambda_ce',0.0)):.4f}->final={float(getattr(cfg,'lambda_ce_final', getattr(cfg,'lambda_ce',0.0))):.4f})"
					)

				if nonfinite_reason is None and epoch <= cfg.warmup_epochs:
					# Warmup: align features, but still run IAAM forward (monitor progress).
					# IAAM is frozen; we run it under no_grad to avoid extra graph memory.
					with torch.no_grad():
						s_logits, _ = iaam(student_feats, scales_t, coords)
						if bool(getattr(cfg, "skip_nonfinite_batches", True)) and (not torch.isfinite(s_logits).all()):
							nonfinite_reason = "student_logits_nonfinite"
						else:
							ce_mon = float(F.cross_entropy(s_logits.unsqueeze(0), label).item())
					# IMPORTANT: compute loss OUTSIDE no_grad so it keeps autograd graph.
					if nonfinite_reason is None:
						loss = float(lambda_feat_now) * loss_feat  # type: ignore[operator]
				else:
					# Stage B: joint feature KD + logit KD + supervised CE.
					if nonfinite_reason is not None:
						pass
					elif cfg.checkpoint_iaam:
						# checkpoint expects tensor-only inputs; capture coords/scales via closure.
						s_logits = grad_checkpoint(lambda feats: iaam(feats, scales_t, coords)[0], student_feats)
					else:
						s_logits, _ = iaam(student_feats, scales_t, coords)
					if nonfinite_reason is None and bool(getattr(cfg, "skip_nonfinite_batches", True)) and (not torch.isfinite(s_logits).all()):
						nonfinite_reason = "student_logits_nonfinite"
					loss_ce = None if nonfinite_reason is not None else F.cross_entropy(s_logits.unsqueeze(0), label)
					# If IAAM is trainable, stop logit-KD by default (IAAM is no longer a fixed teacher).
					iaam_trainable = bool(cfg.iaam_train_start_epoch >= 1 and epoch >= int(cfg.iaam_train_start_epoch))
					disable_kd = bool(cfg.disable_logit_kd_when_training_iaam) and iaam_trainable
					loss = None if loss_ce is None else (float(lambda_feat_now) * loss_feat + float(w_ce) * loss_ce)  # type: ignore[operator]
					if (not disable_kd) and float(w_kd) > 0.0:
						with torch.no_grad():
							assert teacher_feats is not None
							t_logits, _ = iaam(teacher_feats, scales_t, coords)
							if bool(getattr(cfg, "skip_nonfinite_batches", True)) and (not torch.isfinite(t_logits).all()):
								nonfinite_reason = "teacher_logits_nonfinite"
							else:
								loss_kd = kl_kd_loss(s_logits.unsqueeze(0), t_logits.unsqueeze(0), cfg.temperature)
								loss = loss + float(w_kd) * loss_kd  # type: ignore[operator]
					if loss_ce is not None:
						ce_mon = float(loss_ce.item())

			# Training-time robustness: skip non-finite batches.
			if bool(getattr(cfg, "skip_nonfinite_batches", True)):
				loss_bad = (loss is None) or (not torch.isfinite(loss.detach()).all())
				# `cos_mean` is only defined when teacher features are present. In late-stage CE-only
				# training, teacher_feats is None and cos_mean intentionally remains NaN; do NOT
				# treat that as a non-finite failure.
				cos_bad = (teacher_feats is not None) and (not math.isfinite(float(cos_mean)))
				ce_bad = not math.isfinite(float(ce_mon))
				if nonfinite_reason is None and loss_bad:
					nonfinite_reason = "loss_nonfinite"
				if nonfinite_reason is None and cos_bad:
					nonfinite_reason = "cos_nonfinite"
				if nonfinite_reason is None and ce_bad:
					nonfinite_reason = "ce_nonfinite"
				if nonfinite_reason is not None:
					epoch_skipped_nonfinite += 1
					nonfinite_wsi_ids.append(wsi_id)
					print(f"[NonFinite][train] epoch={epoch:03d} step={step_idx} wsi_id={wsi_id} reason={nonfinite_reason}")
					optimizer.zero_grad(set_to_none=True)
					max_skips = int(getattr(cfg, "max_nonfinite_skips_per_epoch", 0) or 0)
					if max_skips > 0 and epoch_skipped_nonfinite > max_skips:
						raise RuntimeError(
							f"Too many non-finite batches in epoch {epoch}: {epoch_skipped_nonfinite} > {max_skips}. "
							"Stopping to avoid wasting compute."
						)
					continue

			assert loss is not None and s_logits is not None

			# Backward + optional grad clipping
			scaler.scale(loss).backward()
			if float(getattr(cfg, "grad_clip_norm", 0.0)) and float(cfg.grad_clip_norm) > 0.0:
				scaler.unscale_(optimizer)
				# If IAAM is trainable, clip both MSFEM+IAAM to avoid occasional spikes.
				if bool(cfg.iaam_train_start_epoch >= 1 and epoch >= int(cfg.iaam_train_start_epoch)):
					params = list(msfem.parameters()) + list(iaam.parameters())
				else:
					params = list(msfem.parameters())
				torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.grad_clip_norm))
			scaler.step(optimizer)
			scaler.update()

			epoch_losses.append(float(loss.item()))
			epoch_cos.append(float(cos_mean))
			epoch_ce_mon.append(float(ce_mon))

			# Real-time WSI-level prediction monitoring.
			# Note: loader batch size is 1 WSI; running acc is over WSIs in this epoch.
			with torch.no_grad():
				logits_det = s_logits.detach().float()
				probs_det = F.softmax(logits_det, dim=-1)
				pred_idx = int(torch.argmax(probs_det).item())
				true_idx = int(label.item())
				epoch_total += 1
				if pred_idx == true_idx:
					epoch_correct += 1
			run_acc = float(epoch_correct / max(1, epoch_total))

			# Format probabilities compactly for tqdm.
			# For binary tasks show p(class1); otherwise show full prob vector if small.
			if num_classes == 2:
				p1 = float(probs_det[1].item())
				prob_str = f"p1={p1:.3f}"
			elif num_classes <= 6:
				prob_str = ",".join(f"{float(p.item()):.2f}" for p in probs_det)
			else:
				p_pred = float(probs_det[pred_idx].item())
				p_true = float(probs_det[true_idx].item())
				prob_str = f"pT={p_true:.3f}|pP={p_pred:.3f}"

			y_name = label_names[true_idx] if 0 <= true_idx < len(label_names) else str(true_idx)
			p_name = label_names[pred_idx] if 0 <= pred_idx < len(label_names) else str(pred_idx)

			# Put categorical info first so it remains visible when the terminal is narrow.
			# NOTE: tqdm sorts kwargs by key; use ordered_dict to preserve this order.
			post = OrderedDict(
				[
					("y", y_name),
					("pred", p_name),
					("p", prob_str),
					("acc", run_acc),
					("loss", float(np.mean(epoch_losses))),
					("cos", float(np.mean(epoch_cos))),
					("ce", float(np.mean(epoch_ce_mon))),
					("lf", float(lambda_feat_now)),
				]
			)
			pbar.set_postfix(ordered_dict=post)
			if bool(cfg.debug) and len(epoch_losses) == 1:
				try:
					t_shape = tuple(teacher_feats.shape) if teacher_feats is not None else tuple(teacher_feats_cpu.shape)
				except Exception:
					t_shape = ("?",)
				print(
					f"[Debug] first_batch patches={tuple(patches.shape)} teacher={t_shape} "
					f"coords={tuple(coords.shape)} scales={tuple(scales_t.shape)}"
				)

		# Save progress BEFORE validation so a slow/failed/aborted val doesn't lose the epoch.
		_save_checkpoint(
			path=last_path,
			msfem=msfem,
			iaam=iaam,
			optimizer=optimizer,
			scaler=scaler,
			epoch=epoch,
			best_val_auc=best_auc,
			cfg=cfg,
			label_names=train_ds.label_names,
		)
		if bool(cfg.save_every_epoch):
			epoch_path = run_dir / f"epoch_{epoch:03d}.pth"
			_save_checkpoint(
				path=epoch_path,
				msfem=msfem,
				iaam=iaam,
				optimizer=optimizer,
				scaler=scaler,
				epoch=epoch,
				best_val_auc=best_auc,
				cfg=cfg,
				label_names=train_ds.label_names,
			)

		val_metrics = evaluate_student(
			msfem=msfem,
			iaam=iaam,
			loader=val_loader,
			device=device,
			amp=cfg.amp,
			num_classes=num_classes,
			epoch=epoch,
			cfg=cfg,
		)
		train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
		if epoch_skipped_nonfinite > 0:
			uniq = len(set(nonfinite_wsi_ids))
			print(f"[NonFinite][train] epoch={epoch:03d} skipped_batches={epoch_skipped_nonfinite} unique_wsi={uniq}")
			try:
				# Append for later offline inspection.
				path = run_dir / "nonfinite_wsi_ids.txt"
				with open(path, "a", encoding="utf-8") as f:
					for wid in nonfinite_wsi_ids:
						f.write(f"epoch={epoch:03d}\twsi_id={wid}\n")
			except Exception:
				pass
		if epoch_skipped_oom > 0:
			uniq = len(set(oom_wsi_ids))
			print(f"[OOM][train] epoch={epoch:03d} skipped_batches={epoch_skipped_oom} unique_wsi={uniq}")
			try:
				path = run_dir / "oom_wsi_ids.txt"
				with open(path, "a", encoding="utf-8") as f:
					for wid in oom_wsi_ids:
						f.write(f"epoch={epoch:03d}\twsi_id={wid}\n")
			except Exception:
				pass

		summary = {
			"epoch": epoch,
			"train_loss": train_loss,
			"train_cos": float(np.mean(epoch_cos)) if epoch_cos else float("nan"),
			"train_ce": float(np.mean(epoch_ce_mon)) if epoch_ce_mon else float("nan"),
			"train_skipped_nonfinite": int(epoch_skipped_nonfinite),
			"train_skipped_oom": int(epoch_skipped_oom),
			"val_loss": val_metrics["loss"],
			"val_acc": val_metrics["acc"],
			"val_auc": val_metrics["auc"],
		}
		_append_metrics(run_dir, summary)

		print(
			f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
			f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} auc={val_metrics['auc']}"
		)

		auc = val_metrics["auc"]
		if isinstance(auc, (int, float)) and not (isinstance(auc, float) and math.isnan(auc)) and float(auc) > best_auc:
			best_auc = float(auc)
			_save_checkpoint(
				path=best_path,
				msfem=msfem,
				iaam=iaam,
				optimizer=optimizer,
				scaler=scaler,
				epoch=epoch,
				best_val_auc=best_auc,
				cfg=cfg,
				label_names=train_ds.label_names,
			)
			print(f"[Best] Updated best_msfem.pth (val_auc={best_auc:.4f})")

		# Save again after validation (overwrites last/epoch with the same weights).
		_save_checkpoint(
			path=last_path,
			msfem=msfem,
			iaam=iaam,
			optimizer=optimizer,
			scaler=scaler,
			epoch=epoch,
			best_val_auc=best_auc,
			cfg=cfg,
			label_names=train_ds.label_names,
		)
		if bool(cfg.save_every_epoch):
			epoch_path = run_dir / f"epoch_{epoch:03d}.pth"
			_save_checkpoint(
				path=epoch_path,
				msfem=msfem,
				iaam=iaam,
				optimizer=optimizer,
				scaler=scaler,
				epoch=epoch,
				best_val_auc=best_auc,
				cfg=cfg,
				label_names=train_ds.label_names,
			)

	if best_path.exists():
		print(f"Done. Best val_auc={best_auc:.4f}. Saved to {best_path}")
	else:
		print(
			"Done. No best checkpoint was saved (val_auc may be NaN or never improved). "
			f"Last checkpoint: {last_path}"
		)


if __name__ == "__main__":
	main()