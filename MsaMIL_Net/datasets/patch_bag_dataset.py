import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# Public constants used by other modules
SCALE_MAP: Dict[str, int] = {"20x": 0, "10x": 1, "5x": 2}
FNAME_REGEX = re.compile(r"_(20x|10x|5x)_(\d+)_(\d+)_\d+\.(png|jpg|jpeg|tif|tiff|bmp)$", re.IGNORECASE)


def _load_label_csv(label_csv: Path) -> Tuple[Dict[str, int], Dict[str, Tuple[int, int]], List[str]]:
    """解析 label.csv → (wsi_id→类别索引, wsi_id→(W,H), 类别名称列表)

    期望列名: image_id,label,image_width,image_height,...
    若缺失文件或列，返回空映射，由调用方决定回退策略。
    """
    labels: Dict[str, int] = {}
    dims: Dict[str, Tuple[int, int]] = {}
    class_names: List[str] = []

    if not label_csv.exists():
        return {}, {}, []

    import csv as _csv

    rows: List[Dict[str, str]] = []
    with open(label_csv, newline='') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(row)
            cls = str(row.get('label', '')).strip()
            if cls and cls not in class_names:
                class_names.append(cls)

    class_to_idx = {c: i for i, c in enumerate(sorted(class_names))}

    for row in rows:
        wid = str(row.get('image_id', '')).strip()
        if not wid:
            continue
        cls = str(row.get('label', '')).strip()
        if cls not in class_to_idx:
            continue
        labels[wid] = class_to_idx[cls]
        try:
            W = int(str(row.get('image_width', '0')).strip() or 0)
            H = int(str(row.get('image_height', '0')).strip() or 0)
            if W > 0 and H > 0:
                dims[wid] = (W, H)
        except Exception:
            pass

    class_names_sorted = [c for c, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    return labels, dims, class_names_sorted


class PatchBagDataset(Dataset):
    """
    基于预切块目录构建的 bag 级数据集。

    目录结构:
      root/
        <wsi_id>/
          20x/*.png
          10x/*.png
          5x/*.png

    文件名格式（来自 Ge.py）：
      "{stem}_{scale}_{cx}_{cy}_{idx:04d}.png"

    __getitem__ 返回某个 WSI 的一个 bag（K 个 patch）：
      images: [K, 3, H, W]
      scales: [K]
      coords: [K, 2]（按 WSI 的 (W,H) 归一化到 [0,1]）
      label: int
      wsi_id: str
    """

    def __init__(self,
                 root_dir: str,
                 label_csv: str,
                 split_ids: Optional[List[str]] = None,
                 select_patches: int = 200,
                 image_size: int = 512,
                 normalize: bool = True,
                 seed: int = 42,
                 do_resize: bool = True):
        super().__init__()
        self.root = Path(root_dir)
        self.select_patches = int(select_patches)
        self.image_size = int(image_size)
        self.normalize = bool(normalize)
        self.rng = random.Random(seed)
        self.do_resize = bool(do_resize)

        self.label_map, self.wsi_dims, self.class_names = _load_label_csv(Path(label_csv))

        # 枚举 root 下存在且带有标签的 WSI 目录
        all_wsi_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        candidate_ids = [d.name for d in all_wsi_dirs if d.name in self.label_map]

        if split_ids is not None:
            split_set = set(split_ids)
            self.wsi_ids = [wid for wid in candidate_ids if wid in split_set]
        else:
            self.wsi_ids = sorted(candidate_ids)

        # 为每个 WSI 构建其 patch 索引
        self.bag_index: Dict[str, List[Path]] = {}
        for wid in self.wsi_ids:
            bag_files: List[Path] = []
            for scale in ("20x", "10x", "5x"):
                scale_dir = self.root / wid / scale
                if not scale_dir.exists():
                    continue
                bag_files.extend(sorted(scale_dir.glob('*.png')))
            self.bag_index[wid] = bag_files

        # 图像预处理变换
        tfms: List = []
        if self.do_resize:
            tfms.append(T.Resize((self.image_size, self.image_size)))
        tfms.append(T.ToTensor())
        if self.normalize:
            tfms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = T.Compose(tfms)

    def __len__(self) -> int:
        return len(self.wsi_ids)

    def _parse_scale_and_coords(self, path: Path) -> Tuple[int, Tuple[float, float]]:
        """从文件名/路径解析尺度 ID 与 (cx, cy)。若文件名缺失信息，则回退用目录名推断尺度。
        """
        m = FNAME_REGEX.search(path.name)
        if m:
            scale_str, cx, cy, _ = m.groups()
            sid = SCALE_MAP[scale_str]
            return sid, (float(cx), float(cy))
        # 回退：用父目录名推断尺度
        scale_str = path.parent.name
        sid = SCALE_MAP.get(scale_str, 0)
        return sid, (0.0, 0.0)

    def __getitem__(self, idx: int):
        wid = self.wsi_ids[idx]
        files = self.bag_index.get(wid, [])
        label = int(self.label_map[wid])

        if len(files) == 0:
            # 空 WSI（罕见）→ 返回全零占位
            images = torch.zeros(self.select_patches, 3, self.image_size, self.image_size)
            scales = torch.zeros(self.select_patches, dtype=torch.long)
            coords = torch.zeros(self.select_patches, 2)
            return images, scales, coords, label, wid

        # 先按 (x, y, scale) 排序，再进行采样
        def _key_by_xy_scale(p: Path):
            m = FNAME_REGEX.search(p.name)
            if m:
                scale_str, cx, cy, _ = m.groups()
                sx = float(cx)
                sy = float(cy)
                sid = SCALE_MAP[scale_str]
                return (sx, sy, sid)
            sid = SCALE_MAP.get(p.parent.name, 0)
            return (0.0, 0.0, sid)

        files_sorted = sorted(files, key=_key_by_xy_scale)

        # 采样 K 个文件（必要时可放回）— 在排序后采样可保持局部邻接
        if len(files_sorted) >= self.select_patches:
            chosen = self.rng.sample(files_sorted, self.select_patches)
            chosen = sorted(chosen, key=_key_by_xy_scale)
        else:
            chosen = [self.rng.choice(files_sorted) for _ in range(self.select_patches)]
            chosen = sorted(chosen, key=_key_by_xy_scale)

        images: List[torch.Tensor] = []
        scales: List[int] = []
        coords_xy: List[Tuple[float, float]] = []

        for p in chosen:
            try:
                with Image.open(p) as img_raw:
                    img = img_raw.convert('RGB')
                img_t = self.transform(img)
                sid, (cx, cy) = self._parse_scale_and_coords(p)
            except Exception:
                # 回退：读图或解析失败时返回零张量
                img_t = torch.zeros(3, self.image_size, self.image_size)
                sid, (cx, cy) = 0, (0.0, 0.0)
            images.append(img_t)
            scales.append(sid)
            coords_xy.append((cx, cy))

        images_t = torch.stack(images, dim=0)
        scales_t = torch.tensor(scales, dtype=torch.long)

        # 若已知 WSI 尺寸，则用其归一化坐标
        W, H = self.wsi_dims.get(
            wid,
            (
                max(1, int(max([c[0] for c in coords_xy] + [1]))),
                max(1, int(max([c[1] for c in coords_xy] + [1])))
            )
        )
        coords_norm = torch.tensor(
            [[c[0] / max(W, 1), c[1] / max(H, 1)] for c in coords_xy],
            dtype=torch.float32
        )

        return images_t, scales_t, coords_norm, label, wid
