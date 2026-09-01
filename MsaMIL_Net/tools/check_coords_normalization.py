#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查数据集中 patch 坐标是否已归一化到 [0, 1]。

支持两种数据来源：
- features: 预提取特征数据集（datasets/feature_dataset.PreExtractedFeatureDataset）
- patches: 直接从切块图像构建的 bag 数据集（datasets/patch_bag_dataset.PatchBagDataset）

用法示例（features 模式，默认）：
  python MsaMIL_Net/tools/check_coords_normalization.py \
    --features_dir /home/bawa/xiangmu/MsaMIL/MsaMIL_Net/data/features \
    --label_file /home/bawa/xiangmu/MsaMIL/MsaMIL_Net/data/label.csv \
    --limit 100

用法示例（patches 模式）：
  python MsaMIL_Net/tools/check_coords_normalization.py \
    --mode patches \
    --patch_root /path/to/patches_root \
    --label_file /home/bawa/xiangmu/MsaMIL/MsaMIL_Net/data/label.csv \
    --select_patches 200 --image_size 512 --limit 50
"""
import os
import sys
import argparse
from typing import Tuple, List, Dict

import torch

# 让脚本可以相对导入 MsaMIL_Net/datasets 下的模块
CUR_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(CUR_DIR)  # MsaMIL_Net
if PROJ_DIR not in sys.path:
    sys.path.insert(0, PROJ_DIR)

try:
    from datasets.feature_dataset import PreExtractedFeatureDataset
except Exception as e:
    PreExtractedFeatureDataset = None  # 延迟报错

try:
    from datasets.patch_bag_dataset import PatchBagDataset
except Exception as e:
    PatchBagDataset = None


def _update_stats(coords: torch.Tensor,
                  global_min: torch.Tensor,
                  global_max: torch.Tensor,
                  out_of_range_counter: Dict[str, int],
                  eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    更新全局统计信息：
    - global_min/global_max: 全局坐标最小/最大
    - out_of_range_counter: 记录超出 [0,1] 的数量（左越界/右越界）
    """
    if coords.numel() == 0:
        return global_min, global_max, out_of_range_counter

    cmin = coords.amin(dim=0)
    cmax = coords.amax(dim=0)
    global_min = torch.minimum(global_min, cmin)
    global_max = torch.maximum(global_max, cmax)

    below0 = (coords < -eps).sum().item()
    above1 = (coords > 1.0 + eps).sum().item()
    out_of_range_counter['below0'] += int(below0)
    out_of_range_counter['above1'] += int(above1)
    return global_min, global_max, out_of_range_counter


def check_features_mode(features_dir: str,
                        label_file: str,
                        limit: int = 0,
                        seed: int = 42,
                        max_patches_per_wsi: int = 8000,
                        show_example: bool = False) -> None:
    # ...existing code...
    # 主循环统计
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    for i in range(n):
        features, coords, scales, label, wsi_id = ds[i]
        # coords: [N, 2]
        if coords is None or coords.numel() == 0:
            continue
        cmin = coords.amin(dim=0)
        cmax = coords.amax(dim=0)
        # 若 cmin/cmax 是 0 维 tensor（即空 patch），直接跳过
        if cmin.dim() == 0 or cmax.dim() == 0:
            continue
        global_min, global_max, oob = _update_stats(coords, global_min, global_max, oob)
        # 记录越界样本
        if (cmin[0] < -1e-6 or cmin[1] < -1e-6 or cmax[0] > 1.0 + 1e-6 or cmax[1] > 1.0 + 1e-6):
            offenders.append((
                f"idx_{i}", float(cmin[0].item()), float(cmin[1].item()), float(cmax[0].item()), float(cmax[1].item())
            ))

    # 随机输出一个样本的 coords 内容
    if show_example and len(ds) > 0:
        import random
        idx = random.randint(0, len(ds)-1)
        features, coords, scales, label, wsi_id = ds[idx]
        print(f"\n【样本示例 idx={idx}】")
        print(f"coords shape: {coords.shape}")
        if coords.numel() > 0:
            print(f"coords 前5行: \n{coords[:5]}")
        else:
            print("coords 为空")
    if PreExtractedFeatureDataset is None:
        raise RuntimeError("无法导入 PreExtractedFeatureDataset，请确认路径和依赖是否正确。")

    ds = PreExtractedFeatureDataset(
        features_dir=features_dir,
        label_file=label_file,
        split='train',              # 只检查全量；内部会自行根据 label_file 组织
        test_size=0.0,
        val_size=0.0,
        max_patches_per_wsi=max_patches_per_wsi,
    )

    print(f"数据集(特征)大小: {len(ds)} 个 WSI")

    global_min = torch.tensor([float('inf'), float('inf')], dtype=torch.float32)
    global_max = torch.tensor([float('-inf'), float('-inf')], dtype=torch.float32)
    oob = {'below0': 0, 'above1': 0}
    offenders: List[Tuple[str, float, float, float, float]] = []  # (wid, minx, miny, maxx, maxy)

    n = len(ds) if limit <= 0 else min(limit, len(ds))
    for i in range(n):
        features, coords, scales, label, wsi_id = ds[i]
        # coords: [N, 2]
        if coords is None or coords.numel() == 0:
            continue
        cmin = coords.amin(dim=0)
        cmax = coords.amax(dim=0)
        # 若 cmin/cmax 是 0 维 tensor（即空 patch），直接跳过
        if cmin.dim() == 0 or cmax.dim() == 0:
            continue
        global_min, global_max, oob = _update_stats(coords, global_min, global_max, oob)
        # 记录越界样本
        if (cmin[0] < -1e-6 or cmin[1] < -1e-6 or cmax[0] > 1.0 + 1e-6 or cmax[1] > 1.0 + 1e-6):
            offenders.append((
                f"idx_{i}", float(cmin[0].item()), float(cmin[1].item()), float(cmax[0].item()), float(cmax[1].item())
            ))

    print("\n=== 统计结果（features 模式） ===")
    print(f"检查数量: {n} 个 WSI")
    print(f"全局最小坐标: x={global_min[0].item():.6f}, y={global_min[1].item():.6f}")
    print(f"全局最大坐标: x={global_max[0].item():.6f}, y={global_max[1].item():.6f}")
    print(f"越界计数: 小于0={oob['below0']}, 大于1={oob['above1']}")

    if offenders:
        print("\n存在越界的 WSI 示例(最多前20个):")
        for wid, minx, miny, maxx, maxy in offenders[:20]:
            print(f"  - {wid}: min=({minx:.6f},{miny:.6f}), max=({maxx:.6f},{maxy:.6f})")
    else:
        print("\n未发现越界样本，坐标看起来已归一化到 [0,1]。")


def check_patches_mode(patch_root: str,
                       label_file: str,
                       limit: int = 0,
                       select_patches: int = 200,
                       image_size: int = 512,
                       seed: int = 42) -> None:
    if PatchBagDataset is None:
        raise RuntimeError("无法导入 PatchBagDataset，请确认路径和依赖是否正确。")

    ds = PatchBagDataset(
        root_dir=patch_root,
        label_csv=label_file,
        split_ids=None,             # 默认检查所有有标签的 WSI
        select_patches=select_patches,
        image_size=image_size,
        normalize=False,            # 不影响坐标计算
        seed=seed,
        do_resize=False,
    )

    print(f"数据集(切块)大小: {len(ds)} 个 WSI")

    global_min = torch.tensor([float('inf'), float('inf')], dtype=torch.float32)
    global_max = torch.tensor([float('-inf'), float('-inf')], dtype=torch.float32)
    oob = {'below0': 0, 'above1': 0}
    offenders: List[Tuple[str, float, float, float, float]] = []

    n = len(ds) if limit <= 0 else min(limit, len(ds))
    for i in range(n):
        _, _, coords, label, wid = ds[i]
        if coords is None or coords.numel() == 0:
            continue
        cmin = coords.amin(dim=0)
        cmax = coords.amax(dim=0)
        global_min, global_max, oob = _update_stats(coords, global_min, global_max, oob)
        if (cmin[0] < -1e-6 or cmin[1] < -1e-6 or cmax[0] > 1.0 + 1e-6 or cmax[1] > 1.0 + 1e-6):
            offenders.append((
                str(wid), float(cmin[0].item()), float(cmin[1].item()), float(cmax[0].item()), float(cmax[1].item())
            ))

    print("\n=== 统计结果（patches 模式） ===")
    print(f"检查数量: {n} 个 WSI")
    print(f"全局最小坐标: x={global_min[0].item():.6f}, y={global_min[1].item():.6f}")
    print(f"全局最大坐标: x={global_max[0].item():.6f}, y={global_max[1].item():.6f}")
    print(f"越界计数: 小于0={oob['below0']}, 大于1={oob['above1']}")

    if offenders:
        print("\n存在越界的 WSI 示例(最多前20个):")
        for wid, minx, miny, maxx, maxy in offenders[:20]:
            print(f"  - {wid}: min=({minx:.6f},{miny:.6f}), max=({maxx:.6f},{maxy:.6f})")
    else:
        print("\n未发现越界样本，坐标看起来已归一化到 [0,1]。")


def main():
    parser = argparse.ArgumentParser(description='检查坐标向量是否归一化到 [0,1]')
    parser.add_argument('--mode', choices=['features', 'patches'], default='features', help='数据来源模式')

    # features 模式参数
    parser.add_argument('--features_dir', type=str,
                        default='/home/bawa/xiangmu/MsaMIL/MsaMIL_Net/data/features',
                        help='预提取特征的根目录')
    parser.add_argument('--label_file', type=str,
                        default='/home/bawa/xiangmu/MsaMIL/MsaMIL_Net/data/label.csv',
                        help='标签 CSV 路径')
    parser.add_argument('--max_patches_per_wsi', type=int, default=8000,
                        help='单个 WSI 最多加载多少 patch 特征（features 模式）')

    # patches 模式参数
    parser.add_argument('--patch_root', type=str, default='', help='切块根目录（patches 模式）')
    parser.add_argument('--select_patches', type=int, default=200, help='每个 WSI 选取多少块进行检查')
    parser.add_argument('--image_size', type=int, default=512, help='读图后 resize 尺寸（仅影响图像，不影响坐标）')

    # 通用
    parser.add_argument('--limit', type=int, default=0, help='最多检查多少个 WSI（0 表示全量）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--show_example', action='store_true', help='随机输出一个样本的 coords 内容')

    args = parser.parse_args()

    if args.mode == 'features':
        check_features_mode(
            features_dir=args.features_dir,
            label_file=args.label_file,
            limit=args.limit,
            seed=args.seed,
            max_patches_per_wsi=args.max_patches_per_wsi,
            show_example=args.show_example,
        )
    else:
        if not args.patch_root:
            raise SystemExit('请提供 --patch_root 指定切块根目录（patches 模式）。')
        check_patches_mode(
            patch_root=args.patch_root,
            label_file=args.label_file,
            limit=args.limit,
            select_patches=args.select_patches,
            image_size=args.image_size,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
