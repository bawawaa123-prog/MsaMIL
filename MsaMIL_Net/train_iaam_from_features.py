from __future__ import annotations

import os
import json
import math
import random
import csv
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import List, Dict, Tuple, Any
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# local imports
from models.IAAM import IAAM
from datasets.feature_dataset import PreExtractedFeatureDataset, collate_fn

# Optional: sklearn for metrics
try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
    _SK_OK = True
except Exception:
    _SK_OK = False


class ModelEMA:
    """简单的模型EMA（权重滑动平均），提升验证稳定性与精度。"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.ema = copy.deepcopy(model.module if isinstance(model, DDP) else model).eval()
        for p in self.ema.parameters():
            p.requires_grad = False
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        src = model.module if isinstance(model, DDP) else model
        msd = src.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

@dataclass
class TrainConfig:
    # 单尺度(20x)特征目录
    features_dir: str = './data/features_phikon_Yi_20x'
    label_file: str = './data/all_data.csv'
    # 输出目录：建议单尺度单独命名，避免覆盖多尺度结果
    save_dir: str = './results/YiYuan/features_phikon_queries_10_20x'

    d_model: int = 512
    input_dim: int = 1024
    mhe_layers: int = 1
    num_heads: int = 8
    low_rank: int = 32
    num_queries: int = 10

    # 消融：若提供，则依次用不同 num_queries 重复训练（每个值一个独立 save_dir）。
    # 例：ablation_num_queries = [6, 8, 10, 12, 14]
    ablation_num_queries: List[int] | None = None

    max_patches_per_wsi: int = 512  # 每个WSI最多使用的patch数量，防止内存爆炸
    batch_size: int = 1  # MIL: bag per batch
    epochs: int = 50
    lr: float = 1e-5
    weight_decay: float = 3e-5
    amp: bool = True
    use_focal_loss: bool = False
    label_smoothing: float = 0.1
    # 默认启用损失类别权重，关闭采样加权，避免双重矫正
    use_class_weights: bool = True
    sampler_weighted: bool = False
    # 模型与训练正则
    dropout: float = 0.01  # IAAM 内部各模块 dropout（原 0.1），适度增大以缓解过拟合
    patch_keep_ratio: float = 0.99  # 训练时对单个WSI随机保留的patch比例(0,1]，<1可作为bag级dropout
    ema_enable: bool = True
    ema_decay: float = 0.999

    # 划分：按需只保留train/val（将test_size设为0.0即可）
    test_size: float = 0.15
    val_size: float = 0.15
    # 固定划分（推荐）：由 tools/make_splits.py 生成。提供后将忽略 test_size/val_size 的随机划分。
    split_csv: str | None = "./splits/YiYuan/splits_phikon_03.csv"
    fold: int = 0
    patience: int = 20  # 早停耐心周期
    seed: int = 42
    # 训练/评估时的排序：与特征生成时的 sort-order 保持一致（xy 或 yx）
    sort_order: str = 'xy'
    # DataLoader配置
    num_workers: int = 4
    pin_memory: bool = True
    train_limit: int | None = None  # 调试模式：仅采样若干WSI 进行训练

    # 指标监控：二分类更建议用 auc 作为 early-stop / best 模型选择
    # 可选: 'acc' | 'auc'
    monitor_metric: str = 'acc'

    # 评估集是否固定子采样（当 max_patches_per_wsi 触发截断时）
    # False: val/test 每次评估都会随机抽取 patch（你当前希望的行为）
    # True: val/test 对每张 WSI 固定抽取同一批 patch（指标更稳定）
    deterministic_eval_subsample: bool = False


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, num_classes: int = 5, reduction: str = 'mean'):
        super().__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(num_classes) * float(alpha)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            raise ValueError('alpha must be float/int/list/None')
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        fl = alpha_t * ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        return fl


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, label_names: List[str] | None = None) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for idx in range(num_classes):
        name = label_names[idx] if label_names and idx < len(label_names) else str(idx)
        mask = (y_true == idx)
        support = int(mask.sum())
        correct = int(((y_pred == idx) & mask).sum())
        pred_total = int((y_pred == idx).sum())
        recall = correct / support if support > 0 else float('nan')
        precision = correct / pred_total if pred_total > 0 else float('nan')
        if math.isnan(recall) or math.isnan(precision) or (recall + precision) == 0:
            f1 = float('nan')
        else:
            f1 = 2 * recall * precision / (recall + precision)
        metrics[name] = {
            'acc': recall,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        }
    return metrics


def compute_f1_macro_weighted(per_class: Dict[str, Dict[str, float]] | None) -> Tuple[float, float]:
    """Compute macro/weighted F1 from per-class metrics.

    Returns:
        (f1_macro, f1_weighted). NaN if not computable.
    """
    if not per_class:
        return float('nan'), float('nan')
    f1_vals: List[float] = []
    f1_weighted_sum = 0.0
    weight_sum = 0.0
    for _, stats in per_class.items():
        support = stats.get('support', 0)
        f1 = stats.get('f1', float('nan'))
        try:
            support_i = int(support)
        except Exception:
            support_i = 0
        if support_i <= 0:
            continue
        if not isinstance(f1, (int, float)) or (isinstance(f1, float) and math.isnan(f1)):
            continue
        f1_f = float(f1)
        f1_vals.append(f1_f)
        f1_weighted_sum += f1_f * support_i
        weight_sum += support_i
    f1_macro = float(sum(f1_vals) / len(f1_vals)) if f1_vals else float('nan')
    f1_weighted = float(f1_weighted_sum / weight_sum) if weight_sum > 0 else float('nan')
    return f1_macro, f1_weighted


def sanitize_per_class_metrics(metrics: Dict[str, Dict[str, float]] | None) -> Dict[str, Dict[str, float | int | None]] | None:
    if not metrics:
        return None
    sanitized: Dict[str, Dict[str, float | int | None]] = {}
    for label, stats in metrics.items():
        sanitized[label] = {}
        for key, value in stats.items():
            if key == 'support':
                sanitized[label][key] = int(value)
            else:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    sanitized[label][key] = None
                else:
                    sanitized[label][key] = float(value)
    return sanitized


def format_per_class_metrics(stage: str, metrics: Dict[str, Dict[str, float]] | None) -> str:
    if not metrics:
        return ''
    def _fmt(val: float | None) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 'nan'
        return f"{val:.3f}"
    parts = []
    for label, stats in metrics.items():
        parts.append(
            f"{label}:acc={_fmt(stats.get('acc'))},prec={_fmt(stats.get('precision'))},"
            f"rec={_fmt(stats.get('recall'))},f1={_fmt(stats.get('f1'))},n={int(stats.get('support', 0))}"
        )
    return f"[{stage}][Per-class] " + " | ".join(parts)


def append_per_class_log(path: str, epoch: int, train_metrics: Dict[str, Dict[str, float]] | None, val_metrics: Dict[str, Dict[str, float]] | None):
    entry = {
        'epoch': epoch,
        'train': sanitize_per_class_metrics(train_metrics),
        'val': sanitize_per_class_metrics(val_metrics),
    }
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"[Warn] 无法写入逐类指标日志: {e}")


def append_epoch_metrics_log(path: str, epoch: int, train_loss: float, train_acc: float, train_auc: float,
                             val_loss: float, val_acc: float, val_auc: float):
    entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_auc': None if math.isnan(train_auc) else train_auc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_auc': None if math.isnan(val_auc) else val_auc,
    }
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"[Warn] 无法写入epoch指标日志: {e}")


def write_epoch_prediction_csv(records: List[Dict[str, Any]] | None, label_names: List[str], csv_path: str):
    """将当前epoch的bag级预测写入CSV，便于离线分析。"""
    if records is None:
        return
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    except Exception:
        pass

    num_classes = len(label_names)
    logit_headers = [f"logit_{label_names[i]}" for i in range(num_classes)]
    prob_headers = [f"prob_{label_names[i]}" for i in range(num_classes)]
    fieldnames = ['wsi_id', 'label_idx', 'label_name', 'pred_idx', 'pred_name'] + logit_headers + prob_headers

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            probs = rec.get('probs', []) or []
            logits = rec.get('logits', []) or []
            label_idx = int(rec.get('label_idx', -1))
            pred_idx = int(np.argmax(probs)) if len(probs) > 0 else -1
            row = {
                'wsi_id': rec.get('wsi_id', ''),
                'label_idx': label_idx,
                'label_name': label_names[label_idx] if 0 <= label_idx < num_classes else '',
                'pred_idx': pred_idx,
                'pred_name': label_names[pred_idx] if 0 <= pred_idx < num_classes else '',
            }
            for i, header in enumerate(logit_headers):
                row[header] = float(logits[i]) if i < len(logits) else ''
            for i, header in enumerate(prob_headers):
                row[header] = float(probs[i]) if i < len(probs) else ''
            writer.writerow(row)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_is_initialized() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def get_rank() -> int:
    if ddp_is_initialized():
        try:
            return dist.get_rank()
        except Exception:
            return 0
    return 0


def get_world_size() -> int:
    if ddp_is_initialized():
        try:
            return dist.get_world_size()
        except Exception:
            return 1
    return 1


def build_dataloaders(cfg: TrainConfig, *, rank: int = 0, world_size: int = 1):
    train_ds = PreExtractedFeatureDataset(
        features_dir=cfg.features_dir,
        label_file=cfg.label_file,
        split='train',
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.seed,
        max_patches_per_wsi=cfg.max_patches_per_wsi,
        deterministic_eval_subsample=cfg.deterministic_eval_subsample,
        split_csv=cfg.split_csv,
        fold=cfg.fold,
    )
    val_ds = PreExtractedFeatureDataset(
        features_dir=cfg.features_dir,
        label_file=cfg.label_file,
        split='val',
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.seed,
        max_patches_per_wsi=cfg.max_patches_per_wsi,
        deterministic_eval_subsample=cfg.deterministic_eval_subsample,
        split_csv=cfg.split_csv,
        fold=cfg.fold,
    )
    test_ds = None
    if cfg.test_size and cfg.test_size > 0.0:
        test_ds = PreExtractedFeatureDataset(
            features_dir=cfg.features_dir,
            label_file=cfg.label_file,
            split='test',
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.seed,
            max_patches_per_wsi=cfg.max_patches_per_wsi,
            deterministic_eval_subsample=cfg.deterministic_eval_subsample,
            split_csv=cfg.split_csv,
            fold=cfg.fold,
        )

    # 可选：调试模式下仅使用部分样本，加速冒烟测试
    if cfg.train_limit is not None and cfg.train_limit > 0 and cfg.train_limit < len(train_ds.samples):
        rng = random.Random(cfg.seed)
        keep_indices = sorted(rng.sample(range(len(train_ds.samples)), cfg.train_limit))
        train_ds.samples = [train_ds.samples[i] for i in keep_indices]
        print(f"[Debug] Train limit enabled: using {len(train_ds.samples)} WSIs for training")

    pin_mem = cfg.pin_memory and torch.cuda.is_available()
    loader_common = dict(num_workers=cfg.num_workers, pin_memory=pin_mem, persistent_workers=cfg.num_workers > 0)

    # Weighted sampler by class frequency (for batch_size=1 MIL)
    if cfg.sampler_weighted:
        # compute weights inversely proportional to class counts
        label_counts = {name: 0 for name in train_ds.label_names}
        for s in train_ds.samples:
            label_counts[s['label']] += 1
        total = len(train_ds.samples)
        class_weights: Dict[str, float] = {}
        for k, v in label_counts.items():
            class_weights[k] = total / max(1, v)
        sample_weights: List[float] = [class_weights[s['label']] for s in train_ds.samples]
        # 为多卡，将每个rank的采样步数缩减为总样本/卡数，避免重复看两遍数据
        # 同时通过不同的随机种子让各rank抽样序列不同
        per_rank_num = math.ceil(len(sample_weights) / max(1, world_size))
        g = torch.Generator()
        g.manual_seed(cfg.seed + rank)
        sampler = WeightedRandomSampler(sample_weights, num_samples=per_rank_num, replacement=True, generator=g)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate_fn, drop_last=False, **loader_common)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False, **loader_common)

    # 验证集不 shuffle，保证评估可复现
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, **loader_common)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, **loader_common)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(cfg: TrainConfig, num_classes: int, device: torch.device) -> IAAM:
    iaam = IAAM(
        d_model=cfg.d_model,
        input_dim=cfg.input_dim,
        mhe_layers=cfg.mhe_layers,
        num_heads=cfg.num_heads,
        low_rank=cfg.low_rank,
        num_queries=cfg.num_queries,
        num_classes=num_classes,
        dropout=cfg.dropout,
        ).to(device)
    # 将排序方式传递给模型（与特征生成的 sort-order 对齐）
    if hasattr(cfg, 'sort_order'):
        iaam.sort_order = cfg.sort_order
    return iaam


def _count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    """Return total params, trainable params and number of param tensors."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_tensors = len(list(model.parameters()))
    return total, trainable, num_tensors


def print_run_debug_info(cfg: TrainConfig, model: nn.Module, device: torch.device,
                         train_ds: PreExtractedFeatureDataset, val_ds: PreExtractedFeatureDataset,
                         train_loader: DataLoader, val_loader: DataLoader, is_main: bool):
    """Print comprehensive debug information about run configuration, dataset and model.

    Only printed by rank0 / main process to avoid noisy multi-process output.
    """
    if not is_main:
        return

    print('\n' + '=' * 60)
    print('Run debug info:')
    # Config summary
    try:
        cfg_dict = asdict(cfg)
    except Exception:
        cfg_dict = dict(cfg.__dict__)
    print('\n[Config]')
    print(json.dumps(cfg_dict, ensure_ascii=False, indent=2))

    # Device
    print('\n[Device]')
    print(f"device={device}, cuda_available={torch.cuda.is_available()}")

    # Dataset and loader
    print('\n[Dataset]')
    print(f"train_samples={len(train_ds)}, val_samples={len(val_ds)}")
    try:
        dist_info = train_ds.get_label_distribution()
        print('label_distribution_train=', dist_info)
    except Exception:
        pass
    print(f"train_loader_len={len(train_loader)}, val_loader_len={len(val_loader)}")

    # Sample shapes (first few) -- helpful to debug N of patches
    try:
        s = train_ds[0]
        print('\n[Sample example (train_ds[0])]')
        print(f"features.shape={s[0].shape}, coords.shape={s[1].shape}, scales.shape={s[2].shape}, label={s[3]}")
    except Exception as e:
        print('[Warn] unable to sample train_ds[0]', e)

    try:
        s2 = val_ds[0]
        print('\n[Sample example (val_ds[0])]')
        print(f"features.shape={s2[0].shape}, coords.shape={s2[1].shape}, scales.shape={s2[2].shape}, label={s2[3]}")
    except Exception:
        pass

    # Model parameter counts
    print('\n[Model]')
    total, trainable, num_tensors = _count_parameters(model if not isinstance(model, DDP) else model.module)
    print(f"num_param_tensors={num_tensors}, total_params={total:,}, trainable_params={trainable:,}")
    print('=' * 60 + '\n')


def build_criterion(cfg: TrainConfig, num_classes: int, class_weights: torch.Tensor = None):
    if cfg.use_focal_loss:
        alpha = class_weights.tolist() if class_weights is not None else None
        return FocalLoss(alpha=alpha, gamma=2.0, num_classes=num_classes, reduction='mean')
    # CrossEntropy with optional class weights and label smoothing
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)


def evaluate(model: IAAM, loader: DataLoader, device: torch.device, amp: bool, num_classes: int, label_names: List[str] | None = None, *, is_main: bool = True):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    ce = nn.CrossEntropyLoss(reduction='none')

    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []
    all_wsi_ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='[Val]'):
            if batch is None:
                continue
            (features, coords, scales, labels, wsi_id) = batch
            features = features.to(device)
            coords = coords.to(device)
            scales = scales.to(device)
            labels = labels.to(device)

            if amp and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits, _ = model(features, scales, coords)
            else:
                logits, _ = model(features, scales, coords)

            loss = ce(logits.unsqueeze(0), labels.unsqueeze(0)).mean()
            total_loss += loss.item()
            total += 1
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == labels).sum().item()

            probs = F.softmax(logits, dim=-1).detach().cpu()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu())
            all_logits.append(logits.detach().cpu())
            all_wsi_ids.append(wsi_id)

    # 分布式聚合 loss/acc
    loss_t = torch.tensor([total_loss], dtype=torch.float64, device=device)
    tot_t = torch.tensor([total], dtype=torch.float64, device=device)
    cor_t = torch.tensor([correct], dtype=torch.float64, device=device)
    if ddp_is_initialized():
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(tot_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor_t, op=dist.ReduceOp.SUM)
    avg_loss = (loss_t.item() / max(1.0, tot_t.item())) if tot_t.item() > 0 else 0.0
    acc = (cor_t.item() / max(1.0, tot_t.item())) if tot_t.item() > 0 else 0.0

    # AUC：在rank0聚合后计算再广播（提供更详细的多分类逐类AUC诊断）
    auc = float('nan')
    # 收集所有rank的labels/probs便于指标统计
    def _gather_predictions():
        if ddp_is_initialized():
            try:
                obj_local = {
                    'labels': [t.numpy() for t in all_labels],
                    'probs': [t.numpy() for t in all_probs],
                    'logits': [t.numpy() for t in all_logits],
                    'wsi_ids': all_wsi_ids,
                }
                gathered = [None for _ in range(get_world_size())]
                dist.all_gather_object(gathered, obj_local)
                if is_main:
                    import numpy as _np
                    y_true_list, y_prob_list, y_logit_list = [], [], []
                    merged_wsi_ids: List[str] = []
                    for obj in gathered:
                        y_true_list.extend(obj['labels'])
                        y_prob_list.extend(obj['probs'])
                        y_logit_list.extend(obj['logits'])
                        merged_wsi_ids.extend(obj['wsi_ids'])
                    if len(y_true_list) > 0:
                        y_true = _np.stack(y_true_list, axis=0)
                        y_prob = _np.stack(y_prob_list, axis=0)
                        y_logits = _np.stack(y_logit_list, axis=0)
                        return y_true, y_prob, y_logits, merged_wsi_ids
                    return None, None, None, merged_wsi_ids
            except Exception as e:
                if is_main:
                    print(f"[Warn] 预测结果聚合失败: {e}")
                return None, None, None, None
        else:
            if len(all_labels) > 0:
                import numpy as _np
                y_true = torch.stack(all_labels, dim=0).numpy()
                y_prob = torch.stack(all_probs, dim=0).numpy()
                y_logits = torch.stack(all_logits, dim=0).numpy()
                return y_true, y_prob, y_logits, list(all_wsi_ids)
            return None, None, None, []

    y_true_np, y_prob_np, y_logits_np, gathered_wsi_ids = _gather_predictions()

    per_class_metrics = None
    if y_true_np is not None and y_prob_np is not None:
        y_pred_np = y_prob_np.argmax(axis=1)
        per_class_metrics = compute_per_class_metrics(y_true_np, y_pred_np, num_classes, label_names)

    f1_macro, f1_weighted = compute_f1_macro_weighted(per_class_metrics)

    auc = float('nan')
    if _SK_OK and y_true_np is not None and y_prob_np is not None:
        import numpy as _np

        def _compute_multiclass_auc_and_report(y_true_np_local, y_prob_np_local) -> float:
            present = _np.unique(y_true_np_local)
            if present.size < 2:
                if is_main:
                    print('[AUC][Val] 本轮验证仅出现单一类别，AUC 无法定义（NaN）。present=', present.tolist())
                return float('nan')

            if y_prob_np_local.shape[1] == 2:
                try:
                    auc_bin = roc_auc_score(y_true_np_local, y_prob_np_local[:, 1])
                    if is_main:
                        print(f"[AUC][Val] Binary AUC={auc_bin:.4f}")
                    return float(auc_bin)
                except Exception as e:
                    if is_main:
                        print(f"[AUC][Val] 二分类AUC计算异常（NaN）。err={e.__class__.__name__}: {e}")
                    return float('nan')

            per_class_scores = {}
            auc_values = []
            for c in present.astype(int).tolist():
                y_true_bin = (y_true_np_local == c).astype(int)
                pos = y_true_bin.sum()
                neg = y_true_bin.size - pos
                if pos == 0 or neg == 0:
                    per_class_scores[c] = float('nan')
                    continue
                try:
                    auc_c = roc_auc_score(y_true_bin, y_prob_np_local[:, c])
                    per_class_scores[c] = float(auc_c)
                    if not math.isnan(per_class_scores[c]):
                        auc_values.append(per_class_scores[c])
                except Exception:
                    per_class_scores[c] = float('nan')
            macro = float(_np.mean(auc_values)) if len(auc_values) > 0 else float('nan')

            if is_main:
                parts = []
                present_list = present.astype(int).tolist()
                for c in present_list:
                    name = f"{c}"
                    if label_names is not None and 0 <= c < len(label_names):
                        name = label_names[c]
                    val = per_class_scores.get(c, float('nan'))
                    parts.append(f"{name}:{'nan' if math.isnan(val) else f'{val:.3f}'}")
                print("[AUC][Val] per-class AUC → " + ", ".join(parts) + f" | macro={ 'nan' if math.isnan(macro) else f'{macro:.4f}' }")
            return macro

        auc = _compute_multiclass_auc_and_report(y_true_np, y_prob_np)
    elif _SK_OK and y_true_np is None and is_main:
        print('[Warn] 验证阶段未收集到完整的预测结果，AUC 将为 NaN。')
    elif not _SK_OK and is_main:
        print('[Warn] 未安装 scikit-learn，无法计算AUC（显示为 NaN）。建议安装：pip install scikit-learn')

    if ddp_is_initialized():
        auc_tensor = torch.tensor([float('nan') if math.isnan(auc) else auc], dtype=torch.float64, device=device)
        dist.broadcast(auc_tensor, src=0)
        auc = float(auc_tensor.item())

    epoch_records: List[Dict[str, Any]] | None = None
    if is_main and y_true_np is not None and y_prob_np is not None and y_logits_np is not None and gathered_wsi_ids is not None:
        if len(gathered_wsi_ids) == y_true_np.shape[0]:
            epoch_records = []
            for idx, wsi_id in enumerate(gathered_wsi_ids):
                epoch_records.append({
                    'wsi_id': wsi_id,
                    'label_idx': int(y_true_np[idx]),
                    'logits': y_logits_np[idx].tolist(),
                    'probs': y_prob_np[idx].tolist(),
                })

    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc,
        'f1': f1_macro,
        'f1_weighted': f1_weighted,
        'per_class': per_class_metrics,
        'labels': torch.stack(all_labels, dim=0) if len(all_labels) > 0 else torch.empty(0),
        'probs': torch.stack(all_probs, dim=0) if len(all_probs) > 0 else torch.empty(0),
        'epoch_records': epoch_records,
    }


def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    # 为当前训练创建带时间戳的输出目录（形如 checkpoints/iaam_from_features_YYYYmmdd_HHMMSS）
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save = cfg.save_dir.rstrip('/\\') or cfg.save_dir
    run_dir = f"{base_save}_{run_ts}"
    os.makedirs(run_dir, exist_ok=True)
    per_wsi_pred_dir = os.path.join(run_dir, 'per_wsi_predictions')
    os.makedirs(per_wsi_pred_dir, exist_ok=True)

    per_class_log_path = os.path.join(run_dir, 'per_class_metrics.log')
    with open(per_class_log_path, 'w', encoding='utf-8') as f:
        f.write('# per-class metrics log (JSON lines)\n')
    epoch_metrics_log_path = os.path.join(run_dir, 'epoch_metrics.log')
    with open(epoch_metrics_log_path, 'w', encoding='utf-8') as f:
        f.write('# aggregated epoch metrics log (JSON lines)\n')
    best_checkpoint_path = os.path.join(run_dir, f'best_model_{run_ts}.pth')

    # 分布式初始化（env变量）
    distributed = False
    local_rank_env = os.environ.get('LOCAL_RANK', None)
    try:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)
            distributed = True
            if torch.cuda.is_available() and local_rank_env is not None:
                torch.cuda.set_device(int(local_rank_env))
    except Exception:
        distributed = False

    rank = get_rank()
    world_size = get_world_size()
    is_main = (rank == 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Data
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(cfg, rank=rank, world_size=world_size)
    num_classes = train_ds.num_classes

    # 若同时启用“加权采样 + 类别权重”，会产生叠加矫正，实践中更偏向采样层面平衡；此处自动关闭类别权重并提示
    if cfg.sampler_weighted and cfg.use_class_weights and is_main:
        print("[Info] 已启用加权采样(sampler_weighted=True)，为避免与类别权重叠加造成过度矫正，自动关闭 use_class_weights。")
        cfg.use_class_weights = False

    # Class weights (per-WSI frequencies)
    class_weights = None
    if cfg.use_class_weights:
        class_weights = train_ds.get_class_weights().to(device)

    # Model
    model = build_model(cfg, num_classes, device)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None, output_device=device.index if device.type == 'cuda' else None, find_unused_parameters=False)

    # EMA
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_enable else None

    # Print debug info (only on main rank)
    print_run_debug_info(cfg, model, device, train_ds, val_ds, train_loader, val_loader, is_main)

    # 将 debug 信息 dump 到 run_dir/debug_info.json(仅 rank0)
    if is_main:
        try:
            total, trainable, _ = _count_parameters(model if not isinstance(model, DDP) else model.module)
            debug_info = {
                'config': asdict(cfg),
                'device': str(device),
                'train_samples': len(train_ds),
                'val_samples': len(val_ds),
                'num_classes': num_classes,
                'total_params': int(total),
                'trainable_params': int(trainable),
                'label_distribution': train_ds.get_label_distribution()
            }
            with open(os.path.join(run_dir, 'debug_info.json'), 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('[Warn] 写入 debug_info.json 失败:', e)

    # Criterion / Optim / Scheduler
    criterion = build_criterion(cfg, num_classes, class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == 'cuda'))

    best_acc = 0.0
    best_metric_name = getattr(cfg, 'monitor_metric', 'acc')
    heldout_test_summary: Dict[str, Any] | None = None
    best_state = None
    epochs_no_improve = 0

    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'train_per_class': [], 'val_per_class': []
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        # 收集训练期预测用于AUC（按本epoch采样到的bag）
        tr_all_probs: List[torch.Tensor] = []
        tr_all_labels: List[torch.Tensor] = []
        tr_all_logits: List[torch.Tensor] = []
        tr_all_wsi_ids: List[str] = []

        pbar = tqdm(train_loader, desc=f'[Train] Epoch {epoch}/{cfg.epochs}', disable=not is_main)
        for batch in pbar:
            if batch is None:
                continue
            (features, coords, scales, labels, wsi_id) = batch
            features = features.to(device)
            coords = coords.to(device)
            scales = scales.to(device)
            labels = labels.to(device)

            # bag级随机丢弃patch，缓解过拟合（仅训练阶段）
            if 0.0 < cfg.patch_keep_ratio < 1.0 and features.size(0) > 1:
                keep = max(1, int(features.size(0) * cfg.patch_keep_ratio))
                idx = torch.randperm(features.size(0), device=features.device)[:keep]
                features = features.index_select(0, idx)
                coords = coords.index_select(0, idx)
                scales = scales.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.amp.autocast('cuda'):
                    logits, _ = model(features, scales, coords)
                    loss = criterion(logits.unsqueeze(0), labels.unsqueeze(0))
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
            else:
                logits, _ = model(features, scales, coords)
                loss = criterion(logits.unsqueeze(0), labels.unsqueeze(0))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            total_loss += loss.item()
            total += 1
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == labels).sum().item()

            if is_main:
                pbar.set_postfix(loss=f"{(total_loss/max(1,total)):.4f}", acc=f"{(correct/max(1,total)):.4f}")

            # 收集训练概率和标签
            probs = F.softmax(logits.detach(), dim=-1).detach().cpu()
            tr_all_probs.append(probs)
            tr_all_labels.append(labels.detach().cpu())
            tr_all_logits.append(logits.detach().cpu())
            tr_all_wsi_ids.append(wsi_id)

        # 分布式聚合训练loss/acc
        loss_t = torch.tensor([total_loss], dtype=torch.float64, device=device)
        tot_t = torch.tensor([total], dtype=torch.float64, device=device)
        cor_t = torch.tensor([correct], dtype=torch.float64, device=device)
        if ddp_is_initialized():
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(tot_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(cor_t, op=dist.ReduceOp.SUM)
        train_loss = (loss_t.item() / max(1.0, tot_t.item())) if tot_t.item() > 0 else 0.0
        train_acc = (cor_t.item() / max(1.0, tot_t.item())) if tot_t.item() > 0 else 0.0

        # 训练集AUC与逐类指标（基于本epoch采样的bag）
        train_auc = float('nan')
        train_per_class = None

        def _gather_train_predictions():
            if ddp_is_initialized():
                try:
                    obj_local = {
                        'labels': [t.numpy() for t in tr_all_labels],
                        'probs': [t.numpy() for t in tr_all_probs],
                        'logits': [t.numpy() for t in tr_all_logits],
                        'wsi_ids': tr_all_wsi_ids,
                    }
                    gathered = [None for _ in range(get_world_size())]
                    dist.all_gather_object(gathered, obj_local)
                    if is_main:
                        import numpy as _np
                        y_true_list, y_prob_list, y_logit_list = [], [], []
                        merged_wsi_ids: List[str] = []
                        for obj in gathered:
                            y_true_list.extend(obj['labels'])
                            y_prob_list.extend(obj['probs'])
                            y_logit_list.extend(obj['logits'])
                            merged_wsi_ids.extend(obj['wsi_ids'])
                        if len(y_true_list) > 0:
                            y_true = _np.stack(y_true_list, axis=0)
                            y_prob = _np.stack(y_prob_list, axis=0)
                            y_logits = _np.stack(y_logit_list, axis=0)
                            return y_true, y_prob, y_logits, merged_wsi_ids
                    return None, None, None, None
                except Exception as e:
                    if is_main:
                        print(f"[Warn] 训练阶段预测结果聚合失败: {e}")
                    return None, None, None, None
            else:
                if len(tr_all_labels) > 0:
                    y_true = torch.stack(tr_all_labels, dim=0).numpy()
                    y_prob = torch.stack(tr_all_probs, dim=0).numpy()
                    y_logits = torch.stack(tr_all_logits, dim=0).numpy()
                    return y_true, y_prob, y_logits, list(tr_all_wsi_ids)
                return None, None, None, []

        y_true_train, y_prob_train, y_logits_train, train_wsi_ids = _gather_train_predictions()

        train_epoch_records: List[Dict[str, Any]] | None = None
        if is_main and y_true_train is not None and y_logits_train is not None and train_wsi_ids:
            if len(train_wsi_ids) == y_true_train.shape[0]:
                train_epoch_records = []
                for idx, wsi_id in enumerate(train_wsi_ids):
                    train_epoch_records.append({
                        'wsi_id': wsi_id,
                        'label_idx': int(y_true_train[idx]),
                        'logits': y_logits_train[idx].tolist(),
                        'probs': y_prob_train[idx].tolist(),
                    })

        if y_true_train is not None and y_prob_train is not None:
            y_pred_train = y_prob_train.argmax(axis=1)
            train_per_class = compute_per_class_metrics(y_true_train, y_pred_train, num_classes, train_ds.label_names)

            if _SK_OK:
                def _compute_multiclass_auc_train(y_true_np, y_prob_np) -> float:
                    import numpy as _np
                    present = _np.unique(y_true_np)
                    if present.size < 2:
                        if is_main:
                            print('[AUC][Train] 本轮训练仅出现单一类别，AUC 无法定义（NaN）。present=', present.tolist())
                        return float('nan')
                    if y_prob_np.shape[1] == 2:
                        try:
                            return float(roc_auc_score(y_true_np, y_prob_np[:, 1]))
                        except Exception as e:
                            if is_main:
                                print(f"[AUC][Train] 二分类AUC计算异常（NaN）。err={e.__class__.__name__}: {e}")
                            return float('nan')
                    vals = []
                    for c in present.astype(int).tolist():
                        y_true_bin = (y_true_np == c).astype(int)
                        pos = y_true_bin.sum()
                        neg = y_true_bin.size - pos
                        if pos == 0 or neg == 0:
                            continue
                        try:
                            vals.append(float(roc_auc_score(y_true_bin, y_prob_np[:, c])))
                        except Exception:
                            pass
                    return float(_np.mean(vals)) if len(vals) > 0 else float('nan')

                train_auc = _compute_multiclass_auc_train(y_true_train, y_prob_train)
        else:
            if is_main and epoch == 1:
                print('[Warn] 训练阶段未收集到完整的预测结果，AUC 将显示为 NaN。')

        if not _SK_OK and is_main and epoch == 1:
            print('[Warn] 未安装 scikit-learn，无法计算训练AUC（显示为 NaN）。建议安装：pip install scikit-learn')

        if ddp_is_initialized():
            auc_tensor = torch.tensor([float('nan') if math.isnan(train_auc) else train_auc], dtype=torch.float64, device=device)
            dist.broadcast(auc_tensor, src=0)
            train_auc = float(auc_tensor.item())

        # Validate（优先使用EMA模型评估）
        eval_model = ema.ema if ema is not None else (model.module if isinstance(model, DDP) else model)
        val_metrics = evaluate(eval_model, val_loader, device, cfg.amp, num_classes, label_names=train_ds.label_names, is_main=is_main)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['acc']
        val_auc = val_metrics['auc']
        val_per_class = val_metrics.get('per_class')
        val_epoch_records = val_metrics.get('epoch_records') if is_main else None

        best_metric_name = getattr(cfg, 'monitor_metric', 'acc')
        monitor = val_acc
        if best_metric_name == 'auc' and not math.isnan(val_auc):
            monitor = val_auc
        scheduler.step(monitor)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['train_auc'].append(float('nan') if math.isnan(train_auc) else train_auc)
        history['train_per_class'].append(sanitize_per_class_metrics(train_per_class))
        history['val_per_class'].append(sanitize_per_class_metrics(val_per_class))

        if is_main:
            print(
                f"Epoch {epoch:03d} | "
                f"Train loss {train_loss:.4f} acc {train_acc:.4f} auc {train_auc if not math.isnan(train_auc) else 'nan'} | "
                f"Val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc if not math.isnan(val_auc) else 'nan'}"
            )
            if train_per_class:
                print(format_per_class_metrics('Train', train_per_class))
            if val_per_class:
                print(format_per_class_metrics('Val', val_per_class))
            append_per_class_log(per_class_log_path, epoch, train_per_class, val_per_class)
            append_epoch_metrics_log(
                epoch_metrics_log_path,
                epoch,
                train_loss,
                train_acc,
                float('nan') if math.isnan(train_auc) else train_auc,
                val_loss,
                val_acc,
                float('nan') if math.isnan(val_auc) else val_auc,
            )
            write_epoch_prediction_csv(
                train_epoch_records,
                train_ds.label_names,
                os.path.join(per_wsi_pred_dir, f'epoch_{epoch:03d}_train_predictions.csv')
            )
            write_epoch_prediction_csv(
                val_epoch_records,
                train_ds.label_names,
                os.path.join(per_wsi_pred_dir, f'epoch_{epoch:03d}_val_predictions.csv')
            )

        # Early stopping & best checkpoint by monitored metric（分布式一致广播）
        improved = monitor > best_acc + 1e-6
        if improved and is_main:
            best_acc = monitor
            best_state = {
                'epoch': epoch,
                'model_state_dict': eval_model.state_dict(),
                'best_acc': best_acc,
                'best_metric': best_acc,
                'best_metric_name': best_metric_name,
                'config': asdict(cfg),
                'label_names': train_ds.label_names,
                'metrics': {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_auc': None if math.isnan(train_auc) else train_auc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_auc': None if math.isnan(val_auc) else val_auc,
                }
            }
            torch.save(best_state, best_checkpoint_path)

            # Save classification report on best epoch
            if _SK_OK and val_metrics['labels'].numel() > 0:
                y_true = val_metrics['labels'].numpy()
                probs = val_metrics['probs'].numpy()
                y_pred = probs.argmax(axis=1)
                report = classification_report(y_true, y_pred, target_names=train_ds.label_names, digits=4)
                cm = confusion_matrix(y_true, y_pred)
                with open(os.path.join(run_dir, f'best_classification_report_{run_ts}.txt'), 'w') as f:
                    f.write(f"Best Model Classification Report (Epoch {epoch}):\n")
                    f.write(f"Validation Accuracy: {best_acc:.4f}\n\n")
                    f.write(report)
                    f.write("\n")
                    f.write("Confusion Matrix:\n")
                    f.write(str(cm))
                    f.write("\n")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 广播早停标志，确保所有rank同时退出
        early_stop = (epochs_no_improve >= cfg.patience)
        if ddp_is_initialized():
            flag = torch.tensor([1 if (early_stop and is_main) else 0], device=device, dtype=torch.int32)
            # 仅以rank0的判断为准
            dist.broadcast(flag, src=0)
            early_stop = bool(flag.item())
        if early_stop:
            if is_main:
                print(f"Early stopping triggered at epoch {epoch}")
            break

        # Save history after each epoch (JSON + CSV) — 仅 rank0 写入，NaN 明确写为 'nan'
        if is_main:
            try:
                with open(os.path.join(run_dir, f'training_history_{run_ts}.json'), 'w') as f:
                    json.dump(history, f)
                # 追加写入CSV，便于快速对比与可视化
                csv_path = os.path.join(run_dir, f'metrics_per_epoch_{run_ts}.csv')
                write_header = (not os.path.exists(csv_path)) or (epoch == 1)
                with open(csv_path, 'a', newline='') as cf:
                    writer = csv.writer(cf)
                    if write_header:
                        writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_auc', 'val_loss', 'val_acc', 'val_auc'])
                    writer.writerow([
                        epoch,
                        round(train_loss, 6),
                        round(train_acc, 6),
                        ('nan' if math.isnan(train_auc) else round(train_auc, 6)),
                        round(val_loss, 6),
                        round(val_acc, 6),
                        ('nan' if math.isnan(val_auc) else round(val_auc, 6)),
                    ])
            except Exception as e:
                print(f'[Warn] 写入度量历史失败: {e}')

    if cfg.test_size and cfg.test_size > 0.0 and test_loader is not None:
        # 在所有 rank 上加载最佳权重以保持 evaluate 中的分布式同步
        best_weights = None
        if os.path.exists(best_checkpoint_path):
            try:
                try:
                    checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=True)
                except TypeError:
                    checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
                best_weights = checkpoint.get('model_state_dict', checkpoint)
                if is_main:
                    print(f"[Test] Loaded best checkpoint from {best_checkpoint_path}")
            except Exception as exc:
                if is_main:
                    print(f"[Test][Warn] 无法加载最佳checkpoint({best_checkpoint_path}): {exc}")
        if best_weights is None:
            source_model = ema.ema if ema is not None else (model.module if isinstance(model, DDP) else model)
            best_weights = source_model.state_dict()
            if is_main:
                print("[Test] 未找到（或无法读取）最佳checkpoint，使用当前模型权重进行测试评估。")

        test_model = build_model(cfg, num_classes, device)
        test_model.load_state_dict(best_weights, strict=False)
        test_model.to(device)
        test_metrics = evaluate(test_model, test_loader, device, cfg.amp, num_classes, label_names=train_ds.label_names, is_main=is_main)
        if is_main:
            test_summary = {
                'loss': test_metrics['loss'],
                'acc': test_metrics['acc'],
                'auc': test_metrics['auc'],
                'f1': test_metrics.get('f1', float('nan')),
                'f1_weighted': test_metrics.get('f1_weighted', float('nan')),
                'per_class': sanitize_per_class_metrics(test_metrics.get('per_class')),
            }
            heldout_test_summary = dict(test_summary)
            print("[Test] Held-out split → loss={loss:.4f} acc={acc:.4f} auc={auc}".format(**test_summary))
            with open(os.path.join(run_dir, f'test_metrics_{run_ts}.json'), 'w', encoding='utf-8') as f:
                json.dump(test_summary, f, ensure_ascii=False, indent=2)

    if is_main:
        print(f"Training finished. Best Val {best_metric_name} = {best_acc:.4f}")

    # 返回摘要（非DDP或rank0更有意义）
    summary = {
        'best_val_acc': float(best_acc),
        'fold': int(getattr(cfg, 'fold', 0)),
        'run_dir': str(run_dir),
    }
    if heldout_test_summary is not None:
        summary.update({
            'test_loss': heldout_test_summary.get('loss'),
            'test_acc': heldout_test_summary.get('acc'),
            'test_auc': heldout_test_summary.get('auc'),
            'test_f1': heldout_test_summary.get('f1'),
            'test_f1_weighted': heldout_test_summary.get('f1_weighted'),
        })
    return summary


def _discover_folds_from_split_csv(path: str) -> List[int]:
    import pandas as pd
    df = pd.read_csv(path)
    if 'fold' not in df.columns:
        return [0]
    folds = sorted({int(x) for x in df['fold'].fillna(0).astype(int).tolist()})
    return folds


def run_all_folds(cfg: TrainConfig):
    """当 cfg.fold < 0 时：遍历 split_csv 里的全部 fold，逐折训练并汇总 mean/std。"""
    if not cfg.split_csv:
        raise ValueError('run_all_folds requires cfg.split_csv')

    folds = _discover_folds_from_split_csv(cfg.split_csv)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = cfg.save_dir.rstrip('/\\') + f"_kfold_{ts}"
    os.makedirs(base_dir, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []
    for fold in folds:
        cfg_i = copy.deepcopy(cfg)
        cfg_i.fold = int(fold)
        # 每折单独目录（train()内部还会加时间戳）
        cfg_i.save_dir = os.path.join(base_dir, f"fold{fold}")
        print(f"\n{'='*24} [KFold] fold={fold} {'='*24}\n")
        summary = train(cfg_i)
        all_summaries.append(summary)

    def _mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return float('nan'), float('nan')
        mean_v = float(sum(values) / len(values))
        if len(values) < 2:
            return mean_v, float('nan')
        var = sum((v - mean_v) ** 2 for v in values) / (len(values) - 1)
        return mean_v, float(math.sqrt(var))

    def _collect_float(key: str) -> List[float]:
        out_vals: List[float] = []
        for s in all_summaries:
            v = s.get(key)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                out_vals.append(float(v))
        return out_vals

    stats = {}
    for key in ('best_val_acc', 'test_loss', 'test_acc', 'test_auc', 'test_f1', 'test_f1_weighted'):
        mean_v, std_v = _mean_std(_collect_float(key))
        stats[f'{key}_mean'] = mean_v
        stats[f'{key}_std'] = std_v

    # CSV 汇总（每折 + mean/std）
    csv_path = os.path.join(base_dir, 'kfold_summary.csv')
    fieldnames = [
        'fold', 'run_dir',
        'best_val_acc',
        'test_loss', 'test_acc', 'test_auc', 'test_f1', 'test_f1_weighted',
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_summaries:
            row = {k: s.get(k) for k in fieldnames}
            writer.writerow(row)
        writer.writerow({
            'fold': 'mean',
            'best_val_acc': stats['best_val_acc_mean'],
            'test_loss': stats['test_loss_mean'],
            'test_acc': stats['test_acc_mean'],
            'test_auc': stats['test_auc_mean'],
            'test_f1': stats['test_f1_mean'],
            'test_f1_weighted': stats['test_f1_weighted_mean'],
        })
        writer.writerow({
            'fold': 'std',
            'best_val_acc': stats['best_val_acc_std'],
            'test_loss': stats['test_loss_std'],
            'test_acc': stats['test_acc_std'],
            'test_auc': stats['test_auc_std'],
            'test_f1': stats['test_f1_std'],
            'test_f1_weighted': stats['test_f1_weighted_std'],
        })

    # 同时保留 JSON（向后兼容）
    out = {
        'split_csv': cfg.split_csv,
        'folds': folds,
        'per_fold': all_summaries,
        'stats': stats,
    }
    with open(os.path.join(base_dir, 'kfold_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[KFold] Summary saved to {csv_path}")


def main():
    # Update TrainConfig() inline if you need to override默认值；脚本将完全依赖该配置。
    cfg = TrainConfig()

    # Ablation: sweep num_queries
    sweep = getattr(cfg, 'ablation_num_queries', None)
    if sweep:
        # 去重+排序，避免重复跑
        uniq = sorted({int(x) for x in sweep if int(x) > 0})
        if not uniq:
            raise ValueError('ablation_num_queries must contain positive integers')
        base_dir = cfg.save_dir.rstrip('/\\')
        for q in uniq:
            cfg_i = copy.deepcopy(cfg)
            cfg_i.num_queries = int(q)
            cfg_i.save_dir = f"{base_dir}_q{q}"
            print(f"\n{'='*24} [Ablation] num_queries={q} {'='*24}\n")
            if getattr(cfg_i, 'fold', 0) is not None and int(cfg_i.fold) < 0:
                run_all_folds(cfg_i)
            else:
                train(cfg_i)
        return

    if getattr(cfg, 'fold', 0) is not None and int(cfg.fold) < 0:
        run_all_folds(cfg)
    else:
        train(cfg)


if __name__ == '__main__':
    main()
 