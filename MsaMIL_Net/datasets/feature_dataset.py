import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import random
import hashlib

class PreExtractedFeatureDataset(Dataset):
    """
    预提取特征数据集 - 用于IAAM模块训练
    处理.pt特征文件和.npy坐标文件
    """
    
    def __init__(self, 
                 features_dir: str,
                 label_file: str,
                 split: str = 'train',
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 max_patches_per_wsi: int = None,
                 deterministic_eval_subsample: bool = True,
                 split_csv: Optional[str] = None,
                 fold: int = 0,
                 skip_broken: bool = True):
        """
        Args:
            features_dir: 特征文件目录
            label_file: 标签CSV文件路径
            split: 数据集划分 ('train', 'val', 'test')
            test_size: 测试集比例
            val_size: 验证集比例（从训练集中分出）
            random_state: 随机种子
            max_patches_per_wsi: 每个WSI最大patch数（用于控制内存）
            split_csv: 可选的固定划分CSV（由 tools/make_splits.py 生成），提供后将完全跳过随机划分。
            fold: 使用 split_csv 时选择的折编号。
            skip_broken: 若特征文件损坏/读取失败，是否跳过该样本（返回None并由collate_fn过滤）。
        """
        self.features_dir = features_dir
        self.split = split
        self.max_patches_per_wsi = max_patches_per_wsi
        self.split_csv = split_csv
        self.fold = int(fold)
        self.skip_broken = bool(skip_broken)
        self.random_state = int(random_state)
        self.deterministic_eval_subsample = bool(deterministic_eval_subsample)
        
        # 读取标签文件
        self.df = pd.read_csv(label_file)
        if 'image_id' in self.df.columns:
            self.id_column = 'image_id'
        elif 'slide_id' in self.df.columns:
            self.id_column = 'slide_id'
        else:
            raise ValueError("label_file 必须包含 'image_id' 或 'slide_id' 列")
        
        # 创建标签映射
        self.label_names = sorted(self.df['label'].unique())
        self.label2idx = {label: idx for idx, label in enumerate(self.label_names)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.num_classes = len(self.label_names)
        
        print(f"✓ 发现 {self.num_classes} 个类别: {self.label_names}")
        print(f"✓ 标签映射: {self.label2idx}")
        
        # 数据集划分：优先使用固定 split_csv
        if self.split_csv:
            self.samples = self._split_dataset_from_csv(self.split_csv, self.fold)
        else:
            self.samples = self._split_dataset(test_size, val_size, random_state)
        
        print(f"✓ {split} 集样本数: {len(self.samples)}")
        
        # 统计当前划分的类别分布
        split_labels = [self.label2idx[sample['label']] for sample in self.samples]
        for idx, label_name in enumerate(self.label_names):
            count = split_labels.count(idx)
            print(f"   {label_name}: {count}")
    
    def _split_dataset(self, test_size: float, val_size: float, random_state: int) -> List[Dict]:
        """数据集划分（支持不划分测试集/验证集）。

        规则：
        - 若 test_size <= 0，则不划分测试集，所有样本进入 train/val 划分。
        - 若 val_size <= 0，则不划分验证集，所有非测试部分进入训练集。
        - stratify 仅在目标划分非空时启用。
        """
        # 获取有特征文件的样本
        valid_samples = []
        missing_files = []
        missing_scales = []
        
        for _, row in self.df.iterrows():
            wsi_id = str(row[self.id_column])
            feature_file = os.path.join(self.features_dir, f"{wsi_id}.pt")
            coord_file = os.path.join(self.features_dir, f"{wsi_id}_coords.npy")
            scale_file = os.path.join(self.features_dir, f"{wsi_id}_scales.npy")

            feature_exists = os.path.exists(feature_file)
            coord_exists = os.path.exists(coord_file)
            scale_exists = os.path.exists(scale_file)

            if feature_exists and coord_exists:
                if not scale_exists:
                    missing_scales.append(wsi_id)
                valid_samples.append({
                    'wsi_id': wsi_id,
                    'label': row['label'],
                    'feature_file': feature_file,
                    'coord_file': coord_file,
                    'scale_file': scale_file if scale_exists else None
                })
            else:
                missing_files.append(wsi_id)
        
        if missing_files:
            print(f"⚠️  缺少特征/坐标文件的样本数: {len(missing_files)}")
            if len(missing_files) < 10:
                print(f"   缺少的ID: {missing_files}")
        if missing_scales:
            print(f"⚠️  缺少scales文件的样本数: {len(missing_scales)} (训练时将回退为0)")
            if len(missing_scales) < 10:
                print(f"   缺少scales的ID: {missing_scales}")
        
        print(f"✓ 有效样本数: {len(valid_samples)}")
        
        # 按标签分层划分
        # 1) 训练/测试
        if test_size and test_size > 0.0:
            train_val_samples, test_samples = train_test_split(
                valid_samples,
                test_size=test_size,
                stratify=[s['label'] for s in valid_samples],
                random_state=random_state
            )
        else:
            train_val_samples = valid_samples
            test_samples = []

        # 2) 从训练中划分验证
        if val_size and val_size > 0.0:
            # 当 train_val_samples 中若某些类别样本过少，stratify 可能失败；因此加一层保护
            try:
                train_samples, val_samples = train_test_split(
                    train_val_samples,
                    test_size=val_size,
                    stratify=[s['label'] for s in train_val_samples],
                    random_state=random_state
                )
            except Exception:
                train_samples, val_samples = train_test_split(
                    train_val_samples,
                    test_size=val_size,
                    random_state=random_state
                )
        else:
            train_samples = train_val_samples
            val_samples = []
        
        print(f"✓ 数据集划分完成:")
        print(f"   训练集: {len(train_samples)}")
        print(f"   验证集: {len(val_samples)}")
        print(f"   测试集: {len(test_samples)}")
        
        if self.split == 'train':
            return train_samples
        elif self.split == 'val':
            return val_samples
        elif self.split == 'test':
            return test_samples
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _split_dataset_from_csv(self, split_csv: str, fold: int) -> List[Dict]:
        """从固定划分CSV中读取样本列表。

        split_csv 期望包含列：slide_id(或image_id/wsi_id)、label、fold、split。
        其中 split ∈ {'train','val','test'}。
        """
        split_path = split_csv
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"split_csv 不存在: {split_path}")
        sdf = pd.read_csv(split_path)
        # 兼容可能的ID列名
        id_col = None
        for cand in ('slide_id', 'image_id', 'wsi_id'):
            if cand in sdf.columns:
                id_col = cand
                break
        if id_col is None:
            raise ValueError("split_csv 必须包含 'slide_id' 或 'image_id' 或 'wsi_id' 列")
        if 'split' not in sdf.columns:
            raise ValueError("split_csv 必须包含 'split' 列")
        if 'fold' not in sdf.columns:
            # 兼容单划分CSV：没有fold列时默认fold=0
            sdf['fold'] = 0
        if 'label' not in sdf.columns:
            raise ValueError("split_csv 必须包含 'label' 列")

        # 过滤当前fold与split
        sdf = sdf.copy()
        sdf[id_col] = sdf[id_col].astype(str)
        sdf['split'] = sdf['split'].astype(str)
        sdf['fold'] = sdf['fold'].fillna(0).astype(int)
        fold = int(fold)

        subset = sdf[(sdf['fold'] == fold) & (sdf['split'] == self.split)]
        if subset.empty:
            raise ValueError(
                f"split_csv 中未找到 fold={fold}, split='{self.split}' 的样本；"
                f"请检查 {split_path} 的 fold/split 列。"
            )

        # 建立从ID到label的映射，用于填充 feature/coord 路径
        label_map = {str(row[id_col]): str(row['label']) for _, row in subset.iterrows()}

        # 仅保留当前 features_dir 下存在特征+坐标文件的样本
        valid_samples: List[Dict] = []
        missing_files: List[str] = []
        missing_scales: List[str] = []
        for wsi_id, label in label_map.items():
            feature_file = os.path.join(self.features_dir, f"{wsi_id}.pt")
            coord_file = os.path.join(self.features_dir, f"{wsi_id}_coords.npy")
            scale_file = os.path.join(self.features_dir, f"{wsi_id}_scales.npy")

            feature_exists = os.path.exists(feature_file)
            coord_exists = os.path.exists(coord_file)
            scale_exists = os.path.exists(scale_file)
            if feature_exists and coord_exists:
                if not scale_exists:
                    missing_scales.append(wsi_id)
                valid_samples.append({
                    'wsi_id': wsi_id,
                    'label': label,
                    'feature_file': feature_file,
                    'coord_file': coord_file,
                    'scale_file': scale_file if scale_exists else None,
                })
            else:
                missing_files.append(wsi_id)

        if missing_files:
            print(f"⚠️  split_csv指定但缺少特征/坐标文件的样本数: {len(missing_files)}")
            if len(missing_files) < 10:
                print(f"   缺少的ID: {missing_files}")
        if missing_scales:
            print(f"⚠️  split_csv指定但缺少scales文件的样本数: {len(missing_scales)} (训练时将回退为0)")
            if len(missing_scales) < 10:
                print(f"   缺少scales的ID: {missing_scales}")

        print(f"✓ [split_csv] fold={fold} split={self.split} 有效样本数: {len(valid_samples)}")
        return valid_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str] | None:
        """
        Returns:
            features: [N, 1024] - patch特征
            coords: [N, 2] - 归一化坐标
            scales: [N,] - 尺度信息（单尺度时全为0）
            label: int - 类别标签
            wsi_id: str - 当前bag对应的WSI ID
        """
        sample = self.samples[idx]
        
        # 加载特征和坐标（健壮处理）
        try:
            try:
                obj = torch.load(sample['feature_file'], map_location='cpu', weights_only=True)
            except TypeError:
                obj = torch.load(sample['feature_file'], map_location='cpu')
        except Exception as e:
            if self.skip_broken:
                if not hasattr(self, '_warned_broken_feature'):
                    print(f"⚠️  发现无法读取的特征文件，将跳过坏样本（后续不再重复提示）。示例: {sample['feature_file']} err={e}")
                    self._warned_broken_feature = True
                return None
            raise
        try:
            if isinstance(obj, torch.Tensor):
                features = obj
            elif isinstance(obj, dict) and 'features' in obj:
                features = obj['features']
            else:
                # 兼容可能保存为numpy的情况
                import numpy as _np
                if isinstance(obj, _np.ndarray):
                    features = torch.from_numpy(obj)
                else:
                    raise TypeError(f"Unsupported feature file format: {type(obj)}")
        except Exception as e:
            if self.skip_broken:
                if not hasattr(self, '_warned_broken_format'):
                    print(f"⚠️  特征文件格式异常，将跳过坏样本（后续不再重复提示）。示例: {sample['feature_file']} err={e}")
                    self._warned_broken_format = True
                return None
            raise RuntimeError(f"Failed to load features from {sample['feature_file']}: {e}")

        # 确保为float32二维 [N, 1024]
        if features.dtype != torch.float32:
            features = features.float()
        if features.ndimension() != 2:
            raise ValueError(f"Expected 2D features [N, D], got shape {tuple(features.shape)} from {sample['feature_file']}")

        coords = np.load(sample['coord_file'])  # [N, 2]，已归一化
        if sample.get('scale_file') and os.path.exists(sample['scale_file']):
            scales_np = np.load(sample['scale_file'])
            if scales_np.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"Scale/coord length mismatch for {sample['wsi_id']}: {scales_np.shape[0]} vs {coords.shape[0]}"
                )
        else:
            if not hasattr(self, '_warned_scale_fallback'):
                print("⚠️  未找到scales文件，将默认全部为0 (20x)")
                self._warned_scale_fallback = True
            scales_np = np.zeros(coords.shape[0], dtype=np.int64)
        
        # 限制patch数量（如果指定）
        if self.max_patches_per_wsi is not None and features.shape[0] > self.max_patches_per_wsi:
            n_total = int(features.shape[0])
            n_keep = int(self.max_patches_per_wsi)
            if self.split == 'train':
                # 训练集：随机子采样作为bag级增强
                indices = torch.randperm(n_total)[:n_keep]
                np_idx = indices.numpy()
                features = features[indices]
                coords = coords[np_idx]
                scales_np = scales_np[np_idx]
            else:
                if self.deterministic_eval_subsample:
                    # 验证/测试：固定子采样，避免每次评估抽到不同patch导致指标抖动
                    # 注意：不能用 Python 内置 hash（每次启动会随机化），这里用 md5 生成稳定种子
                    digest = hashlib.md5(str(sample['wsi_id']).encode('utf-8')).hexdigest()
                    seed = (int(digest[:8], 16) + self.random_state) % (2 ** 32)
                    rng = np.random.RandomState(seed)
                    np_idx = rng.choice(n_total, size=n_keep, replace=False)
                    np_idx.sort()
                    idx_t = torch.from_numpy(np_idx).long()
                    features = features.index_select(0, idx_t)
                    coords = coords[np_idx]
                    scales_np = scales_np[np_idx]
                else:
                    # 验证/测试：随机子采样（每次评估可能抽到不同patch）
                    indices = torch.randperm(n_total)[:n_keep]
                    np_idx = indices.numpy()
                    features = features[indices]
                    coords = coords[np_idx]
                    scales_np = scales_np[np_idx]
        
        # 转换坐标为tensor
        coords = torch.from_numpy(coords).float()
        scales = torch.from_numpy(scales_np).long()
        
        # 获取标签
        label = self.label2idx[sample['label']]

        return features, coords, scales, label, sample['wsi_id']
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重，用于处理数据不平衡"""
        # 统计每个类别的样本数
        label_counts = np.zeros(self.num_classes)
        for sample in self.samples:
            label_idx = self.label2idx[sample['label']]
            label_counts[label_idx] += 1
        
        # 计算权重 (总样本数 / (类别数 * 该类别样本数))
        total_samples = len(self.samples)
        class_weights = total_samples / (self.num_classes * label_counts)
        
        return torch.FloatTensor(class_weights)
    
    def get_label_distribution(self) -> Dict[str, int]:
        """获取当前划分的标签分布"""
        distribution = {}
        for sample in self.samples:
            label = sample['label']
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

def collate_fn(batch):
    """
    自定义批处理函数，处理变长序列
    由于每个WSI的patch数不同，需要特殊处理
    """
    # 允许跳过坏样本：Dataset 可能返回 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # 对于MIL，我们一次只处理一个bag（WSI）
    # 所以batch_size实际上应该是1
    if len(batch) != 1:
        raise ValueError("MIL任务batch_size应该为1，因为每个bag的patch数不同")

    features, coords, scales, label, wsi_id = batch[0]
    return features, coords, scales, torch.tensor(label), wsi_id

# 使用示例和测试代码
if __name__ == "__main__":
    # 初始化数据集
    features_dir = r"d:\FenLei\MsaMIL\MsaMIL_Net\data\features"
    label_file = r"d:\FenLei\MsaMIL\MsaMIL_Net\data\label.csv"
    
    # 创建训练集
    train_dataset = PreExtractedFeatureDataset(
        features_dir=features_dir,
        label_file=label_file,
        split='train',
        max_patches_per_wsi=5000  # 限制每个WSI最大5000个patch
    )
    
    print(f"\n训练集测试:")
    print(f"样本数: {len(train_dataset)}")
    print(f"类别数: {train_dataset.num_classes}")
    print(f"类别权重: {train_dataset.get_class_weights()}")
    
    # 测试数据加载
    sample_features, sample_coords, sample_scales, sample_label, sample_wsi_id = train_dataset[0]
    print(f"\n样本测试:")
    print(f"特征shape: {sample_features.shape}")
    print(f"坐标shape: {sample_coords.shape}")
    print(f"尺度shape: {sample_scales.shape}")
    print(f"标签: {sample_label} ({train_dataset.idx2label[sample_label]}) / WSI: {sample_wsi_id}")
    
    # 创建验证集和测试集
    val_dataset = PreExtractedFeatureDataset(
        features_dir=features_dir,
        label_file=label_file,
        split='val',
        max_patches_per_wsi=5000
    )
    
    test_dataset = PreExtractedFeatureDataset(
        features_dir=features_dir,
        label_file=label_file,
        split='test',
        max_patches_per_wsi=5000
    )
    
    print(f"\n数据集大小:")
    print(f"训练集: {len(train_dataset)}")
    print(f"验证集: {len(val_dataset)}")
    print(f"测试集: {len(test_dataset)}")