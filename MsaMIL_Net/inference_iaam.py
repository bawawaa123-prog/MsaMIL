import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import argparse
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.IAAM import IAAM
from datasets.feature_dataset import PreExtractedFeatureDataset, collate_fn
from torch.utils.data import DataLoader

class IAAMInference:
    """IAAM模型推理类"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Args:
            model_path: 训练好的模型路径
            device: 计算设备
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # 加载模型
        self.model, self.model_config = self._load_model(model_path)
        self.model.eval()
        
        print(f"✓ 模型加载完成，设备: {self.device}")
        print(f"✓ 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从checkpoint中获取模型配置
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # 使用默认配置
            config = {
                'd_model': 512,
                'input_dim': 1024,
                'mhe_layers': 4,
                'num_heads': 8,
                'low_rank': 64,
                'num_queries': 10,
                'num_classes': 5,
                'dropout': 0.1
            }
            print("⚠️  未找到模型配置，使用默认配置")
        
        # 创建模型
        model = IAAM(
            d_model=config['d_model'],
            input_dim=config['input_dim'],
            mhe_layers=config['mhe_layers'],
            num_heads=config['num_heads'],
            low_rank=config['low_rank'],
            num_queries=config['num_queries'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        
        return model, config
    
    def predict_single(self, features, coords, scales=None):
        """
        对单个样本进行预测
        
        Args:
            features: [N, 1024] - patch特征
            coords: [N, 2] - 归一化坐标
            scales: [N,] - 尺度信息（可选，默认全为0）
            
        Returns:
            prediction: int - 预测类别
            confidence: float - 预测置信度
            logits: tensor - 原始logits
        """
        with torch.no_grad():
            # 转换为tensor并移到设备
            if not isinstance(features, torch.Tensor):
                features = torch.from_numpy(features).float()
            if not isinstance(coords, torch.Tensor):
                coords = torch.from_numpy(coords).float()
            
            features = features.to(self.device)
            coords = coords.to(self.device)
            
            # 如果没有提供尺度信息，默认单尺度（全为0）
            if scales is None:
                scales = torch.zeros(features.shape[0], dtype=torch.long).to(self.device)
            else:
                if not isinstance(scales, torch.Tensor):
                    scales = torch.from_numpy(scales).long()
                scales = scales.to(self.device)
            
            # 前向传播
            logits, bag_features = self.model(features, scales, coords)
            
            # 预测
            probs = F.softmax(logits, dim=0)
            prediction = torch.argmax(logits).item()
            confidence = probs[prediction].item()
            
            return prediction, confidence, logits.cpu()
    
    def predict_batch(self, dataloader, return_features=False):
        """
        批量预测
        
        Args:
            dataloader: 数据加载器
            return_features: 是否返回bag级特征
            
        Returns:
            predictions: list - 预测结果
            confidences: list - 置信度
            true_labels: list - 真实标签
            bag_features: list - bag级特征（如果return_features=True）
        """
        predictions = []
        confidences = []
        true_labels = []
        bag_features = [] if return_features else None
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                if len(batch) == 5:
                    features, coords, scales, labels, _ = batch
                else:
                    features, coords, scales, labels = batch
                # 移到设备
                features = features.to(self.device)
                coords = coords.to(self.device)
                scales = scales.to(self.device)
                
                # 前向传播
                logits, bag_feat = self.model(features, scales, coords)
                
                # 预测
                probs = F.softmax(logits, dim=0)
                prediction = torch.argmax(logits).item()
                confidence = probs[prediction].item()
                
                predictions.append(prediction)
                confidences.append(confidence)
                true_labels.append(labels.item())
                
                if return_features:
                    bag_features.append(bag_feat.cpu().numpy())
        
        return predictions, confidences, true_labels, bag_features

def evaluate_model(model_path, features_dir, label_file, save_dir=None):
    """
    评估模型性能
    
    Args:
        model_path: 模型路径
        features_dir: 特征目录
        label_file: 标签文件
        save_dir: 结果保存目录
    """
    # 创建推理器
    inferencer = IAAMInference(model_path)
    
    # 创建测试数据集
    test_dataset = PreExtractedFeatureDataset(
        features_dir=features_dir,
        label_file=label_file,
        split='test',
        max_patches_per_wsi=5000
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    print(f"测试样本数: {len(test_dataset)}")
    print(f"类别: {test_dataset.label_names}")
    
    # 预测
    predictions, confidences, true_labels, _ = inferencer.predict_batch(test_loader)
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average=None)
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    
    # 分类报告
    report = classification_report(
        true_labels, predictions,
        target_names=test_dataset.label_names,
        digits=4
    )
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    
    # 打印结果
    print(f"\n{'='*50}")
    print(f"模型评估结果")
    print(f"{'='*50}")
    print(f"总体准确率: {accuracy:.4f}")
    print(f"宏平均F1: {macro_f1:.4f}")
    print(f"加权平均F1: {weighted_f1:.4f}")
    print(f"\n各类别表现:")
    for i, class_name in enumerate(test_dataset.label_names):
        print(f"  {class_name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    print(f"\n分类报告:")
    print(report)
    
    print(f"\\n混淆矩阵:")
    print(cm)
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存详细结果
        results = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'per_class_metrics': {
                test_dataset.label_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(test_dataset.label_names))
            },
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'confusion_matrix': cm.tolist(),
            'class_names': test_dataset.label_names
        }
        
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存分类报告
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"模型评估结果\\n")
            f.write(f"{'='*50}\\n")
            f.write(f"总体准确率: {accuracy:.4f}\\n")
            f.write(f"宏平均F1: {macro_f1:.4f}\\n")
            f.write(f"加权平均F1: {weighted_f1:.4f}\\n\\n")
            f.write(report)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=test_dataset.label_names,
                    yticklabels=test_dataset.label_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 绘制置信度分布
        plt.figure(figsize=(10, 6))
        
        # 按类别绘制置信度分布
        for i, class_name in enumerate(test_dataset.label_names):
            class_confidences = [confidences[j] for j, pred in enumerate(predictions) if pred == i]
            if class_confidences:
                plt.hist(class_confidences, alpha=0.7, label=f'{class_name} (n={len(class_confidences)})',
                        bins=20)
        
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=300)
        plt.close()
        
        print(f"\\n✓ 评估结果已保存到: {save_dir}")
    
    return results

def predict_single_wsi(model_path, feature_file, coord_file, class_names=None):
    """
    对单个WSI进行预测
    
    Args:
        model_path: 模型路径
        feature_file: 特征文件路径
        coord_file: 坐标文件路径
        class_names: 类别名称列表
    """
    # 创建推理器
    inferencer = IAAMInference(model_path)
    
    # 加载数据
    features = torch.load(feature_file, weights_only=True)
    coords = np.load(coord_file)
    
    print(f"特征形状: {features.shape}")
    print(f"坐标形状: {coords.shape}")
    
    # 预测
    prediction, confidence, logits = inferencer.predict_single(features, coords)
    
    # 输出结果
    if class_names:
        print(f"预测类别: {class_names[prediction]} (ID: {prediction})")
    else:
        print(f"预测类别ID: {prediction}")
    
    print(f"置信度: {confidence:.4f}")
    print(f"所有类别概率: {F.softmax(logits, dim=0).numpy()}")
    
    return prediction, confidence

def main():
    parser = argparse.ArgumentParser(description='IAAM模型推理和评估')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'predict'], default='evaluate',
                       help='运行模式：evaluate=评估模型，predict=单样本预测')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--features_dir', type=str,
                       default=r'd:\\FenLei\\MsaMIL\\MsaMIL_Net\\data\\features',
                       help='特征文件目录')
    parser.add_argument('--label_file', type=str,
                       default=r'd:\\FenLei\\MsaMIL\\MsaMIL_Net\\data\\label.csv',
                       help='标签CSV文件')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='结果保存目录')
    parser.add_argument('--feature_file', type=str,
                       help='单个特征文件路径（predict模式）')
    parser.add_argument('--coord_file', type=str,
                       help='单个坐标文件路径（predict模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        # 评估模式
        evaluate_model(
            model_path=args.model_path,
            features_dir=args.features_dir,
            label_file=args.label_file,
            save_dir=args.save_dir
        )
    
    elif args.mode == 'predict':
        # 预测模式
        if not args.feature_file or not args.coord_file:
            raise ValueError("predict模式需要指定--feature_file和--coord_file")
        
        # 从标签文件获取类别名称
        df = pd.read_csv(args.label_file)
        class_names = sorted(df['label'].unique())
        
        predict_single_wsi(
            model_path=args.model_path,
            feature_file=args.feature_file,
            coord_file=args.coord_file,
            class_names=class_names
        )

if __name__ == '__main__':
    main()