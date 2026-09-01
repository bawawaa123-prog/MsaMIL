#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证训练改进脚本
快速检查所有改进是否正确实现
"""

import sys
import pandas as pd
from pathlib import Path

print("="*80)
print("训练改进验证")
print("="*80)

# 1. 验证label.csv存在且格式正确
print("\n[1/5] 检查label.csv...")
label_file = Path('data/label.csv')
if not label_file.exists():
    print("❌ label.csv不存在！")
    sys.exit(1)

df = pd.read_csv(label_file)
required_cols = ['image_id', 'label']
if not all(col in df.columns for col in required_cols):
    print("❌ label.csv缺少必需列！")
    sys.exit(1)

print(f"✅ label.csv格式正确，共{len(df)}个样本")

# 2. 分析类别分布
print("\n[2/5] 分析类别分布...")
counts = df['label'].value_counts().sort_index()
print(f"类别数: {len(counts)}")
for label, count in counts.items():
    ratio = count / len(df) * 100
    print(f"  {label}: {count}个 ({ratio:.1f}%)")

imbalance_ratio = counts.max() / counts.min()
print(f"不平衡比: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print(f"⚠️  存在类别不平衡（>{3}:1）")
else:
    print(f"✅ 类别分布相对均衡")

# 3. 检查训练脚本关键函数
print("\n[3/5] 检查训练脚本...")
train_script = Path('train_NMFEM_end2end.py')
if not train_script.exists():
    print("❌ train_NMFEM_end2end.py不存在！")
    sys.exit(1)

content = train_script.read_text(encoding='utf-8')

# 检查关键改进是否存在
checks = [
    ('class_chunks', '类别平衡采样'),
    ('per_class_acc', 'Per-class准确率'),
    ('c_pred', 'Chunk预测输出'),
    ('c_acc', 'Chunk级准确率'),
    ('w_acc', 'WSI级准确率'),
    ('classification_report', '分类报告'),
    ('all_gather_object', 'GPU结果汇总'),
    ('pred_distribution', '预测分布监控'),
]

print("关键功能检查:")
all_passed = True
for keyword, description in checks:
    if keyword in content:
        print(f"  ✅ {description}")
    else:
        print(f"  ❌ {description} - 未找到关键字: {keyword}")
        all_passed = False

if not all_passed:
    print("\n⚠️  部分功能可能未正确实现")
else:
    print("\n✅ 所有关键功能已实现")

# 4. 计算建议的类别权重
print("\n[4/5] 计算类别权重...")
total = len(df)
num_classes = len(counts)
print("建议的类别权重 (inverse frequency):")
for label, count in counts.items():
    weight = total / (num_classes * count)
    print(f"  {label}: {weight:.4f}")

# 5. 生成训练命令
print("\n[5/5] 生成训练命令...")
print("\n推荐的训练命令:")
print("-"*80)
print("torchrun --nproc_per_node=2 train_NMFEM_end2end.py \\")
print("    --epochs 50 \\")
print("    --chunk_size 256 \\")
print("    --lr 1e-4 \\")
print("    --label_smoothing 0.05 \\")
print("    --weight_decay 1e-4 \\")
print("    --log_interval 10")
print("-"*80)

print("\n" + "="*80)
print("验证完成！")
print("="*80)
print("\n📝 重要提示:")
print("1. 训练时会自动输出数据分布诊断")
print("2. 进度条显示chunk预测、真实标签和双层准确率")
print("3. 每个epoch结束输出完整的per-class评估报告")
print("4. 使用类别权重处理不平衡问题")
print("5. 类别平衡采样确保GPU间分布均匀")
print("\n⚠️  如果准确率仍停滞在0.4左右，请检查:")
print("   - 日志中的'预测分布'是否过度集中某一类")
print("   - 类别权重是否正确应用")
print("   - 学习率是否合适")
print("="*80)

