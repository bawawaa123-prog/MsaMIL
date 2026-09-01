#!/usr/bin/env python3
"""
重置IAAM分类器头，消除HGSC偏置

使用方法:
python reset_iaam_classifier.py --checkpoint_path /path/to/best_model.pth
"""

import torch
import torch.nn as nn
import argparse
import os
from models.IAAM import IAAM

def reset_classifier_head(checkpoint_path: str, backup: bool = True):
    """重置IAAM分类器头"""
    
    print(f"🔧 重置IAAM分类器头: {checkpoint_path}")
    
    # 备份原始checkpoint
    if backup:
        backup_path = checkpoint_path.replace('.pth', '_backup.pth')
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(checkpoint_path, backup_path)
            print(f"✅ 已备份到: {backup_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 获取配置
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        num_classes = cfg.get('num_classes', 5)
        d_model = cfg.get('d_model', 512)
        print(f"   模型配置: d_model={d_model}, num_classes={num_classes}")
    else:
        print("⚠️ 未找到配置信息，使用默认值")
        num_classes = 5
        d_model = 512
    
    # 创建临时IAAM模型以获取分类器结构
    temp_iaam = IAAM(d_model=d_model, num_classes=num_classes)
    
    # 获取state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 重置分类器相关权重
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
    
    print(f"   找到分类器层: {len(classifier_keys)}个")
    for key in classifier_keys:
        layer_name = key.split('.')[-1]  # weight 或 bias
        layer_idx = key.split('.')[1]    # 第几层
        
        if 'weight' in key:
            # 重新初始化权重
            if '0.weight' in key or '2.weight' in key:  # Linear层
                original_shape = state_dict[key].shape
                new_weight = torch.empty(original_shape)
                nn.init.xavier_uniform_(new_weight, gain=1.0)
                state_dict[key] = new_weight
                print(f"   ✅ 重置 {key}: Xavier uniform, shape={original_shape}")
        
        elif 'bias' in key:
            # 重新初始化偏置
            original_shape = state_dict[key].shape
            new_bias = torch.zeros(original_shape)
            state_dict[key] = new_bias
            print(f"   ✅ 重置 {key}: 全零, shape={original_shape}")
    
    # 保存修改后的checkpoint
    if 'model_state_dict' in checkpoint:
        checkpoint['model_state_dict'] = state_dict
    else:
        checkpoint = state_dict
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ 分类器头已重置并保存")
    
    # 验证修改
    print("\n🔍 验证重置结果:")
    reloaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    reload_state = reloaded['model_state_dict'] if 'model_state_dict' in reloaded else reloaded
    
    for key in classifier_keys:
        if 'weight' in key:
            tensor = reload_state[key]
            mean_val = tensor.float().mean().item()
            std_val = tensor.float().std().item()
            max_abs = tensor.float().abs().max().item()
            print(f"   {key}: mean={mean_val:.6f}, std={std_val:.6f}, max_abs={max_abs:.6f}")

def main():
    parser = argparse.ArgumentParser(description='重置IAAM分类器头消除HGSC偏置')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='IAAM checkpoint路径')
    parser.add_argument('--no-backup', action='store_true',
                        help='不创建备份文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ 文件不存在: {args.checkpoint_path}")
        return
    
    try:
        reset_classifier_head(args.checkpoint_path, backup=not args.no_backup)
        print("\n✅ 重置完成！现在可以重新训练端到端模型")
        print("   预期效果: HGSC过度预测应该得到缓解")
    except Exception as e:
        print(f"❌ 重置失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()