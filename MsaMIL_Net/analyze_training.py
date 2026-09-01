#!/usr/bin/env python3
"""
训练进度监控脚本
用于实时分析训练日志，生成可视化报告

使用方法: python analyze_training.py
会自动查找最新的训练日志文件并分析
"""

import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非GUI后端，避免显示问题
import matplotlib.pyplot as plt

def parse_training_log(log_file):
    """解析训练日志"""

    data = {
        'batches': [],
        'losses': [],
        'accuracies': [],
        'grad_norms': [],
        'lr': [],
        'timestamps': [],
        'grad_clips': []
    }

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配: Epoch 1/50 | Batch 511/3625 | Loss=2.0582 | Acc=0.4305
            match = re.search(r'Batch (\d+)/\d+ \| Loss=([\d.]+) \| Acc=([\d.]+)', line)
            if match:
                batch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))

                data['batches'].append(batch)
                data['losses'].append(loss)
                data['accuracies'].append(acc)

            # 匹配: 梯度范数
            match = re.search(r'Unscale后梯度范数: ([\d.]+)', line)
            if match:
                norm = float(match.group(1))
                data['grad_norms'].append(norm)

            # 匹配: 梯度裁剪
            match = re.search(r'梯度范数过大: ([\d.]+)', line)
            if match:
                clip = float(match.group(1))
                data['grad_clips'].append(clip)

    return data

def analyze_training(data):
    """分析训练状态"""

    if not data['losses']:
        print("❌ 未找到有效数据")
        return

    print("\n" + "="*60)
    print("📊 训练状态分析报告")
    print("="*60 + "\n")

    # 基本统计
    total_batches = len(data['batches'])
    current_batch = data['batches'][-1] if data['batches'] else 0

    print(f"📈 训练进度：")
    print(f"   已完成 batches: {total_batches}")
    print(f"   当前 batch: {current_batch}")

    # Loss 分析
    if len(data['losses']) > 1:
        initial_loss = data['losses'][0]
        current_loss = data['losses'][-1]
        loss_decrease = ((initial_loss - current_loss) / initial_loss) * 100

        print(f"\n📉 Loss 变化：")
        print(f"   初始 Loss: {initial_loss:.4f}")
        print(f"   当前 Loss: {current_loss:.4f}")
        print(f"   下降比例: {loss_decrease:.2f}%")

        # 计算最近100个batch的平均loss
        if len(data['losses']) >= 100:
            recent_loss = sum(data['losses'][-100:]) / 100
            print(f"   近100batch平均: {recent_loss:.4f}")

    # Accuracy 分析
    if len(data['accuracies']) > 1:
        initial_acc = data['accuracies'][0]
        current_acc = data['accuracies'][-1]
        max_acc = max(data['accuracies'])

        print(f"\n🎯 准确率变化：")
        print(f"   初始准确率: {initial_acc*100:.2f}%")
        print(f"   当前准确率: {current_acc*100:.2f}%")
        print(f"   最高准确率: {max_acc*100:.2f}%")
        print(f"   提升幅度: +{(current_acc - initial_acc)*100:.2f}%")

        # 判断是否在平台期
        if len(data['accuracies']) >= 100:
            recent_acc = data['accuracies'][-100:]
            acc_std = (sum([(x - current_acc)**2 for x in recent_acc]) / 100) ** 0.5
            if acc_std < 0.005:  # 标准差小于0.5%
                print(f"   ⚠️  可能进入平台期（波动小于0.5%）")

    # 梯度分析
    if data['grad_norms']:
        avg_norm = sum(data['grad_norms']) / len(data['grad_norms'])
        max_norm = max(data['grad_norms'])
        min_norm = min(data['grad_norms'])

        print(f"\n⚡ 梯度统计：")
        print(f"   平均梯度范数: {avg_norm:.4f}")
        print(f"   最大梯度范数: {max_norm:.4f}")
        print(f"   最小梯度范数: {min_norm:.4f}")

        # 裁剪频率
        if data['grad_clips']:
            clip_rate = len(data['grad_clips']) / total_batches * 100
            print(f"   梯度裁剪次数: {len(data['grad_clips'])}")
            print(f"   裁剪频率: {clip_rate:.2f}%")

            if clip_rate > 10:
                print(f"   ⚠️  裁剪频率过高，建议降低学习率")

    # 训练稳定性
    print(f"\n🔍 训练稳定性：")
    if len(data['losses']) >= 50:
        recent_losses = data['losses'][-50:]
        loss_trend = (recent_losses[-1] - recent_losses[0]) / recent_losses[0]

        if loss_trend < -0.05:
            print(f"   ✅ Loss持续下降（{loss_trend*100:.2f}%）")
        elif loss_trend > 0.05:
            print(f"   ⚠️  Loss上升，可能过拟合或学习率过大")
        else:
            print(f"   ℹ️  Loss趋于稳定（{loss_trend*100:.2f}%）")

    # 预估剩余时间（假设每batch 30秒）
    if current_batch > 0:
        total_expected = 3625  # 根据日志推断
        remaining = total_expected - current_batch
        estimated_hours = remaining * 30 / 3600

        print(f"\n⏱️  预估剩余时间：")
        print(f"   剩余 batches: {remaining}")
        print(f"   预估时间: {estimated_hours:.1f} 小时")

    # 建议
    print(f"\n💡 优化建议：")

    # 基于准确率的建议
    if data['accuracies']:
        current_acc = data['accuracies'][-1]
        if current_acc < 0.3:
            print(f"   📌 准确率较低(<30%)，继续训练")
        elif 0.3 <= current_acc < 0.5:
            print(f"   📌 准确率正常(30-50%)，模型正在学习")
        elif 0.5 <= current_acc < 0.7:
            print(f"   📌 准确率良好(50-70%)，考虑增加正则化")
        else:
            print(f"   📌 准确率优秀(70%+)，检查是否过拟合")

    # 基于梯度的建议
    if data['grad_clips'] and len(data['grad_clips']) / total_batches > 0.2:
        print(f"   📌 梯度频繁被裁剪，建议：")
        print(f"      - 降低学习率至5e-5")
        print(f"      - 增加warmup至2-3 epochs")

    # 基于loss的建议
    if len(data['losses']) > 100:
        recent_loss_std = (sum([(x - data['losses'][-1])**2 for x in data['losses'][-100:]]) / 100) ** 0.5
        if recent_loss_std < 0.01:
            print(f"   📌 Loss波动很小，可能需要：")
            print(f"      - 增大学习率")
            print(f"      - 减少label_smoothing")

    print("\n" + "="*60 + "\n")


def plot_training_progress(data, output_file='training_progress.png'):
    """生成训练进度可视化图表"""

    if not data['losses'] or not data['accuracies']:
        print("⚠️  数据不足，无法生成图表")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练进度监控', fontsize=16, fontweight='bold')

    # 1. Loss 曲线
    if data['losses']:
        axes[0, 0].plot(data['losses'], linewidth=1.5, color='#e74c3c')
        axes[0, 0].set_title('Loss 变化曲线', fontweight='bold')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # 添加移动平均线
        if len(data['losses']) > 50:
            window = 50
            moving_avg = [sum(data['losses'][max(0, i-window):i+1]) / min(i+1, window)
                         for i in range(len(data['losses']))]
            axes[0, 0].plot(moving_avg, linewidth=2, color='#c0392b',
                           label=f'{window}-batch移动平均', alpha=0.8)
            axes[0, 0].legend()

    # 2. Accuracy 曲线
    if data['accuracies']:
        axes[0, 1].plot([acc * 100 for acc in data['accuracies']],
                       linewidth=1.5, color='#3498db')
        axes[0, 1].set_title('准确率变化曲线', fontweight='bold')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # 添加移动平均线
        if len(data['accuracies']) > 50:
            window = 50
            moving_avg = [sum(data['accuracies'][max(0, i-window):i+1]) / min(i+1, window) * 100
                         for i in range(len(data['accuracies']))]
            axes[0, 1].plot(moving_avg, linewidth=2, color='#2980b9',
                           label=f'{window}-batch移动平均', alpha=0.8)
            axes[0, 1].legend()

    # 3. 梯度范数统计
    if data['grad_norms']:
        axes[1, 0].plot(data['grad_norms'], linewidth=1, color='#2ecc71', alpha=0.7)
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--',
                          linewidth=2, label='裁剪阈值')
        axes[1, 0].set_title('梯度范数变化', fontweight='bold')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, '无梯度数据',
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])

    # 4. 训练统计摘要
    axes[1, 1].axis('off')
    summary_text = "📊 训练统计摘要\n" + "─" * 30 + "\n\n"

    if data['losses']:
        summary_text += f"Loss:\n"
        summary_text += f"  当前: {data['losses'][-1]:.4f}\n"
        summary_text += f"  最低: {min(data['losses']):.4f}\n"
        summary_text += f"  下降: {((data['losses'][0] - data['losses'][-1])/data['losses'][0]*100):.1f}%\n\n"

    if data['accuracies']:
        summary_text += f"准确率:\n"
        summary_text += f"  当前: {data['accuracies'][-1]*100:.2f}%\n"
        summary_text += f"  最高: {max(data['accuracies'])*100:.2f}%\n"
        summary_text += f"  提升: +{(data['accuracies'][-1] - data['accuracies'][0])*100:.2f}%\n\n"

    if data['grad_norms']:
        summary_text += f"梯度:\n"
        summary_text += f"  平均范数: {sum(data['grad_norms'])/len(data['grad_norms']):.4f}\n"
        summary_text += f"  最大范数: {max(data['grad_norms']):.4f}\n"

    if data['grad_clips']:
        clip_rate = len(data['grad_clips']) / len(data['losses']) * 100
        summary_text += f"  裁剪率: {clip_rate:.1f}%\n"

    axes[1, 1].text(0.1, 0.9, summary_text,
                   fontsize=11, verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ 训练进度图表已保存: {output_file}")
    plt.close()


def main():
    """主函数"""
    # 尝试找到最新的日志文件
    log_dir = Path(".")
    log_files = list(log_dir.glob("train_*.log"))

    if not log_files:
        # 也尝试在logs目录
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("train_*.log"))

    if not log_files:
        print("❌ 未找到训练日志文件")
        print("请将此脚本放在与日志文件相同的目录下")
        return

    # 使用最新的日志文件
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)

    print(f"📁 正在分析日志文件: {latest_log}")

    data = parse_training_log(latest_log)
    analyze_training(data)

    # 生成可视化图表
    plot_training_progress(data)

if __name__ == '__main__':
    main()
