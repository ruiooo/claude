#!/usr/bin/env python3
"""
plot_training.py - 训练进度可视化工具
========================================

功能：
1. 读取训练日志
2. 绘制训练曲线（胜率、奖励、epsilon）
3. 分析收敛趋势
4. 推荐最佳停止时机

使用方法：
    python python/plot_training.py

依赖：
    pip install matplotlib (可选)
"""

import os
import re
from typing import List, Dict, Tuple
import numpy as np


def parse_training_log(log_file: str = None) -> Dict[str, List]:
    """
    解析训练日志文件

    Args:
        log_file: 日志文件路径，如果为None则使用最新的checkpoint

    Returns:
        包含训练数据的字典
    """
    # TODO: 实现日志解析
    # 这里先返回模拟数据，后续可以从actual log中读取
    return {
        'episodes': [],
        'win_rates': [],
        'rewards': [],
        'epsilons': [],
        'losses': []
    }


def load_from_checkpoints() -> Dict[str, List]:
    """
    从checkpoint文件中加载训练进度

    Returns:
        训练数据字典
    """
    import torch

    data = {
        'episodes': [],
        'win_rates': [],
        'epsilons': [],
    }

    # 查找所有checkpoint文件
    checkpoint_dirs = ['saved_models/checkpoints', 'checkpoints']
    checkpoint_files = []

    for dir_path in checkpoint_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.startswith('checkpoint_ep') and f.endswith('.pth')]
            checkpoint_files.extend([os.path.join(dir_path, f) for f in files])

    if not checkpoint_files:
        print("❌ 未找到checkpoint文件")
        return data

    # 按回合数排序
    checkpoint_files.sort(key=lambda x: int(re.search(r'ep(\d+)', x).group(1)))

    print(f"找到 {len(checkpoint_files)} 个checkpoint文件")

    # 加载每个checkpoint
    for ckpt_file in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            episode = checkpoint.get('episode', 0)
            epsilon = checkpoint.get('epsilon', 0)

            data['episodes'].append(episode)
            data['epsilons'].append(epsilon)
            # win_rate需要从其他地方获取（暂时无法从checkpoint中读取）
            data['win_rates'].append(0.0)

        except Exception as e:
            print(f"⚠️  加载失败 {ckpt_file}: {e}")

    return data


def plot_training_curves(data: Dict[str, List], save_path: str = 'training_curves.png'):
    """
    绘制训练曲线

    Args:
        data: 训练数据
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
    except ImportError:
        print("❌ 未安装matplotlib，无法绘图")
        print("   安装: pip install matplotlib")
        return

    episodes = data['episodes']
    if not episodes:
        print("⚠️  没有数据可绘制")
        return

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('训练进度曲线', fontsize=16, fontweight='bold')

    # 1. 胜率曲线
    if data['win_rates'] and any(wr > 0 for wr in data['win_rates']):
        axes[0].plot(episodes, data['win_rates'], 'b-', linewidth=2, label='胜率')
        axes[0].axhline(y=0.7, color='g', linestyle='--', label='目标胜率 70%')
        axes[0].axhline(y=0.5, color='orange', linestyle='--', label='基准 50%')
        axes[0].set_ylabel('胜率', fontsize=12)
        axes[0].set_title('胜率变化')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

    # 2. 奖励曲线
    if data.get('rewards'):
        axes[1].plot(episodes, data['rewards'], 'r-', linewidth=2, label='平均奖励')
        axes[1].set_ylabel('平均奖励', fontsize=12)
        axes[1].set_title('奖励变化')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # 3. Epsilon曲线
    if data['epsilons']:
        axes[2].plot(episodes, data['epsilons'], 'g-', linewidth=2, label='Epsilon')
        axes[2].set_ylabel('探索率 (Epsilon)', fontsize=12)
        axes[2].set_xlabel('训练回合数', fontsize=12)
        axes[2].set_title('探索率衰减')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存到: {save_path}")


def analyze_convergence(data: Dict[str, List]) -> Dict:
    """
    分析训练收敛情况

    Returns:
        分析结果
    """
    if not data['episodes']:
        return {'converged': False, 'reason': '没有数据'}

    episodes = np.array(data['episodes'])
    win_rates = np.array(data['win_rates']) if data['win_rates'] else np.zeros_like(episodes)

    # 检查最近100个评估点的胜率
    if len(win_rates) < 10:
        return {
            'converged': False,
            'reason': f'数据不足（需要至少10个评估点，当前{len(win_rates)}个）'
        }

    recent_win_rates = win_rates[-10:]
    avg_win_rate = np.mean(recent_win_rates)
    std_win_rate = np.std(recent_win_rates)

    # 收敛判断标准
    converged = (
        avg_win_rate >= 0.7 and  # 平均胜率>70%
        std_win_rate < 0.05       # 标准差<5%（稳定）
    )

    return {
        'converged': converged,
        'avg_win_rate': avg_win_rate,
        'std_win_rate': std_win_rate,
        'total_episodes': episodes[-1] if len(episodes) > 0 else 0,
        'checkpoints': len(episodes),
        'recommendation': _get_recommendation(avg_win_rate, std_win_rate, episodes[-1] if len(episodes) > 0 else 0)
    }


def _get_recommendation(win_rate: float, stability: float, episodes: int) -> str:
    """生成训练建议"""
    if win_rate >= 0.85 and stability < 0.03:
        return "✅ 训练已完全收敛，建议停止训练"
    elif win_rate >= 0.7 and stability < 0.05:
        return "✅ 训练已收敛，可以考虑停止或继续优化"
    elif win_rate >= 0.5 and episodes < 5000:
        return "⏳ 训练进展正常，建议继续到至少5000回合"
    elif win_rate < 0.5 and episodes > 5000:
        return "⚠️  训练效果不理想，建议检查配置或数据质量"
    elif episodes < 2000:
        return "⏳ 训练早期，请至少训练到2000回合"
    else:
        return "⏳ 继续训练，监控收敛情况"


def print_summary(analysis: Dict):
    """打印分析摘要"""
    print(f"\n{'='*80}")
    print(f"📊 训练进度分析报告")
    print(f"{'='*80}")

    if 'reason' in analysis:
        print(f"   状态: ⏳ {analysis['reason']}")
    else:
        print(f"   总回合数: {analysis['total_episodes']:,}")
        print(f"   Checkpoint数: {analysis['checkpoints']}")
        print(f"   平均胜率: {analysis['avg_win_rate']*100:.1f}%")
        print(f"   胜率标准差: {analysis['std_win_rate']*100:.2f}%")
        print(f"   收敛状态: {'✅ 已收敛' if analysis['converged'] else '⏳ 训练中'}")
        print(f"\n   建议: {analysis['recommendation']}")

    print(f"{'='*80}\n")


def main():
    """主函数"""
    print(f"\n🔍 训练进度可视化工具\n")

    # 尝试从checkpoint加载
    print("正在从checkpoint加载数据...")
    data = load_from_checkpoints()

    # 如果没有数据，提示用户
    if not data['episodes']:
        print(f"\n⚠️  未找到训练数据")
        print(f"\n💡 提示：")
        print(f"   1. 先运行训练: make train")
        print(f"   2. 训练会每100回合保存一次checkpoint")
        print(f"   3. 再次运行本工具查看训练曲线")
        return

    # 分析收敛
    analysis = analyze_convergence(data)
    print_summary(analysis)

    # 绘制曲线
    print("正在生成训练曲线...")
    plot_training_curves(data)

    print(f"\n✅ 分析完成！\n")


if __name__ == '__main__':
    main()
