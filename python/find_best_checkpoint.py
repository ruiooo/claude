#!/usr/bin/env python3
"""
批量诊断Checkpoint工具

测试关键时期的模型，找到训练崩溃的时间点和最佳模型
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DQNAgent
from config import MODEL_CONFIG

def quick_diagnose(model_path: str):
    """快速诊断模型Q值分布"""

    # 设置设备
    device = MODEL_CONFIG.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    agent = DQNAgent(
        state_dim=MODEL_CONFIG['state_dim'],
        action_dim=MODEL_CONFIG['action_dim'],
        config=MODEL_CONFIG,
        device=device
    )

    if not os.path.exists(model_path):
        return None

    agent.load(model_path)

    # 构造测试状态（敌人在左上方）
    test_state = np.zeros(43, dtype=np.float32)
    test_state[0] = 0.5   # AI x
    test_state[1] = 0.5   # AI y
    test_state[4] = 1.0   # 满血
    test_state[5] = 0.0   # 可射击
    test_state[6] = -0.2  # 敌人dx
    test_state[7] = -0.2  # 敌人dy
    test_state[10] = 1.0  # 敌人满血

    # 获取Q值
    with torch.no_grad():
        state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(agent.device)
        q_values = agent.policy_net(state_tensor)[0].cpu().numpy()

    best_action = np.argmax(q_values)

    return {
        'episode': agent.train_step // 6,  # 粗略估计回合数
        'epsilon': agent.epsilon,
        'q_mean': q_values.mean(),
        'q_std': q_values.std(),
        'q_min': q_values.min(),
        'q_max': q_values.max(),
        'q_range': q_values.max() - q_values.min(),
        'best_action': best_action,
        'best_q': q_values[best_action],
        'idle_q': q_values[0],
        'q_values': q_values
    }

def main():
    # 要测试的关键时期checkpoint
    test_episodes = [
        10000,   # 早期
        50000,   # 早中期
        100000,  # 中期
        150000,  # 中后期
        200000,  # 后期
        250000,  # 后期
        299000,  # 末期
        300000   # 最终
    ]

    print("=" * 100)
    print("🔍 批量诊断训练Checkpoint - 寻找最佳模型和崩溃时间点")
    print("=" * 100)

    results = []
    action_names = ["IDLE", "UP", "DOWN", "LEFT", "RIGHT", "SHOOT_UP", "SHOOT_DOWN", "SHOOT_LEFT", "SHOOT_RIGHT"]

    for ep in test_episodes:
        model_path = f"saved_models/checkpoints/checkpoint_ep{ep}.pth"
        print(f"\n{'─' * 100}")
        print(f"📂 测试: checkpoint_ep{ep}.pth")

        result = quick_diagnose(model_path)
        if result is None:
            print(f"   ❌ 文件不存在")
            continue

        results.append((ep, result))

        # 打印关键指标
        print(f"   回合数: {ep:,}")
        print(f"   Epsilon: {result['epsilon']:.4f}")
        print(f"   Q值统计: 均值={result['q_mean']:+.2f}, 标准差={result['q_std']:.4f}, 范围={result['q_range']:.4f}")
        print(f"   最佳动作: {action_names[result['best_action']]} (Q={result['best_q']:+.2f})")
        print(f"   IDLE Q值: {result['idle_q']:+.2f}")

        # 健康度评估
        health_score = 0
        issues = []

        # 检查1: Q值标准差
        if result['q_std'] > 1.0:
            health_score += 3
        elif result['q_std'] > 0.5:
            health_score += 2
            issues.append("⚠️  Q值差异偏小")
        else:
            health_score += 0
            issues.append("❌ Q值几乎相同（训练崩溃）")

        # 检查2: 是否选择IDLE
        if result['best_action'] != 0:
            health_score += 2
        else:
            issues.append("❌ 倾向选择IDLE")

        # 检查3: Q值是否合理
        if -50 < result['q_mean'] < 100:
            health_score += 1
        else:
            issues.append(f"⚠️  Q值范围异常 (均值={result['q_mean']:.1f})")

        # 总评
        if health_score >= 5:
            status = "✅ 健康"
        elif health_score >= 3:
            status = "⚠️  一般"
        else:
            status = "❌ 崩溃"

        print(f"   健康度: {status} (评分: {health_score}/6)")
        if issues:
            for issue in issues:
                print(f"      {issue}")

    # 总结报告
    print("\n" + "=" * 100)
    print("📊 训练过程分析")
    print("=" * 100)

    if not results:
        print("❌ 没有找到任何checkpoint文件")
        return

    # 绘制Q值标准差趋势
    print("\n【Q值标准差趋势】（健康指标: >1.0=好, 0.5-1.0=一般, <0.5=崩溃）")
    print("-" * 100)
    for ep, result in results:
        bar_length = int(result['q_std'] * 20)  # 20倍放大显示
        bar = "█" * bar_length
        status = "✅" if result['q_std'] > 1.0 else "⚠️ " if result['q_std'] > 0.5 else "❌"
        print(f"  {ep:>7,}回合: {status} {bar} {result['q_std']:.4f}")

    # 找到崩溃时间点
    print("\n【训练崩溃时间点分析】")
    print("-" * 100)
    for i in range(len(results) - 1):
        ep1, r1 = results[i]
        ep2, r2 = results[i + 1]

        if r1['q_std'] > 0.5 and r2['q_std'] <= 0.5:
            print(f"  🚨 崩溃检测: 在 {ep1:,} ~ {ep2:,} 回合之间")
            print(f"     Q值标准差: {r1['q_std']:.4f} → {r2['q_std']:.4f} (下降 {r1['q_std']-r2['q_std']:.4f})")
            print(f"     Epsilon: {r1['epsilon']:.4f} → {r2['epsilon']:.4f}")
            break
    else:
        if results[0][1]['q_std'] <= 0.5:
            print(f"  ⚠️  从一开始就处于崩溃状态")
        else:
            print(f"  ✅ 未检测到明显崩溃（或在采样点之间）")

    # 推荐最佳模型
    print("\n【推荐模型】")
    print("-" * 100)

    # 按健康度排序
    sorted_results = sorted(results, key=lambda x: x[1]['q_std'], reverse=True)

    print("  Top 3 最健康的模型:")
    for i, (ep, result) in enumerate(sorted_results[:3], 1):
        print(f"    [{i}] checkpoint_ep{ep}.pth")
        print(f"        Q值标准差: {result['q_std']:.4f}")
        print(f"        最佳动作: {action_names[result['best_action']]}")
        print(f"        Epsilon: {result['epsilon']:.4f}")

    if sorted_results[0][1]['q_std'] < 0.5:
        print("\n  ⚠️  警告: 所有模型的Q值标准差都很小，建议重新训练")
    else:
        best_ep = sorted_results[0][0]
        print(f"\n  💡 建议使用: checkpoint_ep{best_ep}.pth")
        print(f"     运行: make run-player 并选择该模型")

    # Q值范围趋势
    print("\n【Q值平均值趋势】")
    print("-" * 100)
    for ep, result in results:
        marker = "→"
        print(f"  {ep:>7,}回合: {marker} {result['q_mean']:+7.2f}")

    print("\n" + "=" * 100)
    print("✅ 诊断完成")
    print("=" * 100)
    print("\n💡 下一步建议:")
    if sorted_results[0][1]['q_std'] >= 0.5:
        best_ep = sorted_results[0][0]
        print(f"  1. 测试最佳模型: make run-player (选择 checkpoint_ep{best_ep}.pth)")
        print(f"  2. 观察效果，如果仍不理想，继续执行步骤3")
        print(f"  3. 修改代码重新训练（参考 FIX_TRAINING_COLLAPSE.md）")
    else:
        print(f"  ❌ 所有模型都已崩溃，必须重新训练:")
        print(f"     1. 修改奖励函数: src/ai_interface.c")
        print(f"     2. 修改训练配置: python/config.py")
        print(f"     3. 重新编译: make clean && make")
        print(f"     4. 开始训练: make train (选择 [0] 创建新模型)")

if __name__ == "__main__":
    main()
