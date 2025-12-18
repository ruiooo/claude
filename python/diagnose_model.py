#!/usr/bin/env python3
"""
模型Q值诊断工具

用于检查训练后的模型为什么一直选择IDLE动作
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DQNAgent
from config import MODEL_CONFIG, ENV_CONFIG

def analyze_q_values(model_path: str = "saved_models/final_model.pth"):
    """分析模型的Q值分布"""

    print("=" * 80)
    print("🔍 DQN模型Q值诊断工具")
    print("=" * 80)

    # 1. 加载模型
    print(f"\n📂 加载模型: {model_path}")

    # 设置设备
    device = MODEL_CONFIG.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - 使用设备: {device}")

    agent = DQNAgent(
        state_dim=MODEL_CONFIG['state_dim'],
        action_dim=MODEL_CONFIG['action_dim'],
        config=MODEL_CONFIG,
        device=device
    )

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    agent.load(model_path)
    print(f"✅ 模型加载成功")
    print(f"   - 训练步数: {agent.train_step}")
    print(f"   - Epsilon: {agent.epsilon:.4f}")

    # 2. 构造测试状态
    print("\n" + "=" * 80)
    print("🧪 测试场景")
    print("=" * 80)

    # 场景1：典型游戏开局状态
    # AI坦克在中央(0.5, 0.5)，敌人在右上角(0.3, 0.3)
    test_state = np.zeros(43, dtype=np.float32)
    test_state[0] = 0.5  # AI x位置 (中央)
    test_state[1] = 0.5  # AI y位置 (中央)
    test_state[2] = 0.0  # AI vx (静止)
    test_state[3] = 0.0  # AI vy (静止)
    test_state[4] = 1.0  # AI 血量满
    test_state[5] = 0.0  # AI 可以射击
    # 第一个敌人
    test_state[6] = -0.2  # 敌人相对x (在左侧)
    test_state[7] = -0.2  # 敌人相对y (在上方)
    test_state[8] = 0.0   # 敌人vx
    test_state[9] = 0.0   # 敌人vy
    test_state[10] = 1.0  # 敌人血量满

    print("\n场景1: 敌人在左上方")
    print(f"  AI位置: (0.5, 0.5) - 地图中央")
    print(f"  AI状态: 静止, 满血, 可射击")
    print(f"  敌人相对位置: (-0.2, -0.2) - 左上方")
    print(f"  敌人状态: 静止, 满血")

    # 3. 获取Q值
    with torch.no_grad():
        state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(agent.device)
        q_values = agent.policy_net(state_tensor)[0].cpu().numpy()

    # 4. 分析Q值
    print("\n" + "-" * 80)
    print("📊 Q值分析")
    print("-" * 80)

    action_names = [
        "0:IDLE(静止)",
        "1:MOVE_UP(上移)",
        "2:MOVE_DOWN(下移)",
        "3:MOVE_LEFT(左移)",
        "4:MOVE_RIGHT(右移)",
        "5:SHOOT_UP(上射)",
        "6:SHOOT_DOWN(下射)",
        "7:SHOOT_LEFT(左射)",
        "8:SHOOT_RIGHT(右射)"
    ]

    print("\n动作Q值排序:")
    sorted_indices = np.argsort(q_values)[::-1]  # 从高到低
    for rank, idx in enumerate(sorted_indices, 1):
        marker = "⭐" if rank == 1 else "  "
        print(f"{marker} [{rank}] {action_names[idx]}: Q={q_values[idx]:+.4f}")

    best_action = np.argmax(q_values)
    print(f"\n最佳动作: {action_names[best_action]} (Q={q_values[best_action]:+.4f})")

    # 5. Q值统计
    print("\n" + "-" * 80)
    print("📈 Q值统计")
    print("-" * 80)
    print(f"  最大Q值: {q_values.max():+.4f}")
    print(f"  最小Q值: {q_values.min():+.4f}")
    print(f"  平均Q值: {q_values.mean():+.4f}")
    print(f"  Q值标准差: {q_values.std():.4f}")
    print(f"  Q值范围: {q_values.max() - q_values.min():.4f}")

    # 6. 问题诊断
    print("\n" + "=" * 80)
    print("🚨 问题诊断")
    print("=" * 80)

    issues = []

    # 检查1: IDLE是否Q值最高
    if best_action == 0:
        issues.append("❌ 模型倾向于选择IDLE（静止不动）")
        issues.append("   原因可能：")
        issues.append("   - 奖励函数设计问题（存活奖励过高）")
        issues.append("   - 训练策略崩溃（陷入局部最优）")
        issues.append("   - Epsilon过早衰减（缺乏探索）")
    else:
        issues.append(f"✅ 模型正常选择动作: {action_names[best_action]}")

    # 检查2: Q值是否都很接近
    if q_values.std() < 0.1:
        issues.append("❌ 所有动作Q值非常接近（标准差<0.1）")
        issues.append("   说明模型未能区分不同动作的价值")

    # 检查3: Q值是否异常（全正或全负）
    if q_values.min() > 0:
        issues.append("⚠️  所有Q值都是正数")
        issues.append("   可能过度估计奖励")
    elif q_values.max() < 0:
        issues.append("⚠️  所有Q值都是负数")
        issues.append("   可能过度惩罚（或未学到有效策略）")

    # 检查4: 射击动作Q值
    shoot_q_values = q_values[5:9]  # 射击动作的Q值
    if shoot_q_values.max() < q_values[0]:  # 如果最好的射击动作不如IDLE
        issues.append("❌ 射击动作Q值都低于IDLE")
        issues.append("   模型认为射击不如静止")

    for issue in issues:
        print(issue)

    # 7. 修复建议
    print("\n" + "=" * 80)
    print("💡 修复建议")
    print("=" * 80)

    if best_action == 0:
        print("\n【问题】模型一直选择IDLE，训练可能已崩溃\n")
        print("【建议方案A】从头重新训练（推荐）:")
        print("  1. 降低存活奖励: 0.02 → 0.005 (src/ai_interface.c)")
        print("  2. 增加动作奖励: 移动/射击时给予小奖励(+0.05)")
        print("  3. 使用更小网络: [512,512,256,128] → [256,256,128]")
        print("  4. 更慢epsilon衰减: 0.9999 → 0.99995")
        print("  5. make clean && make && make train")
        print("  6. 选择 [0] 创建新模型\n")

        print("【建议方案B】检查训练曲线:")
        print("  1. 查看saved_models/checkpoints/目录")
        print("  2. 加载早期模型测试(如checkpoint_ep50000.pth)")
        print("  3. 找到胜率最高的checkpoint")
        print("  4. 可能早期模型比final_model更好\n")

        print("【建议方案C】调整奖励函数:")
        print("  编辑 src/ai_interface.c:")
        print("  - 减少存活奖励: reward += 0.005f  (原0.02)")
        print("  - 增加移动奖励: if(action != IDLE) reward += 0.05f")
        print("  - 增加射击奖励: if(action >= 5) reward += 0.05f")
        print("  然后: make clean && make && make train\n")

    # 8. 测试更多场景
    print("\n" + "=" * 80)
    print("🎮 测试更多场景")
    print("=" * 80)

    scenarios = [
        {
            'name': "场景2: 敌人在正前方",
            'enemy_dx': 0.0,
            'enemy_dy': -0.3,
            'expected': "SHOOT_UP或MOVE_UP"
        },
        {
            'name': "场景3: 敌人在正右方",
            'enemy_dx': 0.3,
            'enemy_dy': 0.0,
            'expected': "SHOOT_RIGHT或MOVE_RIGHT"
        },
        {
            'name': "场景4: 被子弹追击",
            'enemy_dx': -0.2,
            'enemy_dy': 0.0,
            'bullet_dx': 0.1,
            'bullet_dy': 0.0,
            'expected': "MOVE躲避"
        }
    ]

    for scenario in scenarios[:2]:  # 只测试前2个场景，节省时间
        test_state = np.zeros(43, dtype=np.float32)
        test_state[0] = 0.5  # AI位置
        test_state[1] = 0.5
        test_state[4] = 1.0  # 满血
        test_state[5] = 0.0  # 可射击
        test_state[6] = scenario['enemy_dx']
        test_state[7] = scenario['enemy_dy']
        test_state[10] = 1.0  # 敌人满血

        with torch.no_grad():
            state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor)[0].cpu().numpy()

        best_action = np.argmax(q_values)
        print(f"\n{scenario['name']}")
        print(f"  预期: {scenario['expected']}")
        print(f"  实际: {action_names[best_action]} (Q={q_values[best_action]:+.4f})")

        if best_action == 0:
            print("  ❌ 仍然选择IDLE，模型确实有问题！")
        else:
            print("  ✅ 选择了有意义的动作")

    print("\n" + "=" * 80)
    print("✅ 诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='诊断DQN模型Q值')
    parser.add_argument('--model', type=str, default='saved_models/final_model.pth',
                       help='模型文件路径')
    args = parser.parse_args()

    analyze_q_values(args.model)
