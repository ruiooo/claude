"""
play_vs_ai.py - 玩家对战AI模式（使用训练好的模型）

使用方法:
    python play_vs_ai.py                    # 自动选择模型
    python play_vs_ai.py --model path.pth   # 指定模型
"""

import argparse
import os
import time
import numpy as np
import torch
import glob
from datetime import datetime
import pygame

from config import *
from model import DQNAgent
from env_wrapper import TankBattleEnv


def select_ai_model():
    """
    交互式选择AI模型版本

    Returns:
        str: 模型路径
    """
    print("\n" + "="*60)
    print("选择AI模型版本")
    print("="*60)

    # 检查可用的模型
    model_files = []
    if os.path.exists(PATHS['models']):
        model_files = glob.glob(os.path.join(PATHS['models'], '*.pth'))
        model_files.sort(key=os.path.getmtime, reverse=True)

    # 检查checkpoints目录
    checkpoint_files = []
    if os.path.exists(PATHS['checkpoints']):
        checkpoint_files = glob.glob(os.path.join(PATHS['checkpoints'], '*.pth'))
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    all_files = model_files + checkpoint_files

    if not all_files:
        print("⚠ 未找到任何训练模型！")
        print("请先运行训练：python python/train.py")
        return None

    print("\n可用的AI模型:")

    for i, model_path in enumerate(all_files, 1):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"  [{i}] {os.path.basename(model_path)}")
        print(f"      大小: {file_size:.2f} MB | 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 默认使用latest_model.pth
    default_model = None
    for i, model_path in enumerate(all_files, 1):
        if 'latest_model.pth' in model_path:
            default_model = i
            break

    if default_model is None and all_files:
        default_model = 1

    print("\n" + "="*60)

    while True:
        try:
            choice_str = input(f"请选择AI模型 (1-{len(all_files)}, 默认={default_model}): ").strip()

            if not choice_str and default_model:
                choice = default_model
            else:
                choice = int(choice_str)

            if 1 <= choice <= len(all_files):
                selected_model = all_files[choice - 1]
                print(f"✓ 已选择: {os.path.basename(selected_model)}")
                return selected_model
            else:
                print("⚠ 无效选项，请重新输入")
        except (ValueError, KeyboardInterrupt, EOFError):
            if default_model:
                selected_model = all_files[default_model - 1]
                print(f"\n✓ 使用默认模型: {os.path.basename(selected_model)}")
                return selected_model
            else:
                print("\n⚠ 已取消")
                return None


class HumanVsAIGame:
    """
    人类玩家对战AI游戏
    """

    def __init__(self, model_path: str, ai_count: int = 1):
        """
        初始化游戏

        Args:
            model_path: AI模型路径
            ai_count: AI坦克数量
        """
        self.ai_count = ai_count

        # 初始化环境（启用可视化）
        lib_path = './libtankbattle.so'
        self.env = TankBattleEnv(lib_path,
                                ENV_CONFIG['map_width'],
                                ENV_CONFIG['map_height'],
                                visualize=True)

        # 加载AI模型
        self.agent = DQNAgent(MODEL_CONFIG['state_dim'],
                             MODEL_CONFIG['action_dim'],
                             MODEL_CONFIG,
                             TRAINING_CONFIG['device'])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.agent.load(model_path)
        self.agent.eval()  # 设置为评估模式
        print(f"✓ 已加载AI模型: {os.path.basename(model_path)}")
        print(f"  设备: {self.agent.device}")

        # 统计信息
        self.wins = 0
        self.losses = 0
        self.current_round = 1

        print("\n" + "="*60)
        print("玩家 vs AI 对战模式")
        print("="*60)
        print(f"AI模型: {os.path.basename(model_path)}")
        print(f"AI数量: {ai_count}")
        print("\n操作:")
        print("  WASD - 移动")
        print("  空格 - 射击")
        print("  ESC  - 退出")
        print("="*60 + "\n")

    def play(self):
        """
        开始游戏
        """
        running = True

        while running:
            # 开始新回合
            print(f"\n========== 第 {self.current_round} 局 ==========")
            print(f"胜/负: {self.wins}/{self.losses}")

            # 重置环境
            state = self.env.reset(self.ai_count)

            episode_steps = 0
            done = False
            start_time = time.time()

            while not done and episode_steps < 2000:  # 最多2000步
                # 渲染
                self.env.render()

                # AI选择动作
                action = self.agent.select_action(state, training=False)

                # 执行动作
                state, reward, done, info = self.env.step(action)

                episode_steps += 1

                # 控制帧率
                time.sleep(0.016)  # ~60 FPS

            # 回合结束
            elapsed_time = time.time() - start_time
            winner = info.get('winner', -1)

            if winner == 0:
                self.wins += 1
                print(f"✓ 玩家获胜！(用时 {elapsed_time:.1f}s, {episode_steps} 步)")
            elif winner == 1:
                self.losses += 1
                print(f"✗ AI获胜！(用时 {elapsed_time:.1f}s, {episode_steps} 步)")
            else:
                print(f"- 平局 (用时 {elapsed_time:.1f}s, {episode_steps} 步)")

            self.current_round += 1

            # 等待一下，让玩家看结果
            time.sleep(2)

            # 询问是否继续
            print("\n继续下一局？(按ESC退出，其他键继续)")
            # 这里简单处理，自动继续
            # 在实际实现中可以添加更复杂的输入处理

        # 清理
        self.env.close()

        # 打印最终统计
        print("\n" + "="*60)
        print("游戏结束")
        print("="*60)
        print(f"总回合数: {self.current_round - 1}")
        print(f"胜/负: {self.wins}/{self.losses}")
        if self.wins + self.losses > 0:
            win_rate = self.wins / (self.wins + self.losses) * 100
            print(f"胜率: {win_rate:.1f}%")
        print("="*60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='玩家对战AI模式')
    parser.add_argument('--model', type=str, default=None,
                       help='指定AI模型路径')
    parser.add_argument('--ai-count', type=int, default=1,
                       help='AI坦克数量（默认1）')
    args = parser.parse_args()

    # 选择模型
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"⚠ 模型文件不存在: {model_path}")
            return
    else:
        model_path = select_ai_model()
        if not model_path:
            print("未选择模型，退出")
            return

    # 开始游戏
    try:
        game = HumanVsAIGame(model_path, args.ai_count)
        game.play()
    except KeyboardInterrupt:
        print("\n\n游戏被用户中断")
    except Exception as e:
        print(f"\n⚠ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
