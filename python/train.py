"""
train.py - 主训练脚本（支持GPU加速和两种显示模式）

使用方法:
    python train.py --visualize          # 可视化训练
    python train.py --no-visualize       # 纯文本训练
"""

import argparse
import os
import time
import numpy as np
import torch
from datetime import datetime
import glob

from config import *
from model import DQNAgent
from replay_buffer import ReplayBuffer
from env_wrapper import TankBattleEnv
from enemy_manager import EnemyManager


def select_model_interactively():
    """
    交互式选择模型：新建或继续训练

    Returns:
        str: 模型路径，如果新建则返回None
    """
    print("\n" + "="*60)
    print("模型选择")
    print("="*60)

    # 检查是否有可用的模型
    model_files = []
    if os.path.exists(PATHS['models']):
        model_files = glob.glob(os.path.join(PATHS['models'], '*.pth'))
        model_files.sort(key=os.path.getmtime, reverse=True)  # 按修改时间排序

    # 检查checkpoints目录
    checkpoint_files = []
    if os.path.exists(PATHS['checkpoints']):
        checkpoint_files = glob.glob(os.path.join(PATHS['checkpoints'], '*.pth'))
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    all_files = model_files + checkpoint_files

    if not all_files:
        print("未找到已有模型，将创建新模型")
        return None

    # 只显示最近的5个模型
    all_files = all_files[:5]

    print("\n请选择:")
    print("  [0] 创建新模型")
    print("\n可用的模型 (最近5个):")

    for i, model_path in enumerate(all_files, 1):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"  [{i}] {os.path.basename(model_path)}")
        print(f"      大小: {file_size:.2f} MB | 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "="*60)
    print("提示: 按 Ctrl+C 取消并创建新模型")

    while True:
        try:
            choice = input("请输入选项 (0-%d): " % len(all_files))
            choice = int(choice)

            if choice == 0:
                print("✓ 将创建新模型")
                return None
            elif 1 <= choice <= len(all_files):
                selected_model = all_files[choice - 1]
                print(f"✓ 将从 {os.path.basename(selected_model)} 继续训练")
                return selected_model
            else:
                print("⚠ 无效选项，请重新输入")
        except (ValueError, KeyboardInterrupt):
            print("\n⚠ 已取消，将创建新模型")
            return None
        except EOFError:
            print("\n⚠ 已取消，将创建新模型")
            return None


class Trainer:
    """
    训练器类
    管理整个训练流程
    """

    def __init__(self, visualize: bool = False, continue_from: str = None):
        """
        初始化训练器

        Args:
            visualize: 是否可视化
            continue_from: 继续训练的模型路径
        """
        self.visualize = visualize

        # 创建必要目录
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)

        # 初始化环境
        lib_path = './libtankbattle.so'
        self.env = TankBattleEnv(lib_path,
                                 ENV_CONFIG['map_width'],
                                 ENV_CONFIG['map_height'],
                                 visualize)

        # 初始化DQN智能体
        self.agent = DQNAgent(MODEL_CONFIG['state_dim'],
                             MODEL_CONFIG['action_dim'],
                             MODEL_CONFIG,
                             TRAINING_CONFIG['device'])

        # 加载已有模型（如果提供）
        if continue_from and os.path.exists(continue_from):
            self.agent.load(continue_from)
            print(f"从 {continue_from} 继续训练")

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(TRAINING_CONFIG['buffer_size'],
                                         MODEL_CONFIG['state_dim'])

        # 初始化敌人管理器
        config_dict = {
            'SELF_PLAY_CONFIG': SELF_PLAY_CONFIG,
            'DIFFICULTY_CONFIG': DIFFICULTY_CONFIG,
            'ENV_CONFIG': ENV_CONFIG,
            'PATHS': PATHS
        }
        self.enemy_manager = EnemyManager(config_dict)

        # 统计信息
        self.episode = 0
        self.total_steps = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.best_reward = -float('inf')

        # 训练日志
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

        print("\n" + "="*60)
        print("坦克大战 AI 训练系统")
        print("="*60)
        print(f"设备: {self.agent.device}")
        print(f"可视化: {visualize}")
        print(f"GPU加速: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("="*60 + "\n")

    def train_step(self):
        """
        执行一次训练步骤（从经验回放中采样并训练）
        """
        if not self.replay_buffer.is_ready(TRAINING_CONFIG['min_buffer_size']):
            return None

        # 从缓冲区采样批数据
        batch = self.replay_buffer.sample(TRAINING_CONFIG['batch_size'])

        # 训练（利用GPU加速）
        loss = self.agent.train(batch)

        return loss

    def run_episode(self):
        """
        运行一个训练回合
        """
        self.episode += 1

        # 重置环境
        enemy_count = self.enemy_manager.current_enemy_count
        state = self.env.reset(enemy_count)

        episode_reward = 0
        episode_steps = 0
        done = False

        # 回合开始时间
        start_time = time.time()

        while not done and episode_steps < TRAINING_CONFIG['max_steps_per_episode']:
            # 选择动作
            action = self.agent.select_action(state, training=True)

            # 执行动作
            next_state, reward, done, info = self.env.step(action)

            # 存储经验
            self.replay_buffer.push(state, action, reward, next_state, done)

            # 训练
            loss = self.train_step()
            if loss is not None:
                self.training_losses.append(loss)

            # 渲染（如果启用可视化）
            if self.visualize:
                self.env.render()

            # 更新状态
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

        # 更新epsilon
        self.agent.update_epsilon()

        # 回合结束时间
        elapsed_time = time.time() - start_time

        # 统计胜负
        winner = info.get('winner', -1)
        if winner == 0:
            self.wins += 1
            win = True
        elif winner == 1:
            self.losses += 1
            win = False
        else:
            self.draws += 1
            win = False

        # 更新难度
        new_enemy_count = self.enemy_manager.update_difficulty(win)

        # 记录统计
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)

        # 更新最佳奖励
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward

        # 打印进度（纯文本模式或定期打印）
        if not self.visualize or self.episode % TRAINING_CONFIG['log_interval'] == 0:
            self._print_progress(episode_reward, episode_steps, elapsed_time, winner)

        # 保存模型
        if self.episode % TRAINING_CONFIG['save_interval'] == 0:
            self._save_checkpoint()

        # 保存历史版本（用于自我对弈）
        if self.enemy_manager.should_save_history_version(self.episode):
            self.enemy_manager.save_history_version(self.agent, self.episode)

        return episode_reward

    def _print_progress(self, reward: float, steps: int, elapsed_time: float, winner: int):
        """
        打印训练进度（文本模式）

        Args:
            reward: 回合奖励
            steps: 回合步数
            elapsed_time: 回合耗时
            winner: 胜者
        """
        # 计算平均值
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        avg_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0

        # 胜率
        total_games = self.wins + self.losses + self.draws
        win_rate = self.wins / total_games * 100 if total_games > 0 else 0

        # 敌人管理器统计
        enemy_stats = self.enemy_manager.get_stats()

        # 结果标记
        result_str = "WIN" if winner == 0 else "LOSS" if winner == 1 else "DRAW"
        result_color = "\033[92m" if winner == 0 else "\033[91m" if winner == 1 else "\033[93m"
        reset_color = "\033[0m"

        print(f"\n{'='*80}")
        print(f"回合 {self.episode:5d} | {result_color}{result_str:4s}{reset_color} | "
              f"奖励: {reward:7.2f} | 步数: {steps:4d} | 耗时: {elapsed_time:.2f}s")
        print(f"{'='*80}")
        print(f"  平均奖励 (100回合): {avg_reward:7.2f} | 最佳奖励: {self.best_reward:7.2f}")
        print(f"  平均损失 (100步):   {avg_loss:7.4f}")
        print(f"  探索率 (Epsilon):   {self.agent.epsilon:.4f}")
        print(f"  训练步数:           {self.total_steps:7d}")
        print(f"  经验缓冲区:         {len(self.replay_buffer):7d} / {TRAINING_CONFIG['buffer_size']}")
        print(f"  胜/负/平:           {self.wins:4d} / {self.losses:4d} / {self.draws:4d} "
              f"(胜率: {win_rate:.1f}%)")
        print(f"  当前敌人数量:       {enemy_stats['current_enemy_count']}")
        print(f"  连胜次数:           {enemy_stats['consecutive_wins']}")
        print(f"  历史版本数:         {enemy_stats['history_versions']}")
        print(f"{'='*80}\n")

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = os.path.join(PATHS['checkpoints'],
                                      f'checkpoint_ep{self.episode}.pth')
        self.agent.save(checkpoint_path)

        # 同时保存最新模型
        latest_path = os.path.join(PATHS['models'], 'latest_model.pth')
        self.agent.save(latest_path)

    def train(self):
        """
        主训练循环
        """
        print("开始训练...")
        print(f"最大回合数: {TRAINING_CONFIG['max_episodes']}")
        print(f"显示模式: {'可视化' if self.visualize else '纯文本'}\n")

        try:
            for episode in range(TRAINING_CONFIG['max_episodes']):
                self.run_episode()

        except KeyboardInterrupt:
            print("\n\n训练被用户中断")

        finally:
            # 保存最终模型
            final_path = os.path.join(PATHS['models'], 'final_model.pth')
            self.agent.save(final_path)

            # 打印最终统计
            self._print_final_stats()

            # 清理
            self.env.close()

    def _print_final_stats(self):
        """打印最终统计信息"""
        print("\n" + "="*80)
        print("训练完成!")
        print("="*80)
        print(f"总回合数:     {self.episode}")
        print(f"总步数:       {self.total_steps}")
        print(f"胜/负/平:     {self.wins} / {self.losses} / {self.draws}")
        print(f"最终胜率:     {self.wins / (self.wins + self.losses + self.draws) * 100:.2f}%")
        print(f"最佳奖励:     {self.best_reward:.2f}")
        print(f"平均奖励:     {np.mean(self.episode_rewards):.2f}")
        print("="*80 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='坦克大战AI训练')
    parser.add_argument('--visualize', action='store_true',
                       help='启用可视化训练模式')
    parser.add_argument('--no-visualize', action='store_true',
                       help='使用纯文本训练模式（默认）')
    args = parser.parse_args()

    # 确定是否可视化
    visualize = args.visualize and not args.no_visualize

    # 交互式选择模型
    continue_from = select_model_interactively()

    # 创建训练器并开始训练
    trainer = Trainer(visualize=visualize, continue_from=continue_from)
    trainer.train()


if __name__ == '__main__':
    main()
