"""
play_vs_ai.py - 玩家对战AI模式

玩家通过键盘控制自己的坦克，对战训练好的AI坦克

使用方法:
    python python/play_vs_ai.py                          # 对战1个AI（使用最新模型）
    python python/play_vs_ai.py --enemies 3              # 对战3个AI
    python python/play_vs_ai.py --model model.pth        # 使用指定模型

注意: 此模式需要安装pygame来处理键盘输入
    pip install pygame
"""

import argparse
import os
import time
import numpy as np
import torch

from config import *
from model import DQNAgent
from env_wrapper import TankBattleEnv

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("警告: 未安装pygame，将使用自动AI对战模式")
    print("要启用玩家控制，请运行: pip install pygame")


class PlayerVsAI:
    """
    玩家对战AI模式
    """

    def __init__(self, model_path: str = None, enemy_count: int = 1):
        """
        初始化玩家对战AI模式

        Args:
            model_path: AI模型路径
            enemy_count: AI敌人数量
        """
        self.enemy_count = enemy_count
        self.use_player_control = HAS_PYGAME

        # 加载AI模型（用于敌人）
        self.agent = DQNAgent(MODEL_CONFIG['state_dim'],
                             MODEL_CONFIG['action_dim'],
                             MODEL_CONFIG,
                             TRAINING_CONFIG['device'])

        # 确定模型路径
        if model_path is None:
            model_path = os.path.join(PATHS['models'], 'latest_model.pth')

        if not os.path.exists(model_path):
            print(f"警告: 找不到模型文件 {model_path}")
            print("AI将使用未训练的模型（随机动作）")
        else:
            self.agent.load(model_path)
            print(f"已加载AI模型: {model_path}")

        # 初始化环境（可视化模式）
        lib_path = './libtankbattle.so'
        self.env = TankBattleEnv(lib_path,
                                ENV_CONFIG['map_width'],
                                ENV_CONFIG['map_height'],
                                visualize=True)

        # 初始化pygame（仅用于键盘输入）
        if HAS_PYGAME:
            pygame.init()
            # 创建一个小窗口用于捕获键盘事件（不显示，因为SDL已经创建了窗口）
            # 或者我们只使用pygame.key.get_pressed()
            pass

        print("\n" + "="*60)
        print("坦克大战 - 玩家对战AI模式")
        print("="*60)
        print(f"AI敌人数量: {enemy_count}")
        print(f"设备: {self.agent.device}")
        print(f"玩家控制: {'启用' if self.use_player_control else '禁用（AI自动）'}")
        print("\n控制说明:")
        print("  WASD     - 移动")
        print("  方向键   - 射击")
        print("  ESC      - 退出")
        print("="*60 + "\n")

    def get_player_action_from_input(self) -> int:
        """
        从键盘输入获取玩家动作

        Returns:
            动作索引 (0-8)
            0=IDLE, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT,
            5=SHOOT_UP, 6=SHOOT_DOWN, 7=SHOOT_LEFT, 8=SHOOT_RIGHT
        """
        if not HAS_PYGAME:
            # 如果没有pygame，返回IDLE
            return 0

        # 处理pygame事件（必须调用以保持响应）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1  # 特殊值表示退出

        # 获取当前按键状态
        keys = pygame.key.get_pressed()

        # 检查退出
        if keys[pygame.K_ESCAPE]:
            return -1

        # 优先处理射击（方向键）
        if keys[pygame.K_UP]:
            return 5  # SHOOT_UP
        if keys[pygame.K_DOWN]:
            return 6  # SHOOT_DOWN
        if keys[pygame.K_LEFT]:
            return 7  # SHOOT_LEFT
        if keys[pygame.K_RIGHT]:
            return 8  # SHOOT_RIGHT

        # 移动（WASD）
        if keys[pygame.K_w]:
            return 1  # MOVE_UP
        if keys[pygame.K_s]:
            return 2  # MOVE_DOWN
        if keys[pygame.K_a]:
            return 3  # MOVE_LEFT
        if keys[pygame.K_d]:
            return 4  # MOVE_RIGHT

        # 默认空闲
        return 0  # IDLE

    def play(self):
        """
        开始游戏主循环
        """
        print("游戏开始！\n")

        running = True
        game_count = 0

        try:
            while running:
                game_count += 1
                print(f"第 {game_count} 局")

                # 重置环境
                state = self.env.reset(self.enemy_count)

                # 添加AI敌人（使用自我对弈模式）
                # TANK_TYPE_SELF_PLAY = 2
                for i in range(self.enemy_count):
                    self.env.add_enemy(enemy_type=2, model_version=0)

                game_over = False
                total_reward = 0
                steps = 0
                start_time = time.time()

                # 游戏循环
                while not game_over and running:
                    # 获取玩家输入
                    if self.use_player_control:
                        player_action = self.get_player_action_from_input()
                        if player_action == -1:  # 退出信号
                            running = False
                            break
                    else:
                        # 如果没有pygame，让玩家坦克也用AI控制
                        player_action = self.agent.select_action(state, training=False)

                    # 执行动作
                    next_state, reward, done, info = self.env.step(player_action)

                    # 渲染
                    self.env.render()

                    # 更新状态
                    state = next_state
                    total_reward += reward
                    steps += 1
                    game_over = done

                    # 控制帧率
                    time.sleep(1.0 / 60)  # 60 FPS

                # 游戏结束，显示结果
                if game_over:
                    elapsed_time = time.time() - start_time
                    winner = info.get('winner', -1)

                    print("\n" + "="*60)
                    if winner == 0:
                        result = "你赢了！" if self.use_player_control else "玩家方AI赢了！"
                        color = "\033[92m"  # 绿色
                    elif winner == 1:
                        result = "你输了！" if self.use_player_control else "敌方AI赢了！"
                        color = "\033[91m"  # 红色
                    else:
                        result = "平局！"
                        color = "\033[93m"  # 黄色

                    print(f"{color}{result}\033[0m")
                    print(f"总步数: {steps}")
                    print(f"总奖励: {total_reward:.2f}")
                    print(f"耗时: {elapsed_time:.2f}秒")
                    print("="*60 + "\n")

                    # 询问是否继续
                    if self.use_player_control:
                        print("再来一局？(Y/n)")
                        # 简单起见，自动重新开始
                        time.sleep(2)
                    else:
                        # 自动模式，等待一下
                        time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n游戏被中断")

        finally:
            self.env.close()
            if HAS_PYGAME:
                pygame.quit()

            print("\n游戏结束，共进行了 {} 局".format(game_count))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='玩家对战AI模式')
    parser.add_argument('--model', type=str, default=None,
                       help='AI模型路径（默认: saved_models/latest_model.pth）')
    parser.add_argument('--enemies', type=int, default=1,
                       help='AI敌人数量（默认: 1）')
    args = parser.parse_args()

    # 创建游戏并开始
    game = PlayerVsAI(model_path=args.model, enemy_count=args.enemies)
    game.play()


if __name__ == '__main__':
    main()
