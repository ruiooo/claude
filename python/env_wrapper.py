"""
env_wrapper.py - 游戏环境包装器（通过ctypes调用C库）
"""

import ctypes
import numpy as np
import os


class TankBattleEnv:
    """
    坦克大战环境包装器
    通过ctypes调用编译好的C共享库
    """

    def __init__(self, lib_path: str, width: int = 800, height: int = 600, visualize: bool = False):
        """
        初始化环境

        Args:
            lib_path: C共享库路径
            width: 地图宽度
            height: 地图高度
            visualize: 是否可视化
        """
        self.lib_path = lib_path
        self.width = width
        self.height = height
        self.visualize = visualize

        # 加载C共享库
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"找不到共享库: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)

        # 定义C函数接口
        self._setup_function_signatures()

        # 初始化环境
        self.lib.ai_init_env(width, height, 1 if visualize else 0)

        # 状态空间和动作空间
        self.state_dim = 43  # 固定维度
        self.action_dim = 9  # 9个动作

        print(f"环境初始化完成 - 尺寸: {width}x{height}, 可视化: {visualize}")

    def _setup_function_signatures(self):
        """设置C函数签名"""

        # ai_init_env(int width, int height, int visualize)
        self.lib.ai_init_env.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ai_init_env.restype = None

        # ai_reset_env(int enemy_count, float* obs, int* obs_size)
        self.lib.ai_reset_env.argtypes = [ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_int)]
        self.lib.ai_reset_env.restype = None

        # ai_step(int action, float* obs, int* obs_size, float* reward, int* done, int* info)
        self.lib.ai_step.argtypes = [ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_int)]
        self.lib.ai_step.restype = None

        # ai_render()
        self.lib.ai_render.argtypes = []
        self.lib.ai_render.restype = None

        # ai_add_enemy(int enemy_type, int model_version)
        self.lib.ai_add_enemy.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.ai_add_enemy.restype = None

        # ai_cleanup()
        self.lib.ai_cleanup.argtypes = []
        self.lib.ai_cleanup.restype = None

    def reset(self, enemy_count: int = 2) -> np.ndarray:
        """
        重置环境

        Args:
            enemy_count: 敌人数量

        Returns:
            初始状态
        """
        obs = np.zeros(self.state_dim, dtype=np.float32)
        obs_size = ctypes.c_int(0)

        self.lib.ai_reset_env(enemy_count,
                             obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                             ctypes.byref(obs_size))

        return obs[:obs_size.value]

    def step(self, action: int):
        """
        执行一步

        Args:
            action: 动作索引

        Returns:
            (next_state, reward, done, info)
        """
        obs = np.zeros(self.state_dim, dtype=np.float32)
        obs_size = ctypes.c_int(0)
        reward = ctypes.c_float(0.0)
        done = ctypes.c_int(0)
        info = np.zeros(3, dtype=np.int32)  # [winner, ai_health, kills]

        self.lib.ai_step(action,
                        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.byref(obs_size),
                        ctypes.byref(reward),
                        ctypes.byref(done),
                        info.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

        next_state = obs[:obs_size.value]
        reward_val = reward.value
        done_val = bool(done.value)
        info_dict = {
            'winner': info[0],
            'ai_health': info[1],
            'kills': info[2]
        }

        return next_state, reward_val, done_val, info_dict

    def render(self):
        """渲染当前帧（如果启用可视化）"""
        if self.visualize:
            self.lib.ai_render()

    def add_enemy(self, enemy_type: int, model_version: int = 0):
        """
        添加敌人

        Args:
            enemy_type: 敌人类型（1=ENEMY, 2=SELF_PLAY）
            model_version: 模型版本号
        """
        self.lib.ai_add_enemy(enemy_type, model_version)

    def close(self):
        """关闭环境"""
        self.lib.ai_cleanup()

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'lib'):
            self.close()
