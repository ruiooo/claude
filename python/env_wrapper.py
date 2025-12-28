"""
env_wrapper.py - 游戏环境包装器（通过ctypes调用C库）
=========================================================

【核心功能】
将C语言编写的游戏引擎包装成Python可调用的标准RL环境接口。

【技术架构】
Python层 (env_wrapper.py)
    ↓ ctypes外部函数接口
C共享库 (libtankbattle.so)
    ↓ 游戏引擎
游戏状态 (C结构体)

【为什么需要包装?】
1. 性能: C引擎比Pure Python快10-100倍
2. 分离: 游戏逻辑与训练逻辑解耦
3. 复用: C引擎可用于玩家对战、多人模式
4. 训练: Python生态丰富，方便使用PyTorch训练

【OpenAI Gym风格接口】
- reset(): 重置环境，返回初始状态
- step(action): 执行动作，返回(状态, 奖励, 完成, 信息)
- render(): 可视化当前帧
- close(): 清理资源

【ctypes技术说明】
ctypes是Python的外部函数库接口：
- 允许Python调用C/C++动态库(.so/.dll)
- 无需修改C代码或编写扩展模块
- 需要手动定义函数签名（参数类型、返回类型）
"""

import ctypes  # Python的C库调用接口
import numpy as np  # 数值数组操作
import os  # 文件路径操作


class TankBattleEnv:
    """
    坦克大战环境包装器

    功能:
    - 通过ctypes加载C共享库(libtankbattle.so)
    - 提供标准的环境接口（reset/step/render）
    - 管理状态观测和动作执行

    使用示例:
        env = TankBattleEnv("./libtankbattle.so", visualize=False)
        state = env.reset(enemy_count=2)
        next_state, reward, done, info = env.step(action=1)
        env.close()
    """

    def __init__(self, lib_path: str, width: int = 800, height: int = 600, visualize: bool = False, disable_vsync: bool = False):
        """
        初始化游戏环境

        初始化流程:
        1. 检查共享库文件是否存在
        2. 使用ctypes加载C共享库
        3. 定义所有C函数的签名（参数类型和返回类型）
        4. 调用C函数初始化游戏引擎
        5. 设置状态和动作空间维度

        Args:
            lib_path: C共享库路径
                - 通常是 "./libtankbattle.so" (Linux)
                - 或 "./libtankbattle.dll" (Windows)
                - 由Makefile编译生成
            width: 地图宽度（像素）
                - 默认800: 标准游戏地图宽度
                - 必须与config.py中的配置一致
            height: 地图高度（像素）
                - 默认600: 标准游戏地图高度
                - 必须与config.py中的配置一致
            visualize: 是否开启可视化
                - False: 纯后台运行，速度快（训练时）
                - True: 显示SDL窗口，速度慢（调试/录像时）
            disable_vsync: 是否禁用垂直同步
                - False: 启用VSYNC，限制60fps（训练/对战时）
                - True: 禁用VSYNC，支持任意帧率（回放时）
        """
        # 保存配置参数
        self.lib_path = lib_path
        self.width = width
        self.height = height
        self.visualize = visualize
        self.disable_vsync = disable_vsync

        # ========== 加载C共享库 ==========

        # 检查共享库文件是否存在
        # 如果文件不存在，抛出异常（避免后续难以调试的错误）
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"找不到共享库: {lib_path}")

        # 使用ctypes加载共享库
        # CDLL: 加载C动态链接库
        # - Windows: .dll文件
        # - Linux: .so文件
        # - macOS: .dylib文件
        #
        # 加载后，self.lib包含所有导出的C函数
        # 例: self.lib.ai_init_env, self.lib.ai_step 等
        self.lib = ctypes.CDLL(lib_path)

        # ========== 定义C函数接口 ==========

        # 必须先定义函数签名，否则Python不知道如何传参和接收返回值
        # C语言是强类型的，Python需要明确告诉ctypes每个参数的类型
        self._setup_function_signatures()

        # ========== 初始化游戏引擎 ==========

        # 调用C函数 ai_init_env() 初始化游戏
        # 参数:
        # - width: 地图宽度
        # - height: 地图高度
        # - visualize: 1=开启可视化, 0=关闭可视化
        # - disable_vsync: 1=禁用垂直同步, 0=启用垂直同步
        #
        # C函数签名: void ai_init_env(int width, int height, int visualize, int disable_vsync);
        self.lib.ai_init_env(width, height, 1 if visualize else 0, 1 if disable_vsync else 0)

        # ========== 状态空间和动作空间配置 ==========

        # 状态空间维度: 47维（v2.0升级版）
        # 这个值必须与C代码的状态生成函数匹配
        # 组成: 6维AI + 25维敌人(5×5) + 12维子弹(3×4) + 4维战略信息
        self.state_dim = 47

        # 动作空间维度: 9个动作
        # 0: 静止, 1-4: 移动, 5-8: 射击
        self.action_dim = 9

        # 打印初始化信息
        print(f"环境初始化完成 - 尺寸: {width}x{height}, 可视化: {visualize}")

    def _setup_function_signatures(self):
        """
        设置C函数签名（定义参数类型和返回类型）

        为什么需要这一步?
        - Python是动态类型，C是静态类型
        - ctypes需要知道如何打包Python数据传给C
        - 如果类型不匹配，会导致崩溃或错误结果

        函数签名格式:
        - argtypes: 参数类型列表（从左到右）
        - restype: 返回值类型

        常用ctypes类型:
        - ctypes.c_int: C的int类型（32位整数）
        - ctypes.c_float: C的float类型（32位浮点）
        - ctypes.POINTER(T): 指向类型T的指针
        - None: void类型（无返回值）
        """

        # ========== ai_init_env: 初始化环境 ==========

        # C函数原型: void ai_init_env(int width, int height, int visualize, int disable_vsync);
        # 参数:
        # - width: 地图宽度（像素）
        # - height: 地图高度（像素）
        # - visualize: 是否可视化（1=是, 0=否）
        # - disable_vsync: 是否禁用垂直同步（1=是, 0=否）
        # 返回值: 无
        self.lib.ai_init_env.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ai_init_env.restype = None

        # ========== ai_reset_env: 重置环境 ==========

        # C函数原型: void ai_reset_env(int enemy_count, float* obs, int* obs_size);
        # 参数:
        # - enemy_count: 敌人数量
        # - obs: 输出参数，存储观测状态的数组（指针）
        # - obs_size: 输出参数，存储观测状态的实际大小（指针）
        # 返回值: 无
        #
        # POINTER(c_float): float*指针
        # POINTER(c_int): int*指针
        self.lib.ai_reset_env.argtypes = [ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_int)]
        self.lib.ai_reset_env.restype = None

        # ========== ai_step: 执行一步 ==========

        # C函数原型:
        # void ai_step(int action, float* obs, int* obs_size,
        #              float* reward, int* done, int* info);
        #
        # 参数:
        # - action: 动作索引（0-8）
        # - obs: 输出，下一状态
        # - obs_size: 输出，状态大小
        # - reward: 输出，奖励值
        # - done: 输出，是否结束（1=是, 0=否）
        # - info: 输出，额外信息数组（winner, health, kills）
        # 返回值: 无
        self.lib.ai_step.argtypes = [ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_int)]
        self.lib.ai_step.restype = None

        # ========== ai_render: 渲染当前帧 ==========

        # C函数原型: void ai_render();
        # 参数: 无
        # 返回值: 无
        # 作用: 如果可视化开启，渲染SDL窗口
        self.lib.ai_render.argtypes = []
        self.lib.ai_render.restype = None

        # ========== ai_add_enemy: 添加敌人 ==========

        # C函数原型: void ai_add_enemy(int enemy_type, int model_version);
        # 参数:
        # - enemy_type: 敌人类型（1=ENEMY普通AI, 2=SELF_PLAY历史版本）
        # - model_version: 模型版本号（仅SELF_PLAY时有效）
        # 返回值: 无
        self.lib.ai_add_enemy.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.ai_add_enemy.restype = None

        # ========== ai_cleanup: 清理资源 ==========

        # C函数原型: void ai_cleanup();
        # 参数: 无
        # 返回值: 无
        # 作用: 释放SDL资源，关闭窗口
        self.lib.ai_cleanup.argtypes = []
        self.lib.ai_cleanup.restype = None

        # ========== ai_set_difficulty: 设置敌人难度 ==========

        # C函数原型: void ai_set_difficulty(int level);
        # 参数:
        # - level: 难度级别 (0-3)
        #   0 = 假人模式（静止靶，不移动不射击）
        #   1 = 简单模式（随机移动，偶尔射击）
        #   2 = 中等模式（追踪移动，简单射击，不躲避）
        #   3 = 困难模式（完整AI：预判射击+躲避+定位）
        self.lib.ai_set_difficulty.argtypes = [ctypes.c_int]
        self.lib.ai_set_difficulty.restype = None

        # ========== ai_get_difficulty: 获取当前难度 ==========
        self.lib.ai_get_difficulty.argtypes = []
        self.lib.ai_get_difficulty.restype = ctypes.c_int

        # ========== ai_set_seed: 设置随机种子 ==========
        # C函数原型: void ai_set_seed(unsigned int seed);
        # 参数:
        # - seed: 随机种子值
        # 作用: 设置C rand()的随机种子，确保环境初始状态可复现
        self.lib.ai_set_seed.argtypes = [ctypes.c_uint]
        self.lib.ai_set_seed.restype = None

        # ========== ai_poll_key: 检测SDL按键事件 ==========
        # C函数原型: int ai_poll_key();
        # 返回值:
        # - 0: 无按键
        # - 32: 空格键（跳过当前局）
        # - 27: ESC键（退出）
        # - -1: 窗口关闭事件
        self.lib.ai_poll_key.argtypes = []
        self.lib.ai_poll_key.restype = ctypes.c_int

    def set_seed(self, seed: int):
        """
        设置随机种子（用于轨迹回放）

        用途：
        - 轨迹回放：使用相同种子还原训练时的环境初始状态
        - 确保回放时敌人位置和训练时一致
        """
        self.lib.ai_set_seed(seed)

    def poll_key(self) -> int:
        """
        检测SDL按键事件（用于轨迹回放控制）

        Returns:
            按键代码:
            - 0: 无按键
            - 32: 空格键（跳过当前局）
            - 27: ESC键（退出）
            - -1: 窗口关闭事件
        """
        return self.lib.ai_poll_key()

    def set_difficulty(self, level: int):
        """
        设置敌人AI难度级别（课程学习核心）

        难度说明:
        - 0: 假人模式 - 敌人静止不动，不射击（练习射击）
        - 1: 简单模式 - 敌人随机移动，偶尔射击（练习追踪）
        - 2: 中等模式 - 敌人追踪AI，简单射击（练习躲避）
        - 3: 困难模式 - 完整AI，预判射击+躲避（最终挑战）

        建议训练流程:
        1. 难度0训练2000回合（学会射击）
        2. 难度1训练5000回合（学会追踪和射击）
        3. 难度2训练10000回合（学会躲避和战术）
        4. 难度3训练10000+回合（精炼策略）
        """
        self.lib.ai_set_difficulty(level)

    def get_difficulty(self) -> int:
        """获取当前难度级别"""
        return self.lib.ai_get_difficulty()

    def reset(self, enemy_count: int = 2, seed: int = None) -> np.ndarray:
        """
        重置环境（开始新的一局游戏）

        重置流程:
        1. （可选）设置随机种子
        2. 清除所有实体（坦克、子弹）
        3. 重置AI坦克到初始位置
        4. 根据enemy_count生成敌人
        5. 返回初始状态观测

        为什么需要reset?
        - 每局游戏结束后需要重新开始
        - 训练中每个episode都从reset开始
        - 保证每局游戏的初始条件相同

        Args:
            enemy_count: 敌人数量
                - 范围: 1-5（状态表示支持最多5个敌人）
                - 课程学习: 从1个开始，逐步增加
                - 默认: 2个（适中难度）
            seed: 随机种子（可选）
                - 用于轨迹回放，确保环境初始状态一致
                - 默认: None（使用系统随机种子）

        Returns:
            初始状态向量
                - shape: [43] 或 [state_dim]
                - dtype: float32
                - 归一化: 已归一化到合理范围
                - 内容: AI状态 + 敌人状态 + 子弹状态

        示例:
            state = env.reset(enemy_count=1)  # 1个敌人的简单模式
            state = env.reset(enemy_count=5)  # 5个敌人的困难模式
            state = env.reset(enemy_count=1, seed=12345)  # 使用固定种子（回放用）
        """
        # 如果指定了种子，设置随机种子
        if seed is not None:
            self.lib.ai_set_seed(seed)

        # 创建观测数组（C代码会填充这个数组）
        # zeros: 初始化为全0
        # state_dim: 数组长度=43
        # dtype=float32: 与C的float类型匹配
        obs = np.zeros(self.state_dim, dtype=np.float32)

        # 创建观测大小变量（C代码会设置实际大小）
        # c_int(0): 创建一个C int类型，初始值为0
        obs_size = ctypes.c_int(0)

        # 调用C函数重置环境
        # obs.ctypes.data_as(...): 将numpy数组转为C指针
        # ctypes.byref(obs_size): 获取obs_size的地址（相当于&obs_size）
        self.lib.ai_reset_env(enemy_count,
                             obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                             ctypes.byref(obs_size))

        # 返回实际填充的状态
        # obs_size.value: 读取C int的值（Python int）
        # obs[:obs_size.value]: 只返回有效部分（通常是全部43维）
        #
        # 为什么要切片?
        # - 防止C代码填充不足state_dim的情况
        # - obs_size.value通常等于state_dim，但更安全
        return obs[:obs_size.value]

    def step(self, action: int):
        """
        执行一步动作

        交互流程:
        1. AI选择动作（0-8）
        2. 调用C引擎执行动作
        3. 更新游戏状态（坦克移动、子弹发射、碰撞检测）
        4. 计算奖励
        5. 检查是否结束（胜利/失败/超时）
        6. 返回下一状态、奖励、完成标志、额外信息

        奖励计算（在C代码中 src/ai_interface.c):
        - 存活: +0.02/帧
        - 受伤: -15.0
        - 击杀: +100.0
        - 胜利: +300.0
        - 死亡: -100.0
        - 奖励塑形: 距离、瞄准、射击、躲避等

        Args:
            action: 动作索引（0-8）
                - 0: ACTION_IDLE (静止)
                - 1: ACTION_MOVE_UP (向上移动)
                - 2: ACTION_MOVE_DOWN (向下移动)
                - 3: ACTION_MOVE_LEFT (向左移动)
                - 4: ACTION_MOVE_RIGHT (向右移动)
                - 5: ACTION_SHOOT_UP (向上射击)
                - 6: ACTION_SHOOT_DOWN (向下射击)
                - 7: ACTION_SHOOT_LEFT (向左射击)
                - 8: ACTION_SHOOT_RIGHT (向右射击)

        Returns:
            tuple: (next_state, reward, done, info)
                - next_state: 下一状态 [43] float32
                - reward: 奖励值 float
                - done: 是否结束 bool
                - info: 额外信息 dict
                    - 'winner': 胜者（0=AI, 1=敌人, -1=未结束）
                    - 'ai_health': AI血量（0-3）
                    - 'kills': 本步击杀数（0-N）

        示例:
            state, reward, done, info = env.step(action=1)  # 向上移动
            if done:
                if info['winner'] == 0:
                    print("AI获胜!")
                else:
                    print("AI失败!")
        """
        # ========== 准备输出缓冲区 ==========

        # 下一状态数组（C代码填充）
        obs = np.zeros(self.state_dim, dtype=np.float32)

        # 状态大小（C代码设置）
        obs_size = ctypes.c_int(0)

        # 奖励值（C代码设置）
        # c_float: 单精度浮点数
        reward = ctypes.c_float(0.0)

        # 完成标志（C代码设置）
        # 0=未结束, 1=已结束
        done = ctypes.c_int(0)

        # 额外信息数组（C代码填充）
        # [winner, ai_health, kills]
        # - winner: 胜者（0=AI, 1=敌人, -1=未结束）
        # - ai_health: AI血量（0-3）
        # - kills: 本步击杀数
        info = np.zeros(3, dtype=np.int32)

        # ========== 调用C函数执行动作 ==========

        # C函数: void ai_step(int action, float* obs, int* obs_size,
        #                      float* reward, int* done, int* info)
        #
        # 执行流程（C代码）:
        # 1. 根据action移动AI坦克或发射子弹
        # 2. 更新所有敌人AI（追踪、射击）
        # 3. 更新所有子弹（飞行、碰撞检测）
        # 4. 检测碰撞（子弹打中坦克）
        # 5. 计算奖励（基于状态变化）
        # 6. 检查游戏结束条件
        # 7. 生成新状态观测
        # 8. 通过指针返回所有数据
        self.lib.ai_step(action,
                        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.byref(obs_size),
                        ctypes.byref(reward),
                        ctypes.byref(done),
                        info.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

        # ========== 处理返回值 ==========

        # 提取下一状态（只取有效部分）
        next_state = obs[:obs_size.value]

        # 提取奖励值（C float → Python float）
        # .value: 读取ctypes变量的实际值
        reward_val = reward.value

        # 提取完成标志（C int → Python bool）
        # bool(done.value): 将0/1转为False/True
        done_val = bool(done.value)

        # 构建信息字典（更易读）
        # 将C数组转为Python字典，方便访问
        info_dict = {
            'winner': info[0],      # 胜者: 0=AI, 1=敌人, -1=未结束
            'ai_health': info[1],   # AI血量: 0-3
            'kills': info[2]        # 击杀数: 本步击杀的敌人数
        }

        # 返回标准的RL环境输出
        # 这个格式与OpenAI Gym兼容
        return next_state, reward_val, done_val, info_dict

    def render(self):
        """
        渲染当前帧（如果启用可视化）

        渲染内容:
        - 地图背景
        - AI坦克（绿色）
        - 敌人坦克（红色）
        - 子弹（黄色小圆点）
        - 血量条
        - 统计信息（回合数、分数等）

        为什么要render?
        - 调试: 观察AI行为是否符合预期
        - 录制: 生成演示视频
        - 演示: 展示训练结果
        - 理解: 帮助人类理解AI策略

        性能影响:
        - 无可视化: 每秒可执行1000-10000步
        - 有可视化: 每秒只能执行30-60步（受SDL限制）
        - 建议: 训练时关闭，测试时开启

        使用示例:
            env = TankBattleEnv(lib_path, visualize=True)
            for step in range(1000):
                action = agent.select_action(state)
                state, reward, done, info = env.step(action)
                env.render()  # 显示当前帧
                if done:
                    break
        """
        # 只有在初始化时开启可视化才渲染
        # 避免无意义的函数调用
        if self.visualize:
            # 调用C函数渲染SDL窗口
            # C函数内部会:
            # 1. 清空窗口
            # 2. 绘制地图背景
            # 3. 绘制所有实体（坦克、子弹）
            # 4. 更新显示
            # 5. 处理SDL事件（关闭窗口等）
            self.lib.ai_render()

    def add_enemy(self, enemy_type: int, model_version: int = 0):
        """
        动态添加敌人（用于自我对弈和动态难度）

        使用场景:
        1. 自我对弈: 添加使用历史模型的敌人
        2. 动态难度: 根据AI表现动态增加敌人数量
        3. 混合训练: 同时使用不同类型的敌人

        敌人类型:
        - TANK_TYPE_ENEMY (1): 传统追踪AI
            - 简单的追击逻辑
            - 靠近AI后射击
            - 难度固定
        - TANK_TYPE_SELF_PLAY (2): 使用历史模型的AI
            - 加载过去训练的模型
            - 策略与当前AI类似
            - 难度动态变化

        Args:
            enemy_type: 敌人类型
                - 1: TANK_TYPE_ENEMY (传统AI)
                - 2: TANK_TYPE_SELF_PLAY (历史版本AI)
            model_version: 模型版本号（仅enemy_type=2时使用）
                - 0: 不使用模型（默认）
                - 1-N: 历史版本号

        示例:
            # 添加传统AI敌人
            env.add_enemy(enemy_type=1, model_version=0)

            # 添加历史版本AI敌人
            env.add_enemy(enemy_type=2, model_version=5)
        """
        # 调用C函数添加敌人
        # C代码会:
        # 1. 创建新的Tank实体
        # 2. 设置类型和版本
        # 3. 随机初始位置（远离AI）
        # 4. 添加到游戏状态中
        self.lib.ai_add_enemy(enemy_type, model_version)

    def close(self):
        """
        关闭环境，释放资源

        清理内容:
        - 关闭SDL窗口
        - 释放SDL渲染器
        - 释放SDL纹理
        - 释放字体资源

        何时调用?
        - 训练完成后
        - 程序退出前
        - 不再使用环境时

        为什么重要?
        - 防止内存泄漏
        - 释放GPU资源
        - 关闭窗口（避免僵尸窗口）

        使用示例:
            env = TankBattleEnv(lib_path, visualize=True)
            # ... 使用环境 ...
            env.close()  # 清理资源
        """
        # 调用C函数清理资源
        # C函数内部会调用SDL_DestroyRenderer等函数
        self.lib.ai_cleanup()

    def __del__(self):
        """
        析构函数（对象销毁时自动调用）

        Python垃圾回收机制:
        - 当对象引用计数为0时，会调用__del__
        - 不保证何时调用（可能延迟）
        - 不应依赖__del__做关键清理（最好显式调用close）

        作用:
        - 兜底清理: 即使忘记调用close()也能清理资源
        - 防止泄漏: 避免长期运行时资源累积

        注意:
        - __del__不保证一定执行（如程序崩溃时）
        - 推荐显式调用close()，不要完全依赖__del__
        """
        # 检查是否有lib属性（避免初始化失败时报错）
        # hasattr: 检查对象是否有某个属性
        if hasattr(self, 'lib'):
            # 显式调用close()清理资源
            self.close()
