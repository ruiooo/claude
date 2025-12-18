/*
 * ai_interface.h - AI训练接口（用于Python通信）
 *
 * 模块说明：
 *   提供C游戏引擎与Python训练系统之间的接口
 *   通过ctypes暴露C函数给Python调用
 *   实现强化学习环境的标准接口（reset, step）
 *
 * 使用场景：
 *   - DQN训练：Python调用C环境进行高性能训练
 *   - 动态难度调整：Python控制敌人数量
 *   - 自我对弈：Python加载历史模型作为敌人
 *
 * 架构：
 *   Python (ctypes) -> ai_interface.c -> game.c -> 游戏逻辑
 *
 * 数据流向：
 *   1. Python调用ai_reset_env -> 重置游戏 -> 返回观察状态
 *   2. Python调用ai_step(action) -> 执行动作 -> 返回(obs, reward, done)
 *   3. 循环执行直到done=true
 *
 * 对应文件：
 *   - C实现: src/ai_interface.c
 *   - Python封装: python/env_wrapper.py
 *   - 训练脚本: python/train.py
 */

#ifndef AI_INTERFACE_H
#define AI_INTERFACE_H

#include "game.h"

/*
 * C++兼容性声明
 * 确保C++编译器使用C链接规范（Python ctypes需要）
 */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * 全局游戏状态
 * 由ai_interface.c定义，供所有接口函数访问
 */
extern GameState g_game;

/*
 * 初始化游戏环境
 * @param width: 地图宽度（像素）
 * @param height: 地图高度（像素）
 * @param visualize: 是否启用可视化（0=纯逻辑，1=显示窗口）
 * 功能：
 *   - 初始化全局游戏状态g_game
 *   - 如果visualize=1，初始化SDL渲染器
 *   - 设置随机种子
 * 注意：整个训练过程只调用一次
 */
void ai_init_env(int width, int height, int visualize);

/*
 * 重置环境为新回合（强化学习reset接口）
 * @param enemy_count: 敌人数量（1-31）
 * @param obs: 输出观察数组（float[43]）
 * @param obs_size: 输出观察维度（应为43）
 * 功能：
 *   - 调用game_reset重置游戏
 *   - 清除所有坦克和子弹
 *   - 生成新的初始状态
 *   - 返回初始观察状态
 * 用途：每个训练回合开始时调用
 */
void ai_reset_env(int enemy_count, float* obs, int* obs_size);

/*
 * 执行一步游戏逻辑（强化学习step接口）
 * @param action: DQN选择的动作（0-8）
 * @param obs: 输出下一个观察状态（float[43]）
 * @param obs_size: 输出观察维度（应为43）
 * @param reward: 输出奖励值
 * @param done: 输出回合结束标志（0=继续，1=结束）
 * @param info: 输出额外信息（当前未使用，保留）
 * 功能：
 *   - 执行AI坦克的动作（game_execute_action）
 *   - 更新游戏状态（game_update）
 *   - 计算奖励（存活、受伤、击杀、胜利/失败）
 *   - 提取新的观察状态
 *   - 检查游戏结束条件
 * 奖励设计：
 *   - 存活: +0.01/帧
 *   - 受伤: -10
 *   - 击杀: +50/敌人
 *   - 胜利: +200
 *   - 失败: -100
 */
void ai_step(int action, float* obs, int* obs_size, float* reward,
             int* done, int* info);

/*
 * 获取当前游戏状态
 * @param obs: 输出观察数组（float[43]）
 * @param obs_size: 输出观察维度（应为43）
 * 功能：调用game_get_observation提取当前状态
 * 用途：调试和状态检查
 */
void ai_get_state(float* obs, int* obs_size);

/*
 * 添加敌人到游戏（动态难度和自我对弈）
 * @param enemy_type: 敌人类型（1=TANK_TYPE_ENEMY, 2=TANK_TYPE_SELF_PLAY）
 * @param model_version: 模型版本号（自我对弈时使用，0=最新）
 * 功能：调用game_add_enemy添加新敌人
 * 用途：
 *   - 动态难度：连续胜利后增加敌人数量
 *   - 自我对弈：加载历史版本模型作为对手
 */
void ai_add_enemy(int enemy_type, int model_version);

/*
 * 获取统计信息
 * @param ai_health: 输出AI坦克当前血量
 * @param enemy_count: 输出存活敌人数量
 * @param frame_count: 输出当前帧数
 * 功能：提取游戏状态的统计数据
 * 用途：训练日志、调试、监控
 */
void ai_get_stats(int* ai_health, int* enemy_count, int* frame_count);

/*
 * 计算奖励值（完整的奖励塑形系统）
 * @param game: 游戏状态
 * @param ai_tank: AI坦克（或玩家坦克）
 * @param prev_health: 上一帧血量（用于检测受伤）
 * @param prev_enemies: 上一帧敌人数（用于检测击杀）
 * @return 该步的总奖励值
 * 功能：
 *   - 基础奖励：存活(+0.02)、受伤(-15)、击杀(+100)、胜利(+300)、死亡(-100)
 *   - 奖励塑形：距离(+0.5)、瞄准(+0.3)、射击(±0.5)、躲避(±0.4)、墙壁避让(+0.2)、动作多样性(+0.1)
 * 用途：
 *   - AI训练时计算奖励
 *   - 人类数据记录时计算奖励（确保一致性）
 */
float calculate_reward(GameState* game, Tank* ai_tank, int prev_health,
                       int prev_enemies);

/*
 * 渲染当前帧（仅可视化模式）
 * 功能：
 *   - 如果启用了可视化，调用renderer_render_game
 *   - 显示游戏画面和训练信息
 * 注意：只在MODE_TRAINING_VIS模式下有效
 */
void ai_render();

/*
 * 清理资源
 * 功能：
 *   - 清理渲染器（如果已初始化）
 *   - 释放SDL资源
 * 注意：训练结束时必须调用，防止内存泄漏
 */
void ai_cleanup();

#ifdef __cplusplus
}
#endif

#endif // AI_INTERFACE_H
