/*
 * game.h - 游戏核心逻辑和状态管理
 *
 * 模块说明：
 *   游戏引擎的核心模块，管理所有游戏实体和状态
 *   提供游戏主循环、实体更新、碰撞处理等核心功能
 *   作为C游戏引擎与Python强化学习系统的桥梁
 *
 * 使用场景：
 *   - 训练模式：为DQN提供高性能游戏环境（纯逻辑，无渲染）
 *   - 训练可视化模式：训练同时显示游戏画面（调试用）
 *   - 玩家对战模式：玩家与训练好的AI模型对战
 *
 * 架构：
 *   - 所有实体在栈上分配（Tank[32], Bullet[100]）
 *   - 固定时间步（60fps）
 *   - 状态同步（与Python通过ctypes通信）
 */

#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include "tank.h"
#include "bullet.h"
#include "enemy_ai.h"

/*
 * 游戏模式枚举
 * 决定游戏的运行方式和渲染策略
 */
typedef enum {
    MODE_TRAINING,       // 训练模式：纯逻辑无渲染，最高性能，用于DQN训练
    MODE_TRAINING_VIS,   // 训练可视化模式：带渲染的训练，用于调试和观察学习过程
    MODE_PLAYER          // 玩家对战模式：玩家与AI模型对战，完整渲染
} GameMode;

/*
 * 游戏状态结构体
 * 保存游戏的完整状态，是游戏引擎的核心数据结构
 */
typedef struct {
    Tank tanks[32];           // 所有坦克数组（索引0=AI坦克，1-31=敌人坦克）
    int tank_count;           // 当前存活的坦克总数
    Bullet bullets[MAX_BULLETS]; // 子弹池，所有子弹的固定大小数组
    EnemyAI enemy_ais[32];    // 敌人AI状态数组（对应tanks[1-31]）
    int frame_count;          // 当前回合的帧计数器，用于时间和行为延迟
    int episode_count;        // 累计回合数（训练模式）
    bool game_over;           // 游戏结束标志，true=回合结束
    int winner;               // 胜者标识：0=AI胜利，1=敌人胜利，-1=未结束
    int map_width;            // 地图宽度（像素），通常为800
    int map_height;           // 地图高度（像素），通常为600
    GameMode mode;            // 当前游戏模式
} GameState;

/*
 * 游戏核心功能函数声明
 */

/*
 * 初始化游戏状态
 * @param game: 游戏状态指针
 * @param map_width: 地图宽度（像素）
 * @param map_height: 地图高度（像素）
 * @param mode: 游戏模式
 * 功能：
 *   - 初始化子弹池（全部设为未激活）
 *   - 设置地图尺寸
 *   - 重置计数器
 * 注意：只调用一次，之后使用game_reset重置回合
 */
void game_init(GameState* game, int map_width, int map_height, GameMode mode);

/*
 * 重置游戏为新回合
 * @param game: 游戏状态指针
 * @param enemy_count: 敌人数量（1-31）
 * 功能：
 *   - 清除所有坦克和子弹
 *   - 创建1个AI坦克（tanks[0]）
 *   - 创建指定数量的敌人坦克
 *   - 随机生成初始位置（避免重叠）
 *   - 重置帧计数器和游戏结束标志
 * 用途：每个训练回合开始时调用
 */
void game_reset(GameState* game, int enemy_count);

/*
 * 更新游戏状态（每帧调用，60fps）
 * @param game: 游戏状态指针
 * 功能：
 *   - 更新所有坦克状态（位置、冷却等）
 *   - 更新所有子弹位置
 *   - 处理碰撞检测（子弹-坦克）
 *   - 更新敌人AI决策并执行动作
 *   - 检查游戏结束条件：
 *     * AI坦克死亡 -> 敌人胜利
 *     * 所有敌人死亡 -> AI胜利
 *   - 递增帧计数器
 */
void game_update(GameState* game);

/*
 * 执行坦克动作
 * @param game: 游戏状态指针
 * @param tank_id: 坦克ID（通常0=AI坦克）
 * @param action: 动作枚举（移动或射击）
 * 功能：
 *   - 解析动作：移动动作（ACTION_MOVE_*）调用tank_move
 *   - 射击动作（ACTION_SHOOT_*）发射子弹
 *   - 边界检查和冷却检查
 * 用途：
 *   - Python训练系统调用：执行DQN模型的动作
 *   - 玩家输入：执行玩家的键盘操作
 */
void game_execute_action(GameState* game, int tank_id, TankAction action);

/*
 * 添加敌人到游戏
 * @param game: 游戏状态指针
 * @param type: 敌人类型（TANK_TYPE_ENEMY或TANK_TYPE_SELF_PLAY）
 * @param model_version: 模型版本号（自我对弈时使用，0=最新）
 * 功能：
 *   - 在tanks数组中找到空位
 *   - 初始化敌人坦克（随机位置）
 *   - 如果是追踪型敌人，初始化enemy_ai
 *   - 递增tank_count
 * 用途：动态难度调整、自我对弈系统
 */
void game_add_enemy(GameState* game, TankType type, int model_version);

/*
 * 获取AI坦克指针
 * @param game: 游戏状态指针
 * @return: AI坦克指针（通常是&game->tanks[0]）
 * 功能：返回第一个TANK_TYPE_AI类型的坦克
 * 用途：快速访问主角坦克
 */
Tank* game_get_ai_tank(GameState* game);

/*
 * 统计指定类型的存活坦克数量
 * @param game: 游戏状态指针
 * @param type: 坦克类型
 * @return: 存活数量
 * 功能：遍历所有坦克，统计alive=true且类型匹配的数量
 * 用途：判断游戏结束条件、动态难度调整
 */
int game_get_alive_count(GameState* game, TankType type);

/*
 * 获取游戏观察状态（DQN训练接口）
 * @param game: 游戏状态指针
 * @param obs: 输出观察数组（float[43]）
 * @param obs_size: 输出观察维度（应为43）
 * 功能：
 *   - 提取AI坦克状态：x, y, vx, vy, health, shoot_cooldown（6维）
 *   - 提取最近5个敌人状态：dx, dy, vx, vy, health（5×5=25维）
 *   - 提取最近3个子弹状态：dx, dy, vx, vy（3×4=12维）
 *   - 归一化所有数值到合理范围
 * 重要：此函数的输出维度必须与Python模型输入维度一致
 * 对应文件：python/config.py MODEL_CONFIG['state_dim']=43
 */
void game_get_observation(GameState* game, float* obs, int* obs_size);

#endif // GAME_H
