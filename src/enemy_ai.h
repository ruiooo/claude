/*
 * enemy_ai.h - 敌人AI逻辑（追踪型）
 *
 * 模块说明：
 *   实现传统的追踪型敌人AI，用于对抗训练中的DQN模型
 *   提供追踪、躲避、预测射击等智能行为
 *   为DQN训练提供动态的对手，增强训练效果
 *
 * 使用场景：
 *   - 训练模式：作为DQN模型的对手，提供挑战性
 *   - 动态难度：多个追踪型敌人同时追击AI坦克
 *   - 自我对弈：与历史版本AI模型混合使用
 *
 * AI特性：
 *   - 主动追踪：计算与目标的距离，选择最优移动方向
 *   - 预测射击：根据目标速度提前瞄准
 *   - 躲避子弹：检测来袭子弹并进行规避
 *   - 卡住检测：防止陷入重复移动的死循环
 */

#ifndef ENEMY_AI_H
#define ENEMY_AI_H

#include "tank.h"
#include "bullet.h"

/*
 * 敌人AI状态结构体
 * 保存AI决策所需的历史信息和状态数据
 */
typedef struct {
    Tank* target;           // 追踪目标坦克指针（通常指向TANK_TYPE_AI）
    int last_action_time;   // 上次执行动作的帧数，用于行为延迟和冷却
    Direction last_move_dir;// 上次移动方向，用于检测卡住状态
    int stuck_counter;      // 卡住计数器，连续多帧位置不变时触发随机移动
    float last_x, last_y;   // 上一帧的位置，用于计算是否移动成功
    int dodge_cooldown;     // 躲避行为冷却计数器（帧数），>0时处于躲避模式
    Direction dodge_dir;    // 当前躲避方向，躲避冷却期间保持该方向移动
} EnemyAI;

/*
 * 敌人AI功能函数声明
 */

/*
 * 初始化敌人AI
 * @param ai: 待初始化的敌人AI指针
 * @param target: 追踪目标坦克指针（通常是玩家或DQN控制的坦克）
 * 功能：
 *   - 设置追踪目标
 *   - 清空历史状态（计数器、位置等）
 *   - 重置冷却时间
 */
void enemy_ai_init(EnemyAI* ai, Tank* target);

/*
 * 更新敌人AI并返回决策动作（每帧调用）
 * @param ai: 敌人AI状态指针
 * @param enemy_tank: 该AI控制的敌人坦克指针
 * @param tanks: 所有坦克数组，用于避免碰撞
 * @param tank_count: 坦克总数
 * @param bullets: 所有子弹数组，用于躲避检测
 * @param bullet_count: 子弹数量
 * @param current_frame: 当前游戏帧数，用于行为延迟
 * @param map_width: 地图宽度，用于墙壁避让
 * @param map_height: 地图高度，用于墙壁避让
 * @return: 决策的动作（移动或射击）
 * 决策流程：
 *   1. 检测危险子弹，优先躲避
 *   2. 尝试预测射击目标
 *   3. 检测卡住状态，随机移动脱困
 *   4. 计算与目标距离，选择追踪方向
 * AI策略：
 *   - 近距离：优先射击
 *   - 中远距离：接近目标
 *   - 子弹来袭：垂直躲避
 *   - 靠近墙壁：自动避让
 */
TankAction enemy_ai_update(EnemyAI* ai, Tank* enemy_tank, Tank* tanks,
                           int tank_count, Bullet* bullets, int bullet_count,
                           int current_frame, int map_width, int map_height);

/*
 * 尝试预测射击（提前量算法）
 * @param ai: 敌人AI状态指针
 * @param enemy_tank: 该AI控制的敌人坦克指针
 * @param target: 射击目标坦克指针
 * @return: true=应该射击，false=不应射击
 * 算法：
 *   - 计算目标当前位置和速度
 *   - 预测子弹飞行时间内目标的未来位置
 *   - 判断是否在有效射程和角度内
 * 射击条件：
 *   - 目标存活（alive=true）
 *   - 距离在有效范围内
 *   - 预测命中率较高
 *   - 射击冷却已完成
 */
bool enemy_ai_try_shoot(EnemyAI* ai, Tank* enemy_tank, Tank* target);

#endif // ENEMY_AI_H
