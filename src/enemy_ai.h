/*
 * enemy_ai.h - 敌人AI逻辑（追踪型）
 */

#ifndef ENEMY_AI_H
#define ENEMY_AI_H

#include "tank.h"
#include "bullet.h"

// 敌人AI状态
typedef struct {
    Tank* target;           // 目标坦克（通常是AI坦克）
    int last_action_time;   // 上次动作时间
    Direction last_move_dir;// 上次移动方向
    int stuck_counter;      // 卡住计数器
    float last_x, last_y;   // 上次位置
    int dodge_cooldown;     // 躲避冷却
    Direction dodge_dir;    // 躲避方向
} EnemyAI;

// 初始化敌人AI
void enemy_ai_init(EnemyAI* ai, Tank* target);

// 更新敌人AI并返回动作
TankAction enemy_ai_update(EnemyAI* ai, Tank* enemy_tank, Tank* tanks,
                           int tank_count, Bullet* bullets, int bullet_count,
                           int current_frame);

// 预测射击（提前量）
bool enemy_ai_try_shoot(EnemyAI* ai, Tank* enemy_tank, Tank* target);

#endif // ENEMY_AI_H
