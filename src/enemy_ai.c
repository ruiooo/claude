/*
 * enemy_ai.c - 敌人AI实现（追踪型，带优化）
 */

#include "enemy_ai.h"
#include <math.h>
#include <stdlib.h>

#define ACTION_INTERVAL 5  // 每5帧更新一次决策

// 初始化敌人AI
void enemy_ai_init(EnemyAI* ai, Tank* target) {
    ai->target = target;
    ai->last_action_time = 0;
    ai->last_move_dir = DIR_UP;
    ai->stuck_counter = 0;
    ai->last_x = 0;
    ai->last_y = 0;
}

// 检测是否卡住
static bool is_stuck(EnemyAI* ai, Tank* tank) {
    float dx = tank->x - ai->last_x;
    float dy = tank->y - ai->last_y;
    float dist = sqrtf(dx * dx + dy * dy);

    // 如果移动距离很小，增加卡住计数器
    if (dist < 0.5f) {
        ai->stuck_counter++;
    } else {
        ai->stuck_counter = 0;
    }

    ai->last_x = tank->x;
    ai->last_y = tank->y;

    return ai->stuck_counter > 20;  // 连续20帧没有移动
}

// 尝试射击
bool enemy_ai_try_shoot(EnemyAI* ai, Tank* enemy_tank, Tank* target) {
    if (!target || !target->alive) return false;
    if (!tank_can_shoot(enemy_tank)) return false;

    float dx = target->x - enemy_tank->x;
    float dy = target->y - enemy_tank->y;
    float dist = sqrtf(dx * dx + dy * dy);

    // 射程内
    if (dist < 300.0f) {
        // 判断是否在射击线上（带一定容错）
        float tolerance = 40.0f;

        if (fabs(dy) < tolerance && fabs(dx) > 20.0f) {
            // 水平射击
            if (dx > 0) {
                enemy_tank->direction = DIR_RIGHT;
                return true;
            } else {
                enemy_tank->direction = DIR_LEFT;
                return true;
            }
        } else if (fabs(dx) < tolerance && fabs(dy) > 20.0f) {
            // 垂直射击
            if (dy > 0) {
                enemy_tank->direction = DIR_DOWN;
                return true;
            } else {
                enemy_tank->direction = DIR_UP;
                return true;
            }
        }
    }

    return false;
}

// 获取朝向目标的最佳移动方向
static Direction get_best_direction(Tank* enemy_tank, Tank* target) {
    float dx = target->x - enemy_tank->x;
    float dy = target->y - enemy_tank->y;

    // 选择距离较大的轴优先移动
    if (fabs(dx) > fabs(dy)) {
        return dx > 0 ? DIR_RIGHT : DIR_LEFT;
    } else {
        return dy > 0 ? DIR_DOWN : DIR_UP;
    }
}

// 更新敌人AI
TankAction enemy_ai_update(EnemyAI* ai, Tank* enemy_tank, Tank* tanks,
                           int tank_count, int current_frame) {
    if (!enemy_tank->alive) return ACTION_IDLE;

    // 检测是否卡住
    if (is_stuck(ai, enemy_tank)) {
        // 随机改变方向以脱困
        int random_dir = rand() % 4;
        ai->last_move_dir = (Direction)random_dir;
        ai->stuck_counter = 0;
    }

    // 定期更新决策
    if (current_frame - ai->last_action_time < ACTION_INTERVAL) {
        // 继续上次的动作
        switch (ai->last_move_dir) {
            case DIR_UP: return ACTION_MOVE_UP;
            case DIR_DOWN: return ACTION_MOVE_DOWN;
            case DIR_LEFT: return ACTION_MOVE_LEFT;
            case DIR_RIGHT: return ACTION_MOVE_RIGHT;
        }
    }

    ai->last_action_time = current_frame;

    // 寻找AI坦克作为目标
    Tank* ai_tank = NULL;
    for (int i = 0; i < tank_count; i++) {
        if (tanks[i].alive && tanks[i].type == TANK_TYPE_AI) {
            ai_tank = &tanks[i];
            break;
        }
    }

    if (!ai_tank) return ACTION_IDLE;

    ai->target = ai_tank;

    // 优先尝试射击
    if (enemy_ai_try_shoot(ai, enemy_tank, ai_tank)) {
        // 根据方向返回射击动作
        switch (enemy_tank->direction) {
            case DIR_UP: return ACTION_SHOOT_UP;
            case DIR_DOWN: return ACTION_SHOOT_DOWN;
            case DIR_LEFT: return ACTION_SHOOT_LEFT;
            case DIR_RIGHT: return ACTION_SHOOT_RIGHT;
        }
    }

    // 追踪移动
    Direction best_dir = get_best_direction(enemy_tank, ai_tank);
    ai->last_move_dir = best_dir;

    switch (best_dir) {
        case DIR_UP: return ACTION_MOVE_UP;
        case DIR_DOWN: return ACTION_MOVE_DOWN;
        case DIR_LEFT: return ACTION_MOVE_LEFT;
        case DIR_RIGHT: return ACTION_MOVE_RIGHT;
    }

    return ACTION_IDLE;
}
