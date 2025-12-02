/*
 * enemy_ai.c - 敌人AI实现（智能射击和躲避型）
 */

#include "enemy_ai.h"
#include <math.h>
#include <stdlib.h>

#define ACTION_INTERVAL 5   // 每5帧更新一次决策
#define IDEAL_MIN_DIST 150.0f  // 理想最小距离
#define IDEAL_MAX_DIST 250.0f  // 理想最大距离
#define DANGER_DIST 80.0f   // 子弹危险距离

// 初始化敌人AI
void enemy_ai_init(EnemyAI* ai, Tank* target) {
    ai->target = target;
    ai->last_action_time = 0;
    ai->last_move_dir = DIR_UP;
    ai->stuck_counter = 0;
    ai->last_x = 0;
    ai->last_y = 0;
    ai->dodge_cooldown = 0;
    ai->dodge_dir = DIR_UP;
}

// 检测是否卡住
static bool is_stuck(EnemyAI* ai, Tank* tank) {
    float dx = tank->x - ai->last_x;
    float dy = tank->y - ai->last_y;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < 0.5f) {
        ai->stuck_counter++;
    } else {
        ai->stuck_counter = 0;
    }

    ai->last_x = tank->x;
    ai->last_y = tank->y;

    return ai->stuck_counter > 20;
}

// 检测危险子弹并返回躲避方向
static bool detect_danger_bullet(Tank* enemy_tank, Bullet* bullets, int bullet_count, Direction* dodge_dir) {
    for (int i = 0; i < bullet_count; i++) {
        if (!bullets[i].active) continue;

        // 只关心AI坦克的子弹
        if (bullets[i].owner_type != TANK_TYPE_AI) continue;

        float dx = bullets[i].x - enemy_tank->x;
        float dy = bullets[i].y - enemy_tank->y;
        float dist = sqrtf(dx * dx + dy * dy);

        // 子弹在危险范围内
        if (dist < DANGER_DIST) {
            // 判断子弹的飞行方向
            float bullet_vx = bullets[i].vx;
            float bullet_vy = bullets[i].vy;

            // 向垂直于子弹方向躲避
            if (fabs(bullet_vx) > fabs(bullet_vy)) {
                // 子弹横向飞行，上下躲避
                *dodge_dir = (dy > 0) ? DIR_DOWN : DIR_UP;
                return true;
            } else {
                // 子弹纵向飞行，左右躲避
                *dodge_dir = (dx > 0) ? DIR_RIGHT : DIR_LEFT;
                return true;
            }
        }
    }
    return false;
}

// 尝试射击（改进的射击判定 - 减少不必要的转向）
bool enemy_ai_try_shoot(EnemyAI* ai, Tank* enemy_tank, Tank* target) {
    if (!target || !target->alive) return false;
    if (!tank_can_shoot(enemy_tank)) return false;

    float dx = target->x - enemy_tank->x;
    float dy = target->y - enemy_tank->y;
    float dist = sqrtf(dx * dx + dy * dy);

    // 在理想射程内
    if (dist >= IDEAL_MIN_DIST && dist <= IDEAL_MAX_DIST + 100.0f) {
        // 判断是否在射击线上（更严格的容错，减少转向）
        float tolerance = 40.0f;

        if (fabs(dy) < tolerance && fabs(dx) > 40.0f) {
            // 水平射击 - 只有在当前方向不对时才转向
            Direction desired_dir = (dx > 0) ? DIR_RIGHT : DIR_LEFT;
            if (enemy_tank->direction != desired_dir) {
                enemy_tank->direction = desired_dir;
            }
            return true;
        } else if (fabs(dx) < tolerance && fabs(dy) > 40.0f) {
            // 垂直射击 - 只有在当前方向不对时才转向
            Direction desired_dir = (dy > 0) ? DIR_DOWN : DIR_UP;
            if (enemy_tank->direction != desired_dir) {
                enemy_tank->direction = desired_dir;
            }
            return true;
        }
    }

    return false;
}

// 获取定位移动方向（保持理想射击距离 - 优化以减少原地转向）
static Direction get_positioning_direction(Tank* enemy_tank, Tank* target) {
    float dx = target->x - enemy_tank->x;
    float dy = target->y - enemy_tank->y;
    float dist = sqrtf(dx * dx + dy * dy);

    // 距离太近，后退
    if (dist < IDEAL_MIN_DIST) {
        if (fabs(dx) > fabs(dy)) {
            return dx > 0 ? DIR_LEFT : DIR_RIGHT;
        } else {
            return dy > 0 ? DIR_UP : DIR_DOWN;
        }
    }

    // 距离太远，靠近
    if (dist > IDEAL_MAX_DIST) {
        if (fabs(dx) > fabs(dy)) {
            return dx > 0 ? DIR_RIGHT : DIR_LEFT;
        } else {
            return dy > 0 ? DIR_DOWN : DIR_UP;
        }
    }

    // 距离合适，调整位置以瞄准
    // 只有偏差较大时才移动
    if (fabs(dx) > 80.0f) {
        return dx > 0 ? DIR_RIGHT : DIR_LEFT;
    } else if (fabs(dy) > 80.0f) {
        return dy > 0 ? DIR_DOWN : DIR_UP;
    }

    // 已经在理想位置且对准较好，保持当前方向继续移动
    // 不进行随机转向，而是保持当前方向或微调
    return enemy_tank->direction;
}

// 更新敌人AI
TankAction enemy_ai_update(EnemyAI* ai, Tank* enemy_tank, Tank* tanks,
                           int tank_count, Bullet* bullets, int bullet_count,
                           int current_frame) {
    if (!enemy_tank->alive) return ACTION_IDLE;

    // 检测是否卡住
    if (is_stuck(ai, enemy_tank)) {
        int random_dir = rand() % 4;
        ai->last_move_dir = (Direction)random_dir;
        ai->stuck_counter = 0;
    }

    // 定期更新决策
    if (current_frame - ai->last_action_time < ACTION_INTERVAL) {
        if (ai->dodge_cooldown > 0) {
            ai->dodge_cooldown--;
            // 继续躲避
            switch (ai->dodge_dir) {
                case DIR_UP: return ACTION_MOVE_UP;
                case DIR_DOWN: return ACTION_MOVE_DOWN;
                case DIR_LEFT: return ACTION_MOVE_LEFT;
                case DIR_RIGHT: return ACTION_MOVE_RIGHT;
            }
        }
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

    // 优先级1: 检测并躲避子弹
    Direction dodge_direction;
    if (detect_danger_bullet(enemy_tank, bullets, bullet_count, &dodge_direction)) {
        ai->dodge_dir = dodge_direction;
        ai->dodge_cooldown = 10;  // 躲避10帧
        ai->last_move_dir = dodge_direction;

        switch (dodge_direction) {
            case DIR_UP: return ACTION_MOVE_UP;
            case DIR_DOWN: return ACTION_MOVE_DOWN;
            case DIR_LEFT: return ACTION_MOVE_LEFT;
            case DIR_RIGHT: return ACTION_MOVE_RIGHT;
        }
    }

    // 优先级2: 尝试射击
    if (enemy_ai_try_shoot(ai, enemy_tank, ai_tank)) {
        switch (enemy_tank->direction) {
            case DIR_UP: return ACTION_SHOOT_UP;
            case DIR_DOWN: return ACTION_SHOOT_DOWN;
            case DIR_LEFT: return ACTION_SHOOT_LEFT;
            case DIR_RIGHT: return ACTION_SHOOT_RIGHT;
        }
    }

    // 优先级3: 定位移动（保持理想射击距离）
    Direction best_dir = get_positioning_direction(enemy_tank, ai_tank);
    ai->last_move_dir = best_dir;

    switch (best_dir) {
        case DIR_UP: return ACTION_MOVE_UP;
        case DIR_DOWN: return ACTION_MOVE_DOWN;
        case DIR_LEFT: return ACTION_MOVE_LEFT;
        case DIR_RIGHT: return ACTION_MOVE_RIGHT;
    }

    return ACTION_IDLE;
}
