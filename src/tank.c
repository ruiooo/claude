/*
 * tank.c - 坦克实体实现
 */

#include "tank.h"
#include <stdlib.h>
#include <string.h>

// 初始化坦克
void tank_init(Tank* tank, float x, float y, TankType type) {
    tank->x = x;
    tank->y = y;
    tank->vx = 0;
    tank->vy = 0;
    tank->width = TANK_SIZE;
    tank->height = TANK_SIZE;
    tank->health = TANK_MAX_HEALTH;
    tank->alive = true;
    tank->type = type;
    tank->direction = DIR_UP;
    tank->shoot_cooldown = 0;
    tank->model_version = 0;
    tank->id = rand() % 10000;
}

// 更新坦克状态
void tank_update(Tank* tank) {
    if (!tank->alive) return;

    // 更新射击冷却
    if (tank->shoot_cooldown > 0) {
        tank->shoot_cooldown--;
    }

    // 应用速度
    tank->x += tank->vx;
    tank->y += tank->vy;

    // 平滑衰减速度 - 改为更缓慢的衰减
    tank->vx *= 0.95f;
    tank->vy *= 0.95f;

    // 速度太小时归零，避免持续微小抖动
    if (tank->vx > -0.1f && tank->vx < 0.1f) tank->vx = 0;
    if (tank->vy > -0.1f && tank->vy < 0.1f) tank->vy = 0;
}

// 移动坦克（带边界检查）
void tank_move(Tank* tank, Direction dir, int map_width, int map_height) {
    if (!tank->alive) return;

    tank->direction = dir;

    // 设置速度，不直接修改位置
    switch (dir) {
        case DIR_UP:
            tank->vy = -TANK_SPEED;
            tank->vx = 0;
            break;
        case DIR_DOWN:
            tank->vy = TANK_SPEED;
            tank->vx = 0;
            break;
        case DIR_LEFT:
            tank->vx = -TANK_SPEED;
            tank->vy = 0;
            break;
        case DIR_RIGHT:
            tank->vx = TANK_SPEED;
            tank->vy = 0;
            break;
    }

    // 预测下一帧位置并边界检查
    float next_x = tank->x + tank->vx;
    float next_y = tank->y + tank->vy;

    // 如果即将碰到边界，停止该方向的移动
    if (next_x < 0 || next_x + tank->width > map_width) {
        tank->vx = 0;
    }

    if (next_y < 0 || next_y + tank->height > map_height) {
        tank->vy = 0;
    }
}

// 坦克受到伤害
void tank_take_damage(Tank* tank, int damage) {
    if (!tank->alive) return;

    tank->health -= damage;
    if (tank->health <= 0) {
        tank->health = 0;
        tank->alive = false;
    }
}

// 检查是否可以射击
bool tank_can_shoot(Tank* tank) {
    return tank->alive && tank->shoot_cooldown == 0;
}

// 开始射击冷却
void tank_start_shoot_cooldown(Tank* tank) {
    tank->shoot_cooldown = SHOOT_COOLDOWN;
}

// 设置模型版本（用于历史版本敌人）
void tank_set_model_version(Tank* tank, int version) {
    tank->model_version = version;
}
