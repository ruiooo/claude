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

    // 衰减速度
    tank->vx *= 0.85f;
    tank->vy *= 0.85f;
}

// 移动坦克（带边界检查）
void tank_move(Tank* tank, Direction dir, int map_width, int map_height) {
    if (!tank->alive) return;

    tank->direction = dir;

    float new_x = tank->x;
    float new_y = tank->y;

    switch (dir) {
        case DIR_UP:
            new_y = tank->y - TANK_SPEED;
            tank->vy = -TANK_SPEED;
            tank->vx = 0;
            break;
        case DIR_DOWN:
            new_y = tank->y + TANK_SPEED;
            tank->vy = TANK_SPEED;
            tank->vx = 0;
            break;
        case DIR_LEFT:
            new_x = tank->x - TANK_SPEED;
            tank->vx = -TANK_SPEED;
            tank->vy = 0;
            break;
        case DIR_RIGHT:
            new_x = tank->x + TANK_SPEED;
            tank->vx = TANK_SPEED;
            tank->vy = 0;
            break;
    }

    // 边界检查
    if (new_x >= 0 && new_x + tank->width <= map_width) {
        tank->x = new_x;
    } else {
        tank->vx = 0;
    }

    if (new_y >= 0 && new_y + tank->height <= map_height) {
        tank->y = new_y;
    } else {
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
