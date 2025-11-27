/*
 * bullet.c - 子弹实体实现
 */

#include "bullet.h"
#include <math.h>

// 初始化子弹
void bullet_init(Bullet* bullet) {
    bullet->x = 0;
    bullet->y = 0;
    bullet->vx = 0;
    bullet->vy = 0;
    bullet->width = BULLET_SIZE;
    bullet->height = BULLET_SIZE;
    bullet->active = false;
    bullet->owner_id = -1;
    bullet->owner_type = TANK_TYPE_ENEMY;
}

// 从坦克发射子弹
void bullet_fire(Bullet* bullet, Tank* tank) {
    if (!tank->alive) return;

    bullet->active = true;
    bullet->owner_id = tank->id;
    bullet->owner_type = tank->type;

    // 根据坦克朝向设置子弹位置和速度
    // 炮筒长度：16像素
    #define BARREL_LENGTH 16

    switch (tank->direction) {
        case DIR_UP:
            bullet->x = tank->x + tank->width / 2 - BULLET_SIZE / 2;
            bullet->y = tank->y - BULLET_SIZE - BARREL_LENGTH;
            bullet->vx = 0;
            bullet->vy = -BULLET_SPEED;
            break;
        case DIR_DOWN:
            bullet->x = tank->x + tank->width / 2 - BULLET_SIZE / 2;
            bullet->y = tank->y + tank->height + BARREL_LENGTH;
            bullet->vx = 0;
            bullet->vy = BULLET_SPEED;
            break;
        case DIR_LEFT:
            bullet->x = tank->x - BULLET_SIZE - BARREL_LENGTH;
            bullet->y = tank->y + tank->height / 2 - BULLET_SIZE / 2;
            bullet->vx = -BULLET_SPEED;
            bullet->vy = 0;
            break;
        case DIR_RIGHT:
            bullet->x = tank->x + tank->width + BARREL_LENGTH;
            bullet->y = tank->y + tank->height / 2 - BULLET_SIZE / 2;
            bullet->vx = BULLET_SPEED;
            bullet->vy = 0;
            break;
    }
}

// 更新子弹
void bullet_update(Bullet* bullet, int map_width, int map_height) {
    if (!bullet->active) return;

    bullet->x += bullet->vx;
    bullet->y += bullet->vy;

    // 边界检查，出界则失效
    if (bullet->x < 0 || bullet->x > map_width ||
        bullet->y < 0 || bullet->y > map_height) {
        bullet->active = false;
    }
}
