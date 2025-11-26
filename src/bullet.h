/*
 * bullet.h - 子弹实体定义
 */

#ifndef BULLET_H
#define BULLET_H

#include <stdbool.h>
#include "tank.h"

// 子弹结构
typedef struct {
    float x, y;          // 位置
    float vx, vy;        // 速度
    int width, height;   // 尺寸
    bool active;         // 是否激活
    int owner_id;        // 发射者ID
    TankType owner_type; // 发射者类型
} Bullet;

// 常量
#define BULLET_SIZE 8
#define BULLET_SPEED 5.0f
#define MAX_BULLETS 100

// 函数声明
void bullet_init(Bullet* bullet);
void bullet_fire(Bullet* bullet, Tank* tank);
void bullet_update(Bullet* bullet, int map_width, int map_height);

#endif // BULLET_H
