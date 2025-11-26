/*
 * collision.h - 碰撞检测系统
 */

#ifndef COLLISION_H
#define COLLISION_H

#include <stdbool.h>
#include "tank.h"
#include "bullet.h"

// AABB碰撞检测
bool check_aabb_collision(float x1, float y1, int w1, int h1,
                          float x2, float y2, int w2, int h2);

// 坦克之间的碰撞检测
bool check_tank_collision(Tank* t1, Tank* t2);

// 子弹和坦克的碰撞检测
bool check_bullet_tank_collision(Bullet* bullet, Tank* tank);

// 处理所有碰撞
void handle_collisions(Tank* tanks, int tank_count, Bullet* bullets, int bullet_count);

#endif // COLLISION_H
