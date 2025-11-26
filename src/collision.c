/*
 * collision.c - 碰撞检测实现
 */

#include "collision.h"
#include <math.h>

// AABB (Axis-Aligned Bounding Box) 碰撞检测
bool check_aabb_collision(float x1, float y1, int w1, int h1,
                          float x2, float y2, int w2, int h2) {
    return (x1 < x2 + w2 &&
            x1 + w1 > x2 &&
            y1 < y2 + h2 &&
            y1 + h1 > y2);
}

// 坦克之间的碰撞检测
bool check_tank_collision(Tank* t1, Tank* t2) {
    if (!t1->alive || !t2->alive) return false;

    return check_aabb_collision(t1->x, t1->y, t1->width, t1->height,
                                t2->x, t2->y, t2->width, t2->height);
}

// 检查坦克在指定位置是否会与其他坦克碰撞
bool check_tank_position_collision(float x, float y, int width, int height,
                                   Tank* tanks, int tank_count, int exclude_id) {
    for (int i = 0; i < tank_count; i++) {
        if (!tanks[i].alive || tanks[i].id == exclude_id) continue;

        if (check_aabb_collision(x, y, width, height,
                                tanks[i].x, tanks[i].y,
                                tanks[i].width, tanks[i].height)) {
            return true;
        }
    }
    return false;
}

// 子弹和坦克的碰撞检测
bool check_bullet_tank_collision(Bullet* bullet, Tank* tank) {
    if (!bullet->active || !tank->alive) return false;

    // 子弹不能击中发射者
    if (bullet->owner_id == tank->id) return false;

    return check_aabb_collision(bullet->x, bullet->y, bullet->width, bullet->height,
                                tank->x, tank->y, tank->width, tank->height);
}

// 处理所有碰撞
void handle_collisions(Tank* tanks, int tank_count, Bullet* bullets, int bullet_count) {
    // 处理子弹-坦克碰撞
    for (int i = 0; i < bullet_count; i++) {
        if (!bullets[i].active) continue;

        for (int j = 0; j < tank_count; j++) {
            if (!tanks[j].alive) continue;

            if (check_bullet_tank_collision(&bullets[i], &tanks[j])) {
                // 子弹击中坦克
                tank_take_damage(&tanks[j], 1);
                bullets[i].active = false;
                break;
            }
        }
    }

    // 处理坦克-坦克碰撞（简单的分离）
    for (int i = 0; i < tank_count; i++) {
        if (!tanks[i].alive) continue;

        for (int j = i + 1; j < tank_count; j++) {
            if (!tanks[j].alive) continue;

            if (check_tank_collision(&tanks[i], &tanks[j])) {
                // 简单的分离：向相反方向推开
                float dx = tanks[j].x - tanks[i].x;
                float dy = tanks[j].y - tanks[i].y;
                float dist = sqrtf(dx * dx + dy * dy);

                if (dist > 0) {
                    float nx = dx / dist;
                    float ny = dy / dist;

                    float separation = (tanks[i].width + tanks[j].width) / 2.0f - dist;
                    tanks[i].x -= nx * separation * 0.5f;
                    tanks[i].y -= ny * separation * 0.5f;
                    tanks[j].x += nx * separation * 0.5f;
                    tanks[j].y += ny * separation * 0.5f;
                }
            }
        }
    }
}
