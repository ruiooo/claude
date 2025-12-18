/*
 * collision.c - 碰撞检测实现
 *
 * 模块说明：
 * 本模块实现了游戏中所有实体的碰撞检测系统，包括：
 * - AABB（轴对齐包围盒）碰撞检测算法
 * - 坦克与坦克的碰撞检测和分离
 * - 子弹与坦克的碰撞检测
 * - 统一的碰撞处理接口
 *
 * 碰撞检测算法：
 * - 使用AABB（Axis-Aligned Bounding Box）算法
 * - 时间复杂度：O(n²) 用于坦克-坦克，O(n*m) 用于子弹-坦克
 * - 优点：简单高效，适合2D游戏
 *
 * 碰撞响应：
 * - 子弹击中坦克：坦克受伤，子弹失效
 * - 坦克碰撞坦克：物理分离（推开）
 */

#include "collision.h"
#include <math.h>

/**
 * AABB (Axis-Aligned Bounding Box) 碰撞检测
 *
 * 功能：检测两个矩形是否相交
 *
 * 参数：
 *   x1, y1, w1, h1 - 第一个矩形的位置和尺寸
 *   x2, y2, w2, h2 - 第二个矩形的位置和尺寸
 *
 * 返回值：
 *   true  - 两个矩形相交（发生碰撞）
 *   false - 两个矩形不相交
 *
 * 算法原理：
 *   两个矩形相交的充要条件：
 *   - 水平方向重叠：x1 < x2+w2 AND x1+w1 > x2
 *   - 垂直方向重叠：y1 < y2+h2 AND y1+h1 > y2
 *
 * 时间复杂度：O(1)
 */
bool check_aabb_collision(float x1, float y1, int w1, int h1,
                          float x2, float y2, int w2, int h2) {
    return (x1 < x2 + w2 &&           // 矩形1的左边在矩形2的右边之左
            x1 + w1 > x2 &&           // 矩形1的右边在矩形2的左边之右
            y1 < y2 + h2 &&           // 矩形1的上边在矩形2的下边之上
            y1 + h1 > y2);            // 矩形1的下边在矩形2的上边之下
}

/**
 * 坦克之间的碰撞检测
 *
 * 功能：检测两个坦克是否碰撞
 *
 * 参数：
 *   t1 - 第一个坦克指针
 *   t2 - 第二个坦克指针
 *
 * 返回值：
 *   true  - 发生碰撞
 *   false - 未碰撞或至少一方已死亡
 *
 * 实现细节：
 *   - 死亡的坦克不参与碰撞检测
 *   - 使用AABB算法检测
 */
bool check_tank_collision(Tank* t1, Tank* t2) {
    if (!t1->alive || !t2->alive) return false;  // 死亡坦克不碰撞

    return check_aabb_collision(t1->x, t1->y, t1->width, t1->height,
                                t2->x, t2->y, t2->width, t2->height);
}

/**
 * 子弹和坦克的碰撞检测
 *
 * 功能：检测子弹是否击中坦克
 *
 * 参数：
 *   bullet - 子弹指针
 *   tank   - 坦克指针
 *
 * 返回值：
 *   true  - 子弹击中坦克
 *   false - 未击中或不应碰撞
 *
 * 特殊规则：
 *   - 子弹不能击中发射者（通过owner_id判断）
 *   - 未激活的子弹不参与碰撞
 *   - 死亡的坦克不参与碰撞
 */
bool check_bullet_tank_collision(Bullet* bullet, Tank* tank) {
    if (!bullet->active || !tank->alive) return false;  // 必须都是活跃状态

    // 子弹不能击中发射者
    // 这样设计避免了"自杀"，提升游戏体验
    if (bullet->owner_id == tank->id) return false;

    // ✅ AI坦克之间不互相伤害（玩家对战模式优化）
    // 如果子弹发射者是AI（ENEMY或SELF_PLAY），目标也是AI，则不碰撞
    bool owner_is_ai = (bullet->owner_type == TANK_TYPE_ENEMY ||
                        bullet->owner_type == TANK_TYPE_SELF_PLAY);
    bool target_is_ai = (tank->type == TANK_TYPE_ENEMY ||
                         tank->type == TANK_TYPE_SELF_PLAY);

    if (owner_is_ai && target_is_ai) {
        return false;  // AI坦克之间不互相伤害
    }

    return check_aabb_collision(bullet->x, bullet->y, bullet->width, bullet->height,
                                tank->x, tank->y, tank->width, tank->height);
}

/**
 * 处理所有碰撞（统一碰撞处理接口）
 *
 * 功能：
 *   - 检测并处理子弹-坦克碰撞
 *   - 检测并处理坦克-坦克碰撞
 *
 * 参数：
 *   tanks        - 坦克数组
 *   tank_count   - 坦克数量
 *   bullets      - 子弹数组
 *   bullet_count - 子弹数量
 *
 * 碰撞处理顺序：
 *   1. 先处理子弹-坦克碰撞（伤害计算）
 *   2. 再处理坦克-坦克碰撞（物理分离）
 *
 * 实现细节：
 *   - 使用双层循环遍历所有可能的碰撞对
 *   - 子弹碰撞后立即失效，避免穿透
 *   - 坦克碰撞使用物理分离算法
 */
void handle_collisions(Tank* tanks, int tank_count, Bullet* bullets, int bullet_count) {
    // ========== 处理子弹-坦克碰撞 ==========
    // 遍历所有活跃子弹
    for (int i = 0; i < bullet_count; i++) {
        if (!bullets[i].active) continue;  // 跳过未激活的子弹

        // 检测该子弹是否击中任何坦克
        for (int j = 0; j < tank_count; j++) {
            if (!tanks[j].alive) continue;  // 跳过死亡坦克

            // 碰撞检测
            if (check_bullet_tank_collision(&bullets[i], &tanks[j])) {
                // 子弹击中坦克
                tank_take_damage(&tanks[j], 1);  // 造成1点伤害
                bullets[i].active = false;       // 子弹失效
                break;  // 子弹已失效，跳出内层循环
            }
        }
    }

    // ========== 处理坦克-坦克碰撞（物理分离） ==========
    // 使用双层循环遍历所有坦克对，避免重复检测
    for (int i = 0; i < tank_count; i++) {
        if (!tanks[i].alive) continue;

        // 只检测 j > i 的坦克，避免重复（i,j）和（j,i）
        for (int j = i + 1; j < tank_count; j++) {
            if (!tanks[j].alive) continue;

            // 碰撞检测
            if (check_tank_collision(&tanks[i], &tanks[j])) {
                // 简单的物理分离算法：向相反方向推开
                // 计算两个坦克的中心距离向量
                float dx = tanks[j].x - tanks[i].x;
                float dy = tanks[j].y - tanks[i].y;
                float dist = sqrtf(dx * dx + dy * dy);  // 欧几里得距离

                if (dist > 0) {
                    // 归一化距离向量，得到方向向量
                    float nx = dx / dist;
                    float ny = dy / dist;

                    // 计算重叠量（期望距离 - 实际距离）
                    // 期望距离：两个坦克的半宽度之和
                    float separation = (tanks[i].width + tanks[j].width) / 2.0f - dist;

                    // 将两个坦克沿方向向量推开（各承担一半的分离量）
                    // 这样保证质量相等的坦克分离是对称的
                    tanks[i].x -= nx * separation * 0.5f;
                    tanks[i].y -= ny * separation * 0.5f;
                    tanks[j].x += nx * separation * 0.5f;
                    tanks[j].y += ny * separation * 0.5f;
                }
            }
        }
    }
}
