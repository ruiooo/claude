/*
 * bullet.c - 子弹实体实现
 *
 * 模块说明：
 * 本模块负责子弹实体的核心逻辑，包括：
 * - 子弹的初始化和状态管理
 * - 从坦克发射子弹（根据坦克朝向）
 * - 子弹的移动和边界检测
 *
 * 关键特性：
 * - 子弹速度固定为 BULLET_SPEED = 10.0f
 * - 子弹出界自动失效
 * - 记录发射者ID，用于碰撞检测（避免击中自己）
 */

#include "bullet.h"
#include <math.h>

/**
 * 初始化子弹
 *
 * 功能：将子弹设置为初始状态（未激活）
 *
 * 参数：
 *   bullet - 要初始化的子弹指针
 *
 * 实现细节：
 *   - 初始位置为(0,0)
 *   - 初始速度为0
 *   - 尺寸固定为 BULLET_SIZE (8x8像素)
 *   - active = false 表示子弹未使用
 *   - owner_id = -1 表示无主
 */
void bullet_init(Bullet* bullet) {
    bullet->x = 0;
    bullet->y = 0;
    bullet->vx = 0;
    bullet->vy = 0;
    bullet->width = BULLET_SIZE;       // 8像素
    bullet->height = BULLET_SIZE;
    bullet->active = false;            // 未激活
    bullet->owner_id = -1;             // 无主
    bullet->owner_type = TANK_TYPE_ENEMY;
}

/**
 * 从坦克发射子弹
 *
 * 功能：根据坦克的位置和朝向，初始化子弹的位置和速度
 *
 * 参数：
 *   bullet - 要发射的子弹指针
 *   tank   - 发射子弹的坦克指针
 *
 * 实现细节：
 *   - 子弹位置：坦克中心偏移，位于炮筒前方
 *   - 子弹速度：根据朝向设置（BULLET_SPEED = 10）
 *   - 记录发射者ID和类型（用于碰撞检测）
 *
 * 位置计算：
 *   - 向上：子弹在坦克上方，水平居中
 *   - 向下：子弹在坦克下方，水平居中
 *   - 向左：子弹在坦克左侧，垂直居中
 *   - 向右：子弹在坦克右侧，垂直居中
 */
void bullet_fire(Bullet* bullet, Tank* tank) {
    if (!tank->alive) return;  // 死亡的坦克不能射击

    bullet->active = true;                // 激活子弹
    bullet->owner_id = tank->id;          // 记录发射者ID
    bullet->owner_type = tank->type;      // 记录发射者类型

    // 根据坦克朝向设置子弹位置和速度
    // 目标：子弹从炮筒前方发射，速度方向与炮筒一致
    switch (tank->direction) {
        case DIR_UP:
            // 向上发射：子弹在坦克上方，水平居中
            bullet->x = tank->x + tank->width / 2 - BULLET_SIZE / 2;
            bullet->y = tank->y - BULLET_SIZE;
            bullet->vx = 0;
            bullet->vy = -BULLET_SPEED;  // Y轴负方向
            break;
        case DIR_DOWN:
            // 向下发射：子弹在坦克下方，水平居中
            bullet->x = tank->x + tank->width / 2 - BULLET_SIZE / 2;
            bullet->y = tank->y + tank->height;
            bullet->vx = 0;
            bullet->vy = BULLET_SPEED;   // Y轴正方向
            break;
        case DIR_LEFT:
            // 向左发射：子弹在坦克左侧，垂直居中
            bullet->x = tank->x - BULLET_SIZE;
            bullet->y = tank->y + tank->height / 2 - BULLET_SIZE / 2;
            bullet->vx = -BULLET_SPEED;  // X轴负方向
            bullet->vy = 0;
            break;
        case DIR_RIGHT:
            // 向右发射：子弹在坦克右侧，垂直居中
            bullet->x = tank->x + tank->width;
            bullet->y = tank->y + tank->height / 2 - BULLET_SIZE / 2;
            bullet->vx = BULLET_SPEED;   // X轴正方向
            bullet->vy = 0;
            break;
    }
}

/**
 * 更新子弹（每帧调用）
 *
 * 功能：
 *   - 应用速度到位置（移动子弹）
 *   - 检测边界，出界则失效
 *
 * 参数：
 *   bullet     - 要更新的子弹指针
 *   map_width  - 地图宽度（像素）
 *   map_height - 地图高度（像素）
 *
 * 实现细节：
 *   - 子弹速度恒定（无衰减）
 *   - 出界判定：任何边界
 *   - 出界后设置 active = false，可被复用
 */
void bullet_update(Bullet* bullet, int map_width, int map_height) {
    if (!bullet->active) return;  // 未激活的子弹不更新

    // 应用速度到位置（匀速运动）
    bullet->x += bullet->vx;
    bullet->y += bullet->vy;

    // 边界检查：出界则失效
    // 出界条件：任何一边超出地图边界
    if (bullet->x < 0 || bullet->x > map_width ||
        bullet->y < 0 || bullet->y > map_height) {
        bullet->active = false;  // 失效，可被复用
    }
}
