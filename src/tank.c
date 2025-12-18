/*
 * tank.c - 坦克实体实现
 *
 * 模块说明：
 * 本模块负责坦克实体的核心逻辑，包括：
 * - 坦克的初始化和状态管理
 * - 移动控制和速度衰减
 * - 伤害处理和生命值管理
 * - 射击冷却机制
 *
 * 关键特性：
 * - 支持多种坦克类型（玩家、AI、敌人、历史版本等）
 * - 平滑的移动物理（速度衰减）
 * - 玩家坦克和AI坦克使用不同的移动感
 * - 自动边界检查，防止出界
 */

#include "tank.h"
#include <stdlib.h>
#include <string.h>

/**
 * 初始化坦克
 *
 * 功能：设置坦克的初始状态，包括位置、血量、类型等
 *
 * 参数：
 *   tank - 要初始化的坦克指针
 *   x    - 初始X坐标（像素）
 *   y    - 初始Y坐标（像素）
 *   type - 坦克类型（玩家/AI/敌人/自我对弈）
 *
 * 实现细节：
 *   - 坦克尺寸固定为 TANK_SIZE (32x32像素)
 *   - 初始血量为 TANK_MAX_HEALTH (3点)
 *   - 初始方向朝上
 *   - 随机分配唯一ID（用于子弹碰撞检测）
 */
void tank_init(Tank* tank, float x, float y, TankType type) {
    tank->x = x;                          // 设置X坐标
    tank->y = y;                          // 设置Y坐标
    tank->vx = 0;                         // 初始速度为0
    tank->vy = 0;
    tank->width = TANK_SIZE;              // 宽度32像素
    tank->height = TANK_SIZE;             // 高度32像素
    tank->health = TANK_MAX_HEALTH;       // 满血（3点）
    tank->alive = true;                   // 存活状态
    tank->type = type;                    // 坦克类型
    tank->direction = DIR_UP;             // 初始朝向：向上
    tank->shoot_cooldown = 0;             // 射击冷却为0（可以立即射击）
    tank->action_cooldown = 0;            // 动作切换冷却为0
    tank->direction_change_cooldown = 0;  // 方向转变冷却为0（可以立即转向）
    tank->model_version = 0;              // 模型版本（用于自我对弈）
    tank->id = rand() % 10000;            // 随机ID，用于区分不同坦克
}

/**
 * 更新坦克状态（每帧调用）
 *
 * 功能：
 *   - 更新射击冷却计时
 *   - 应用速度到位置（物理更新）
 *   - 实现速度衰减（模拟摩擦力）
 *
 * 参数：
 *   tank - 要更新的坦克指针
 *
 * 物理模型：
 *   - 玩家坦克：快速停止（0.5倍衰减），提供精确控制感
 *   - AI/敌人坦克：平滑移动（0.95倍衰减），更自然的运动
 *   - 速度阈值：小于0.1时归零，避免持续微小抖动
 */
void tank_update(Tank* tank) {
    if (!tank->alive) return;  // 死亡的坦克不更新

    // 更新射击冷却计时器（每帧递减1）
    if (tank->shoot_cooldown > 0) {
        tank->shoot_cooldown--;
    }

    // 更新动作切换冷却计时器（每帧递减1）
    if (tank->action_cooldown > 0) {
        tank->action_cooldown--;
    }

    // 更新方向转变冷却计时器（每帧递减1）
    if (tank->direction_change_cooldown > 0) {
        tank->direction_change_cooldown--;
    }

    // 应用速度到位置（欧拉积分法）
    tank->x += tank->vx;
    tank->y += tank->vy;

    // 平滑衰减速度 - 玩家坦克快速停止，AI/敌人坦克平滑移动
    // 设计理念：不同类型的坦克有不同的操作感
    if (tank->type == TANK_TYPE_PLAYER || tank->type == TANK_TYPE_PLAYER2) {
        // 玩家坦克：快速停止，无滑行感（衰减系数0.5）
        // 这样玩家可以精确控制位置，适合人类操作
        tank->vx *= 0.5f;
        tank->vy *= 0.5f;
    } else {
        // AI和敌人坦克：平滑移动（衰减系数0.95）
        // 更自然的运动轨迹，更符合AI的连续决策特性
        tank->vx *= 0.95f;
        tank->vy *= 0.95f;
    }

    // 速度太小时归零，避免持续微小抖动
    // 阈值0.1：低于此值视为静止
    if (tank->vx > -0.1f && tank->vx < 0.1f) tank->vx = 0;
    if (tank->vy > -0.1f && tank->vy < 0.1f) tank->vy = 0;
}

/**
 * 移动坦克（带边界检查）
 *
 * 功能：设置坦克的移动方向和速度，并进行边界碰撞检测
 *
 * 参数：
 *   tank       - 要移动的坦克指针
 *   dir        - 移动方向（上/下/左/右）
 *   map_width  - 地图宽度（像素）
 *   map_height - 地图高度（像素）
 *
 * 实现细节：
 *   - 检测方向是否改变（与当前朝向比较）
 *   - 设置坦克朝向（用于渲染炮筒和射击）
 *   - 根据方向设置速度（TANK_SPEED = 2.5）
 *   - 预测下一帧位置，进行边界碰撞检测
 *   - 如果即将出界，停止该方向的移动（速度归零）
 *   - 只有改变方向时才设置动作冷却（同方向连续移动可以射击）
 *
 * 边界检查：
 *   - 左边界：x < 0
 *   - 右边界：x + width > map_width
 *   - 上边界：y < 0
 *   - 下边界：y + height > map_height
 */
void tank_move(Tank* tank, Direction dir, int map_width, int map_height) {
    if (!tank->alive) return;  // 死亡的坦克不能移动

    // ✅ 只有改变方向时才设置动作冷却
    // 同一方向连续移动时不设置冷却，可以射击
    bool direction_changed = (tank->direction != dir);

    // ✅ 如果要改变方向，检查方向转变冷却
    // 如果冷却中，不允许改变方向，保持当前方向移动
    if (direction_changed && tank->direction_change_cooldown > 0) {
        dir = tank->direction;  // 保持当前方向
        direction_changed = false;  // 标记为未改变方向
    }

    tank->direction = dir;  // 更新朝向（影响渲染和射击）

    // 设置速度，不直接修改位置（位置由tank_update中的速度积分更新）
    // 使用速度系统的优点：支持平滑衰减和碰撞响应
    switch (dir) {
        case DIR_UP:
            tank->vy = -TANK_SPEED;  // 向上：Y轴负方向
            tank->vx = 0;            // 停止水平移动
            break;
        case DIR_DOWN:
            tank->vy = TANK_SPEED;   // 向下：Y轴正方向
            tank->vx = 0;
            break;
        case DIR_LEFT:
            tank->vx = -TANK_SPEED;  // 向左：X轴负方向
            tank->vy = 0;            // 停止垂直移动
            break;
        case DIR_RIGHT:
            tank->vx = TANK_SPEED;   // 向右：X轴正方向
            tank->vy = 0;
            break;
    }

    // 预测下一帧位置并进行边界检查
    // 预测法优点：避免穿墙，提前阻止非法移动
    float next_x = tank->x + tank->vx;
    float next_y = tank->y + tank->vy;

    // 如果即将碰到边界，停止该方向的移动
    // 这样坦克会贴着边界停下，而不是穿出去
    if (next_x < 0 || next_x + tank->width > map_width) {
        tank->vx = 0;  // 水平方向停止
    }

    if (next_y < 0 || next_y + tank->height > map_height) {
        tank->vy = 0;  // 垂直方向停止
    }

    // ✅ 只有改变方向时才设置动作冷却，同方向连续移动时不设置
    if (direction_changed) {
        tank->action_cooldown = ACTION_COOLDOWN;
        // ✅ 设置方向转变冷却，防止频繁转向（如快速左右摇摆）
        tank->direction_change_cooldown = DIR_CHANGE_COOLDOWN;
    }
}

/**
 * 坦克受到伤害
 *
 * 功能：减少坦克血量，血量耗尽时标记为死亡
 *
 * 参数：
 *   tank   - 受伤的坦克指针
 *   damage - 伤害值（通常为1）
 *
 * 实现细节：
 *   - 死亡的坦克不再受伤
 *   - 血量不会低于0
 *   - 血量归零时设置alive为false
 */
void tank_take_damage(Tank* tank, int damage) {
    if (!tank->alive) return;  // 死亡的坦克不再受伤

    tank->health -= damage;
    if (tank->health <= 0) {
        tank->health = 0;      // 血量不低于0
        tank->alive = false;   // 标记为死亡
    }
}

/**
 * 检查是否可以射击
 *
 * 功能：检查坦克是否满足射击条件
 *
 * 参数：
 *   tank - 要检查的坦克指针
 *
 * 返回值：
 *   true  - 可以射击（存活、射击冷却完毕、动作冷却完毕）
 *   false - 不可射击（死亡、射击冷却中、或刚移动后）
 */
bool tank_can_shoot(Tank* tank) {
    // ✅ 修改：增加动作冷却检查，刚移动后不能立即射击
    return tank->alive && tank->shoot_cooldown == 0 && tank->action_cooldown == 0;
}

/**
 * 开始射击冷却
 *
 * 功能：设置射击冷却计时器，防止连续射击
 *
 * 参数：
 *   tank - 刚射击的坦克指针
 *
 * 冷却时间：
 *   SHOOT_COOLDOWN = 45帧（约0.75秒 @ 60fps）- 射击冷却
 *
 * 注意：
 *   射击后不设置action_cooldown，允许立即移动
 */
void tank_start_shoot_cooldown(Tank* tank) {
    tank->shoot_cooldown = SHOOT_COOLDOWN;
}

/**
 * 设置模型版本（用于自我对弈系统）
 *
 * 功能：标记坦克使用的AI模型版本号
 *
 * 参数：
 *   tank    - 坦克指针
 *   version - 模型版本号（如200表示第200回合保存的模型）
 *
 * 用途：
 *   - 在自我对弈中，历史版本AI作为敌人
 *   - 渲染时显示版本号，便于调试
 */
void tank_set_model_version(Tank* tank, int version) {
    tank->model_version = version;
}
