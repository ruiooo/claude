/*
 * bullet.h - 子弹实体定义和相关函数
 *
 * 模块说明：
 *   定义子弹实体的数据结构和行为
 *   提供子弹的初始化、发射、更新等功能
 *   管理子弹池（固定大小数组）和碰撞检测
 *
 * 使用场景：
 *   - 坦克射击：根据坦克朝向发射子弹
 *   - 子弹飞行：每帧更新子弹位置
 *   - 边界检测：超出地图边界时销毁子弹
 *   - 碰撞检测：与坦克碰撞时造成伤害
 */

#ifndef BULLET_H
#define BULLET_H

#include <stdbool.h>
#include "tank.h"

/*
 * 子弹实体结构体
 * 使用对象池模式管理，固定分配MAX_BULLETS个实例
 */
typedef struct {
    float x, y;          // 子弹中心位置（像素坐标，浮点数用于精确飞行轨迹）
    float vx, vy;        // 速度分量（像素/帧），通常为 [-BULLET_SPEED, BULLET_SPEED]
    int width, height;   // 子弹尺寸（像素），通常为 BULLET_SIZE x BULLET_SIZE
    bool active;         // 激活标志，true=正在飞行，false=已销毁或未使用
    int owner_id;        // 发射者坦克ID，用于防止自伤和归属判定
    TankType owner_type; // 发射者类型，用于判断敌我关系（AI子弹不伤害AI坦克）
} Bullet;

/*
 * 子弹相关常量定义
 */
#define BULLET_SIZE 8           // 子弹尺寸（像素），正方形边长
#define BULLET_SPEED 5.0f       // 子弹飞行速度（像素/帧），速度越快越难躲避
#define MAX_BULLETS 100         // 子弹池最大容量，超过此数量无法发射新子弹

/*
 * 子弹功能函数声明
 */

/*
 * 初始化子弹为未激活状态
 * @param bullet: 待初始化的子弹指针
 * 功能：设置 active=false，清空位置和速度
 * 用途：游戏启动时初始化子弹池中的所有子弹
 */
void bullet_init(Bullet* bullet);

/*
 * 从坦克发射子弹
 * @param bullet: 待激活的子弹指针（必须是未激活的）
 * @param tank: 发射子弹的坦克指针
 * 功能：
 *   - 根据坦克朝向设置子弹速度（vx, vy）
 *   - 设置子弹初始位置为坦克中心
 *   - 记录发射者信息（owner_id, owner_type）
 *   - 激活子弹（active=true）
 * 注意：调用前应先检查坦克射击冷却是否完成
 */
void bullet_fire(Bullet* bullet, Tank* tank);

/*
 * 更新子弹状态（每帧调用）
 * @param bullet: 待更新的子弹指针
 * @param map_width: 地图宽度（像素），用于边界检测
 * @param map_height: 地图高度（像素），用于边界检测
 * 功能：
 *   - 更新位置（x += vx, y += vy）
 *   - 边界检测：超出地图范围时设置 active=false
 * 注意：只更新激活状态的子弹（active=true）
 */
void bullet_update(Bullet* bullet, int map_width, int map_height);

#endif // BULLET_H
