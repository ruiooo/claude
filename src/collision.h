/*
 * collision.h - 碰撞检测系统
 *
 * 模块说明：
 *   提供2D游戏中的碰撞检测功能
 *   使用AABB（轴对齐包围盒）算法进行高效检测
 *   处理坦克-坦克、子弹-坦克的所有碰撞情况
 *
 * 使用场景：
 *   - 坦克移动时防止穿透其他坦克
 *   - 子弹命中坦克时造成伤害
 *   - 每帧统一处理所有碰撞事件
 *
 * 算法：
 *   AABB碰撞检测 - 通过比较两个矩形的边界判断是否重叠
 *   时间复杂度：O(n*m)，n=坦克数，m=子弹数
 */

#ifndef COLLISION_H
#define COLLISION_H

#include <stdbool.h>
#include "tank.h"
#include "bullet.h"

/*
 * AABB（轴对齐包围盒）碰撞检测
 * @param x1, y1: 第一个矩形的中心坐标
 * @param w1, h1: 第一个矩形的宽度和高度
 * @param x2, y2: 第二个矩形的中心坐标
 * @param w2, h2: 第二个矩形的宽度和高度
 * @return: true=发生碰撞，false=未碰撞
 * 算法：检查两个矩形在X轴和Y轴上是否同时重叠
 * 公式：abs(x1-x2) < (w1+w2)/2 && abs(y1-y2) < (h1+h2)/2
 */
bool check_aabb_collision(float x1, float y1, int w1, int h1,
                          float x2, float y2, int w2, int h2);

/*
 * 坦克之间的碰撞检测
 * @param t1: 第一个坦克指针
 * @param t2: 第二个坦克指针
 * @return: true=发生碰撞，false=未碰撞
 * 功能：
 *   - 只检测存活的坦克（alive=true）
 *   - 调用AABB算法检测矩形重叠
 * 用途：防止坦克重叠，实现推挤效果
 */
bool check_tank_collision(Tank* t1, Tank* t2);

/*
 * 子弹和坦克的碰撞检测
 * @param bullet: 子弹指针
 * @param tank: 坦克指针
 * @return: true=命中，false=未命中
 * 功能：
 *   - 只检测激活的子弹（active=true）和存活的坦克（alive=true）
 *   - 防止自伤：子弹发射者不会被自己的子弹击中
 *   - 调用AABB算法检测碰撞
 * 判断条件：bullet->owner_id != tank->id
 */
bool check_bullet_tank_collision(Bullet* bullet, Tank* tank);

/*
 * 处理所有碰撞事件（每帧调用）
 * @param tanks: 坦克数组指针
 * @param tank_count: 坦克数量
 * @param bullets: 子弹数组指针
 * @param bullet_count: 子弹数量（通常为MAX_BULLETS）
 * @param map_width: 地图宽度（用于边界检查）
 * @param map_height: 地图高度（用于边界检查）
 * 功能：
 *   - 遍历所有激活的子弹
 *   - 检测每个子弹与所有存活坦克的碰撞
 *   - 发生碰撞时：
 *     * 坦克受伤（tank_take_damage）
 *     * 销毁子弹（bullet->active=false）
 *   - 坦克碰撞后进行边界限制，防止被推进墙里
 * 性能：O(n*m)，n=坦克数，m=激活子弹数
 */
void handle_collisions(Tank* tanks, int tank_count, Bullet* bullets, int bullet_count,
                       int map_width, int map_height);

#endif // COLLISION_H
