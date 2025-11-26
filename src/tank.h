/*
 * tank.h - 坦克实体定义和相关函数
 */

#ifndef TANK_H
#define TANK_H

#include <stdbool.h>

// 坦克类型
typedef enum {
    TANK_TYPE_AI,           // AI控制的坦克（蓝色）
    TANK_TYPE_ENEMY,        // 追踪型敌人（红色）
    TANK_TYPE_SELF_PLAY,    // 历史版本AI敌人（红白色）
    TANK_TYPE_PLAYER        // 玩家控制
} TankType;

// 坦克动作
typedef enum {
    ACTION_IDLE = 0,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_SHOOT_UP,
    ACTION_SHOOT_DOWN,
    ACTION_SHOOT_LEFT,
    ACTION_SHOOT_RIGHT,
    ACTION_COUNT
} TankAction;

// 坦克方向
typedef enum {
    DIR_UP = 0,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT
} Direction;

// 坦克结构
typedef struct {
    float x, y;              // 位置
    float vx, vy;            // 速度
    int width, height;       // 尺寸
    int health;              // 血量（最大3）
    bool alive;              // 是否存活
    TankType type;           // 坦克类型
    Direction direction;     // 当前朝向
    int shoot_cooldown;      // 射击冷却（帧数）
    int model_version;       // 模型版本（用于历史版本敌人）
    int id;                  // 坦克ID
} Tank;

// 常量
#define TANK_SIZE 32
#define TANK_SPEED 1.8f
#define TANK_MAX_HEALTH 3
#define SHOOT_COOLDOWN 30  // 30帧约0.5秒（60fps）

// 函数声明
void tank_init(Tank* tank, float x, float y, TankType type);
void tank_update(Tank* tank);
void tank_move(Tank* tank, Direction dir, int map_width, int map_height,
               Tank* all_tanks, int tank_count);
void tank_take_damage(Tank* tank, int damage);
bool tank_can_shoot(Tank* tank);
void tank_start_shoot_cooldown(Tank* tank);
void tank_set_model_version(Tank* tank, int version);

#endif // TANK_H
