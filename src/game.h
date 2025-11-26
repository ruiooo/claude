/*
 * game.h - 游戏核心逻辑
 */

#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include "tank.h"
#include "bullet.h"
#include "enemy_ai.h"

// 游戏模式
typedef enum {
    MODE_TRAINING,       // 训练模式
    MODE_TRAINING_VIS,   // 训练可视化模式
    MODE_PLAYER          // 玩家对战模式
} GameMode;

// 游戏状态
typedef struct {
    Tank tanks[32];           // 所有坦克（1个AI + 最多31个敌人）
    int tank_count;           // 坦克数量
    Bullet bullets[MAX_BULLETS]; // 所有子弹
    EnemyAI enemy_ais[32];    // 敌人AI
    int frame_count;          // 帧计数器
    int episode_count;        // 回合计数
    bool game_over;           // 游戏结束
    int winner;               // 胜者（0=AI, 1=敌人, -1=未结束）
    int map_width;            // 地图宽度
    int map_height;           // 地图高度
    GameMode mode;            // 游戏模式
} GameState;

// 函数声明
void game_init(GameState* game, int map_width, int map_height, GameMode mode);
void game_reset(GameState* game, int enemy_count);
void game_update(GameState* game);
void game_execute_action(GameState* game, int tank_id, TankAction action);
void game_add_enemy(GameState* game, TankType type, int model_version);
Tank* game_get_ai_tank(GameState* game);
int game_get_alive_count(GameState* game, TankType type);

// 获取游戏状态（用于AI观察）
void game_get_observation(GameState* game, float* obs, int* obs_size);

#endif // GAME_H
