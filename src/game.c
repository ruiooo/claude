/*
 * game.c - 游戏核心逻辑实现
 */

#include "game.h"
#include "collision.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 初始化游戏
void game_init(GameState* game, int map_width, int map_height, GameMode mode) {
    memset(game, 0, sizeof(GameState));

    game->map_width = map_width;
    game->map_height = map_height;
    game->mode = mode;
    game->frame_count = 0;
    game->episode_count = 0;
    game->game_over = false;
    game->winner = -1;
    game->tank_count = 0;

    // 初始化所有子弹
    for (int i = 0; i < MAX_BULLETS; i++) {
        bullet_init(&game->bullets[i]);
    }
}

// 重置游戏回合
void game_reset(GameState* game, int enemy_count) {
    game->tank_count = 0;
    game->frame_count = 0;
    game->game_over = false;
    game->winner = -1;
    game->episode_count++;

    // 清空所有子弹
    for (int i = 0; i < MAX_BULLETS; i++) {
        game->bullets[i].active = false;
    }

    // 创建AI坦克（蓝色）
    float ai_x = game->map_width / 2.0f;
    float ai_y = game->map_height / 2.0f;
    tank_init(&game->tanks[0], ai_x, ai_y, TANK_TYPE_AI);
    game->tank_count = 1;

    // 创建敌人坦克（在四周生成）
    for (int i = 0; i < enemy_count; i++) {
        float x, y;
        int edge = rand() % 4;

        switch (edge) {
            case 0: // 上边
                x = rand() % (game->map_width - TANK_SIZE);
                y = TANK_SIZE;
                break;
            case 1: // 下边
                x = rand() % (game->map_width - TANK_SIZE);
                y = game->map_height - TANK_SIZE * 2;
                break;
            case 2: // 左边
                x = TANK_SIZE;
                y = rand() % (game->map_height - TANK_SIZE);
                break;
            case 3: // 右边
                x = game->map_width - TANK_SIZE * 2;
                y = rand() % (game->map_height - TANK_SIZE);
                break;
        }

        tank_init(&game->tanks[game->tank_count], x, y, TANK_TYPE_ENEMY);
        enemy_ai_init(&game->enemy_ais[game->tank_count], &game->tanks[0]);
        game->tank_count++;
    }
}

// 添加敌人
void game_add_enemy(GameState* game, TankType type, int model_version) {
    if (game->tank_count >= 32) return;

    float x = rand() % (game->map_width - TANK_SIZE);
    float y = rand() % (game->map_height - TANK_SIZE);

    tank_init(&game->tanks[game->tank_count], x, y, type);
    tank_set_model_version(&game->tanks[game->tank_count], model_version);

    if (type == TANK_TYPE_ENEMY) {
        enemy_ai_init(&game->enemy_ais[game->tank_count], &game->tanks[0]);
    }

    game->tank_count++;
}

// 获取AI坦克
Tank* game_get_ai_tank(GameState* game) {
    for (int i = 0; i < game->tank_count; i++) {
        if (game->tanks[i].type == TANK_TYPE_AI) {
            return &game->tanks[i];
        }
    }
    return NULL;
}

// 获取特定类型的存活坦克数量
int game_get_alive_count(GameState* game, TankType type) {
    int count = 0;
    for (int i = 0; i < game->tank_count; i++) {
        if (game->tanks[i].alive && game->tanks[i].type == type) {
            count++;
        }
    }
    return count;
}

// 执行动作
void game_execute_action(GameState* game, int tank_id, TankAction action) {
    if (tank_id < 0 || tank_id >= game->tank_count) return;

    Tank* tank = &game->tanks[tank_id];
    if (!tank->alive) return;

    switch (action) {
        case ACTION_IDLE:
            break;
        case ACTION_MOVE_UP:
            tank_move(tank, DIR_UP, game->map_width, game->map_height);
            break;
        case ACTION_MOVE_DOWN:
            tank_move(tank, DIR_DOWN, game->map_width, game->map_height);
            break;
        case ACTION_MOVE_LEFT:
            tank_move(tank, DIR_LEFT, game->map_width, game->map_height);
            break;
        case ACTION_MOVE_RIGHT:
            tank_move(tank, DIR_RIGHT, game->map_width, game->map_height);
            break;
        case ACTION_SHOOT_UP:
        case ACTION_SHOOT_DOWN:
        case ACTION_SHOOT_LEFT:
        case ACTION_SHOOT_RIGHT:
            if (tank_can_shoot(tank)) {
                // 设置射击方向
                if (action == ACTION_SHOOT_UP) tank->direction = DIR_UP;
                else if (action == ACTION_SHOOT_DOWN) tank->direction = DIR_DOWN;
                else if (action == ACTION_SHOOT_LEFT) tank->direction = DIR_LEFT;
                else if (action == ACTION_SHOOT_RIGHT) tank->direction = DIR_RIGHT;

                // 寻找空闲子弹
                for (int i = 0; i < MAX_BULLETS; i++) {
                    if (!game->bullets[i].active) {
                        bullet_fire(&game->bullets[i], tank);
                        tank_start_shoot_cooldown(tank);
                        break;
                    }
                }
            }
            break;
    }
}

// 更新游戏
void game_update(GameState* game) {
    if (game->game_over) return;

    game->frame_count++;

    // 更新所有坦克
    for (int i = 0; i < game->tank_count; i++) {
        tank_update(&game->tanks[i]);

        // 更新敌人AI
        if (game->tanks[i].alive && game->tanks[i].type == TANK_TYPE_ENEMY) {
            TankAction action = enemy_ai_update(&game->enemy_ais[i], &game->tanks[i],
                                               game->tanks, game->tank_count,
                                               game->bullets, MAX_BULLETS,
                                               game->frame_count);
            game_execute_action(game, i, action);
        }
    }

    // 更新所有子弹
    for (int i = 0; i < MAX_BULLETS; i++) {
        bullet_update(&game->bullets[i], game->map_width, game->map_height);
    }

    // 处理碰撞
    handle_collisions(game->tanks, game->tank_count, game->bullets, MAX_BULLETS);

    // 检查游戏结束条件
    if (game->mode == MODE_PLAYER) {
        // 多人对战模式：检查玩家坦克
        int player1_alive = 0;
        int player2_alive = 0;

        for (int i = 0; i < game->tank_count; i++) {
            if (game->tanks[i].type == TANK_TYPE_PLAYER && game->tanks[i].alive) {
                player1_alive = 1;
            }
            if (game->tanks[i].type == TANK_TYPE_PLAYER2 && game->tanks[i].alive) {
                player2_alive = 1;
            }
        }

        if (!player1_alive) {
            // 玩家1死亡，玩家2获胜
            game->game_over = true;
            game->winner = 1;
        } else if (!player2_alive) {
            // 玩家2死亡，玩家1获胜
            game->game_over = true;
            game->winner = 0;
        }
    } else {
        // AI训练模式：检查AI坦克和敌人
        Tank* ai_tank = game_get_ai_tank(game);
        int enemy_alive = game_get_alive_count(game, TANK_TYPE_ENEMY) +
                         game_get_alive_count(game, TANK_TYPE_SELF_PLAY);

        if (ai_tank && !ai_tank->alive) {
            // AI坦克死亡，敌人获胜
            game->game_over = true;
            game->winner = 1;
        } else if (enemy_alive == 0 && game->tank_count > 1) {
            // 所有敌人死亡，AI获胜
            game->game_over = true;
            game->winner = 0;
        }
    }

    // 超时判定（避免无限循环）
    if (game->frame_count > 3600) {  // 60秒超时（60fps）
        game->game_over = true;
        game->winner = -1;  // 平局
    }
}

// 获取观察状态（用于AI）
void game_get_observation(GameState* game, float* obs, int* obs_size) {
    int idx = 0;

    Tank* ai_tank = game_get_ai_tank(game);
    if (!ai_tank) {
        *obs_size = 0;
        return;
    }

    // AI坦克状态 (6个值)
    obs[idx++] = ai_tank->x / game->map_width;
    obs[idx++] = ai_tank->y / game->map_height;
    obs[idx++] = ai_tank->vx / TANK_SPEED;
    obs[idx++] = ai_tank->vy / TANK_SPEED;
    obs[idx++] = (float)ai_tank->health / TANK_MAX_HEALTH;
    obs[idx++] = ai_tank->shoot_cooldown > 0 ? 1.0f : 0.0f;

    // 最近的5个敌人位置和状态 (5 * 5 = 25个值)
    float enemy_info[5][5] = {0};  // x, y, vx, vy, health
    int enemy_found = 0;

    for (int i = 0; i < game->tank_count && enemy_found < 5; i++) {
        if (game->tanks[i].type != TANK_TYPE_AI && game->tanks[i].alive) {
            float dx = game->tanks[i].x - ai_tank->x;
            float dy = game->tanks[i].y - ai_tank->y;
            float dist = sqrtf(dx * dx + dy * dy);

            enemy_info[enemy_found][0] = dx / game->map_width;
            enemy_info[enemy_found][1] = dy / game->map_height;
            enemy_info[enemy_found][2] = game->tanks[i].vx / TANK_SPEED;
            enemy_info[enemy_found][3] = game->tanks[i].vy / TANK_SPEED;
            enemy_info[enemy_found][4] = (float)game->tanks[i].health / TANK_MAX_HEALTH;
            enemy_found++;
        }
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            obs[idx++] = enemy_info[i][j];
        }
    }

    // 最近的3个子弹 (3 * 4 = 12个值)
    float bullet_info[3][4] = {0};  // x, y, vx, vy
    int bullet_found = 0;

    for (int i = 0; i < MAX_BULLETS && bullet_found < 3; i++) {
        if (game->bullets[i].active && game->bullets[i].owner_id != ai_tank->id) {
            float dx = game->bullets[i].x - ai_tank->x;
            float dy = game->bullets[i].y - ai_tank->y;

            bullet_info[bullet_found][0] = dx / game->map_width;
            bullet_info[bullet_found][1] = dy / game->map_height;
            bullet_info[bullet_found][2] = game->bullets[i].vx / BULLET_SPEED;
            bullet_info[bullet_found][3] = game->bullets[i].vy / BULLET_SPEED;
            bullet_found++;
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            obs[idx++] = bullet_info[i][j];
        }
    }

    *obs_size = idx;  // 总共 6 + 25 + 12 = 43 维
}
