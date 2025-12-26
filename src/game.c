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

        // ✅ 修复：玩家模式下，AI由main.c控制，这里不再重复控制
        // 只在训练模式下更新敌人AI
        if (game->mode != MODE_PLAYER &&
            game->tanks[i].alive && game->tanks[i].type == TANK_TYPE_ENEMY) {
            TankAction action = enemy_ai_update(&game->enemy_ais[i], &game->tanks[i],
                                               game->tanks, game->tank_count,
                                               game->bullets, MAX_BULLETS,
                                               game->frame_count,
                                               game->map_width, game->map_height);
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
        // 玩家模式：需要区分多人对战和玩家vs AI
        int player1_alive = 0;
        int player2_alive = 0;
        int enemy_alive = 0;

        for (int i = 0; i < game->tank_count; i++) {
            if (game->tanks[i].type == TANK_TYPE_PLAYER && game->tanks[i].alive) {
                player1_alive = 1;
            }
            if (game->tanks[i].type == TANK_TYPE_PLAYER2 && game->tanks[i].alive) {
                player2_alive = 1;
            }
            if ((game->tanks[i].type == TANK_TYPE_ENEMY ||
                 game->tanks[i].type == TANK_TYPE_SELF_PLAY) && game->tanks[i].alive) {
                enemy_alive = 1;
            }
        }

        // 检查是否有PLAYER2（多人对战）
        bool has_player2 = false;
        for (int i = 0; i < game->tank_count; i++) {
            if (game->tanks[i].type == TANK_TYPE_PLAYER2) {
                has_player2 = true;
                break;
            }
        }

        if (has_player2) {
            // 多人对战模式：PLAYER vs PLAYER2
            if (!player1_alive) {
                game->game_over = true;
                game->winner = 1;
            } else if (!player2_alive) {
                game->game_over = true;
                game->winner = 0;
            }
        } else {
            // 玩家挑战模式：PLAYER vs ENEMY
            if (!player1_alive) {
                // 玩家死亡，敌人获胜
                game->game_over = true;
                game->winner = 1;
            } else if (!enemy_alive && game->tank_count > 1) {
                // 所有敌人死亡，玩家获胜
                game->game_over = true;
                game->winner = 0;
            }
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

/*
 * 获取观察状态 v2.0（用于AI训练和推理）
 *
 * 【核心函数】此函数构建DQN神经网络的输入状态向量
 *
 * 【v2.0升级】增加战略信息维度
 * - 射击冷却改为连续值（更精确的时机判断）
 * - 新增敌人总数（战术意识）
 * - 新增最近敌人距离（快速威胁评估）
 * - 新增墙壁距离信息（位置感知）
 *
 * 状态维度分解：共47维（原43维+4维新增）
 * ┌─────────────────────────────────────────────────────────────┐
 * │ 第1部分：AI坦克自身状态 (6维)                               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ [0] x归一化位置         = x / map_width      范围: [0, 1]  │
 * │ [1] y归一化位置         = y / map_height     范围: [0, 1]  │
 * │ [2] x方向速度归一化     = vx / TANK_SPEED    范围: [-1,1]  │
 * │ [3] y方向速度归一化     = vy / TANK_SPEED    范围: [-1,1]  │
 * │ [4] 血量归一化          = health / MAX_HEALTH 范围: [0,1]  │
 * │ [5] 射击冷却归一化      = cooldown/MAX_CD    范围: [0,1]   │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 第2部分：最近5个敌人状态 (5×5 = 25维)                      │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 每个敌人5维：dx, dy, vx, vy, health（归一化）              │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 第3部分：最近3个敌方子弹状态 (3×4 = 12维)                  │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 每个子弹4维：dx, dy, vx, vy（归一化）                      │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 第4部分：战略信息 (4维) 【新增】                            │
 * ├─────────────────────────────────────────────────────────────┤
 * │ [43] 存活敌人数量归一化  = count / 10       范围: [0, 1]   │
 * │ [44] 最近敌人距离归一化  = dist / 800       范围: [0, 1]   │
 * │ [45] 墙壁距离(水平最近)  = min(x,w-x)/w     范围: [0, 0.5] │
 * │ [46] 墙壁距离(垂直最近)  = min(y,h-y)/h     范围: [0, 0.5] │
 * └─────────────────────────────────────────────────────────────┘
 *
 * 关键点：
 * - 此函数用于训练时生成状态（通过ctypes调用）
 * - 必须与 model_ai.c:game_state_to_observation() 完全一致
 * - 必须与 python/config.py:MODEL_CONFIG['state_dim']=47 匹配
 */
void game_get_observation(GameState* game, float* obs, int* obs_size) {
    int idx = 0;

    Tank* ai_tank = game_get_ai_tank(game);
    if (!ai_tank) {
        *obs_size = 0;
        return;
    }

    // ========== 第1部分：AI坦克自身状态 (6维) ==========

    // [0-1] 位置归一化
    obs[idx++] = ai_tank->x / game->map_width;
    obs[idx++] = ai_tank->y / game->map_height;

    // [2-3] 速度归一化
    obs[idx++] = ai_tank->vx / TANK_SPEED;
    obs[idx++] = ai_tank->vy / TANK_SPEED;

    // [4] 血量归一化
    obs[idx++] = (float)ai_tank->health / TANK_MAX_HEALTH;

    // [5] 射击冷却归一化（改为连续值，提供更精确的时机信息）
    obs[idx++] = (float)ai_tank->shoot_cooldown / SHOOT_COOLDOWN;

    // ========== 第2部分：最近5个敌人状态 (25维) ==========

    float enemy_info[5][5] = {0};
    int enemy_found = 0;
    float min_enemy_dist = 9999.0f;  // 记录最近敌人距离

    // 先收集所有敌人信息并按距离排序
    typedef struct {
        float dx, dy, vx, vy, health, dist;
    } EnemyData;
    EnemyData all_enemies[32];
    int total_enemies = 0;

    for (int i = 0; i < game->tank_count; i++) {
        if (game->tanks[i].type != TANK_TYPE_AI && game->tanks[i].alive) {
            float dx = game->tanks[i].x - ai_tank->x;
            float dy = game->tanks[i].y - ai_tank->y;
            float dist = sqrtf(dx*dx + dy*dy);

            all_enemies[total_enemies].dx = dx;
            all_enemies[total_enemies].dy = dy;
            all_enemies[total_enemies].vx = game->tanks[i].vx;
            all_enemies[total_enemies].vy = game->tanks[i].vy;
            all_enemies[total_enemies].health = (float)game->tanks[i].health;
            all_enemies[total_enemies].dist = dist;

            if (dist < min_enemy_dist) {
                min_enemy_dist = dist;
            }
            total_enemies++;
        }
    }

    // 简单冒泡排序（按距离从近到远）
    for (int i = 0; i < total_enemies - 1; i++) {
        for (int j = 0; j < total_enemies - i - 1; j++) {
            if (all_enemies[j].dist > all_enemies[j+1].dist) {
                EnemyData temp = all_enemies[j];
                all_enemies[j] = all_enemies[j+1];
                all_enemies[j+1] = temp;
            }
        }
    }

    // 取最近的5个敌人
    for (int i = 0; i < 5 && i < total_enemies; i++) {
        enemy_info[enemy_found][0] = all_enemies[i].dx / game->map_width;
        enemy_info[enemy_found][1] = all_enemies[i].dy / game->map_height;
        enemy_info[enemy_found][2] = all_enemies[i].vx / TANK_SPEED;
        enemy_info[enemy_found][3] = all_enemies[i].vy / TANK_SPEED;
        enemy_info[enemy_found][4] = all_enemies[i].health / TANK_MAX_HEALTH;
        enemy_found++;
    }

    // 展开敌人信息到状态向量（固定25维）
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            obs[idx++] = enemy_info[i][j];
        }
    }

    // ========== 第3部分：最近3个敌方子弹状态 (12维) ==========

    // 收集所有敌方子弹并按距离排序
    typedef struct {
        float dx, dy, vx, vy, dist;
    } BulletData;
    BulletData all_bullets[MAX_BULLETS];
    int total_bullets = 0;

    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active && game->bullets[i].owner_id != ai_tank->id) {
            float dx = game->bullets[i].x - ai_tank->x;
            float dy = game->bullets[i].y - ai_tank->y;
            float dist = sqrtf(dx*dx + dy*dy);

            all_bullets[total_bullets].dx = dx;
            all_bullets[total_bullets].dy = dy;
            all_bullets[total_bullets].vx = game->bullets[i].vx;
            all_bullets[total_bullets].vy = game->bullets[i].vy;
            all_bullets[total_bullets].dist = dist;
            total_bullets++;
        }
    }

    // 按距离排序
    for (int i = 0; i < total_bullets - 1; i++) {
        for (int j = 0; j < total_bullets - i - 1; j++) {
            if (all_bullets[j].dist > all_bullets[j+1].dist) {
                BulletData temp = all_bullets[j];
                all_bullets[j] = all_bullets[j+1];
                all_bullets[j+1] = temp;
            }
        }
    }

    // 取最近的3个子弹
    float bullet_info[3][4] = {0};
    for (int i = 0; i < 3 && i < total_bullets; i++) {
        bullet_info[i][0] = all_bullets[i].dx / game->map_width;
        bullet_info[i][1] = all_bullets[i].dy / game->map_height;
        bullet_info[i][2] = all_bullets[i].vx / BULLET_SPEED;
        bullet_info[i][3] = all_bullets[i].vy / BULLET_SPEED;
    }

    // 展开子弹信息（固定12维）
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            obs[idx++] = bullet_info[i][j];
        }
    }

    // ========== 第4部分：战略信息 (4维) 【新增】 ==========

    // [43] 存活敌人数量（归一化到0-1，假设最多10个敌人）
    obs[idx++] = (float)total_enemies / 10.0f;

    // [44] 最近敌人距离（归一化，对角线约1000像素）
    float normalized_min_dist = (min_enemy_dist < 9999.0f) ?
                                 min_enemy_dist / 800.0f : 1.0f;
    if (normalized_min_dist > 1.0f) normalized_min_dist = 1.0f;
    obs[idx++] = normalized_min_dist;

    // [45] 水平墙壁距离（距离左右边界最近的距离）
    float dist_to_left = ai_tank->x;
    float dist_to_right = game->map_width - ai_tank->x;
    float min_horizontal_wall = (dist_to_left < dist_to_right) ?
                                 dist_to_left : dist_to_right;
    obs[idx++] = min_horizontal_wall / game->map_width;

    // [46] 垂直墙壁距离（距离上下边界最近的距离）
    float dist_to_top = ai_tank->y;
    float dist_to_bottom = game->map_height - ai_tank->y;
    float min_vertical_wall = (dist_to_top < dist_to_bottom) ?
                               dist_to_top : dist_to_bottom;
    obs[idx++] = min_vertical_wall / game->map_height;

    // 最终验证：确保维度正确
    *obs_size = idx;  // 必须等于 6 + 25 + 12 + 4 = 47 维
}
