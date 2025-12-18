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
 * 获取观察状态（用于AI训练和推理）
 *
 * 【核心函数】此函数构建DQN神经网络的输入状态向量
 * 状态维度必须与Python训练代码和推理代码保持一致
 *
 * 状态表示设计原则：
 * 1. 归一化：所有值归一化到 [0, 1] 或 [-1, 1] 范围，便于神经网络学习
 * 2. 相对性：使用相对位置（dx, dy）而非绝对位置，增强泛化能力
 * 3. 固定维度：即使实际敌人/子弹数量变化，始终保持固定维度（填充0）
 *
 * 状态维度分解：共43维
 * ┌─────────────────────────────────────────────────────────────┐
 * │ 第1部分：AI坦克自身状态 (6维)                               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ [0] x归一化位置         = x / map_width      范围: [0, 1]  │
 * │ [1] y归一化位置         = y / map_height     范围: [0, 1]  │
 * │ [2] x方向速度归一化     = vx / TANK_SPEED    范围: [-1,1]  │
 * │ [3] y方向速度归一化     = vy / TANK_SPEED    范围: [-1,1]  │
 * │ [4] 血量归一化          = health / MAX_HEALTH 范围: [0,1]   │
 * │ [5] 射击冷却标志        = cooldown>0 ? 1:0   范围: {0,1}   │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 第2部分：最近5个敌人状态 (5×5 = 25维)                      │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 每个敌人5维信息（未找到敌人则填充0）：                      │
 * │   [0] 相对x距离归一化   = dx / map_width    范围: [-1,1]   │
 * │   [1] 相对y距离归一化   = dy / map_height   范围: [-1,1]   │
 * │   [2] 敌人vx归一化      = vx / TANK_SPEED   范围: [-1,1]   │
 * │   [3] 敌人vy归一化      = vy / TANK_SPEED   范围: [-1,1]   │
 * │   [4] 敌人血量归一化    = health / MAX_HEALTH 范围: [0,1]  │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 第3部分：最近3个敌方子弹状态 (3×4 = 12维)                  │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 每个子弹4维信息（未找到子弹则填充0）：                      │
 * │   [0] 相对x距离归一化   = dx / map_width    范围: [-1,1]   │
 * │   [1] 相对y距离归一化   = dy / map_height   范围: [-1,1]   │
 * │   [2] 子弹vx归一化      = vx / BULLET_SPEED 范围: [-1,1]   │
 * │   [3] 子弹vy归一化      = vy / BULLET_SPEED 范围: [-1,1]   │
 * └─────────────────────────────────────────────────────────────┘
 *
 * 关键点：
 * - 此函数用于训练时生成状态（通过ctypes调用）
 * - 必须与 model_ai.c:game_state_to_observation() 完全一致（推理时使用）
 * - 必须与 python/config.py:MODEL_CONFIG['state_dim']=43 匹配
 * - 任何修改都必须同步更新以上3处！
 */
void game_get_observation(GameState* game, float* obs, int* obs_size) {
    int idx = 0;

    Tank* ai_tank = game_get_ai_tank(game);
    if (!ai_tank) {
        *obs_size = 0;
        return;
    }

    // ========== 第1部分：AI坦克自身状态 (6维) ==========

    // [0-1] 位置归一化：将位置映射到 [0, 1]，让网络知道坦克在地图的哪个区域
    obs[idx++] = ai_tank->x / game->map_width;        // [0] x位置 (0=左边界, 1=右边界)
    obs[idx++] = ai_tank->y / game->map_height;       // [1] y位置 (0=上边界, 1=下边界)

    // [2-3] 速度归一化：映射到 [-1, 1]，表示移动方向和速度
    obs[idx++] = ai_tank->vx / TANK_SPEED;            // [2] x方向速度 (-1=左移, 0=静止, 1=右移)
    obs[idx++] = ai_tank->vy / TANK_SPEED;            // [3] y方向速度 (-1=上移, 0=静止, 1=下移)

    // [4] 血量归一化：0=死亡, 1=满血
    obs[idx++] = (float)ai_tank->health / TANK_MAX_HEALTH;  // [4] 血量 (范围: [0, 1])

    // [5] 射击冷却：1=冷却中不能射击, 0=可以射击
    obs[idx++] = ai_tank->shoot_cooldown > 0 ? 1.0f : 0.0f; // [5] 冷却标志

    // ========== 第2部分：最近5个敌人状态 (25维) ==========

    // 为什么是5个？平衡信息量与计算效率
    // - 太少：AI无法感知多敌人包围
    // -太多：网络输入维度过大，训练慢且容易过拟合

    float enemy_info[5][5] = {0};  // 预分配5个敌人位置，未找到的填充0
    int enemy_found = 0;

    // 遍历所有坦克，收集敌方坦克信息
    for (int i = 0; i < game->tank_count && enemy_found < 5; i++) {
        // 只关心存活的非AI坦克（敌人）
        if (game->tanks[i].type != TANK_TYPE_AI && game->tanks[i].alive) {
            // 计算相对位置（使用相对坐标而非绝对坐标）
            // 原因：AI需要知道"敌人在我的哪个方向"，而不是"敌人的绝对位置"
            float dx = game->tanks[i].x - ai_tank->x;  // 正值=敌人在右侧, 负值=左侧
            float dy = game->tanks[i].y - ai_tank->y;  // 正值=敌人在下方, 负值=上方

            // 归一化到 [-1, 1] 范围（除以地图尺寸）
            enemy_info[enemy_found][0] = dx / game->map_width;   // 相对x距离
            enemy_info[enemy_found][1] = dy / game->map_height;  // 相对y距离

            // 敌人速度归一化：帮助AI预测敌人移动轨迹
            enemy_info[enemy_found][2] = game->tanks[i].vx / TANK_SPEED;
            enemy_info[enemy_found][3] = game->tanks[i].vy / TANK_SPEED;

            // 敌人血量：帮助AI优先攻击残血敌人
            enemy_info[enemy_found][4] = (float)game->tanks[i].health / TANK_MAX_HEALTH;

            enemy_found++;
        }
    }

    // 展开敌人信息到状态向量（固定25维，不足的位置为0）
    // 即使只有2个敌人，仍然输出25维（后15维为0）
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            obs[idx++] = enemy_info[i][j];  // [6-30] 敌人信息
        }
    }

    // ========== 第3部分：最近3个敌方子弹状态 (12维) ==========

    // 为什么需要子弹信息？躲避子弹是生存的关键技能
    // 为什么只要3个？距离AI最近的子弹威胁最大，远处子弹可以忽略

    float bullet_info[3][4] = {0};  // 预分配3个子弹槽位
    int bullet_found = 0;

    // 遍历所有子弹，收集敌方子弹信息
    for (int i = 0; i < MAX_BULLETS && bullet_found < 3; i++) {
        // 只关心存活的敌方子弹（owner_id != AI坦克id）
        if (game->bullets[i].active && game->bullets[i].owner_id != ai_tank->id) {
            // 计算子弹相对位置
            float dx = game->bullets[i].x - ai_tank->x;
            float dy = game->bullets[i].y - ai_tank->y;

            // 归一化相对位置
            bullet_info[bullet_found][0] = dx / game->map_width;   // 子弹相对x
            bullet_info[bullet_found][1] = dy / game->map_height;  // 子弹相对y

            // 归一化子弹速度：非常重要！
            // AI需要根据子弹飞行方向判断是否会击中自己
            // 例如：dx>0 且 vx>0 意味着子弹正在靠近
            bullet_info[bullet_found][2] = game->bullets[i].vx / BULLET_SPEED;
            bullet_info[bullet_found][3] = game->bullets[i].vy / BULLET_SPEED;

            bullet_found++;
        }
    }

    // 展开子弹信息到状态向量（固定12维）
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            obs[idx++] = bullet_info[i][j];  // [31-42] 子弹信息
        }
    }

    // 最终验证：确保维度正确
    *obs_size = idx;  // 必须等于 6 + 25 + 12 = 43 维
    // 如果不等于43，说明代码有bug！
}
