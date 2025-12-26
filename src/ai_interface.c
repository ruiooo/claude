/*
 * ai_interface.c - AI训练接口实现
 */

#include "ai_interface.h"
#include "rendering.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 全局游戏状态
GameState g_game;
Renderer g_renderer;
bool g_visualize = false;

// 全局难度级别（课程学习用）
// 0=假人模式（静止靶，不反击）
// 1=简单模式（随机移动，不预判）
// 2=中等模式（追踪但不躲避）
// 3=困难模式（完整AI）
static int g_difficulty_level = 3;  // 默认困难

// 初始化游戏环境
void ai_init_env(int width, int height, int visualize) {
    game_init(&g_game, width, height, visualize ? MODE_TRAINING_VIS : MODE_TRAINING);

    g_visualize = visualize;
    if (g_visualize) {
        renderer_init(&g_renderer, width, height, "Tank Battle AI Training");
    }
}

// 重置环境
void ai_reset_env(int enemy_count, float* obs, int* obs_size) {
    game_reset(&g_game, enemy_count);
    game_get_observation(&g_game, obs, obs_size);
}

/*
 * 计算奖励 v5.0 - 平衡式奖励塑形 (Balanced Reward Shaping)
 *
 * 【核心优化】解决之前版本的问题：
 * 1. 奖励尺度不平衡（击杀1000 vs 存活0.002）导致Q值膨胀
 * 2. 早期训练稀疏奖励问题（前100回合几乎无信号）
 * 3. 缺少中间行为引导（瞄准、躲避、合理射击）
 *
 * 【设计原则】
 * 1. 奖励归一化：所有奖励控制在 [-50, +100] 范围
 * 2. 密集引导：每帧都有行为塑形信号
 * 3. 进攻+防守：平衡击杀和存活两个目标
 * 4. 渐进式奖励：距离越近、瞄准越准，奖励越高
 *
 * 【奖励结构】
 * ┌─────────────────────────────────────────────────────────────┐
 * │ 核心奖励（稀疏）         | 数值      | 占比  | 作用        │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 击杀敌人                 | +80.0     | 40%   | 主要目标    │
 * │ 胜利                     | +50.0     | 25%   | 最终目标    │
 * │ 死亡                     | -50.0     | 25%   | 失败惩罚    │
 * │ 受伤                     | -15.0     | 10%   | 风险惩罚    │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 行为塑形（密集）         | 数值      | 累积  | 作用        │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 1. 距离控制              | 0~1.0     | ~60/ep| 接近敌人    │
 * │ 2. 瞄准奖励              | 0~0.8     | ~30/ep| 面向敌人    │
 * │ 3. 射击时机              | 0~2.0     | ~20/ep| 合理射击    │
 * │ 4. 躲避子弹              | 0~1.5     | ~15/ep| 安全意识    │
 * │ 5. 墙壁避让              | -0.3~0    | ~-5/ep| 避免卡边    │
 * │ 6. 主动移动              | 0~0.1     | ~10/ep| 避免静止    │
 * │ 7. 生存奖励              | 0.02      | ~72/ep| 鼓励存活    │
 * └─────────────────────────────────────────────────────────────┘
 *
 * 【预期效果】
 * - 早期训练：密集奖励引导AI学习基础行为（~50-100/回合）
 * - 中期训练：击杀奖励驱动进攻策略（~150-300/回合）
 * - 后期训练：稳定的高胜率（~200-400/回合）
 */
float calculate_reward(GameState* game, Tank* ai_tank, int prev_health,
                       int prev_enemies, int action) {
    float reward = 0.0f;

    // ========== 第一部分：核心稀疏奖励 ==========

    // 1. 死亡惩罚（终止奖励）
    if (!ai_tank->alive) {
        return -50.0f;  // 降低到-50，避免Q值过度负向
    }

    // 2. 击杀奖励（核心目标）
    int current_enemies = game_get_alive_count(game, TANK_TYPE_ENEMY) +
                         game_get_alive_count(game, TANK_TYPE_SELF_PLAY);
    int enemies_killed = prev_enemies - current_enemies;
    if (enemies_killed > 0) {
        reward += 80.0f * enemies_killed;  // 降低到80，与其他奖励平衡
    }

    // 3. 受伤惩罚（风险意识）
    int health_change = ai_tank->health - prev_health;
    if (health_change < 0) {
        reward -= 15.0f * (-health_change);  // 每点血量损失-15
    }

    // 4. 胜利奖励（最终目标）
    if (game->game_over && game->winner == 0) {
        reward += 50.0f;  // 降低到50，与击杀奖励平衡
    }

    // 5. 平局惩罚（鼓励主动进攻，避免被动消耗时间）
    if (game->game_over && game->winner == -1) {
        reward -= 30.0f;  // 平局也是失败，鼓励进攻
    }

    // 6. IDLE惩罚（解决AI被动问题）
    if (action == 0) {  // ACTION_IDLE = 0
        reward -= 0.3f;  // 每次静止都有惩罚，鼓励移动和射击
    }

    // ========== 第二部分：行为塑形奖励 ==========

    // 7. 基础生存奖励（降低以平衡IDLE惩罚）
    reward += 0.01f;  // 每帧0.01，3600帧=36分（降低50%避免被动生存）

    // 8. 找到最近的敌人（用于后续奖励计算）
    Tank* nearest_enemy = NULL;
    float min_dist = 9999.0f;
    float dx_nearest = 0, dy_nearest = 0;

    for (int i = 0; i < game->tank_count; i++) {
        Tank* tank = &game->tanks[i];
        if (!tank->alive || tank->type == TANK_TYPE_AI) continue;

        float dx = tank->x - ai_tank->x;
        float dy = tank->y - ai_tank->y;
        float dist = sqrtf(dx*dx + dy*dy);

        if (dist < min_dist) {
            min_dist = dist;
            nearest_enemy = tank;
            dx_nearest = dx;
            dy_nearest = dy;
        }
    }

    if (nearest_enemy != NULL) {
        // 9. 距离控制奖励（渐进式）
        // 理想距离：100-200像素（射击范围内但有躲避空间）
        if (min_dist < 100.0f) {
            // 太近：小奖励（鼓励但不过度）
            reward += 0.5f;
        } else if (min_dist < 200.0f) {
            // 理想距离：最高奖励
            reward += 1.0f;
        } else if (min_dist < 300.0f) {
            // 稍远：中等奖励
            reward += 0.3f;
        } else {
            // 太远：鼓励接近
            reward += 0.1f * (1.0f - min_dist / 800.0f);  // 越近越高
        }

        // 10. 瞄准奖励（面向敌人方向）
        // 检查AI坦克是否面向敌人
        bool aiming_at_enemy = false;
        float alignment_tolerance = 50.0f;  // 对齐容忍度

        switch (ai_tank->direction) {
            case DIR_UP:
                aiming_at_enemy = (dy_nearest < 0) && (fabsf(dx_nearest) < alignment_tolerance);
                break;
            case DIR_DOWN:
                aiming_at_enemy = (dy_nearest > 0) && (fabsf(dx_nearest) < alignment_tolerance);
                break;
            case DIR_LEFT:
                aiming_at_enemy = (dx_nearest < 0) && (fabsf(dy_nearest) < alignment_tolerance);
                break;
            case DIR_RIGHT:
                aiming_at_enemy = (dx_nearest > 0) && (fabsf(dy_nearest) < alignment_tolerance);
                break;
        }

        if (aiming_at_enemy && min_dist < 300.0f) {
            // 距离越近，瞄准奖励越高
            reward += 0.8f * (1.0f - min_dist / 300.0f);
        }

        // 11. 射击时机奖励
        // 只有在合适条件下射击才给奖励（替代之前的"准备好就给奖励"）
        if (ai_tank->shoot_cooldown == 0) {
            // 冷却完成
            if (aiming_at_enemy && min_dist < 250.0f) {
                // 瞄准敌人且在射击范围内：高奖励
                reward += 2.0f;
            } else if (min_dist < 200.0f) {
                // 在射击范围内但未瞄准：中等奖励（鼓励调整方向）
                reward += 0.5f;
            }
        }

        // 12. 主动进攻奖励（正在移动且接近敌人）
        bool is_moving = (fabsf(ai_tank->vx) > 0.1f || fabsf(ai_tank->vy) > 0.1f);
        if (is_moving) {
            reward += 0.1f;  // 基础移动奖励

            // 检查是否在接近敌人
            bool approaching = false;
            if (ai_tank->vx > 0 && dx_nearest > 0) approaching = true;
            if (ai_tank->vx < 0 && dx_nearest < 0) approaching = true;
            if (ai_tank->vy > 0 && dy_nearest > 0) approaching = true;
            if (ai_tank->vy < 0 && dy_nearest < 0) approaching = true;

            if (approaching && min_dist > 150.0f) {
                reward += 0.3f;  // 主动接近敌人
            }
        }
    }

    // 13. 子弹躲避奖励
    // 检测附近的危险子弹并奖励躲避行为
    int dangerous_bullets = 0;
    bool is_dodging = false;

    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!game->bullets[i].active) continue;
        if (game->bullets[i].owner_id == ai_tank->id) continue;  // 忽略自己的子弹

        float bx = game->bullets[i].x - ai_tank->x;
        float by = game->bullets[i].y - ai_tank->y;
        float bullet_dist = sqrtf(bx*bx + by*by);

        if (bullet_dist < 80.0f) {
            dangerous_bullets++;

            // 检查是否在躲避（移动方向垂直于子弹方向）
            float bvx = game->bullets[i].vx;
            float bvy = game->bullets[i].vy;

            // 如果子弹主要水平飞行，垂直移动是躲避
            if (fabsf(bvx) > fabsf(bvy)) {
                if (fabsf(ai_tank->vy) > 0.1f) is_dodging = true;
            } else {
                // 子弹主要垂直飞行，水平移动是躲避
                if (fabsf(ai_tank->vx) > 0.1f) is_dodging = true;
            }
        }
    }

    if (dangerous_bullets > 0) {
        if (is_dodging) {
            reward += 1.5f * dangerous_bullets;  // 成功躲避
        } else {
            reward -= 0.5f;  // 有危险但未躲避
        }
    }

    // 14. 墙壁避让惩罚（避免卡边）
    float wall_margin = 40.0f;
    if (ai_tank->x < wall_margin || ai_tank->x > game->map_width - wall_margin ||
        ai_tank->y < wall_margin || ai_tank->y > game->map_height - wall_margin) {
        reward -= 0.3f;  // 靠近墙壁惩罚
    }

    return reward;
}

// 执行一步
void ai_step(int action, float* obs, int* obs_size, float* reward,
             int* done, int* info) {
    Tank* ai_tank = game_get_ai_tank(&g_game);
    if (!ai_tank) {
        *done = 1;
        *reward = -100.0f;
        return;
    }

    int prev_health = ai_tank->health;
    int prev_enemies = game_get_alive_count(&g_game, TANK_TYPE_ENEMY) +
                      game_get_alive_count(&g_game, TANK_TYPE_SELF_PLAY);

    // 执行AI动作
    game_execute_action(&g_game, 0, (TankAction)action);

    // 更新游戏
    game_update(&g_game);

    // 获取新状态
    game_get_observation(&g_game, obs, obs_size);

    // 计算奖励（传入action用于IDLE惩罚）
    *reward = calculate_reward(&g_game, ai_tank, prev_health, prev_enemies, action);

    // 检查是否结束
    *done = g_game.game_over ? 1 : 0;

    // 额外信息
    info[0] = g_game.winner;  // 胜者
    info[1] = ai_tank->health;  // AI血量
    info[2] = prev_enemies - (game_get_alive_count(&g_game, TANK_TYPE_ENEMY) +
                              game_get_alive_count(&g_game, TANK_TYPE_SELF_PLAY));  // 击杀数
}

// 获取当前状态
void ai_get_state(float* obs, int* obs_size) {
    game_get_observation(&g_game, obs, obs_size);
}

// 添加敌人
void ai_add_enemy(int enemy_type, int model_version) {
    game_add_enemy(&g_game, (TankType)enemy_type, model_version);
}

// 获取统计信息
void ai_get_stats(int* ai_health, int* enemy_count, int* frame_count) {
    Tank* ai_tank = game_get_ai_tank(&g_game);
    *ai_health = ai_tank ? ai_tank->health : 0;
    *enemy_count = game_get_alive_count(&g_game, TANK_TYPE_ENEMY) +
                  game_get_alive_count(&g_game, TANK_TYPE_SELF_PLAY);
    *frame_count = g_game.frame_count;
}

// 渲染当前帧
void ai_render() {
    if (g_visualize) {
        renderer_render_game(&g_renderer, &g_game);

        // 处理SDL事件
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                g_visualize = false;
            }
        }
    }
}

// 清理
void ai_cleanup() {
    if (g_visualize) {
        renderer_cleanup(&g_renderer);
    }
}

// 设置假人模式（课程学习用）- 兼容旧接口
void ai_set_dummy_mode(int dummy_mode) {
    g_difficulty_level = dummy_mode ? 0 : 3;
}

// 获取假人模式状态 - 兼容旧接口
int ai_get_dummy_mode() {
    return g_difficulty_level == 0 ? 1 : 0;
}

// 新接口：设置难度级别 (0-4)
// 0=假人（静止），1=慢移动（不射击），2=移动+射击，3=追踪，4=完整AI
void ai_set_difficulty(int level) {
    if (level < 0) level = 0;
    if (level > 4) level = 4;
    g_difficulty_level = level;
}

int ai_get_difficulty() {
    return g_difficulty_level;
}
