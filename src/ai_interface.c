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

// 全局假人模式开关（课程学习用）
// 0=正常模式，1=假人模式（静止靶，不反击，1点血）
static int g_dummy_mode = 0;

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
 * 计算奖励（优化版 - 奖励塑形 Reward Shaping）
 *
 * 【核心函数】这是强化学习中最关键的函数！
 * 奖励函数设计直接决定AI学到什么行为策略
 *
 * 奖励设计哲学：
 * 1. 稀疏奖励（基础）：击杀、受伤、胜利等关键事件
 * 2. 密集奖励（塑形）：距离、瞄准、躲避等持续性指导
 * 3. 平衡探索与利用：不能过于惩罚探索，否则AI学不到新策略
 *
 * 数值设计原则：
 * - 大事件用大奖励（击杀100、胜利300、死亡-100）
 * - 小引导用小奖励（距离0.5、瞄准0.3、存活0.02）
 * - 避免奖励过于稀疏或过于密集
 *
 * 奖励分类：
 * ┌─────────────────────────────────────────────────────────────┐
 * │ 基础奖励（稀疏）         | 数值      | 作用               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 存活每帧                 | +0.02     | 鼓励生存           │
 * │ 受伤                     | -15.0     | 惩罚被击中         │
 * │ 击杀敌人                 | +100.0    | 主要目标           │
 * │ 胜利                     | +300.0    | 最终目标           │
 * │ 死亡                     | -100.0    | 严重惩罚           │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 奖励塑形（密集）         | 数值      | 作用               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 1. 距离奖励              | 0-0.5     | 引导接近敌人       │
 * │ 2. 瞄准奖励              | 0-0.3     | 引导朝向敌人移动   │
 * │ 3. 射击奖励              | ±0.5      | 引导合理时机射击   │
 * │ 4. 躲避子弹              | ±0.4      | 引导躲避危险       │
 * │ 5. 墙壁避让              | +0.2      | 避免卡边           │
 * │ 6. 动作多样性            | +0.1      | 避免静止不动       │
 * └─────────────────────────────────────────────────────────────┘
 *
 * 参数说明：
 * @param game        - 游戏状态
 * @param ai_tank     - AI坦克对象
 * @param prev_health - 上一帧血量（用于检测受伤）
 * @param prev_enemies- 上一帧敌人数（用于检测击杀）
 * @return 该步的总奖励值
 */
float calculate_reward(GameState* game, Tank* ai_tank, int prev_health,
                       int prev_enemies) {
    float reward = 0.0f;

    // ========== 击杀导向奖励函数 v4.0 ==========
    // 核心目标：让AI明白"击杀敌人"才是唯一目标
    // 策略：大幅降低生存/追击奖励，大幅提高击杀奖励

    // ===== 第一部分：核心奖励（击杀为王） =====

    // 1. 死亡惩罚：失败
    if (!ai_tank->alive) {
        return -100.0f;
    }

    // 2. 击杀奖励：唯一重要的事（暴增到1000！）
    int current_enemies = game_get_alive_count(game, TANK_TYPE_ENEMY) +
                         game_get_alive_count(game, TANK_TYPE_SELF_PLAY);
    int enemies_killed = prev_enemies - current_enemies;
    if (enemies_killed > 0) {
        reward += 1000.0f * enemies_killed;  // 暴增到1000，碾压其他所有奖励
    }

    // 3. 受伤惩罚：轻微（只有-5）
    int health_change = ai_tank->health - prev_health;
    if (health_change < 0) {
        reward -= 5.0f;  // 降低到-5，允许AI冒险进攻
    }

    // 4. 胜利奖励：额外奖励
    if (game->game_over && game->winner == 0) {
        reward += 500.0f;  // 提高到500
    }

    // ===== 第二部分：最小引导（避免刷分） =====

    // 5. 存活奖励：极小（防止AI只追不打）
    reward += 0.002f;  // 从0.05降到0.002，3600帧才7.2分

    // 6. 找到最近的敌人
    Tank* nearest_enemy = NULL;
    float min_dist = 9999.0f;
    for (int i = 0; i < game->tank_count; i++) {
        Tank* tank = &game->tanks[i];
        if (!tank->alive || tank->type == TANK_TYPE_AI) continue;

        float dx = tank->x - ai_tank->x;
        float dy = tank->y - ai_tank->y;
        float dist = sqrtf(dx*dx + dy*dy);

        if (dist < min_dist) {
            min_dist = dist;
            nearest_enemy = tank;
        }
    }

    if (nearest_enemy != NULL) {
        // 7. 距离奖励：极小（只在非常近时给）
        if (min_dist < 150.0f) {
            reward += 0.05f;  // 从1.0降到0.05
        }

        // 8. 射击准备高额奖励：强烈鼓励在近距离时准备射击
        if (ai_tank->shoot_cooldown == 0 && min_dist < 250.0f) {
            reward += 50.0f;  // 大幅提高：从0.1 → 50，让AI知道"准备射击"很重要
        }

        // 9. 极近距离额外奖励：非常接近时更高奖励
        if (min_dist < 100.0f) {
            reward += 10.0f;  // 贴脸时的超高奖励
        }
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

    // 计算奖励
    *reward = calculate_reward(&g_game, ai_tank, prev_health, prev_enemies);

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

// 设置假人模式（课程学习用）
void ai_set_dummy_mode(int dummy_mode) {
    g_dummy_mode = dummy_mode;
}

// 获取假人模式状态（供game.c使用）
int ai_get_dummy_mode() {
    return g_dummy_mode;
}
