/*
 * ai_interface.c - AI训练接口实现
 */

#include "ai_interface.h"
#include "rendering.h"
#include <stdlib.h>
#include <string.h>

// 全局游戏状态
GameState g_game;
Renderer g_renderer;
bool g_visualize = false;

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

// 计算奖励
static float calculate_reward(GameState* game, Tank* ai_tank, int prev_health,
                              int prev_enemies) {
    float reward = 0.0f;

    if (!ai_tank->alive) {
        // AI死亡，大惩罚
        return -100.0f;
    }

    // 存活奖励（每帧）
    reward += 0.01f;

    // 血量变化
    int health_change = ai_tank->health - prev_health;
    if (health_change < 0) {
        reward -= 10.0f;  // 受伤惩罚
    }

    // 击杀敌人奖励
    int current_enemies = game_get_alive_count(game, TANK_TYPE_ENEMY) +
                         game_get_alive_count(game, TANK_TYPE_SELF_PLAY);
    int enemies_killed = prev_enemies - current_enemies;
    if (enemies_killed > 0) {
        reward += 50.0f * enemies_killed;  // 击杀奖励
    }

    // 胜利奖励
    if (game->game_over && game->winner == 0) {
        reward += 200.0f;
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
