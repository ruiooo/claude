/*
 * ai_interface.h - AI训练接口（用于Python通信）
 */

#ifndef AI_INTERFACE_H
#define AI_INTERFACE_H

#include "game.h"

// Python调用的C接口
#ifdef __cplusplus
extern "C" {
#endif

// 全局游戏状态（用于Python访问）
extern GameState g_game;

// 初始化游戏环境
void ai_init_env(int width, int height, int visualize);

// 重置环境
void ai_reset_env(int enemy_count, float* obs, int* obs_size);

// 执行一步
void ai_step(int action, float* obs, int* obs_size, float* reward,
             int* done, int* info);

// 获取当前状态
void ai_get_state(float* obs, int* obs_size);

// 添加敌人
void ai_add_enemy(int enemy_type, int model_version);

// 获取统计信息
void ai_get_stats(int* ai_health, int* enemy_count, int* frame_count);

// 渲染当前帧（如果启用可视化）
void ai_render();

// 清理
void ai_cleanup();

#ifdef __cplusplus
}
#endif

#endif // AI_INTERFACE_H
