/*
 * model_ai.h - 基于Python训练模型的AI控制器
 *
 * 在C中加载和使用PyTorch训练的DQN模型
 */

#ifndef MODEL_AI_H
#define MODEL_AI_H

#include "game.h"
#include <stdbool.h>

// 模型AI结构体
typedef struct {
    void* py_agent;      // Python DQNAgent对象 (PyObject*)
    void* py_module;     // Python模块 (PyObject*)
    bool initialized;     // 是否已初始化
    char model_path[256]; // 模型文件路径
} ModelAI;

// 初始化模型AI系统（启动Python解释器）
bool model_ai_system_init(void);

// 清理模型AI系统（关闭Python解释器）
void model_ai_system_cleanup(void);

// 初始化单个模型AI实例
// model_path: 模型文件路径，例如 "models/latest_model.pth"
bool model_ai_init(ModelAI* ai, const char* model_path);

// 清理模型AI实例
void model_ai_cleanup(ModelAI* ai);

// 获取模型AI的决策
// game: 游戏状态
// tank_id: AI控制的坦克ID
// 返回: 动作（ACTION_IDLE, ACTION_MOVE_UP等）
TankAction model_ai_get_action(ModelAI* ai, const GameState* game, int tank_id);

// 列出可用的模型文件
// models: 输出模型路径数组
// max_count: 最大数量
// 返回: 实际找到的模型数量
int model_ai_list_models(char models[][256], int max_count);

#endif // MODEL_AI_H
