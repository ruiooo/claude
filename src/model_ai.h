/*
 * model_ai.h - 基于Python训练模型的AI控制器
 *
 * 模块说明：
 *   在C环境中加载和使用PyTorch训练的DQN模型
 *   通过Python C API嵌入Python解释器
 *   实现玩家对战模式中AI对手的智能决策
 *
 * 使用场景：
 *   - 玩家对战模式：加载训练好的模型与玩家对战
 *   - 模型评估：测试不同版本模型的表现
 *   - 自我对弈：历史版本AI对抗当前训练AI
 *
 * 架构：
 *   C游戏引擎 -> Python C API -> Python DQNAgent -> PyTorch模型 -> 返回动作
 *
 * 依赖：
 *   - Python C API：嵌入Python解释器
 *   - PyTorch：加载.pth模型文件
 *   - 虚拟环境：自动检测并使用项目虚拟环境
 *
 * 对应文件：
 *   - C实现: src/model_ai.c
 *   - Python模型: python/model.py (DQNAgent类)
 *   - 玩家模式: src/main.c
 */

#ifndef MODEL_AI_H
#define MODEL_AI_H

#include "game.h"
#include <stdbool.h>

/*
 * 模型AI结构体
 * 封装Python模型实例和状态
 */
typedef struct {
    void* py_agent;      // Python DQNAgent对象指针 (PyObject*)，模型推理接口
    void* py_module;     // Python模块指针 (PyObject*)，保持模块引用
    bool initialized;     // 初始化标志，防止未初始化使用
    char model_path[256]; // 模型文件路径，用于日志和调试
} ModelAI;

/*
 * 模型AI系统功能函数声明
 */

/*
 * 初始化模型AI系统（全局初始化）
 * @return: true=成功，false=失败
 * 功能：
 *   - 启动Python解释器
 *   - 设置虚拟环境路径（自动检测VENV_DIR）
 *   - 初始化Python系统路径
 *   - 导入必要的Python模块（torch, model等）
 * 注意：
 *   - 整个程序只调用一次（在main函数开始时）
 *   - 失败时打印详细错误信息
 *   - 必须在任何model_ai_init调用之前执行
 */
bool model_ai_system_init(void);

/*
 * 清理模型AI系统（全局清理）
 * 功能：
 *   - 清理所有Python对象引用
 *   - 关闭Python解释器
 *   - 释放Python系统资源
 * 注意：
 *   - 程序退出前必须调用
 *   - 调用后不能再使用任何model_ai函数
 */
void model_ai_system_cleanup(void);

/*
 * 初始化单个模型AI实例
 * @param ai: 模型AI结构体指针
 * @param model_path: 模型文件路径（相对或绝对路径）
 *                    例如: "saved_models/final_model.pth"
 *                          "saved_models/checkpoints/checkpoint_ep1000.pth"
 * @return: true=成功，false=失败
 * 功能：
 *   - 导入Python model模块
 *   - 创建DQNAgent实例
 *   - 加载模型权重（调用agent.load_model）
 *   - 设置为评估模式（非训练模式）
 * 注意：
 *   - 模型文件必须存在且格式正确
 *   - 模型维度必须匹配（state_dim=43, action_dim=9）
 *   - 失败时打印详细错误信息和Python异常堆栈
 */
bool model_ai_init(ModelAI* ai, const char* model_path);

/*
 * 清理模型AI实例
 * @param ai: 模型AI结构体指针
 * 功能：
 *   - 释放Python DQNAgent对象
 *   - 释放Python模块引用
 *   - 重置initialized标志
 * 注意：不再使用模型时必须调用，防止内存泄漏
 */
void model_ai_cleanup(ModelAI* ai);

/*
 * 获取模型AI的决策动作
 * @param ai: 模型AI结构体指针（必须已初始化）
 * @param game: 游戏状态指针
 * @param tank_id: AI控制的坦克ID（通常为0）
 * @return: 决策的动作（ACTION_IDLE到ACTION_SHOOT_RIGHT）
 * 功能：
 *   - 从游戏状态提取观察（43维）
 *   - 将观察数据传递给Python DQNAgent
 *   - 调用agent.select_action（贪心模式，epsilon=0）
 *   - 返回模型输出的动作
 * 性能：
 *   - 每帧调用一次
 *   - Python C API开销约0.1-0.5ms
 *   - PyTorch推理（CPU）约1-3ms
 * 错误处理：
 *   - 失败时返回ACTION_IDLE（安全默认动作）
 *   - 打印Python异常信息
 */
TankAction model_ai_get_action(ModelAI* ai, const GameState* game, int tank_id);

/*
 * 列出可用的模型文件
 * @param models: 输出模型路径数组（每个路径最长256字符）
 * @param max_count: 最大返回数量
 * @return: 实际找到的模型数量
 * 功能：
 *   - 扫描以下目录：
 *     * saved_models/
 *     * saved_models/checkpoints/
 *     * checkpoints/
 *     * models/
 *   - 查找.pth文件
 *   - 优先返回：
 *     1. final_model.pth（最终模型）
 *     2. latest_model.pth（最新检查点）
 *     3. checkpoint_ep*.pth（按时间排序，最新的前3个）
 *   - 最多返回max_count个模型
 * 用途：
 *   - 玩家对战模式：显示可用模型供玩家选择
 *   - 模型管理：列出所有训练好的模型
 */
int model_ai_list_models(char models[][256], int max_count);

#endif // MODEL_AI_H
