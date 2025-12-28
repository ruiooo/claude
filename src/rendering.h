/*
 * rendering.h - SDL2渲染系统
 *
 * 模块说明：
 *   使用SDL2实现游戏的图形渲染功能
 *   提供窗口管理、坦克渲染、文本显示等可视化功能
 *   支持训练可视化和玩家对战模式
 *
 * 使用场景：
 *   - 训练可视化模式：实时显示训练过程和游戏画面
 *   - 玩家对战模式：完整的游戏图形界面
 *   - 调试工具：观察AI行为和游戏状态
 *
 * 依赖：
 *   - SDL2：窗口和图形渲染
 *   - SDL2_ttf：文本和字体渲染
 */

#ifndef RENDERING_H
#define RENDERING_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "game.h"

/*
 * 渲染器结构体
 * 封装SDL2的窗口、渲染器和字体资源
 */
typedef struct {
    SDL_Window* window;         // SDL窗口句柄
    SDL_Renderer* renderer;     // SDL渲染器句柄，硬件加速
    TTF_Font* font;             // 主字体（用于标题和大文本）
    TTF_Font* small_font;       // 小字体（用于状态信息）
    int width;                  // 窗口宽度（像素）
    int height;                 // 窗口高度（像素）
    bool initialized;           // 初始化标志，防止重复初始化
} Renderer;

/*
 * 渲染器功能函数声明
 */

/*
 * 初始化渲染器（默认窗口位置）
 * @param r: 渲染器结构体指针
 * @param width: 窗口宽度（像素）
 * @param height: 窗口高度（像素）
 * @param title: 窗口标题字符串
 * @return: true=成功，false=失败
 * 功能：
 *   - 初始化SDL2视频子系统
 *   - 创建窗口（居中显示）
 *   - 创建硬件加速渲染器
 *   - 初始化SDL_ttf并加载字体
 * 注意：失败时需调用renderer_cleanup清理资源
 */
bool renderer_init(Renderer* r, int width, int height, const char* title);

/*
 * 初始化渲染器（控制VSYNC和窗口位置）
 * @param r: 渲染器结构体指针
 * @param width: 窗口宽度（像素）
 * @param height: 窗口高度（像素）
 * @param title: 窗口标题字符串
 * @param disable_vsync: 是否禁用垂直同步（1=禁用，0=启用）
 * @return: true=成功，false=失败
 * 功能：与renderer_init相同，但允许控制VSYNC
 * 用途：回放模式下禁用VSYNC以支持任意帧率
 */
bool renderer_init_no_vsync(Renderer* r, int width, int height, const char* title, int disable_vsync);

/*
 * 初始化渲染器（指定窗口位置）
 * @param r: 渲染器结构体指针
 * @param width: 窗口宽度（像素）
 * @param height: 窗口高度（像素）
 * @param title: 窗口标题字符串
 * @param x: 窗口左上角X坐标（像素）
 * @param y: 窗口左上角Y坐标（像素）
 * @return: true=成功，false=失败
 * 功能：与renderer_init相同，但允许指定窗口位置
 * 用途：多窗口布局（如多人对战时的双窗口）
 */
bool renderer_init_with_pos(Renderer* r, int width, int height, const char* title, int x, int y);

/*
 * 清理渲染器资源
 * @param r: 渲染器结构体指针
 * 功能：
 *   - 释放字体资源
 *   - 销毁渲染器
 *   - 销毁窗口
 *   - 清理SDL_ttf和SDL
 * 注意：程序退出前必须调用，防止内存泄漏
 */
void renderer_cleanup(Renderer* r);

/*
 * 渲染游戏画面（每帧调用）
 * @param r: 渲染器结构体指针
 * @param game: 游戏状态指针
 * 功能：
 *   - 清空画面（黑色背景）
 *   - 渲染所有存活的坦克（根据类型使用不同颜色）
 *     * AI坦克：蓝色
 *     * 追踪型敌人：红色
 *     * 历史版本AI：红白相间
 *     * 玩家1：绿色
 *     * 玩家2：深绿色
 *   - 渲染所有激活的子弹（黄色或白色）
 *   - 显示坦克血量条（顶部血量显示）
 *   - 显示坦克朝向（通过旋转渲染）
 *   - 呈现到屏幕
 * 性能：硬件加速，60fps稳定
 */
void renderer_render_game(Renderer* r, GameState* game);

/*
 * 渲染训练信息（训练可视化模式）
 * @param r: 渲染器结构体指针
 * @param episode: 当前回合数
 * @param step: 当前步数
 * @param reward: 当前累计奖励
 * @param wins: 累计胜利次数
 * @param losses: 累计失败次数
 * @param current_enemies: 当前敌人数量
 * @param status: 状态字符串（如"训练中"、"回合结束"等）
 * 功能：
 *   - 在游戏画面上方显示训练统计信息
 *   - 格式：回合数、步数、奖励、胜率、敌人数等
 *   - 使用半透明背景，不遮挡游戏画面
 * 用途：实时监控训练进度和AI表现
 */
void renderer_render_training_info(Renderer* r, int episode, int step,
                                   float reward, int wins, int losses,
                                   int current_enemies, const char* status);

#endif // RENDERING_H
