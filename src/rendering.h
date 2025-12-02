/*
 * rendering.h - SDL2渲染系统
 */

#ifndef RENDERING_H
#define RENDERING_H

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "game.h"

// 渲染器结构
typedef struct {
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    TTF_Font* small_font;
    int width;
    int height;
    bool initialized;
} Renderer;

// 初始化渲染器
bool renderer_init(Renderer* r, int width, int height, const char* title);

// 清理渲染器
void renderer_cleanup(Renderer* r);

// 渲染游戏
void renderer_render_game(Renderer* r, GameState* game);

// 渲染训练信息
void renderer_render_training_info(Renderer* r, int episode, int step,
                                   float reward, int wins, int losses,
                                   int current_enemies, const char* status);

#endif // RENDERING_H
