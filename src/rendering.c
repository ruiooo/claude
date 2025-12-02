/*
 * rendering.c - SDL2渲染实现
 */

#include "rendering.h"
#include <stdio.h>

// 初始化渲染器
bool renderer_init(Renderer* r, int width, int height, const char* title) {
    r->width = width;
    r->height = height;
    r->initialized = false;

    // 初始化SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL初始化失败: %s\n", SDL_GetError());
        return false;
    }

    // 初始化TTF
    if (TTF_Init() < 0) {
        fprintf(stderr, "TTF初始化失败: %s\n", TTF_GetError());
        SDL_Quit();
        return false;
    }

    // 创建窗口
    r->window = SDL_CreateWindow(title,
                                  SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED,
                                  width, height,
                                  SDL_WINDOW_SHOWN);
    if (!r->window) {
        fprintf(stderr, "窗口创建失败: %s\n", SDL_GetError());
        TTF_Quit();
        SDL_Quit();
        return false;
    }

    // 创建渲染器
    r->renderer = SDL_CreateRenderer(r->window, -1,
                                     SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!r->renderer) {
        fprintf(stderr, "渲染器创建失败: %s\n", SDL_GetError());
        SDL_DestroyWindow(r->window);
        TTF_Quit();
        SDL_Quit();
        return false;
    }

    // 加载字体（使用系统默认字体）
    r->font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24);
    r->small_font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14);

    if (!r->font) {
        // 如果系统字体不存在，尝试其他路径
        r->font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 24);
        r->small_font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 14);
    }

    r->initialized = true;
    return true;
}

// 清理渲染器
void renderer_cleanup(Renderer* r) {
    if (!r->initialized) return;

    if (r->font) TTF_CloseFont(r->font);
    if (r->small_font) TTF_CloseFont(r->small_font);
    if (r->renderer) SDL_DestroyRenderer(r->renderer);
    if (r->window) SDL_DestroyWindow(r->window);

    TTF_Quit();
    SDL_Quit();

    r->initialized = false;
}

// 渲染文本
static void render_text(Renderer* r, const char* text, int x, int y,
                       SDL_Color color, TTF_Font* font) {
    if (!font) return;

    SDL_Surface* surface = TTF_RenderText_Solid(font, text, color);
    if (!surface) return;

    SDL_Texture* texture = SDL_CreateTextureFromSurface(r->renderer, surface);
    if (texture) {
        SDL_Rect rect = {x, y, surface->w, surface->h};
        SDL_RenderCopy(r->renderer, texture, NULL, &rect);
        SDL_DestroyTexture(texture);
    }

    SDL_FreeSurface(surface);
}

// 渲染坦克
static void render_tank(Renderer* r, Tank* tank) {
    if (!tank->alive) return;

    SDL_Rect rect = {
        (int)tank->x,
        (int)tank->y,
        tank->width,
        tank->height
    };

    // 根据坦克类型设置颜色
    switch (tank->type) {
        case TANK_TYPE_AI:
            // 蓝色（AI坦克）
            SDL_SetRenderDrawColor(r->renderer, 0, 100, 255, 255);
            break;
        case TANK_TYPE_ENEMY:
            // 红色（敌人）
            SDL_SetRenderDrawColor(r->renderer, 255, 50, 50, 255);
            break;
        case TANK_TYPE_SELF_PLAY:
            // 红白色（历史版本）
            SDL_SetRenderDrawColor(r->renderer, 255, 150, 150, 255);
            break;
        case TANK_TYPE_PLAYER:
            // 绿色（玩家）
            SDL_SetRenderDrawColor(r->renderer, 50, 255, 50, 255);
            break;
    }

    SDL_RenderFillRect(r->renderer, &rect);

    // 绘制方向指示（小矩形）
    SDL_Rect dir_rect;
    switch (tank->direction) {
        case DIR_UP:
            dir_rect = (SDL_Rect){(int)tank->x + 12, (int)tank->y, 8, 4};
            break;
        case DIR_DOWN:
            dir_rect = (SDL_Rect){(int)tank->x + 12, (int)tank->y + 28, 8, 4};
            break;
        case DIR_LEFT:
            dir_rect = (SDL_Rect){(int)tank->x, (int)tank->y + 12, 4, 8};
            break;
        case DIR_RIGHT:
            dir_rect = (SDL_Rect){(int)tank->x + 28, (int)tank->y + 12, 4, 8};
            break;
    }
    SDL_SetRenderDrawColor(r->renderer, 255, 255, 0, 255);
    SDL_RenderFillRect(r->renderer, &dir_rect);

    // 渲染血量条
    int bar_width = tank->width;
    int bar_height = 4;
    int bar_x = (int)tank->x;
    int bar_y = (int)tank->y - 8;

    // 背景（红色）
    SDL_SetRenderDrawColor(r->renderer, 100, 0, 0, 255);
    SDL_Rect bg_rect = {bar_x, bar_y, bar_width, bar_height};
    SDL_RenderFillRect(r->renderer, &bg_rect);

    // 血量（绿色）
    int health_width = (bar_width * tank->health) / TANK_MAX_HEALTH;
    SDL_SetRenderDrawColor(r->renderer, 0, 255, 0, 255);
    SDL_Rect health_rect = {bar_x, bar_y, health_width, bar_height};
    SDL_RenderFillRect(r->renderer, &health_rect);

    // 渲染模型版本（对于历史版本敌人）
    if (tank->type == TANK_TYPE_SELF_PLAY && r->small_font) {
        char version_text[16];
        snprintf(version_text, sizeof(version_text), "v%d", tank->model_version);
        SDL_Color white = {255, 255, 255, 255};
        render_text(r, version_text, (int)tank->x, (int)tank->y - 20, white, r->small_font);
    }
}

// 渲染子弹
static void render_bullet(Renderer* r, Bullet* bullet) {
    if (!bullet->active) return;

    SDL_Rect rect = {
        (int)bullet->x,
        (int)bullet->y,
        bullet->width,
        bullet->height
    };

    SDL_SetRenderDrawColor(r->renderer, 255, 255, 0, 255);
    SDL_RenderFillRect(r->renderer, &rect);
}

// 渲染游戏
void renderer_render_game(Renderer* r, GameState* game) {
    if (!r->initialized) return;

    // 清屏（黑色背景）
    SDL_SetRenderDrawColor(r->renderer, 20, 20, 20, 255);
    SDL_RenderClear(r->renderer);

    // 绘制网格
    SDL_SetRenderDrawColor(r->renderer, 40, 40, 40, 255);
    for (int x = 0; x < game->map_width; x += 50) {
        SDL_RenderDrawLine(r->renderer, x, 0, x, game->map_height);
    }
    for (int y = 0; y < game->map_height; y += 50) {
        SDL_RenderDrawLine(r->renderer, 0, y, game->map_width, y);
    }

    // 渲染所有子弹
    for (int i = 0; i < MAX_BULLETS; i++) {
        render_bullet(r, &game->bullets[i]);
    }

    // 渲染所有坦克
    for (int i = 0; i < game->tank_count; i++) {
        render_tank(r, &game->tanks[i]);
    }

    // 渲染游戏结束信息
    if (game->game_over && r->font) {
        SDL_Color color = {255, 255, 255, 255};
        const char* text;

        if (game->winner == 0) {
            text = "AI WINS!";
            color = (SDL_Color){0, 255, 0, 255};
        } else if (game->winner == 1) {
            text = "ENEMIES WIN!";
            color = (SDL_Color){255, 0, 0, 255};
        } else {
            text = "DRAW!";
            color = (SDL_Color){255, 255, 0, 255};
        }

        render_text(r, text, game->map_width / 2 - 80, game->map_height / 2 - 20,
                   color, r->font);
    }

    SDL_RenderPresent(r->renderer);
}

// 渲染训练信息
void renderer_render_training_info(Renderer* r, int episode, int step,
                                   float reward, int wins, int losses,
                                   int current_enemies, const char* status) {
    if (!r->initialized || !r->font) return;

    char text[256];
    SDL_Color white = {255, 255, 255, 255};
    SDL_Color green = {0, 255, 0, 255};
    SDL_Color red = {255, 0, 0, 255};

    int y = 10;

    // 回合信息
    snprintf(text, sizeof(text), "Episode: %d", episode);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // 步数
    snprintf(text, sizeof(text), "Steps: %d", step);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // 奖励
    snprintf(text, sizeof(text), "Reward: %.2f", reward);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // 胜率
    snprintf(text, sizeof(text), "Wins: %d | Losses: %d", wins, losses);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // 敌人数量
    snprintf(text, sizeof(text), "Enemies: %d", current_enemies);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // 状态
    if (status) {
        render_text(r, status, 10, y, green, r->font);
    }
}
