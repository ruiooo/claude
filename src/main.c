/*
 * main.c - 主程序入口（玩家对战模式）
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "game.h"
#include "rendering.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define FPS 60

// 玩家输入处理
void handle_player_input(GameState* game, const Uint8* keys) {
    Tank* player_tank = game_get_ai_tank(game);  // 使用AI坦克作为玩家坦克
    if (!player_tank) return;

    player_tank->type = TANK_TYPE_PLAYER;  // 设置为玩家类型

    TankAction action = ACTION_IDLE;

    // WASD移动
    if (keys[SDL_SCANCODE_W]) {
        action = ACTION_MOVE_UP;
    } else if (keys[SDL_SCANCODE_S]) {
        action = ACTION_MOVE_DOWN;
    } else if (keys[SDL_SCANCODE_A]) {
        action = ACTION_MOVE_LEFT;
    } else if (keys[SDL_SCANCODE_D]) {
        action = ACTION_MOVE_RIGHT;
    }

    // 箭头键射击
    if (keys[SDL_SCANCODE_UP]) {
        action = ACTION_SHOOT_UP;
    } else if (keys[SDL_SCANCODE_DOWN]) {
        action = ACTION_SHOOT_DOWN;
    } else if (keys[SDL_SCANCODE_LEFT]) {
        action = ACTION_SHOOT_LEFT;
    } else if (keys[SDL_SCANCODE_RIGHT]) {
        action = ACTION_SHOOT_RIGHT;
    }

    if (action != ACTION_IDLE) {
        game_execute_action(game, 0, action);
    }
}

int main(int argc, char* argv[]) {
    printf("坦克大战 - 玩家对战模式\n");
    printf("控制: WASD移动, 方向键射击\n");
    printf("按ESC退出\n\n");

    // 初始化渲染器
    Renderer renderer;
    if (!renderer_init(&renderer, WINDOW_WIDTH, WINDOW_HEIGHT, "Tank Battle - Player Mode")) {
        fprintf(stderr, "渲染器初始化失败\n");
        return 1;
    }

    // 初始化游戏
    GameState game;
    game_init(&game, WINDOW_WIDTH, WINDOW_HEIGHT, MODE_PLAYER);

    // 开始游戏，3个敌人
    game_reset(&game, 3);

    bool running = true;
    Uint32 frame_start, frame_time;
    const int frame_delay = 1000 / FPS;

    while (running) {
        frame_start = SDL_GetTicks();

        // 处理事件
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                running = false;
            }
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) {
                running = false;
            }
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_RETURN) {
                // 回车重新开始
                if (game.game_over) {
                    game_reset(&game, 3);
                }
            }
        }

        // 获取键盘状态
        const Uint8* keys = SDL_GetKeyboardState(NULL);

        // 处理玩家输入
        if (!game.game_over) {
            handle_player_input(&game, keys);
        }

        // 更新游戏
        if (!game.game_over) {
            game_update(&game);
        }

        // 渲染
        renderer_render_game(&renderer, &game);

        // 显示提示信息
        if (game.game_over) {
            char text[128];
            snprintf(text, sizeof(text), "Press ENTER to restart");
            SDL_Color white = {255, 255, 255, 255};
            // render_text在rendering.c中是static的，这里简化处理
        }

        // 帧率控制
        frame_time = SDL_GetTicks() - frame_start;
        if (frame_delay > frame_time) {
            SDL_Delay(frame_delay - frame_time);
        }
    }

    // 清理
    renderer_cleanup(&renderer);

    printf("游戏结束\n");
    return 0;
}
