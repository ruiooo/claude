/*
 * main.c - 玩家对战模式（玩家 vs AI训练模型）
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "game.h"
#include "rendering.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define FPS 60
#define WIN_TARGET 10  // 连胜10局获胜

// 游戏状态跟踪
typedef struct {
    int wins;                // 玩家连胜次数
    int current_ai_count;    // 当前AI坦克数量
    bool game_active;        // 当前回合是否进行中
    char message[256];       // 显示消息
} GameProgress;

// 处理玩家输入并返回动作
TankAction get_player_input(const Uint8* keys, Tank* player_tank) {
    // 空格键射击 - 在当前方向开炮
    if (keys[SDL_SCANCODE_SPACE]) {
        switch (player_tank->direction) {
            case DIR_UP: return ACTION_SHOOT_UP;
            case DIR_DOWN: return ACTION_SHOOT_DOWN;
            case DIR_LEFT: return ACTION_SHOOT_LEFT;
            case DIR_RIGHT: return ACTION_SHOOT_RIGHT;
        }
    }

    // WASD移动
    if (keys[SDL_SCANCODE_W]) {
        return ACTION_MOVE_UP;
    } else if (keys[SDL_SCANCODE_S]) {
        return ACTION_MOVE_DOWN;
    } else if (keys[SDL_SCANCODE_A]) {
        return ACTION_MOVE_LEFT;
    } else if (keys[SDL_SCANCODE_D]) {
        return ACTION_MOVE_RIGHT;
    }

    return ACTION_IDLE;
}

// 开始新回合
void start_new_round(GameState* game, GameProgress* progress) {
    printf("\n========== 第 %d 局 ==========\n", progress->wins + 1);
    printf("AI坦克数量: %d\n", progress->current_ai_count);

    snprintf(progress->message, sizeof(progress->message),
             "Round %d - AI Tanks: %d | Wins: %d/%d",
             progress->wins + 1, progress->current_ai_count, progress->wins, WIN_TARGET);

    // 重置游戏
    game_reset(game, 0);  // 先不创建敌人

    // 创建玩家坦克（绿色）
    game->tanks[0].type = TANK_TYPE_PLAYER;
    game->tanks[0].x = WINDOW_WIDTH / 4;
    game->tanks[0].y = WINDOW_HEIGHT / 2;
    game->tank_count = 1;

    // 创建AI训练模型控制的坦克（蓝色，使用AI类型）
    for (int i = 0; i < progress->current_ai_count; i++) {
        float x, y;
        int edge = rand() % 4;

        switch (edge) {
            case 0: // 上边
                x = rand() % (game->map_width - TANK_SIZE);
                y = TANK_SIZE;
                break;
            case 1: // 下边
                x = rand() % (game->map_width - TANK_SIZE);
                y = game->map_height - TANK_SIZE * 2;
                break;
            case 2: // 左边
                x = game->map_width - TANK_SIZE * 2;
                y = rand() % (game->map_height - TANK_SIZE);
                break;
            case 3: // 右边
                x = game->map_width - TANK_SIZE * 2;
                y = rand() % (game->map_height - TANK_SIZE);
                break;
        }

        tank_init(&game->tanks[game->tank_count], x, y, TANK_TYPE_ENEMY);
        enemy_ai_init(&game->enemy_ais[game->tank_count], &game->tanks[0]);
        game->tank_count++;
    }

    progress->game_active = true;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    printf("========================================\n");
    printf("  坦克大战 - 玩家 vs AI 挑战模式\n");
    printf("========================================\n");
    printf("\n目标: 连胜 %d 局获得最终胜利！\n", WIN_TARGET);
    printf("\n规则:\n");
    printf("  - 战胜AI后，AI数量增加1\n");
    printf("  - 输掉任何一局，挑战失败\n");
    printf("\n操作:\n");
    printf("  WASD - 移动\n");
    printf("  空格 - 在当前方向射击\n");
    printf("  ESC  - 退出游戏\n");
    printf("========================================\n\n");

    // 初始化渲染器
    Renderer renderer;
    if (!renderer_init(&renderer, WINDOW_WIDTH, WINDOW_HEIGHT, "Tank Battle - Player vs AI")) {
        fprintf(stderr, "渲染器初始化失败\n");
        return 1;
    }

    // 初始化游戏
    GameState game;
    game_init(&game, WINDOW_WIDTH, WINDOW_HEIGHT, MODE_PLAYER);

    // 初始化进度跟踪
    GameProgress progress = {0};
    progress.wins = 0;
    progress.current_ai_count = 1;  // 从1个AI开始

    // 开始第一回合
    start_new_round(&game, &progress);

    bool running = true;
    bool show_result = false;
    Uint32 result_start_time = 0;
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
        }

        // 获取键盘状态
        const Uint8* keys = SDL_GetKeyboardState(NULL);

        // 游戏逻辑
        if (progress.game_active && !game.game_over) {
            // 获取玩家坦克
            Tank* player_tank = NULL;
            int player_id = -1;
            for (int i = 0; i < game.tank_count; i++) {
                if (game.tanks[i].type == TANK_TYPE_PLAYER && game.tanks[i].alive) {
                    player_tank = &game.tanks[i];
                    player_id = i;
                    break;
                }
            }

            // 处理玩家输入
            if (player_tank) {
                TankAction action = get_player_input(keys, player_tank);
                if (action != ACTION_IDLE) {
                    game_execute_action(&game, player_id, action);
                }
            }

            // 更新游戏
            game_update(&game);
        }

        // 检查回合结束
        if (progress.game_active && game.game_over) {
            progress.game_active = false;
            show_result = true;
            result_start_time = SDL_GetTicks();

            // 检查玩家是否获胜
            bool player_alive = false;
            for (int i = 0; i < game.tank_count; i++) {
                if (game.tanks[i].type == TANK_TYPE_PLAYER && game.tanks[i].alive) {
                    player_alive = true;
                    break;
                }
            }

            if (player_alive) {
                // 玩家获胜
                progress.wins++;
                printf("✓ 第 %d 局胜利！ (连胜: %d/%d)\n", progress.wins, progress.wins, WIN_TARGET);

                if (progress.wins >= WIN_TARGET) {
                    // 最终胜利
                    printf("\n========================================\n");
                    printf("  🎉 恭喜！你战胜了所有挑战！\n");
                    printf("  最终连胜: %d 局\n", progress.wins);
                    printf("  最高AI数量: %d\n", progress.current_ai_count);
                    printf("========================================\n");
                    snprintf(progress.message, sizeof(progress.message),
                             "VICTORY! You defeated all challenges!");
                } else {
                    // 准备下一回合
                    progress.current_ai_count++;
                    snprintf(progress.message, sizeof(progress.message),
                             "Round %d WIN! Next: %d AI tanks",
                             progress.wins, progress.current_ai_count);
                }
            } else {
                // 玩家失败
                printf("✗ 挑战失败！(在第 %d 局)\n", progress.wins + 1);
                printf("\n========================================\n");
                printf("  游戏结束\n");
                printf("  最终成绩: 连胜 %d 局\n", progress.wins);
                printf("  最高AI数量: %d\n", progress.current_ai_count);
                printf("========================================\n");
                snprintf(progress.message, sizeof(progress.message),
                         "GAME OVER - Final Score: %d wins", progress.wins);
            }
        }

        // 显示结果后自动开始下一回合或结束
        if (show_result && SDL_GetTicks() - result_start_time > 3000) {
            show_result = false;

            // 检查是否继续
            bool player_alive = false;
            for (int i = 0; i < game.tank_count; i++) {
                if (game.tanks[i].type == TANK_TYPE_PLAYER && game.tanks[i].alive) {
                    player_alive = true;
                    break;
                }
            }

            if (player_alive && progress.wins < WIN_TARGET) {
                // 开始下一回合
                start_new_round(&game, &progress);
            } else {
                // 游戏结束
                running = false;
            }
        }

        // 渲染
        renderer_render_game(&renderer, &game);

        // 渲染进度信息（简化版，直接在控制台输出）
        // TODO: 可以添加SDL文本渲染显示进度

        // 帧率控制
        frame_time = SDL_GetTicks() - frame_start;
        if (frame_delay > frame_time) {
            SDL_Delay(frame_delay - frame_time);
        }
    }

    // 清理
    renderer_cleanup(&renderer);

    printf("\n感谢游戏！\n");
    return 0;
}
