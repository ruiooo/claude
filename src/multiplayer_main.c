/*
 * multiplayer_main.c - 多人联机对战模式
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <SDL2/SDL.h>
#include "game.h"
#include "rendering.h"
#include "network.h"

#define MAP_WIDTH 800
#define MAP_HEIGHT 600

// 玩家控制
typedef struct {
    TankType type;
    int tank_id;
    bool is_local;
    // 键盘映射
    SDL_Keycode key_up;
    SDL_Keycode key_down;
    SDL_Keycode key_left;
    SDL_Keycode key_right;
    SDL_Keycode key_shoot;
} PlayerControl;

// 处理键盘输入并返回动作
TankAction get_player_action(PlayerControl* control, const Uint8* keyboard_state) {
    // 射击优先
    if (keyboard_state[SDL_GetScancodeFromKey(control->key_shoot)]) {
        // 根据移动键判断射击方向
        if (keyboard_state[SDL_GetScancodeFromKey(control->key_up)]) {
            return ACTION_SHOOT_UP;
        } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_down)]) {
            return ACTION_SHOOT_DOWN;
        } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_left)]) {
            return ACTION_SHOOT_LEFT;
        } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_right)]) {
            return ACTION_SHOOT_RIGHT;
        }
        // 如果没有方向键，根据当前方向射击
        return ACTION_SHOOT_UP;  // 默认向上
    }

    // 移动
    if (keyboard_state[SDL_GetScancodeFromKey(control->key_up)]) {
        return ACTION_MOVE_UP;
    } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_down)]) {
        return ACTION_MOVE_DOWN;
    } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_left)]) {
        return ACTION_MOVE_LEFT;
    } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_right)]) {
        return ACTION_MOVE_RIGHT;
    }

    return ACTION_IDLE;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("使用方法:\n");
        printf("  服务器（玩家1）: %s server [port]\n", argv[0]);
        printf("  客户端（玩家2）: %s client <服务器IP> [port]\n", argv[0]);
        return 1;
    }

    // 解析命令行参数
    NetRole role;
    char server_ip[64] = "127.0.0.1";
    int port = DEFAULT_PORT;

    if (strcmp(argv[1], "server") == 0) {
        role = NET_ROLE_SERVER;
        if (argc >= 3) {
            port = atoi(argv[2]);
        }
        printf("启动服务器模式（玩家1 - 绿色）\n");
    } else if (strcmp(argv[1], "client") == 0) {
        role = NET_ROLE_CLIENT;
        if (argc < 3) {
            fprintf(stderr, "错误: 客户端模式需要指定服务器IP\n");
            return 1;
        }
        strncpy(server_ip, argv[2], sizeof(server_ip) - 1);
        if (argc >= 4) {
            port = atoi(argv[3]);
        }
        printf("启动客户端模式（玩家2 - 深绿色）\n");
    } else {
        fprintf(stderr, "错误: 未知模式 '%s'\n", argv[1]);
        return 1;
    }

    // 初始化网络
    NetworkConnection net_conn;
    if (!network_init(&net_conn, role, server_ip, port)) {
        fprintf(stderr, "网络初始化失败\n");
        return 1;
    }

    // 建立连接
    if (role == NET_ROLE_SERVER) {
        if (!network_listen(&net_conn)) {
            fprintf(stderr, "监听失败\n");
            return 1;
        }
    } else {
        if (!network_connect(&net_conn)) {
            fprintf(stderr, "连接失败\n");
            return 1;
        }
    }

    // 初始化游戏
    GameState game;
    game_init(&game, MAP_WIDTH, MAP_HEIGHT, MODE_PLAYER);

    // 创建两个玩家坦克
    if (role == NET_ROLE_SERVER) {
        // 服务器：玩家1在左边（绿色）
        tank_init(&game.tanks[0], 100, MAP_HEIGHT / 2, TANK_TYPE_PLAYER);
        game.tank_count = 1;

        // 玩家2在右边（深绿色）
        tank_init(&game.tanks[1], MAP_WIDTH - 100, MAP_HEIGHT / 2, TANK_TYPE_PLAYER2);
        game.tank_count = 2;
    } else {
        // 客户端：玩家1在左边（绿色）
        tank_init(&game.tanks[0], 100, MAP_HEIGHT / 2, TANK_TYPE_PLAYER);
        game.tank_count = 1;

        // 玩家2在右边（深绿色）
        tank_init(&game.tanks[1], MAP_WIDTH - 100, MAP_HEIGHT / 2, TANK_TYPE_PLAYER2);
        game.tank_count = 2;
    }

    // 设置玩家控制
    PlayerControl local_control;
    if (role == NET_ROLE_SERVER) {
        // 服务器控制玩家1（绿色）
        local_control.type = TANK_TYPE_PLAYER;
        local_control.tank_id = 0;
    } else {
        // 客户端控制玩家2（深绿色）
        local_control.type = TANK_TYPE_PLAYER2;
        local_control.tank_id = 1;
    }
    local_control.is_local = true;
    local_control.key_up = SDLK_w;
    local_control.key_down = SDLK_s;
    local_control.key_left = SDLK_a;
    local_control.key_right = SDLK_d;
    local_control.key_shoot = SDLK_SPACE;

    // 初始化渲染器
    Renderer renderer;
    const char* window_title = (role == NET_ROLE_SERVER) ?
        "坦克大战 - 玩家1 (绿色)" : "坦克大战 - 玩家2 (深绿色)";

    if (!renderer_init(&renderer, MAP_WIDTH, MAP_HEIGHT, window_title)) {
        fprintf(stderr, "渲染器初始化失败\n");
        network_close(&net_conn);
        return 1;
    }

    // 游戏主循环
    bool running = true;
    SDL_Event event;
    Uint32 last_time = SDL_GetTicks();
    const int FRAME_DELAY = 16;  // 约60fps

    printf("\n========== 游戏开始 ==========\n");
    printf("操作说明:\n");
    printf("  WASD - 移动\n");
    printf("  空格 - 射击\n");
    printf("  ESC - 退出\n");
    printf("=============================\n\n");

    while (running && net_conn.connected) {
        // 处理事件
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }
            }
        }

        // 获取键盘状态
        const Uint8* keyboard_state = SDL_GetKeyboardState(NULL);

        // 获取本地玩家动作
        TankAction local_action = get_player_action(&local_control, keyboard_state);

        // 发送本地动作到对方
        network_send_action(&net_conn, local_action);

        // 接收对方动作
        TankAction remote_action = ACTION_IDLE;
        network_recv_action(&net_conn, &remote_action);

        // 执行动作
        if (role == NET_ROLE_SERVER) {
            game_execute_action(&game, 0, local_action);   // 玩家1
            game_execute_action(&game, 1, remote_action);  // 玩家2
        } else {
            game_execute_action(&game, 0, remote_action);  // 玩家1
            game_execute_action(&game, 1, local_action);   // 玩家2
        }

        // 更新游戏
        game_update(&game);

        // 渲染
        renderer_render_game(&renderer, &game);

        // 检查游戏结束
        if (game.game_over) {
            SDL_Delay(3000);  // 显示结果3秒
            running = false;
        }

        // 帧率控制
        Uint32 current_time = SDL_GetTicks();
        Uint32 elapsed = current_time - last_time;
        if (elapsed < FRAME_DELAY) {
            SDL_Delay(FRAME_DELAY - elapsed);
        }
        last_time = SDL_GetTicks();
    }

    // 清理
    printf("\n游戏结束！\n");
    renderer_cleanup(&renderer);
    network_close(&net_conn);

    return 0;
}
