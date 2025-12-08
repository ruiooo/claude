/*
 * multiplayer_main.c - 多人联机对战模式
 *
 * =================================================================
 * 核心功能: 基于TCP socket的局域网双人对战
 * =================================================================
 *
 * 游戏模式:
 * - 1 vs 1人类玩家对战
 * - 通过网络同步坦克动作
 * - 服务端/客户端架构
 *
 * 网络架构:
 * - 服务端（玩家1）：监听端口8888，绿色坦克
 * - 客户端（玩家2）：连接到服务端IP，深绿色坦克
 * - 同步策略：每帧发送本地动作，接收对端动作
 *
 * 操作方式:
 * - 两个玩家都使用WASD移动
 * - 空格键射击（当前朝向）
 * - ESC键退出
 *
 * 使用方法:
 *   服务端: ./tank_battle_multiplayer server [port]
 *   客户端: ./tank_battle_multiplayer client <server_ip> [port]
 *   或
 *   make run-multiplayer-server  # 终端1
 *   make run-multiplayer-client  # 终端2
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <SDL2/SDL.h>
#include "game.h"
#include "rendering.h"
#include "network.h"

// 地图尺寸常量
#define MAP_WIDTH 800       // 地图宽度（像素）
#define MAP_HEIGHT 600      // 地图高度（像素）

/**
 * PlayerControl - 玩家控制结构
 *
 * 职责: 管理单个玩家的输入映射和状态
 *
 * 字段说明:
 * - type: 坦克类型（TANK_TYPE_PLAYER或TANK_TYPE_PLAYER2）
 * - tank_id: 坦克在游戏中的索引（0或1）
 * - is_local: 是否是本地玩家（本地=键盘控制，远程=网络控制）
 * - key_*: 键盘映射（WASD + 空格）
 */
typedef struct {
    TankType type;          // 坦克类型
    int tank_id;            // 坦克ID（0-31）
    bool is_local;          // 是否本地玩家
    // 键盘映射（可自定义）
    SDL_Keycode key_up;     // 上移键（W）
    SDL_Keycode key_down;   // 下移键（S）
    SDL_Keycode key_left;   // 左移键（A）
    SDL_Keycode key_right;  // 右移键（D）
    SDL_Keycode key_shoot;  // 射击键（空格）
} PlayerControl;

/**
 * 获取玩家动作（输入处理）
 *
 * 功能：将键盘状态转换为坦克动作
 *
 * 参数：
 *   control        - 玩家控制配置
 *   keyboard_state - SDL键盘状态数组
 *   player_tank    - 玩家坦克对象
 *
 * 返回值：
 *   TankAction - 坦克动作枚举（IDLE, MOVE_*, SHOOT_*）
 *
 * 优先级：
 *   1. 射击（空格） - 在当前朝向射击
 *   2. 移动（WASD） - 四方向移动
 *   3. 无操作（IDLE）
 *
 * 设计说明：
 *   - 射击优先于移动（更符合游戏习惯）
 *   - 同时按下移动键时，优先级为：上>下>左>右
 *   - 射击不改变朝向（使用坦克当前direction）
 */
TankAction get_player_action(PlayerControl* control, const Uint8* keyboard_state, Tank* player_tank) {
    // ========== 优先级1：射击检测 ==========
    // 空格键直接在当前朝向开炮（不改变方向）
    if (keyboard_state[SDL_GetScancodeFromKey(control->key_shoot)]) {
        // 根据坦克当前朝向选择射击动作
        switch (player_tank->direction) {
            case DIR_UP:    return ACTION_SHOOT_UP;
            case DIR_DOWN:  return ACTION_SHOOT_DOWN;
            case DIR_LEFT:  return ACTION_SHOOT_LEFT;
            case DIR_RIGHT: return ACTION_SHOOT_RIGHT;
        }
    }

    // ========== 优先级2：移动检测 ==========
    // 检查移动键（WASD），优先级：上>下>左>右
    if (keyboard_state[SDL_GetScancodeFromKey(control->key_up)]) {
        return ACTION_MOVE_UP;
    } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_down)]) {
        return ACTION_MOVE_DOWN;
    } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_left)]) {
        return ACTION_MOVE_LEFT;
    } else if (keyboard_state[SDL_GetScancodeFromKey(control->key_right)]) {
        return ACTION_MOVE_RIGHT;
    }

    // ========== 优先级3：无操作 ==========
    return ACTION_IDLE;
}

/**
 * 主函数 - 多人对战模式入口
 *
 * 流程：
 * 1. 解析命令行参数（服务端/客户端）
 * 2. 初始化网络连接
 * 3. 初始化游戏和渲染器
 * 4. 主游戏循环（输入→同步→更新→渲染）
 * 5. 清理资源
 */
int main(int argc, char* argv[]) {
    // ========== 步骤1：检查命令行参数 ==========
    if (argc < 2) {
        printf("使用方法:\n");
        printf("  服务器（玩家1）: %s server [port]\n", argv[0]);
        printf("  客户端（玩家2）: %s client <服务器IP> [port]\n", argv[0]);
        return 1;
    }

    // ========== 步骤2：解析命令行参数 ==========
    NetRole role;                           // 网络角色（服务端/客户端）
    char server_ip[64] = "127.0.0.1";       // 服务端IP（客户端需要）
    int port = DEFAULT_PORT;                // 端口号（默认8888）

    // 解析模式参数（argv[1]）
    if (strcmp(argv[1], "server") == 0) {
        // 服务端模式
        role = NET_ROLE_SERVER;
        if (argc >= 3) {
            port = atoi(argv[2]);  // 可选：自定义端口
        }
        printf("启动服务器模式（玩家1 - 绿色）\n");
    } else if (strcmp(argv[1], "client") == 0) {
        // 客户端模式
        role = NET_ROLE_CLIENT;
        // 客户端必须指定服务端IP（argv[2]）
        if (argc < 3) {
            fprintf(stderr, "错误: 客户端模式需要指定服务器IP\n");
            return 1;
        }
        strncpy(server_ip, argv[2], sizeof(server_ip) - 1);
        if (argc >= 4) {
            port = atoi(argv[3]);  // 可选：自定义端口
        }
        printf("启动客户端模式（玩家2 - 深绿色）\n");
    } else {
        // 未知模式
        fprintf(stderr, "错误: 未知模式 '%s'\n", argv[1]);
        return 1;
    }

    // ========== 步骤3：初始化网络连接 ==========
    NetworkConnection net_conn;
    if (!network_init(&net_conn, role, server_ip, port)) {
        fprintf(stderr, "网络初始化失败\n");
        return 1;
    }

    // 建立连接（阻塞直到连接成功）
    if (role == NET_ROLE_SERVER) {
        // 服务端：监听并等待客户端连接
        if (!network_listen(&net_conn)) {
            fprintf(stderr, "监听失败\n");
            return 1;
        }
    } else {
        // 客户端：连接到服务端
        if (!network_connect(&net_conn)) {
            fprintf(stderr, "连接失败\n");
            return 1;
        }
    }

    // ========== 步骤4：初始化游戏 ==========
    GameState game;
    game_init(&game, MAP_WIDTH, MAP_HEIGHT, MODE_PLAYER);  // 玩家模式

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

    // 初始化渲染器（不同位置避免窗口重叠）
    Renderer renderer;
    const char* window_title;
    int window_x, window_y;

    if (role == NET_ROLE_SERVER) {
        window_title = "坦克大战 - 玩家1 (绿色)";
        window_x = 50;   // 服务器窗口在左边
        window_y = 100;
    } else {
        window_title = "坦克大战 - 玩家2 (深绿色)";
        window_x = 850;  // 客户端窗口在右边
        window_y = 100;
    }

    if (!renderer_init_with_pos(&renderer, MAP_WIDTH, MAP_HEIGHT, window_title, window_x, window_y)) {
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

    // ========== 步骤7：主游戏循环 ==========
    // 运行条件：程序未退出 且 网络连接正常
    while (running && net_conn.connected) {
        // --- 7.1：处理SDL事件 ---
        // 检查窗口关闭、ESC键等系统事件
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;  // 窗口关闭
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;  // ESC键退出
                }
            }
        }

        // --- 7.2：获取本地玩家输入 ---
        // SDL_GetKeyboardState：获取所有键的当前状态
        const Uint8* keyboard_state = SDL_GetKeyboardState(NULL);

        // 获取本地玩家坦克对象
        Tank* local_tank = &game.tanks[local_control.tank_id];

        // 将键盘状态转换为坦克动作
        TankAction local_action = get_player_action(&local_control, keyboard_state, local_tank);

        // --- 7.3：网络同步动作 ---
        // 发送本地玩家动作到对端（非阻塞）
        network_send_action(&net_conn, local_action);

        // 接收对端玩家动作（非阻塞，可能无数据）
        TankAction remote_action = ACTION_IDLE;
        network_recv_action(&net_conn, &remote_action);

        // --- 7.4：执行动作 ---
        // 重要：服务端和客户端对坦克ID的理解不同
        // 服务端：tank[0]=玩家1（本地），tank[1]=玩家2（远程）
        // 客户端：tank[0]=玩家1（远程），tank[1]=玩家2（本地）
        if (role == NET_ROLE_SERVER) {
            game_execute_action(&game, 0, local_action);   // 玩家1（本地）
            game_execute_action(&game, 1, remote_action);  // 玩家2（远程）
        } else {
            game_execute_action(&game, 0, remote_action);  // 玩家1（远程）
            game_execute_action(&game, 1, local_action);   // 玩家2（本地）
        }

        // --- 7.5：更新游戏逻辑 ---
        // 坦克移动、子弹更新、碰撞检测、胜负判定等
        game_update(&game);

        // --- 7.6：渲染画面 ---
        // 绘制坦克、子弹、UI等
        renderer_render_game(&renderer, &game);

        // --- 7.7：检查游戏结束 ---
        if (game.game_over) {
            SDL_Delay(3000);  // 显示胜负结果3秒
            running = false;  // 结束游戏循环
        }

        // --- 7.8：帧率控制（60fps） ---
        // 目标：每帧16ms（1000ms/60fps ≈ 16.67ms）
        Uint32 current_time = SDL_GetTicks();
        Uint32 elapsed = current_time - last_time;
        if (elapsed < FRAME_DELAY) {
            SDL_Delay(FRAME_DELAY - elapsed);  // 休眠补足时间
        }
        last_time = SDL_GetTicks();
    }

    // ========== 步骤8：清理资源 ==========
    // 游戏结束后，按顺序释放所有资源
    printf("\n游戏结束！\n");

    // 清理SDL渲染器资源（窗口、渲染器、字体）
    // 注意：必须在SDL_Quit之前调用
    renderer_cleanup(&renderer);

    // 关闭网络连接
    // 功能：发送PACKET_DISCONNECT通知对端，关闭所有socket
    network_close(&net_conn);

    return 0;  // 程序正常退出
}
