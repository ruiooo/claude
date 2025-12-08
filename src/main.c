/*
 * main.c - 玩家对战模式（玩家 vs AI训练模型）
 *
 * =================================================================
 * 核心功能: 人类玩家挑战AI模型的对战系统
 * =================================================================
 *
 * 游戏模式:
 * - 玩家从对抗1个AI开始
 * - 每次胜利后AI数量+1（动态难度）
 * - 连胜10局获得最终胜利
 * - 任何一局失败即挑战结束
 *
 * 人类数据收集（模仿学习）:
 * - 可选启用经验记录功能
 * - 记录玩家的状态-动作-奖励-下一状态四元组
 * - 自动保存为二进制文件(.dat格式)
 * - 用于训练时的模仿学习（见human_data_loader.py）
 *
 * 状态表示（43维，与训练时完全一致）:
 * - 玩家坦克: 6维 (x, y, vx, vy, health, shoot_cooldown)
 * - 最近5个敌人: 25维 (每个5维: x, y, vx, vy, health)
 * - 最近3个子弹: 12维 (每个4维: x, y, vx, vy)
 *
 * 动作空间（9维）:
 * - 0: 静止, 1-4: 移动(上下左右), 5-8: 射击(上下左右)
 *
 * 使用方法:
 *   ./tank_battle_player [model_path]  # 可选指定模型路径
 *   或
 *   make run-player  # 交互式选择模型
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "game.h"
#include "rendering.h"
#include "model_ai.h"

// ====================================================================
// 常量定义
// ====================================================================

#define WINDOW_WIDTH 800                    // 窗口宽度
#define WINDOW_HEIGHT 600                   // 窗口高度
#define FPS 60                              // 帧率(60FPS)
#define WIN_TARGET 10                       // 连胜目标(10局)
#define MAX_AI_TANKS 10                     // 最大AI坦克数量
#define MAX_EXPERIENCE_RECORDS 10000        // 最大经验记录数(单回合)

// ====================================================================
// 数据结构定义
// ====================================================================

/**
 * GameProgress - 游戏进度跟踪结构
 *
 * 职责: 跟踪挑战赛的整体进度
 * - 胜利次数统计
 * - 动态难度管理(AI数量)
 * - AI模型实例管理
 */
typedef struct {
    int wins;                           // 玩家连胜次数(0-WIN_TARGET)
    int current_ai_count;               // 当前回合的AI坦克数量
    bool game_active;                   // 当前回合是否进行中
    char message[256];                  // 显示给用户的消息
    ModelAI ai_models[MAX_AI_TANKS];   // AI模型实例数组(每个AI一个实例)
    char model_path[256];               // 选定的PyTorch模型文件路径
} GameProgress;

/**
 * ExperienceRecorder - 人类经验记录器
 *
 * 功能: 收集玩家游戏过程中的经验数据，用于模仿学习
 *
 * 数据格式:
 * - 与强化学习的transition完全一致
 * - (state, action, reward, next_state, done)
 *
 * 存储格式:
 * - 二进制文件(.dat)
 * - 结构: [count(int32)] [states(float[count][43])] [actions(int32[count])]
 *         [rewards(float[count])] [next_states(float[count][43])] [dones(int32[count])]
 *
 * 使用流程:
 * 1. recorder_init(): 初始化并分配内存
 * 2. recorder_add(): 每帧添加一条经验
 * 3. recorder_save(): 回合结束时保存到文件
 * 4. recorder_reset(): 准备新回合(生成新文件名)
 * 5. recorder_cleanup(): 释放内存
 */
typedef struct {
    float (*states)[43];        // 状态数组(每个状态43维)
    int* actions;               // 动作数组(每个动作是0-8的整数)
    float* rewards;             // 奖励数组
    float (*next_states)[43];   // 下一状态数组(每个状态43维)
    int* dones;                 // 结束标志数组(0=继续, 1=回合结束)
    int count;                  // 当前记录数(单回合)
    int capacity;               // 容量上限(防止内存溢出)
    char filename[256];         // 当前回合的保存文件名
    bool enabled;               // 是否启用记录功能
    int total_recorded;         // 总记录数(跨所有回合的累计)
} ExperienceRecorder;

// 初始化经验记录器
bool recorder_init(ExperienceRecorder* recorder, int capacity, bool enabled) {
    recorder->capacity = capacity;
    recorder->count = 0;
    recorder->total_recorded = 0;
    recorder->enabled = enabled;

    if (!enabled) {
        recorder->states = NULL;
        recorder->actions = NULL;
        recorder->rewards = NULL;
        recorder->next_states = NULL;
        recorder->dones = NULL;
        return true;
    }

    // 分配内存
    recorder->states = (float(*)[43])malloc(sizeof(float[43]) * capacity);
    recorder->actions = (int*)malloc(sizeof(int) * capacity);
    recorder->rewards = (float*)malloc(sizeof(float) * capacity);
    recorder->next_states = (float(*)[43])malloc(sizeof(float[43]) * capacity);
    recorder->dones = (int*)malloc(sizeof(int) * capacity);

    if (!recorder->states || !recorder->actions || !recorder->rewards ||
        !recorder->next_states || !recorder->dones) {
        fprintf(stderr, "经验记录器内存分配失败\n");
        recorder_cleanup(recorder);
        return false;
    }

    // 创建数据目录
    mkdir("human_data", 0755);

    // 生成文件名
    time_t now = time(NULL);
    snprintf(recorder->filename, sizeof(recorder->filename),
             "human_data/experience_%ld.dat", now);

    printf("\n🔴 经验记录已启用 - 你的操作将用于AI训练\n");
    printf("   数据将保存到: %s\n\n", recorder->filename);

    return true;
}

// 清理经验记录器
void recorder_cleanup(ExperienceRecorder* recorder) {
    if (recorder->states) free(recorder->states);
    if (recorder->actions) free(recorder->actions);
    if (recorder->rewards) free(recorder->rewards);
    if (recorder->next_states) free(recorder->next_states);
    if (recorder->dones) free(recorder->dones);

    recorder->states = NULL;
    recorder->actions = NULL;
    recorder->rewards = NULL;
    recorder->next_states = NULL;
    recorder->dones = NULL;
}

// 添加一条经验
bool recorder_add(ExperienceRecorder* recorder,
                  const float* state, int action, float reward,
                  const float* next_state, int done) {
    if (!recorder->enabled || recorder->count >= recorder->capacity) {
        return false;
    }

    memcpy(recorder->states[recorder->count], state, 43 * sizeof(float));
    recorder->actions[recorder->count] = action;
    recorder->rewards[recorder->count] = reward;
    memcpy(recorder->next_states[recorder->count], next_state, 43 * sizeof(float));
    recorder->dones[recorder->count] = done;

    recorder->count++;
    recorder->total_recorded++;

    return true;
}

// 保存经验到文件
bool recorder_save(ExperienceRecorder* recorder) {
    if (!recorder->enabled || recorder->count == 0) {
        return false;
    }

    FILE* fp = fopen(recorder->filename, "wb");
    if (!fp) {
        fprintf(stderr, "⚠ 无法创建文件: %s\n", recorder->filename);
        return false;
    }

    // 写入数据
    fwrite(&recorder->count, sizeof(int), 1, fp);
    fwrite(recorder->states, sizeof(float), recorder->count * 43, fp);
    fwrite(recorder->actions, sizeof(int), recorder->count, fp);
    fwrite(recorder->rewards, sizeof(float), recorder->count, fp);
    fwrite(recorder->next_states, sizeof(float), recorder->count * 43, fp);
    fwrite(recorder->dones, sizeof(int), recorder->count, fp);

    fclose(fp);

    printf("\n✓ 已保存 %d 条人类经验到 %s\n", recorder->count, recorder->filename);
    printf("  总计记录: %d 条经验\n", recorder->total_recorded);

    return true;
}

// 重置记录器（用于新回合）
void recorder_reset(ExperienceRecorder* recorder) {
    if (!recorder->enabled) return;

    // 如果有数据，先保存
    if (recorder->count > 0) {
        recorder_save(recorder);
    }

    // 重置计数
    recorder->count = 0;

    // 生成新文件名
    time_t now = time(NULL);
    snprintf(recorder->filename, sizeof(recorder->filename),
             "human_data/experience_%ld.dat", now);
}

// 前向声明
void recorder_cleanup(ExperienceRecorder* recorder);

// 将游戏状态转换为观察向量（43维，与训练时一致）
static void game_state_to_observation_for_recording(const GameState* game, int tank_id,
                                                     float* obs) {
    int idx = 0;
    const Tank* player_tank = &game->tanks[tank_id];

    // 玩家坦克状态 (6个值)
    obs[idx++] = player_tank->x / (float)game->map_width;
    obs[idx++] = player_tank->y / (float)game->map_height;
    obs[idx++] = player_tank->vx / TANK_SPEED;
    obs[idx++] = player_tank->vy / TANK_SPEED;
    obs[idx++] = (float)player_tank->health / (float)TANK_MAX_HEALTH;
    obs[idx++] = player_tank->shoot_cooldown > 0 ? 1.0f : 0.0f;

    // 找到最近的5个敌人 (5×5=25个值)
    typedef struct {
        float dist;
        int index;
    } EnemyDist;
    EnemyDist enemies[32];  // GameState支持最多32个坦克
    int enemy_count = 0;

    for (int i = 0; i < game->tank_count; i++) {
        if (i != tank_id && game->tanks[i].alive &&
            game->tanks[i].type != TANK_TYPE_PLAYER) {
            float dx = game->tanks[i].x - player_tank->x;
            float dy = game->tanks[i].y - player_tank->y;
            enemies[enemy_count].dist = dx * dx + dy * dy;
            enemies[enemy_count].index = i;
            enemy_count++;
        }
    }

    // 排序找最近的5个
    for (int i = 0; i < enemy_count - 1; i++) {
        for (int j = i + 1; j < enemy_count; j++) {
            if (enemies[j].dist < enemies[i].dist) {
                EnemyDist temp = enemies[i];
                enemies[i] = enemies[j];
                enemies[j] = temp;
            }
        }
    }

    // 添加最近的5个敌人的信息
    for (int i = 0; i < 5; i++) {
        if (i < enemy_count) {
            const Tank* enemy = &game->tanks[enemies[i].index];
            obs[idx++] = enemy->x / (float)game->map_width;
            obs[idx++] = enemy->y / (float)game->map_height;
            obs[idx++] = enemy->vx / TANK_SPEED;
            obs[idx++] = enemy->vy / TANK_SPEED;
            obs[idx++] = (float)enemy->health / (float)TANK_MAX_HEALTH;
        } else {
            // 填充0
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
        }
    }

    // 找到最近的3个子弹 (3×4=12个值)
    typedef struct {
        float dist;
        int index;
    } BulletDist;
    BulletDist bullets[MAX_BULLETS];
    int bullet_count = 0;

    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active) {
            float dx = game->bullets[i].x - player_tank->x;
            float dy = game->bullets[i].y - player_tank->y;
            bullets[bullet_count].dist = dx * dx + dy * dy;
            bullets[bullet_count].index = i;
            bullet_count++;
        }
    }

    // 排序找最近的3个
    for (int i = 0; i < bullet_count - 1; i++) {
        for (int j = i + 1; j < bullet_count; j++) {
            if (bullets[j].dist < bullets[i].dist) {
                BulletDist temp = bullets[i];
                bullets[i] = bullets[j];
                bullets[j] = temp;
            }
        }
    }

    // 添加最近的3个子弹的信息
    for (int i = 0; i < 3; i++) {
        if (i < bullet_count) {
            const Bullet* bullet = &game->bullets[bullets[i].index];
            obs[idx++] = bullet->x / (float)game->map_width;
            obs[idx++] = bullet->y / (float)game->map_height;
            obs[idx++] = bullet->vx / 10.0f;  // BULLET_SPEED = 10
            obs[idx++] = bullet->vy / 10.0f;
        } else {
            // 填充0
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
        }
    }

    // 总共应该是 6 + 25 + 12 = 43 维
}

// 计算玩家奖励（简化版）
float calculate_player_reward(const GameState* game, int player_id) {
    const Tank* player = &game->tanks[player_id];

    // 存活奖励
    if (!player->alive) {
        return -100.0f;
    }

    // 基础存活奖励
    float reward = 0.01f;

    // 这里可以添加更复杂的奖励计算
    // 比如: 击中敌人、躲避子弹等

    return reward;
}

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
    printf("AI坦克数量: %d (使用模型: %s)\n",
           progress->current_ai_count, progress->model_path);

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

    // 清理旧的AI模型实例
    for (int i = 0; i < MAX_AI_TANKS; i++) {
        if (progress->ai_models[i].initialized) {
            model_ai_cleanup(&progress->ai_models[i]);
        }
    }

    // 创建AI训练模型控制的坦克
    for (int i = 0; i < progress->current_ai_count && i < MAX_AI_TANKS; i++) {
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

        // 初始化AI模型
        if (!model_ai_init(&progress->ai_models[i], progress->model_path)) {
            fprintf(stderr, "警告: AI模型 %d 初始化失败\n", i);
        }

        game->tank_count++;
    }

    progress->game_active = true;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    printf("========================================\n");
    printf("  坦克大战 - 玩家 vs AI 挑战模式\n");
    printf("========================================\n");

    // 初始化Python模型AI系统
    if (!model_ai_system_init()) {
        fprintf(stderr, "模型AI系统初始化失败\n");
        return 1;
    }

    // 选择AI模型
    char selected_model[256] = "";

    if (argc > 1) {
        // 命令行指定模型
        strncpy(selected_model, argv[1], sizeof(selected_model) - 1);
        printf("使用指定模型: %s\n", selected_model);
    } else {
        // 交互式选择
        char models[50][256];
        int model_count = model_ai_list_models(models, 50);

        if (model_count == 0) {
            printf("\n⚠ 未找到训练模型！\n");
            printf("\n请先训练AI模型：\n");
            printf("  1. 使用虚拟环境训练: ./tank/bin/python3 python/train.py\n");
            printf("  2. 或使用Make命令:    make train\n");
            printf("\n模型将保存到 saved_models/checkpoints/ 目录\n");
            printf("训练完成后，再次运行 make run-player 即可开始游戏\n\n");
            model_ai_system_cleanup();
            return 1;
        }

        // 显示模型列表（最多5个最新的）
        printf("\n可用的AI模型 (最近%d个):\n", model_count);
        for (int i = 0; i < model_count; i++) {
            printf("  [%d] %s\n", i + 1, models[i]);
        }

        int choice = 1;  // 默认选择第一个
        printf("\n请选择模型 (1-%d, 默认=1): ", model_count);
        printf("\n提示: 按 Ctrl+C 退出\n");

        char input[32];
        if (fgets(input, sizeof(input), stdin)) {
            if (input[0] != '\n') {
                choice = atoi(input);
            }
        }

        if (choice < 1 || choice > model_count) {
            printf("无效选择，使用默认模型\n");
            choice = 1;
        }

        strncpy(selected_model, models[choice - 1], sizeof(selected_model) - 1);
        printf("✓ 已选择: %s\n", selected_model);
    }

    printf("\n目标: 连胜 %d 局获得最终胜利！\n", WIN_TARGET);
    printf("\n规则:\n");
    printf("  - 战胜AI后，AI数量增加1\n");
    printf("  - 输掉任何一局，挑战失败\n");
    printf("\n操作:\n");
    printf("  WASD - 移动\n");
    printf("  空格 - 在当前方向射击\n");
    printf("  ESC  - 退出游戏\n");
    printf("========================================\n\n");

    // 询问是否启用经验记录
    printf("是否启用经验记录功能? (y/n, 默认=n): ");
    char record_choice[32] = "";
    bool enable_recording = false;
    if (fgets(record_choice, sizeof(record_choice), stdin)) {
        if (record_choice[0] == 'y' || record_choice[0] == 'Y') {
            enable_recording = true;
            printf("✓ 已启用经验记录，数据将保存到 human_data/ 目录\n\n");
        } else {
            printf("经验记录已禁用\n\n");
        }
    }

    // 初始化经验记录器
    ExperienceRecorder recorder;
    if (!recorder_init(&recorder, MAX_EXPERIENCE_RECORDS, enable_recording)) {
        fprintf(stderr, "经验记录器初始化失败\n");
        return 1;
    }

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
    strncpy(progress.model_path, selected_model, sizeof(progress.model_path) - 1);

    // 初始化AI模型数组
    for (int i = 0; i < MAX_AI_TANKS; i++) {
        progress.ai_models[i].initialized = false;
    }

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

            // 记录当前状态（如果启用了记录）
            float current_state[43] = {0};
            TankAction player_action = ACTION_IDLE;
            bool should_record = false;

            if (recorder.enabled && player_tank) {
                // 捕获当前状态（与训练时一致的43维）
                game_state_to_observation_for_recording(&game, player_id, current_state);
                should_record = true;
            }

            // 处理玩家输入
            if (player_tank) {
                player_action = get_player_input(keys, player_tank);
                if (player_action != ACTION_IDLE) {
                    game_execute_action(&game, player_id, player_action);
                } else {
                    // 没有输入时立即停止移动
                    player_tank->vx = 0;
                    player_tank->vy = 0;
                }
            }

            // AI坦克控制
            int ai_index = 0;
            for (int i = 0; i < game.tank_count; i++) {
                if (game.tanks[i].type == TANK_TYPE_ENEMY && game.tanks[i].alive) {
                    if (ai_index < MAX_AI_TANKS && progress.ai_models[ai_index].initialized) {
                        // 使用模型AI获取动作
                        TankAction action = model_ai_get_action(&progress.ai_models[ai_index],
                                                                &game, i);
                        if (action != ACTION_IDLE) {
                            game_execute_action(&game, i, action);
                        }
                    }
                    ai_index++;
                }
            }

            // 更新游戏
            game_update(&game);

            // 记录经验（如果启用了记录且有动作）
            if (should_record && player_action != ACTION_IDLE && player_tank && player_tank->alive) {
                float next_state[43] = {0};
                game_state_to_observation_for_recording(&game, player_id, next_state);

                float reward = calculate_player_reward(&game, player_id);
                int done = game.game_over ? 1 : 0;

                recorder_add(&recorder, current_state, player_action, reward, next_state, done);
            }
        }

        // 检查回合结束
        if (progress.game_active && game.game_over) {
            progress.game_active = false;
            show_result = true;
            result_start_time = SDL_GetTicks();

            // 保存本回合的经验数据
            if (recorder.enabled && recorder.count > 0) {
                recorder_save(&recorder);
            }

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
                // 重置经验记录器准备下一回合
                if (recorder.enabled) {
                    recorder_reset(&recorder);
                }
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
    // 清理经验记录器
    recorder_cleanup(&recorder);

    // 清理所有AI模型实例
    for (int i = 0; i < MAX_AI_TANKS; i++) {
        if (progress.ai_models[i].initialized) {
            model_ai_cleanup(&progress.ai_models[i]);
        }
    }

    // 清理Python模型AI系统
    model_ai_system_cleanup();

    renderer_cleanup(&renderer);

    printf("\n感谢游戏！\n");
    return 0;
}
