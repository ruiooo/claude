/*
 * ai_interface.c - AI训练接口实现
 */

#include "ai_interface.h"
#include "rendering.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 全局游戏状态
GameState g_game;
Renderer g_renderer;
bool g_visualize = false;

// 初始化游戏环境
void ai_init_env(int width, int height, int visualize) {
    game_init(&g_game, width, height, visualize ? MODE_TRAINING_VIS : MODE_TRAINING);

    g_visualize = visualize;
    if (g_visualize) {
        renderer_init(&g_renderer, width, height, "Tank Battle AI Training");
    }
}

// 重置环境
void ai_reset_env(int enemy_count, float* obs, int* obs_size) {
    game_reset(&g_game, enemy_count);
    game_get_observation(&g_game, obs, obs_size);
}

/*
 * 计算奖励（优化版 - 奖励塑形 Reward Shaping）
 *
 * 【核心函数】这是强化学习中最关键的函数！
 * 奖励函数设计直接决定AI学到什么行为策略
 *
 * 奖励设计哲学：
 * 1. 稀疏奖励（基础）：击杀、受伤、胜利等关键事件
 * 2. 密集奖励（塑形）：距离、瞄准、躲避等持续性指导
 * 3. 平衡探索与利用：不能过于惩罚探索，否则AI学不到新策略
 *
 * 数值设计原则：
 * - 大事件用大奖励（击杀100、胜利300、死亡-100）
 * - 小引导用小奖励（距离0.5、瞄准0.3、存活0.02）
 * - 避免奖励过于稀疏或过于密集
 *
 * 奖励分类：
 * ┌─────────────────────────────────────────────────────────────┐
 * │ 基础奖励（稀疏）         | 数值      | 作用               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 存活每帧                 | +0.02     | 鼓励生存           │
 * │ 受伤                     | -15.0     | 惩罚被击中         │
 * │ 击杀敌人                 | +100.0    | 主要目标           │
 * │ 胜利                     | +300.0    | 最终目标           │
 * │ 死亡                     | -100.0    | 严重惩罚           │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 奖励塑形（密集）         | 数值      | 作用               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 1. 距离奖励              | 0-0.5     | 引导接近敌人       │
 * │ 2. 瞄准奖励              | 0-0.3     | 引导朝向敌人移动   │
 * │ 3. 射击奖励              | ±0.5      | 引导合理时机射击   │
 * │ 4. 躲避子弹              | ±0.4      | 引导躲避危险       │
 * │ 5. 墙壁避让              | +0.2      | 避免卡边           │
 * │ 6. 动作多样性            | +0.1      | 避免静止不动       │
 * └─────────────────────────────────────────────────────────────┘
 *
 * 参数说明：
 * @param game        - 游戏状态
 * @param ai_tank     - AI坦克对象
 * @param prev_health - 上一帧血量（用于检测受伤）
 * @param prev_enemies- 上一帧敌人数（用于检测击杀）
 * @return 该步的总奖励值
 */
static float calculate_reward(GameState* game, Tank* ai_tank, int prev_health,
                              int prev_enemies) {
    float reward = 0.0f;

    // 死亡直接返回巨大惩罚
    if (!ai_tank->alive) {
        return -100.0f;  // 死亡是最坏的结果
    }

    // ========== 基础奖励（稀疏信号）==========

    // 存活奖励：每帧+0.02，鼓励AI尽可能活得久
    // 计算：60fps下，存活1秒=60帧=1.2分
    reward += 0.02f;

    // 血量变化检测
    int health_change = ai_tank->health - prev_health;
    if (health_change < 0) {
        // 受伤惩罚：-15分/次，让AI学会躲避攻击
        // 为什么是-15？比存活奖励大很多（750帧才能抵消），迫使AI重视防御
        reward -= 15.0f;
    }

    // 击杀检测
    int current_enemies = game_get_alive_count(game, TANK_TYPE_ENEMY) +
                         game_get_alive_count(game, TANK_TYPE_SELF_PLAY);
    int enemies_killed = prev_enemies - current_enemies;
    if (enemies_killed > 0) {
        // 击杀奖励：+100分/个，这是主要目标
        // 为什么是100？远大于受伤惩罚（-15），鼓励主动进攻
        reward += 100.0f * enemies_killed;
    }

    // 胜利奖励
    if (game->game_over && game->winner == 0) {
        // 胜利：+300分，最高荣誉
        // 等价于击杀3个敌人，或存活15000帧（250秒）
        reward += 300.0f;
    }

    // ========== 奖励塑形（密集引导信号）==========

    // 奖励塑形的目的：在稀疏奖励之间提供梯度信号
    // 问题：如果只有击杀/死亡奖励，AI很难学习（太稀疏）
    // 解决：提供距离、瞄准等中间指标，像"游戏提示"一样引导AI

    // --- 1. 距离奖励：引导AI接近敌人 ---

    float min_enemy_dist = 99999.0f;
    Tank* nearest_enemy = NULL;

    // 找到最近的敌人
    for (int i = 0; i < game->tank_count; i++) {
        Tank* tank = &game->tanks[i];
        if (tank->alive && (tank->type == TANK_TYPE_ENEMY || tank->type == TANK_TYPE_SELF_PLAY)) {
            float dx = tank->x - ai_tank->x;
            float dy = tank->y - ai_tank->y;
            float dist = sqrtf(dx * dx + dy * dy);  // 欧几里得距离

            if (dist < min_enemy_dist) {
                min_enemy_dist = dist;
                nearest_enemy = tank;
            }
        }
    }

    if (nearest_enemy != NULL) {
        // 距离归一化：假设最大距离为1000像素（对角线距离）
        float normalized_dist = min_enemy_dist / 1000.0f;  // 范围: [0, 1]

        // 距离奖励公式：(1 - normalized_dist) * 0.5
        // 距离0时：奖励=0.5，距离1000时：奖励=0
        // 效果：越靠近敌人，奖励越高
        reward += (1.0f - normalized_dist) * 0.5f;

        // 安全距离惩罚：太近反而危险（容易被击中）
        if (min_enemy_dist < 100.0f) {
            reward -= 0.3f;  // 小惩罚，教AI保持合理距离
        }
    }

    // --- 2. 瞄准奖励：引导AI朝向敌人移动 ---

    if (nearest_enemy != NULL) {
        // 计算到敌人的角度
        float dx = nearest_enemy->x - ai_tank->x;
        float dy = nearest_enemy->y - ai_tank->y;
        float angle_to_enemy = atan2f(dy, dx);  // 弧度，范围: [-π, π]

        // 只有在移动时才计算瞄准奖励（静止时没有方向）
        if (fabsf(ai_tank->vx) > 0.1f || fabsf(ai_tank->vy) > 0.1f) {
            // AI当前移动方向
            float ai_angle = atan2f(ai_tank->vy, ai_tank->vx);

            // 计算角度差（绝对值）
            float angle_diff = fabsf(angle_to_enemy - ai_angle);

            // 处理角度环绕：180° 和 -180° 其实是同一个方向
            if (angle_diff > M_PI) {
                angle_diff = 2 * M_PI - angle_diff;
            }

            // 对齐度：0度=1.0（完全对准），180度=0.0（背向）
            float alignment = 1.0f - (angle_diff / M_PI);

            // 瞄准奖励：最多+0.3
            // 效果：AI倾向于朝敌人方向移动
            reward += alignment * 0.3f;
        }
    }

    // --- 3. 射击奖励：引导AI在合适时机射击 ---

    // shoot_cooldown == 15 意味着刚刚射击
    if (ai_tank->shoot_cooldown == 15) {
        if (nearest_enemy != NULL && min_enemy_dist < 300.0f) {
            // 近距离射击：+0.5奖励
            // 原因：近距离命中率高
            reward += 0.5f;
        } else {
            // 远距离射击：-0.2惩罚
            // 原因：浪费子弹，命中率低
            reward -= 0.2f;
        }
    }

    // --- 4. 躲避子弹奖励：引导AI躲避危险 ---

    int dangerous_bullets = 0;

    // 检测危险子弹（飞向AI的子弹）
    for (int i = 0; i < MAX_BULLETS; i++) {
        Bullet* bullet = &game->bullets[i];

        // 跳过：不存在的子弹、AI自己的子弹
        if (!bullet->active || bullet->owner_id == 0) {
            continue;
        }

        // 子弹相对位置
        float dx = bullet->x - ai_tank->x;
        float dy = bullet->y - ai_tank->y;
        float dist = sqrtf(dx * dx + dy * dy);

        // 点积：判断子弹是否飞向AI
        // dot > 0: 子弹正在靠近，dot < 0: 子弹正在远离
        float dot = dx * bullet->vx + dy * bullet->vy;

        // 危险判定：距离<150且正在靠近
        if (dist < 150.0f && dot > 0) {
            dangerous_bullets++;
        }
    }

    // 根据AI的应对行为给予奖励/惩罚
    if (dangerous_bullets > 0) {
        // 检查AI是否在移动
        if (fabsf(ai_tank->vx) > 0.1f || fabsf(ai_tank->vy) > 0.1f) {
            // 移动躲避：+0.4奖励/弹
            // 效果：鼓励AI在危险时主动移动
            reward += 0.4f * dangerous_bullets;
        } else {
            // 静止不动：-0.3惩罚/弹
            // 效果：惩罚AI站桩挨打
            reward -= 0.3f * dangerous_bullets;
        }
    }

    // --- 5. 墙壁避让：避免AI卡在地图边缘 ---

    // 检测是否靠近墙壁（距离边界<50像素）
    bool near_wall = (ai_tank->x < 50 || ai_tank->x > game->map_width - 50 ||
                     ai_tank->y < 50 || ai_tank->y > game->map_height - 50);

    if (near_wall && (fabsf(ai_tank->vx) > 0.1f || fabsf(ai_tank->vy) > 0.1f)) {
        // 计算地图中心点
        float center_x = game->map_width / 2.0f;
        float center_y = game->map_height / 2.0f;

        // 朝向中心的向量
        float dx_to_center = center_x - ai_tank->x;
        float dy_to_center = center_y - ai_tank->y;

        // 点积：判断是否朝向中心移动
        float dot_product = ai_tank->vx * dx_to_center + ai_tank->vy * dy_to_center;

        if (dot_product > 0) {
            // 朝向中心移动：+0.2奖励
            // 效果：鼓励AI远离墙壁
            reward += 0.2f;
        }
        // 注意：这里没有else惩罚，避免过度限制AI行为
    }

    // --- 6. 动作多样性：避免AI完全静止 ---

    // 检查AI是否有任何动作（移动或射击）
    if (fabsf(ai_tank->vx) > 0.1f || fabsf(ai_tank->vy) > 0.1f || ai_tank->shoot_cooldown > 0) {
        // 有动作：+0.1小奖励
        // 效果：轻微鼓励探索，避免AI学成"木桩"
        reward += 0.1f;
    }

    return reward;
}

// 执行一步
void ai_step(int action, float* obs, int* obs_size, float* reward,
             int* done, int* info) {
    Tank* ai_tank = game_get_ai_tank(&g_game);
    if (!ai_tank) {
        *done = 1;
        *reward = -100.0f;
        return;
    }

    int prev_health = ai_tank->health;
    int prev_enemies = game_get_alive_count(&g_game, TANK_TYPE_ENEMY) +
                      game_get_alive_count(&g_game, TANK_TYPE_SELF_PLAY);

    // 执行AI动作
    game_execute_action(&g_game, 0, (TankAction)action);

    // 更新游戏
    game_update(&g_game);

    // 获取新状态
    game_get_observation(&g_game, obs, obs_size);

    // 计算奖励
    *reward = calculate_reward(&g_game, ai_tank, prev_health, prev_enemies);

    // 检查是否结束
    *done = g_game.game_over ? 1 : 0;

    // 额外信息
    info[0] = g_game.winner;  // 胜者
    info[1] = ai_tank->health;  // AI血量
    info[2] = prev_enemies - (game_get_alive_count(&g_game, TANK_TYPE_ENEMY) +
                              game_get_alive_count(&g_game, TANK_TYPE_SELF_PLAY));  // 击杀数
}

// 获取当前状态
void ai_get_state(float* obs, int* obs_size) {
    game_get_observation(&g_game, obs, obs_size);
}

// 添加敌人
void ai_add_enemy(int enemy_type, int model_version) {
    game_add_enemy(&g_game, (TankType)enemy_type, model_version);
}

// 获取统计信息
void ai_get_stats(int* ai_health, int* enemy_count, int* frame_count) {
    Tank* ai_tank = game_get_ai_tank(&g_game);
    *ai_health = ai_tank ? ai_tank->health : 0;
    *enemy_count = game_get_alive_count(&g_game, TANK_TYPE_ENEMY) +
                  game_get_alive_count(&g_game, TANK_TYPE_SELF_PLAY);
    *frame_count = g_game.frame_count;
}

// 渲染当前帧
void ai_render() {
    if (g_visualize) {
        renderer_render_game(&g_renderer, &g_game);

        // 处理SDL事件
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                g_visualize = false;
            }
        }
    }
}

// 清理
void ai_cleanup() {
    if (g_visualize) {
        renderer_cleanup(&g_renderer);
    }
}
