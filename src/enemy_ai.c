/*
 * enemy_ai.c - 敌人AI实现（智能射击和躲避型）
 *
 * 【设计理念】
 * 这个AI作为DQN训练的对手，需要足够强才能逼迫DQN学习，但不能强到无法战胜
 *
 * AI行为策略（优先级从高到低）：
 * 1. 躲避子弹（生存第一）
 * 2. 射击敌人（合适时机）
 * 3. 保持理想距离（战术定位）
 * 4. 防止卡住（兜底机制）
 *
 * 核心算法：
 * - 追踪算法：计算到目标的最短路径
 * - 躲避算法：检测危险子弹并垂直躲避
 * - 预测射击：在理想射程内对齐射击
 * - 防卡机制：检测移动停滞并随机换向
 */

#include "game.h"
#include "enemy_ai.h"
#include "ai_interface.h"  // 获取难度级别
#include <math.h>
#include <stdlib.h>

// ========== 前向声明 ==========
static bool will_hit_wall(Tank* tank, Direction dir, int map_width, int map_height);

// ========== AI参数配置 ==========

// ✅ 玩家对战模式优化：提升AI智商
#define ACTION_INTERVAL 2      // 每2帧更新一次决策（更快反应）
#define IDEAL_MIN_DIST 100.0f  // 理想最小距离（激进近战）
#define IDEAL_MAX_DIST 180.0f  // 理想最大距离（保持压迫感）
#define DANGER_DIST 100.0f     // 子弹危险距离阈值（更早躲避）
#define SHOOT_RANGE 300.0f     // 射击范围（远距离射击）
// 注意：BULLET_SPEED已在bullet.h中定义为5.0f

// 初始化敌人AI
void enemy_ai_init(EnemyAI* ai, Tank* target) {
    ai->target = target;
    ai->last_action_time = 0;
    ai->last_move_dir = DIR_UP;
    ai->stuck_counter = 0;
    ai->last_x = 0;
    ai->last_y = 0;
    ai->dodge_cooldown = 0;
    ai->dodge_dir = DIR_UP;
}

// 检测是否卡住
static bool is_stuck(EnemyAI* ai, Tank* tank) {
    float dx = tank->x - ai->last_x;
    float dy = tank->y - ai->last_y;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < 0.5f) {
        ai->stuck_counter++;
    } else {
        ai->stuck_counter = 0;
    }

    ai->last_x = tank->x;
    ai->last_y = tank->y;

    return ai->stuck_counter > 20;
}

/*
 * 检测危险子弹并计算躲避方向
 *
 * 躲避算法设计：
 * 1. 检测所有AI发射的子弹
 * 2. 计算距离，判断是否在危险范围（<80像素）
 * 3. 根据子弹飞行方向，选择垂直躲避
 *
 * 为什么垂直躲避？
 * - 如果子弹横向飞行（左右），向上/下躲避最快离开弹道
 * - 如果子弹纵向飞行（上下），向左/右躲避最快离开弹道
 * - 垂直方向比沿着弹道方向逃跑更有效（最短路径）
 *
 * 参数：
 * @param enemy_tank   - 敌人坦克（需要躲避的坦克）
 * @param bullets      - 所有子弹数组
 * @param bullet_count - 子弹数量
 * @param dodge_dir    - 输出参数：推荐的躲避方向
 * @return true=检测到危险子弹, false=安全
 */
static bool detect_danger_bullet(Tank* enemy_tank, Bullet* bullets, int bullet_count,
                                 Direction* dodge_dir, int map_width, int map_height) {
    for (int i = 0; i < bullet_count; i++) {
        if (!bullets[i].active) continue;

        // 只关心对手坦克的子弹（AI坦克、玩家坦克的子弹都需要躲避）
        // 玩家模式：躲避 TANK_TYPE_PLAYER 的子弹
        // 训练模式：躲避 TANK_TYPE_AI 的子弹
        if (bullets[i].owner_type != TANK_TYPE_AI &&
            bullets[i].owner_type != TANK_TYPE_PLAYER &&
            bullets[i].owner_type != TANK_TYPE_PLAYER2) continue;

        // 计算子弹到坦克的距离
        float dx = bullets[i].x - enemy_tank->x;  // x方向距离
        float dy = bullets[i].y - enemy_tank->y;  // y方向距离
        float dist = sqrtf(dx * dx + dy * dy);    // 欧几里得距离

        // 危险判定：子弹在危险范围内（<80像素）
        if (dist < DANGER_DIST) {
            // 获取子弹飞行速度向量
            float bullet_vx = bullets[i].vx;
            float bullet_vy = bullets[i].vy;

            // 选择躲避方向：垂直于子弹主要飞行方向，同时检查墙壁
            Direction primary, alternative;

            if (fabs(bullet_vx) > fabs(bullet_vy)) {
                // 情况1：子弹主要横向飞行（左右）
                // 策略：向上或向下躲避（垂直于弹道）
                primary = (dy > 0) ? DIR_DOWN : DIR_UP;
                alternative = (dy > 0) ? DIR_UP : DIR_DOWN;  // 反向作为备选
            } else {
                // 情况2：子弹主要纵向飞行（上下）
                // 策略：向左或向右躲避（垂直于弹道）
                primary = (dx > 0) ? DIR_RIGHT : DIR_LEFT;
                alternative = (dx > 0) ? DIR_LEFT : DIR_RIGHT;  // 反向作为备选
            }

            // 墙壁检查：优先使用主方向，如果会撞墙则使用备选方向
            if (!will_hit_wall(enemy_tank, primary, map_width, map_height)) {
                *dodge_dir = primary;
            } else if (!will_hit_wall(enemy_tank, alternative, map_width, map_height)) {
                *dodge_dir = alternative;
            } else {
                // 两个方向都会撞墙，选择一个安全方向（任意非墙方向）
                Direction safe_dirs[] = {DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};
                for (int j = 0; j < 4; j++) {
                    if (!will_hit_wall(enemy_tank, safe_dirs[j], map_width, map_height)) {
                        *dodge_dir = safe_dirs[j];
                        return true;
                    }
                }
                // 极端情况：所有方向都是墙（不应该发生）
                return false;
            }
            return true;
        }
    }
    return false;  // 没有检测到危险子弹
}

/*
 * 尝试射击目标（智能射击判定 v2.0 - 带预判）
 *
 * 【核心升级】增加预判射击功能
 * 1. 基础射击：目标静止时直接对齐射击
 * 2. 预判射击：目标移动时预测其位置，提前射击
 *
 * 射击算法设计：
 * 1. 检查射击条件（目标存活、冷却完成）
 * 2. 计算预测位置（根据目标速度）
 * 3. 判断是否在理想射程（100-300）
 * 4. 检查是否在射击线上（水平或垂直对齐）
 * 5. 如果满足条件，调整坦克方向并返回true
 *
 * 预判射击原理：
 * - 子弹飞行时间 = 距离 / 子弹速度
 * - 预测位置 = 当前位置 + 速度 × 飞行时间
 * - 这样可以命中移动中的目标
 *
 * @return true=可以射击并已调整方向, false=不满足射击条件
 */
bool enemy_ai_try_shoot(EnemyAI* ai, Tank* enemy_tank, Tank* target) {
    // 前置条件检查
    if (!target || !target->alive) return false;      // 目标不存在或已死亡
    if (!tank_can_shoot(enemy_tank)) return false;    // 射击冷却中

    // 计算到目标的相对位置
    float dx = target->x - enemy_tank->x;
    float dy = target->y - enemy_tank->y;
    float dist = sqrtf(dx * dx + dy * dy);

    // 射程检查
    if (dist < IDEAL_MIN_DIST || dist > SHOOT_RANGE) {
        return false;
    }

    // ========== 预判射击计算 ==========
    // 计算子弹飞行时间（帧数）
    float bullet_travel_time = dist / BULLET_SPEED;

    // 预测目标位置（考虑目标移动）
    float predicted_dx = dx + target->vx * bullet_travel_time;
    float predicted_dy = dy + target->vy * bullet_travel_time;

    // 根据目标是否移动选择使用哪个位置
    bool target_moving = (fabsf(target->vx) > 0.1f || fabsf(target->vy) > 0.1f);

    // 预判因子：移动目标使用预测位置，静止目标使用当前位置
    float aim_dx = target_moving ? (dx * 0.3f + predicted_dx * 0.7f) : dx;
    float aim_dy = target_moving ? (dy * 0.3f + predicted_dy * 0.7f) : dy;

    // ========== 射击判定 ==========
    // 动态容忍度：距离越近要求越精确，距离越远容忍度越大
    float base_tolerance = 50.0f;
    float tolerance = base_tolerance + (dist / 10.0f);  // 最大约80像素
    if (tolerance > 80.0f) tolerance = 80.0f;

    // 情况1：水平对齐（可以左右射击）
    if (fabsf(aim_dy) < tolerance && fabsf(aim_dx) > 30.0f) {
        Direction desired_dir = (aim_dx > 0) ? DIR_RIGHT : DIR_LEFT;

        if (enemy_tank->direction != desired_dir) {
            enemy_tank->direction = desired_dir;
        }
        return true;
    }
    // 情况2：垂直对齐（可以上下射击）
    else if (fabsf(aim_dx) < tolerance && fabsf(aim_dy) > 30.0f) {
        Direction desired_dir = (aim_dy > 0) ? DIR_DOWN : DIR_UP;

        if (enemy_tank->direction != desired_dir) {
            enemy_tank->direction = desired_dir;
        }
        return true;
    }

    // ========== 主动对齐射击 ==========
    // 如果距离合适但未对齐，尝试主动调整方向射击
    if (dist < 200.0f && dist > 80.0f) {
        // 选择偏差较小的轴进行射击
        if (fabsf(aim_dx) < fabsf(aim_dy) && fabsf(aim_dx) < tolerance * 1.5f) {
            // x偏差小，尝试垂直射击
            Direction desired_dir = (aim_dy > 0) ? DIR_DOWN : DIR_UP;
            enemy_tank->direction = desired_dir;
            return true;
        } else if (fabsf(aim_dy) < fabsf(aim_dx) && fabsf(aim_dy) < tolerance * 1.5f) {
            // y偏差小，尝试水平射击
            Direction desired_dir = (aim_dx > 0) ? DIR_RIGHT : DIR_LEFT;
            enemy_tank->direction = desired_dir;
            return true;
        }
    }

    return false;
}

/*
 * 检查某个方向是否会撞墙（改进版：动态预测）
 *
 * @param tank       - 坦克对象
 * @param dir        - 要检查的方向
 * @param map_width  - 地图宽度
 * @param map_height - 地图高度
 * @return true=会撞墙, false=安全
 *
 * 改进逻辑：
 * - 预测移动几帧后的位置
 * - 检查是否会超出安全边界
 * - 即使当前未贴墙，如果朝墙移动也返回true
 */
static bool will_hit_wall(Tank* tank, Direction dir, int map_width, int map_height) {
    const float WALL_MARGIN = 50.0f;  // 墙壁安全距离（增加到50像素）
    const float PREDICT_FRAMES = 5.0f;  // 预测5帧后的位置
    const float PREDICT_DIST = TANK_SPEED * PREDICT_FRAMES;  // 预测移动距离

    // 计算预测位置
    float next_x = tank->x;
    float next_y = tank->y;

    switch (dir) {
        case DIR_UP:
            next_y -= PREDICT_DIST;
            return next_y < WALL_MARGIN;  // 上边界
        case DIR_DOWN:
            next_y += PREDICT_DIST;
            return next_y + tank->height > map_height - WALL_MARGIN;  // 下边界
        case DIR_LEFT:
            next_x -= PREDICT_DIST;
            return next_x < WALL_MARGIN;  // 左边界
        case DIR_RIGHT:
            next_x += PREDICT_DIST;
            return next_x + tank->width > map_width - WALL_MARGIN;  // 右边界
    }
    return false;
}

/*
 * 计算战术定位方向（保持理想战斗距离）
 *
 * 战术定位算法：
 * 这是一个距离控制算法，让AI保持在最有利的战斗位置
 *
 * 策略分层：
 * 1. 距离<150：太近危险 → 后退（远离目标）
 * 2. 距离>250：太远无效 → 前进（靠近目标）
 * 3. 距离150-250：理想距离 → 微调对齐
 *
 * 移动方向选择：
 * - 比较dx和dy的绝对值，优先沿着距离更大的轴移动
 * - 这样可以最快接近/远离目标
 *
 * 防抖设计：
 * - 只有偏差>80像素时才调整位置
 * - 避免在理想位置附近频繁抖动
 * - 如果已经对准较好，保持当前移动方向
 *
 * 墙壁避让：
 * - 检查选择的方向是否会撞墙
 * - 如果会撞墙，选择替代方向（垂直方向）
 *
 * @param enemy_tank - 敌人坦克
 * @param target     - 目标坦克（通常是DQN训练的AI）
 * @param map_width  - 地图宽度
 * @param map_height - 地图高度
 * @return 推荐的移动方向
 */
static Direction get_positioning_direction(Tank* enemy_tank, Tank* target,
                                           int map_width, int map_height) {
    // 计算到目标的相对位置和距离
    float dx = target->x - enemy_tank->x;
    float dy = target->y - enemy_tank->y;
    float dist = sqrtf(dx * dx + dy * dy);

    Direction primary_dir, alternative_dir;

    // 策略1：距离太近（<150），后退保持安全距离
    if (dist < IDEAL_MIN_DIST) {
        // 选择主要轴向后退
        if (fabs(dx) > fabs(dy)) {
            // x轴偏差更大，左右后退
            primary_dir = dx > 0 ? DIR_LEFT : DIR_RIGHT;
            alternative_dir = dy > 0 ? DIR_UP : DIR_DOWN;  // 备选：垂直方向
        } else {
            // y轴偏差更大，上下后退
            primary_dir = dy > 0 ? DIR_UP : DIR_DOWN;
            alternative_dir = dx > 0 ? DIR_LEFT : DIR_RIGHT;  // 备选：水平方向
        }

        // 墙壁检查：如果主方向会撞墙，使用备选方向
        if (!will_hit_wall(enemy_tank, primary_dir, map_width, map_height)) {
            return primary_dir;
        } else if (!will_hit_wall(enemy_tank, alternative_dir, map_width, map_height)) {
            return alternative_dir;
        }
        // 两个方向都撞墙，返回反向
        return (primary_dir == DIR_UP) ? DIR_DOWN :
               (primary_dir == DIR_DOWN) ? DIR_UP :
               (primary_dir == DIR_LEFT) ? DIR_RIGHT : DIR_LEFT;
    }

    // 策略2：距离太远（>250），前进靠近目标
    if (dist > IDEAL_MAX_DIST) {
        // 选择主要轴向前进
        if (fabs(dx) > fabs(dy)) {
            // x轴偏差更大，左右靠近
            primary_dir = dx > 0 ? DIR_RIGHT : DIR_LEFT;
            alternative_dir = dy > 0 ? DIR_DOWN : DIR_UP;
        } else {
            // y轴偏差更大，上下靠近
            primary_dir = dy > 0 ? DIR_DOWN : DIR_UP;
            alternative_dir = dx > 0 ? DIR_RIGHT : DIR_LEFT;
        }

        // 墙壁检查
        if (!will_hit_wall(enemy_tank, primary_dir, map_width, map_height)) {
            return primary_dir;
        } else if (!will_hit_wall(enemy_tank, alternative_dir, map_width, map_height)) {
            return alternative_dir;
        }
        // 两个方向都撞墙，返回反向
        return (primary_dir == DIR_UP) ? DIR_DOWN :
               (primary_dir == DIR_DOWN) ? DIR_UP :
               (primary_dir == DIR_LEFT) ? DIR_RIGHT : DIR_LEFT;
    }

    // 策略3：距离合适（150-250），微调对齐以便射击
    // 如果水平偏差较大（>80），优先对齐水平
    if (fabs(dx) > 80.0f) {
        primary_dir = dx > 0 ? DIR_RIGHT : DIR_LEFT;
        alternative_dir = dy > 0 ? DIR_DOWN : DIR_UP;

        if (!will_hit_wall(enemy_tank, primary_dir, map_width, map_height)) {
            return primary_dir;
        } else if (!will_hit_wall(enemy_tank, alternative_dir, map_width, map_height)) {
            return alternative_dir;
        }
    }
    // 如果垂直偏差较大（>80），优先对齐垂直
    else if (fabs(dy) > 80.0f) {
        primary_dir = dy > 0 ? DIR_DOWN : DIR_UP;
        alternative_dir = dx > 0 ? DIR_RIGHT : DIR_LEFT;

        if (!will_hit_wall(enemy_tank, primary_dir, map_width, map_height)) {
            return primary_dir;
        } else if (!will_hit_wall(enemy_tank, alternative_dir, map_width, map_height)) {
            return alternative_dir;
        }
    }

    // 策略4：已经在理想位置且对齐良好
    // 保持当前移动方向，但要避开墙壁
    if (!will_hit_wall(enemy_tank, enemy_tank->direction, map_width, map_height)) {
        return enemy_tank->direction;
    }

    // 当前方向会撞墙，选择一个安全方向
    Direction safe_dirs[] = {DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};
    for (int i = 0; i < 4; i++) {
        if (!will_hit_wall(enemy_tank, safe_dirs[i], map_width, map_height)) {
            return safe_dirs[i];
        }
    }

    // 极端情况：所有方向都撞墙（理论上不应该发生）
    return enemy_tank->direction;
}

/*
 * 更新敌人AI（主决策函数）
 *
 * 【核心决策流程】
 * 这是敌人AI的大脑，每帧都会被调用来决定下一步行动
 *
 * 决策优先级（从高到低）：
 * ┌────────────────────────────────────────────────────────────┐
 * │ 优先级 | 条件           | 行为         | 原因             │
 * ├────────────────────────────────────────────────────────────┤
 * │ 0      | 坦克死亡       | IDLE         | 死亡无法行动     │
 * │ 1      | 检测到危险子弹 | 躲避移动     | 生存第一         │
 * │ 2      | 在理想射程内   | 射击         | 攻击是目标       │
 * │ 3      | 其他情况       | 战术定位     | 调整到有利位置   │
 * │ 4      | 卡住检测       | 随机换向     | 兜底机制         │
 * └────────────────────────────────────────────────────────────┘
 *
 * 性能优化：
 * - 每5帧更新一次决策（ACTION_INTERVAL），减少计算开销
 * - 躲避状态持续10帧，避免频繁切换
 * - 其他时候沿用上次决策，保证行为连贯性
 *
 * 参数：
 * @param ai           - AI状态对象
 * @param enemy_tank   - 敌人坦克对象
 * @param tanks        - 所有坦克数组
 * @param tank_count   - 坦克数量
 * @param bullets      - 所有子弹数组
 * @param bullet_count - 子弹数量
 * @param current_frame- 当前帧数（用于时间判断）
 * @return 推荐的坦克动作
 */
TankAction enemy_ai_update(EnemyAI* ai, Tank* enemy_tank, Tank* tanks,
                           int tank_count, Bullet* bullets, int bullet_count,
                           int current_frame, int map_width, int map_height) {
    // 优先级0：死亡检测
    if (!enemy_tank->alive) return ACTION_IDLE;

    // ========== 课程学习：根据难度级别调整行为 ==========
    int difficulty = ai_get_difficulty();

    // ========== 5级难度系统（更平滑的课程学习 v6.3）==========
    //
    // 难度0：假人模式 - 静止不动，1点血量（一击必杀）
    // 难度1：慢移动 - 缓慢随机移动，不射击，1点血量
    // 难度2：正常移动 - 正常速度移动，不射击，2点血量
    // 难度3：移动+射击 - 随机移动+偶尔射击，3点血量
    // 难度4：完整AI - 预判射击+躲避+定位（最终挑战）

    // 难度0-2：降低敌人血量，让AI更容易获得击杀奖励
    if (difficulty <= 2 && enemy_tank->health > 1) {
        // 难度0-1：强制1点血量（一击必杀）
        if (difficulty <= 1) {
            enemy_tank->health = 1;
        }
        // 难度2：最多2点血量
        else if (difficulty == 2 && enemy_tank->health > 2) {
            enemy_tank->health = 2;
        }
    }

    // 难度0：假人模式 - 静止不动
    if (difficulty == 0) {
        return ACTION_IDLE;
    }

    // 难度1：慢移动 - 只移动不射击（过渡难度）
    if (difficulty == 1) {
        // 每20帧决策一次（非常慢，更容易追踪）
        if (current_frame - ai->last_action_time < 20) {
            return ACTION_IDLE;
        }
        ai->last_action_time = current_frame;

        // 只随机移动，不射击
        int move_dir = rand() % 4;
        return ACTION_MOVE_UP + move_dir;
    }

    // 难度2：正常移动 - 正常速度移动，不射击
    if (difficulty == 2) {
        // 每10帧决策一次
        if (current_frame - ai->last_action_time < 10) {
            return ACTION_IDLE;
        }
        ai->last_action_time = current_frame;

        // 只移动，不射击
        int move_dir = rand() % 4;
        return ACTION_MOVE_UP + move_dir;
    }

    // 难度3：移动+射击 - 随机移动，偶尔射击
    if (difficulty == 3) {
        // 每8帧决策一次
        if (current_frame - ai->last_action_time < 8) {
            return ACTION_IDLE;
        }
        ai->last_action_time = current_frame;

        // 15%概率射击，85%概率随机移动
        if (rand() % 100 < 15) {
            int shoot_dir = rand() % 4;
            return ACTION_SHOOT_UP + shoot_dir;
        } else {
            int move_dir = rand() % 4;
            return ACTION_MOVE_UP + move_dir;
        }
    }

    // 难度4+：完整AI - 继续执行下面的完整AI逻辑

    // 防卡机制：检测是否长时间移动距离过小
    if (is_stuck(ai, enemy_tank)) {
        // 选择一个不会撞墙的随机方向，强制打破卡住状态
        Direction safe_dirs[4] = {DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};
        Direction candidate_dirs[4];
        int safe_count = 0;

        // 收集所有不会撞墙的方向
        for (int i = 0; i < 4; i++) {
            if (!will_hit_wall(enemy_tank, safe_dirs[i], map_width, map_height)) {
                candidate_dirs[safe_count++] = safe_dirs[i];
            }
        }

        // 从安全方向中随机选择
        if (safe_count > 0) {
            ai->last_move_dir = candidate_dirs[rand() % safe_count];
        }
        ai->stuck_counter = 0;  // 重置卡住计数器
    }

    // 性能优化：不是每帧都重新决策
    // 只有当距离上次决策超过ACTION_INTERVAL（5帧）时才重新思考
    if (current_frame - ai->last_action_time < ACTION_INTERVAL) {
        Direction dir_to_use;

        // 如果正在躲避状态，使用躲避方向
        if (ai->dodge_cooldown > 0) {
            ai->dodge_cooldown--;  // 躲避冷却倒计时
            dir_to_use = ai->dodge_dir;
        } else {
            // 否则，使用上次的移动方向
            dir_to_use = ai->last_move_dir;
        }

        // ✅ 关键修复：即使在非决策帧，也要检查方向是否会撞墙
        if (will_hit_wall(enemy_tank, dir_to_use, map_width, map_height)) {
            // 如果会撞墙，立即找一个安全方向
            Direction safe_dirs[] = {DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT};
            for (int i = 0; i < 4; i++) {
                if (!will_hit_wall(enemy_tank, safe_dirs[i], map_width, map_height)) {
                    dir_to_use = safe_dirs[i];
                    break;
                }
            }
        }

        // 返回安全的移动动作
        switch (dir_to_use) {
            case DIR_UP: return ACTION_MOVE_UP;
            case DIR_DOWN: return ACTION_MOVE_DOWN;
            case DIR_LEFT: return ACTION_MOVE_LEFT;
            case DIR_RIGHT: return ACTION_MOVE_RIGHT;
        }
    }

    // 更新决策时间戳
    ai->last_action_time = current_frame;

    // 检查目标是否有效
    // 注意：ai->target 已经由 enemy_ai_get_action() 或 enemy_ai_init() 设置
    // 在玩家模式下指向 TANK_TYPE_PLAYER，在训练模式下指向 TANK_TYPE_AI
    if (!ai->target || !ai->target->alive) {
        return ACTION_IDLE;  // 目标无效或已死亡
    }

    // ========== 决策树开始 ==========

    // 优先级1：检测并躲避危险子弹（仅困难模式）
    if (difficulty >= 3) {
        Direction dodge_direction;
        if (detect_danger_bullet(enemy_tank, bullets, bullet_count, &dodge_direction,
                                map_width, map_height)) {
            // 检测到危险！立即进入躲避状态
            ai->dodge_dir = dodge_direction;      // 记录躲避方向
            ai->dodge_cooldown = 10;              // 设置躲避持续时间（10帧）
            ai->last_move_dir = dodge_direction;  // 更新上次移动方向

            // 立即执行躲避动作
            switch (dodge_direction) {
                case DIR_UP: return ACTION_MOVE_UP;
                case DIR_DOWN: return ACTION_MOVE_DOWN;
                case DIR_LEFT: return ACTION_MOVE_LEFT;
                case DIR_RIGHT: return ACTION_MOVE_RIGHT;
            }
        }
    }

    // 优先级2：尝试射击
    // 困难模式(3)：预判射击  中等模式(2)：简单射击
    if (difficulty >= 3) {
        // 困难模式：使用预判射击
        if (enemy_ai_try_shoot(ai, enemy_tank, ai->target)) {
            switch (enemy_tank->direction) {
                case DIR_UP: return ACTION_SHOOT_UP;
                case DIR_DOWN: return ACTION_SHOOT_DOWN;
                case DIR_LEFT: return ACTION_SHOOT_LEFT;
                case DIR_RIGHT: return ACTION_SHOOT_RIGHT;
            }
        }
    } else if (difficulty == 2) {
        // 中等模式：简单射击（对准就射，不预判）
        float dx = ai->target->x - enemy_tank->x;
        float dy = ai->target->y - enemy_tank->y;
        float dist = sqrtf(dx*dx + dy*dy);

        // 在射程内且冷却完成
        if (dist < SHOOT_RANGE && enemy_tank->shoot_cooldown == 0) {
            // 简单对齐检查（不预判）
            if (fabsf(dx) < 40.0f) {
                // 垂直对齐，上下射击
                if (dy > 0) return ACTION_SHOOT_DOWN;
                else return ACTION_SHOOT_UP;
            } else if (fabsf(dy) < 40.0f) {
                // 水平对齐，左右射击
                if (dx > 0) return ACTION_SHOOT_RIGHT;
                else return ACTION_SHOOT_LEFT;
            }
        }
    }

    // 优先级3：战术定位（追踪目标）
    // 中等和困难模式都使用定位，但困难模式更精确
    Direction best_dir = get_positioning_direction(enemy_tank, ai->target, map_width, map_height);
    ai->last_move_dir = best_dir;

    switch (best_dir) {
        case DIR_UP: return ACTION_MOVE_UP;
        case DIR_DOWN: return ACTION_MOVE_DOWN;
        case DIR_LEFT: return ACTION_MOVE_LEFT;
        case DIR_RIGHT: return ACTION_MOVE_RIGHT;
    }

    return ACTION_IDLE;
}

/*
 * 获取追踪AI的动作（简化版，用于玩家对战模式）
 *
 * 功能：简化enemy_ai_update的调用，自动设置参数
 *
 * 参数：
 *   game - 游戏状态指针
 *   tank_index - 敌人坦克索引
 *
 * 返回：
 *   决策的动作
 *
 * 实现：
 *   - 查找玩家坦克作为追踪目标
 *   - 创建临时AI状态（无状态版本）
 *   - 调用完整的enemy_ai_update
 */
TankAction enemy_ai_get_action(GameState* game, int tank_index) {
    // 获取敌人坦克
    Tank* enemy_tank = &game->tanks[tank_index];

    // 查找玩家坦克作为目标
    Tank* player_tank = NULL;
    for (int i = 0; i < game->tank_count; i++) {
        if (game->tanks[i].type == TANK_TYPE_PLAYER && game->tanks[i].alive) {
            player_tank = &game->tanks[i];
            break;
        }
    }

    // 如果没有玩家，就待机
    if (!player_tank) {
        return ACTION_IDLE;
    }

    // 创建临时AI状态（简化版，每次重新初始化）
    static EnemyAI ai_states[32] = {0};  // 静态存储，保持状态

    // 初始化或更新AI状态
    if (ai_states[tank_index].target == NULL) {
        enemy_ai_init(&ai_states[tank_index], player_tank);
    } else {
        ai_states[tank_index].target = player_tank;  // 更新目标
    }

    // 调用完整的AI更新函数
    return enemy_ai_update(
        &ai_states[tank_index],
        enemy_tank,
        game->tanks,
        game->tank_count,
        game->bullets,
        MAX_BULLETS,  // 子弹数组的最大大小
        game->frame_count,  // 使用游戏的帧计数
        game->map_width,    // 地图宽度（用于墙壁检测）
        game->map_height    // 地图高度（用于墙壁检测）
    );
}
