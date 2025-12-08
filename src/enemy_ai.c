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

#include "enemy_ai.h"
#include <math.h>
#include <stdlib.h>

// ========== AI参数配置 ==========

#define ACTION_INTERVAL 5      // 每5帧更新一次决策（避免抖动）
#define IDEAL_MIN_DIST 150.0f  // 理想最小距离（太近会被打）
#define IDEAL_MAX_DIST 250.0f  // 理想最大距离（太远打不准）
#define DANGER_DIST 80.0f      // 子弹危险距离阈值

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
static bool detect_danger_bullet(Tank* enemy_tank, Bullet* bullets, int bullet_count, Direction* dodge_dir) {
    for (int i = 0; i < bullet_count; i++) {
        if (!bullets[i].active) continue;

        // 只关心AI坦克的子弹（训练对手的子弹才需要躲避）
        if (bullets[i].owner_type != TANK_TYPE_AI) continue;

        // 计算子弹到坦克的距离
        float dx = bullets[i].x - enemy_tank->x;  // x方向距离
        float dy = bullets[i].y - enemy_tank->y;  // y方向距离
        float dist = sqrtf(dx * dx + dy * dy);    // 欧几里得距离

        // 危险判定：子弹在危险范围内（<80像素）
        if (dist < DANGER_DIST) {
            // 获取子弹飞行速度向量
            float bullet_vx = bullets[i].vx;
            float bullet_vy = bullets[i].vy;

            // 选择躲避方向：垂直于子弹主要飞行方向
            if (fabs(bullet_vx) > fabs(bullet_vy)) {
                // 情况1：子弹主要横向飞行（左右）
                // 策略：向上或向下躲避（垂直于弹道）
                // 判断：如果子弹在下方(dy>0)就继续向下躲，否则向上
                *dodge_dir = (dy > 0) ? DIR_DOWN : DIR_UP;
                return true;
            } else {
                // 情况2：子弹主要纵向飞行（上下）
                // 策略：向左或向右躲避（垂直于弹道）
                // 判断：如果子弹在右方(dx>0)就继续向右躲，否则向左
                *dodge_dir = (dx > 0) ? DIR_RIGHT : DIR_LEFT;
                return true;
            }
        }
    }
    return false;  // 没有检测到危险子弹
}

/*
 * 尝试射击目标（智能射击判定）
 *
 * 射击算法设计：
 * 1. 检查射击条件（目标存活、冷却完成）
 * 2. 计算到目标的距离，判断是否在理想射程（150-350）
 * 3. 检查是否在射击线上（水平或垂直对齐，容忍度40）
 * 4. 如果满足条件，调整坦克方向并返回true
 *
 * 为什么需要容忍度？
 * - 完美对齐（dx=0或dy=0）几乎不可能
 * - 40像素容忍度允许"差不多对准"时就射击
 * - 避免AI反复微调方向（抖动）
 *
 * 为什么只在方向不对时才转向？
 * - 减少不必要的转向动作
 * - 如果已经朝着正确方向，直接射击
 * - 提高AI响应速度
 *
 * 射程设计：
 * - 最小距离150：太近容易被反击
 * - 最大距离350：太远命中率低
 * - 理想距离200左右：平衡安全和命中率
 *
 * @return true=可以射击并已调整方向, false=不满足射击条件
 */
bool enemy_ai_try_shoot(EnemyAI* ai, Tank* enemy_tank, Tank* target) {
    // 前置条件检查
    if (!target || !target->alive) return false;      // 目标不存在或已死亡
    if (!tank_can_shoot(enemy_tank)) return false;    // 射击冷却中

    // 计算到目标的相对位置
    float dx = target->x - enemy_tank->x;  // x方向距离（正=目标在右，负=在左）
    float dy = target->y - enemy_tank->y;  // y方向距离（正=目标在下，负=在上）
    float dist = sqrtf(dx * dx + dy * dy); // 直线距离

    // 理想射程判定：150-350像素
    if (dist >= IDEAL_MIN_DIST && dist <= IDEAL_MAX_DIST + 100.0f) {
        // 对齐容忍度：允许40像素的偏差
        float tolerance = 40.0f;

        // 情况1：水平对齐（上下偏差小，可以左右射击）
        if (fabs(dy) < tolerance && fabs(dx) > 40.0f) {
            // 确定应该朝哪个方向射击
            Direction desired_dir = (dx > 0) ? DIR_RIGHT : DIR_LEFT;

            // 只有在当前方向不对时才转向（避免重复转向）
            if (enemy_tank->direction != desired_dir) {
                enemy_tank->direction = desired_dir;
            }
            return true;  // 可以射击
        }
        // 情况2：垂直对齐（左右偏差小，可以上下射击）
        else if (fabs(dx) < tolerance && fabs(dy) > 40.0f) {
            // 确定应该朝哪个方向射击
            Direction desired_dir = (dy > 0) ? DIR_DOWN : DIR_UP;

            // 只有在当前方向不对时才转向
            if (enemy_tank->direction != desired_dir) {
                enemy_tank->direction = desired_dir;
            }
            return true;  // 可以射击
        }
    }

    return false;  // 不满足射击条件（距离不合适或未对齐）
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
 * @param enemy_tank - 敌人坦克
 * @param target     - 目标坦克（通常是DQN训练的AI）
 * @return 推荐的移动方向
 */
static Direction get_positioning_direction(Tank* enemy_tank, Tank* target) {
    // 计算到目标的相对位置和距离
    float dx = target->x - enemy_tank->x;
    float dy = target->y - enemy_tank->y;
    float dist = sqrtf(dx * dx + dy * dy);

    // 策略1：距离太近（<150），后退保持安全距离
    if (dist < IDEAL_MIN_DIST) {
        // 选择主要轴向后退
        if (fabs(dx) > fabs(dy)) {
            // x轴偏差更大，左右后退
            return dx > 0 ? DIR_LEFT : DIR_RIGHT;  // 目标在右→往左退，目标在左→往右退
        } else {
            // y轴偏差更大，上下后退
            return dy > 0 ? DIR_UP : DIR_DOWN;     // 目标在下→往上退，目标在上→往下退
        }
    }

    // 策略2：距离太远（>250），前进靠近目标
    if (dist > IDEAL_MAX_DIST) {
        // 选择主要轴向前进
        if (fabs(dx) > fabs(dy)) {
            // x轴偏差更大，左右靠近
            return dx > 0 ? DIR_RIGHT : DIR_LEFT;  // 目标在右→往右，目标在左→往左
        } else {
            // y轴偏差更大，上下靠近
            return dy > 0 ? DIR_DOWN : DIR_UP;     // 目标在下→往下，目标在上→往上
        }
    }

    // 策略3：距离合适（150-250），微调对齐以便射击
    // 目标：让dx或dy接近0，形成水平/垂直射击线

    // 如果水平偏差较大（>80），优先对齐水平
    if (fabs(dx) > 80.0f) {
        return dx > 0 ? DIR_RIGHT : DIR_LEFT;
    }
    // 如果垂直偏差较大（>80），优先对齐垂直
    else if (fabs(dy) > 80.0f) {
        return dy > 0 ? DIR_DOWN : DIR_UP;
    }

    // 策略4：已经在理想位置且对齐良好
    // 保持当前移动方向，避免静止（移动目标更难打）
    // 不进行随机转向，保证行为一致性
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
                           int current_frame) {
    // 优先级0：死亡检测
    if (!enemy_tank->alive) return ACTION_IDLE;

    // 防卡机制：检测是否长时间移动距离过小
    if (is_stuck(ai, enemy_tank)) {
        // 随机选择一个方向，强制打破卡住状态
        int random_dir = rand() % 4;
        ai->last_move_dir = (Direction)random_dir;
        ai->stuck_counter = 0;  // 重置卡住计数器
    }

    // 性能优化：不是每帧都重新决策
    // 只有当距离上次决策超过ACTION_INTERVAL（5帧）时才重新思考
    if (current_frame - ai->last_action_time < ACTION_INTERVAL) {
        // 如果正在躲避状态，继续躲避
        if (ai->dodge_cooldown > 0) {
            ai->dodge_cooldown--;  // 躲避冷却倒计时
            // 继续沿着躲避方向移动
            switch (ai->dodge_dir) {
                case DIR_UP: return ACTION_MOVE_UP;
                case DIR_DOWN: return ACTION_MOVE_DOWN;
                case DIR_LEFT: return ACTION_MOVE_LEFT;
                case DIR_RIGHT: return ACTION_MOVE_RIGHT;
            }
        }
        // 否则，继续上次的移动动作（保持行为连贯性）
        switch (ai->last_move_dir) {
            case DIR_UP: return ACTION_MOVE_UP;
            case DIR_DOWN: return ACTION_MOVE_DOWN;
            case DIR_LEFT: return ACTION_MOVE_LEFT;
            case DIR_RIGHT: return ACTION_MOVE_RIGHT;
        }
    }

    // 更新决策时间戳
    ai->last_action_time = current_frame;

    // 寻找目标：找到训练中的AI坦克
    Tank* ai_tank = NULL;
    for (int i = 0; i < tank_count; i++) {
        if (tanks[i].alive && tanks[i].type == TANK_TYPE_AI) {
            ai_tank = &tanks[i];
            break;
        }
    }

    // 如果没有找到目标，无事可做
    if (!ai_tank) return ACTION_IDLE;
    ai->target = ai_tank;  // 更新目标指针

    // ========== 决策树开始 ==========

    // 优先级1：检测并躲避危险子弹（生存优先）
    Direction dodge_direction;
    if (detect_danger_bullet(enemy_tank, bullets, bullet_count, &dodge_direction)) {
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

    // 优先级2：尝试射击（攻击优先，但不优先于生存）
    if (enemy_ai_try_shoot(ai, enemy_tank, ai_tank)) {
        // 满足射击条件（距离合适且对齐），执行射击
        switch (enemy_tank->direction) {
            case DIR_UP: return ACTION_SHOOT_UP;
            case DIR_DOWN: return ACTION_SHOOT_DOWN;
            case DIR_LEFT: return ACTION_SHOOT_LEFT;
            case DIR_RIGHT: return ACTION_SHOOT_RIGHT;
        }
    }

    // 优先级3：战术定位（保持理想战斗距离）
    Direction best_dir = get_positioning_direction(enemy_tank, ai_tank);
    ai->last_move_dir = best_dir;  // 记录这次的移动方向

    // 执行定位移动
    switch (best_dir) {
        case DIR_UP: return ACTION_MOVE_UP;
        case DIR_DOWN: return ACTION_MOVE_DOWN;
        case DIR_LEFT: return ACTION_MOVE_LEFT;
        case DIR_RIGHT: return ACTION_MOVE_RIGHT;
    }

    // 兜底：理论上不应该到这里
    return ACTION_IDLE;
}
