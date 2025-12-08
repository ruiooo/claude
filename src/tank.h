/*
 * tank.h - 坦克实体定义和相关函数
 *
 * 模块说明：
 *   定义坦克实体的数据结构、类型、动作和方向枚举
 *   提供坦克的初始化、更新、移动、受伤、射击等核心功能
 *   支持多种坦克类型：AI训练、传统敌人、历史版本AI、玩家控制
 *
 * 使用场景：
 *   - 游戏核心逻辑：坦克的基础行为和状态管理
 *   - AI训练：DQN模型控制的坦克实体
 *   - 玩家对战：玩家控制的坦克实体
 *   - 自我对弈：历史版本模型控制的敌人坦克
 */

#ifndef TANK_H
#define TANK_H

#include <stdbool.h>

/*
 * 坦克类型枚举
 * 用于区分不同的坦克角色和行为模式
 */
typedef enum {
    TANK_TYPE_AI,           // AI控制的坦克（蓝色） - DQN模型训练的主角，通常只有1个
    TANK_TYPE_ENEMY,        // 追踪型敌人（红色） - 传统AI控制，追踪并攻击AI坦克
    TANK_TYPE_SELF_PLAY,    // 历史版本AI敌人（红白色） - 使用历史模型的自我对弈敌人
    TANK_TYPE_PLAYER,       // 玩家1控制（绿色） - 由键盘WASD+空格控制
    TANK_TYPE_PLAYER2       // 玩家2控制（深绿色） - 多人对战模式下的第二玩家
} TankType;

/*
 * 坦克动作枚举
 * 定义DQN模型的动作空间，共9个离散动作
 * 重要：此枚举的顺序和数量影响DQN网络的输出层维度
 */
typedef enum {
    ACTION_IDLE = 0,        // 动作0：静止不动，保持原地
    ACTION_MOVE_UP,         // 动作1：向上移动
    ACTION_MOVE_DOWN,       // 动作2：向下移动
    ACTION_MOVE_LEFT,       // 动作3：向左移动
    ACTION_MOVE_RIGHT,      // 动作4：向右移动
    ACTION_SHOOT_UP,        // 动作5：向上射击
    ACTION_SHOOT_DOWN,      // 动作6：向下射击
    ACTION_SHOOT_LEFT,      // 动作7：向左射击
    ACTION_SHOOT_RIGHT,     // 动作8：向右射击
    ACTION_COUNT            // 总动作数：9，用于DQN网络输出维度
} TankAction;

/*
 * 坦克朝向枚举
 * 用于渲染和逻辑判断，决定坦克图像的旋转方向
 */
typedef enum {
    DIR_UP = 0,             // 朝向上方（0度）
    DIR_DOWN,               // 朝向下方（180度）
    DIR_LEFT,               // 朝向左方（270度）
    DIR_RIGHT               // 朝向右方（90度）
} Direction;

/*
 * 坦克实体结构体
 * 存储坦克的完整状态信息
 */
typedef struct {
    float x, y;              // 坦克中心位置（像素坐标，浮点数用于平滑移动）
    float vx, vy;            // 速度分量（像素/帧），范围通常为 [-TANK_SPEED, TANK_SPEED]
    int width, height;       // 坦克尺寸（像素），通常为 TANK_SIZE x TANK_SIZE
    int health;              // 当前血量，范围 [0, TANK_MAX_HEALTH]，0表示死亡
    bool alive;              // 存活标志，false表示已死亡，不再参与游戏逻辑
    TankType type;           // 坦克类型，决定控制方式和渲染颜色
    Direction direction;     // 当前朝向，影响渲染和子弹发射方向
    int shoot_cooldown;      // 射击冷却计数器（帧数），>0时无法射击，每帧递减
    int model_version;       // 模型版本号，用于自我对弈时标识历史版本AI（0表示最新）
    int id;                  // 坦克唯一标识符，用于子弹所有权追踪和碰撞检测
} Tank;

/*
 * 坦克相关常量定义
 */
#define TANK_SIZE 32            // 坦克尺寸（像素），正方形边长
#define TANK_SPEED 2.5f         // 坦克移动速度（像素/帧），影响移动的流畅度和灵活性
#define TANK_MAX_HEALTH 3       // 坦克最大血量，初始血量和血量上限
#define SHOOT_COOLDOWN 15       // 射击冷却时间（帧数），15帧约0.25秒（60fps下）

/*
 * 坦克功能函数声明
 */

/*
 * 初始化坦克
 * @param tank: 待初始化的坦克指针
 * @param x: 初始X坐标（像素）
 * @param y: 初始Y坐标（像素）
 * @param type: 坦克类型
 * 功能：设置坦克的初始位置、类型、血量、速度等属性
 */
void tank_init(Tank* tank, float x, float y, TankType type);

/*
 * 更新坦克状态（每帧调用）
 * @param tank: 待更新的坦克指针
 * 功能：
 *   - 更新位置（x += vx, y += vy）
 *   - 递减射击冷却计数器
 *   - 处理其他每帧更新逻辑
 */
void tank_update(Tank* tank);

/*
 * 移动坦克
 * @param tank: 待移动的坦克指针
 * @param dir: 移动方向
 * @param map_width: 地图宽度（像素），用于边界检测
 * @param map_height: 地图高度（像素），用于边界检测
 * 功能：
 *   - 设置坦克速度分量（vx, vy）
 *   - 更新朝向（direction）
 *   - 确保不超出地图边界
 */
void tank_move(Tank* tank, Direction dir, int map_width, int map_height);

/*
 * 坦克受伤处理
 * @param tank: 受伤的坦克指针
 * @param damage: 伤害值（通常为1）
 * 功能：
 *   - 减少血量（health -= damage）
 *   - 如果血量<=0，设置alive=false
 */
void tank_take_damage(Tank* tank, int damage);

/*
 * 检查坦克是否可以射击
 * @param tank: 待检查的坦克指针
 * @return: true=可以射击，false=冷却中
 * 判断条件：shoot_cooldown <= 0
 */
bool tank_can_shoot(Tank* tank);

/*
 * 启动射击冷却
 * @param tank: 刚发射子弹的坦克指针
 * 功能：设置 shoot_cooldown = SHOOT_COOLDOWN
 */
void tank_start_shoot_cooldown(Tank* tank);

/*
 * 设置模型版本号
 * @param tank: 坦克指针
 * @param version: 模型版本号（0=最新，>0=历史版本）
 * 用途：自我对弈系统中标识敌人使用的历史模型
 */
void tank_set_model_version(Tank* tank, int version);

#endif // TANK_H
