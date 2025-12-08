/*
 * rendering.c - SDL2渲染实现
 *
 * 模块说明：
 * 本模块负责游戏的视觉呈现，包括：
 * - SDL2窗口和渲染器的初始化
 * - 坦克、子弹、UI元素的绘制
 * - 文本渲染（TTF字体）
 * - 训练信息显示
 *
 * 技术栈：
 * - SDL2: 跨平台图形库
 * - SDL2_ttf: TrueType字体渲染扩展
 *
 * 坐标系统：
 * - 原点(0,0)在左上角
 * - X轴向右递增
 * - Y轴向下递增
 */

#include "rendering.h"
#include <stdio.h>

/**
 * 初始化渲染器（内部实现）
 *
 * 功能：创建SDL窗口、渲染器和字体资源
 *
 * 参数：
 *   r      - 渲染器结构体指针
 *   width  - 窗口宽度（像素）
 *   height - 窗口高度（像素）
 *   title  - 窗口标题
 *   x      - 窗口X坐标（或SDL_WINDOWPOS_CENTERED）
 *   y      - 窗口Y坐标（或SDL_WINDOWPOS_CENTERED）
 *
 * 返回值：
 *   true  - 初始化成功
 *   false - 初始化失败
 *
 * 错误处理：
 *   - 任何步骤失败都会清理已分配的资源
 *   - 使用stderr输出错误信息
 */
static bool renderer_init_internal(Renderer* r, int width, int height, const char* title, int x, int y) {
    // 记录窗口尺寸
    r->width = width;
    r->height = height;
    r->initialized = false;

    // ========== 步骤1：初始化SDL视频子系统 ==========
    // SDL_INIT_VIDEO：只初始化视频功能（不需要音频等）
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL初始化失败: %s\n", SDL_GetError());
        return false;
    }

    // ========== 步骤2：初始化TTF字体系统 ==========
    // SDL_ttf用于渲染TrueType字体（如DejaVu Sans）
    if (TTF_Init() < 0) {
        fprintf(stderr, "TTF初始化失败: %s\n", TTF_GetError());
        SDL_Quit();  // 清理SDL
        return false;
    }

    // ========== 步骤3：创建SDL窗口 ==========
    // SDL_WINDOW_SHOWN：窗口立即显示（vs SDL_WINDOW_HIDDEN）
    r->window = SDL_CreateWindow(title, x, y, width, height, SDL_WINDOW_SHOWN);
    if (!r->window) {
        fprintf(stderr, "窗口创建失败: %s\n", SDL_GetError());
        TTF_Quit();  // 清理TTF
        SDL_Quit();  // 清理SDL
        return false;
    }

    // ========== 步骤4：创建SDL渲染器 ==========
    // SDL_RENDERER_ACCELERATED：使用GPU加速渲染
    // SDL_RENDERER_PRESENTVSYNC：开启垂直同步（60fps，避免撕裂）
    r->renderer = SDL_CreateRenderer(r->window, -1,
                                     SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!r->renderer) {
        fprintf(stderr, "渲染器创建失败: %s\n", SDL_GetError());
        SDL_DestroyWindow(r->window);
        TTF_Quit();
        SDL_Quit();
        return false;
    }

    // ========== 步骤5：加载字体 ==========
    // 尝试加载系统默认字体（DejaVu Sans）
    // 大字体：24px用于标题和游戏结束信息
    // 小字体：14px用于调试信息和模型版本号
    r->font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24);
    r->small_font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14);

    if (!r->font) {
        // 如果Debian/Ubuntu路径不存在，尝试Arch Linux路径
        r->font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 24);
        r->small_font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 14);
    }
    // 注意：即使字体加载失败也继续运行（文本渲染会跳过）

    r->initialized = true;
    return true;
}

/**
 * 初始化渲染器（默认居中位置）
 *
 * 功能：创建窗口并将其居中显示在屏幕上
 *
 * 参数：
 *   r      - 渲染器结构体指针
 *   width  - 窗口宽度
 *   height - 窗口高度
 *   title  - 窗口标题
 *
 * 返回值：
 *   true  - 成功
 *   false - 失败
 */
bool renderer_init(Renderer* r, int width, int height, const char* title) {
    // 使用SDL_WINDOWPOS_CENTERED让窗口居中
    return renderer_init_internal(r, width, height, title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
}

/**
 * 初始化渲染器（指定窗口位置）
 *
 * 功能：创建窗口并指定其在屏幕上的位置
 *
 * 参数：
 *   r      - 渲染器结构体指针
 *   width  - 窗口宽度
 *   height - 窗口高度
 *   title  - 窗口标题
 *   x      - 窗口左上角X坐标（屏幕坐标系）
 *   y      - 窗口左上角Y坐标（屏幕坐标系）
 *
 * 用途：
 *   - 多人对战模式：让两个玩家的窗口并排显示
 */
bool renderer_init_with_pos(Renderer* r, int width, int height, const char* title, int x, int y) {
    return renderer_init_internal(r, width, height, title, x, y);
}

/**
 * 清理渲染器
 *
 * 功能：释放所有SDL资源
 *
 * 参数：
 *   r - 渲染器结构体指针
 *
 * 清理顺序：
 *   1. 字体资源（TTF_CloseFont）
 *   2. 渲染器（SDL_DestroyRenderer）
 *   3. 窗口（SDL_DestroyWindow）
 *   4. TTF系统（TTF_Quit）
 *   5. SDL系统（SDL_Quit）
 *
 * 注意：必须在程序退出前调用，否则内存泄漏
 */
void renderer_cleanup(Renderer* r) {
    if (!r->initialized) return;  // 未初始化则跳过

    // 按从高层到低层的顺序清理资源
    if (r->font) TTF_CloseFont(r->font);               // 关闭大字体
    if (r->small_font) TTF_CloseFont(r->small_font);   // 关闭小字体
    if (r->renderer) SDL_DestroyRenderer(r->renderer); // 销毁渲染器
    if (r->window) SDL_DestroyWindow(r->window);       // 销毁窗口

    TTF_Quit();  // 关闭TTF字体系统
    SDL_Quit();  // 关闭SDL系统

    r->initialized = false;  // 标记为未初始化
}

/**
 * 渲染文本（内部辅助函数）
 *
 * 功能：将UTF-8文本渲染到屏幕
 *
 * 参数：
 *   r     - 渲染器指针
 *   text  - 要渲染的文本（UTF-8编码）
 *   x     - 文本左上角X坐标
 *   y     - 文本左上角Y坐标
 *   color - 文本颜色（RGB+Alpha）
 *   font  - 使用的字体（NULL则跳过渲染）
 *
 * 渲染流程：
 *   1. 文本 → Surface（CPU内存中的位图）
 *   2. Surface → Texture（GPU内存中的纹理）
 *   3. Texture → 屏幕（硬件加速渲染）
 *
 * 性能注意：
 *   - 每次调用都重新生成纹理（适合动态文本）
 *   - 频繁渲染同一文本时，应缓存纹理
 */
static void render_text(Renderer* r, const char* text, int x, int y,
                       SDL_Color color, TTF_Font* font) {
    if (!font) return;  // 字体未加载则跳过

    // 步骤1：渲染文本到Surface（CPU内存位图）
    // TTF_RenderText_Solid：最快，但边缘锯齿
    // （还有RenderText_Blended：抗锯齿但较慢）
    SDL_Surface* surface = TTF_RenderText_Solid(font, text, color);
    if (!surface) return;

    // 步骤2：转换为GPU纹理
    SDL_Texture* texture = SDL_CreateTextureFromSurface(r->renderer, surface);
    if (texture) {
        // 步骤3：渲染到屏幕
        // 矩形：{x, y, width, height}
        SDL_Rect rect = {x, y, surface->w, surface->h};
        SDL_RenderCopy(r->renderer, texture, NULL, &rect);
        SDL_DestroyTexture(texture);  // 立即销毁纹理（下次重新生成）
    }

    SDL_FreeSurface(surface);  // 释放Surface内存
}

/**
 * 渲染坦克（内部辅助函数）
 *
 * 功能：绘制坦克主体、炮筒、血量条和版本号
 *
 * 参数：
 *   r    - 渲染器指针
 *   tank - 要渲染的坦克指针
 *
 * 渲染内容：
 *   1. 坦克主体（32x32彩色矩形）
 *   2. 炮筒（朝向当前direction的黄色矩形）
 *   3. 血量条（坦克上方的红/绿条）
 *   4. 模型版本号（自我对弈坦克专用）
 *
 * 颜色编码（用于区分不同阵营）：
 *   - AI坦克：蓝色（训练中的AI）
 *   - 敌人：红色（追踪型AI）
 *   - 历史版本：浅红色（自我对弈的旧模型）
 *   - 玩家1：绿色
 *   - 玩家2：深绿色
 */
static void render_tank(Renderer* r, Tank* tank) {
    if (!tank->alive) return;  // 死亡坦克不渲染

    // ========== 1. 渲染坦克主体（彩色矩形）==========

    // 定义矩形：{x, y, width, height}
    // 坦克尺寸固定为32x32像素
    SDL_Rect rect = {
        (int)tank->x,       // 左上角X坐标（浮点转整数）
        (int)tank->y,       // 左上角Y坐标
        tank->width,        // 宽度（32）
        tank->height        // 高度（32）
    };

    // 根据坦克类型设置颜色
    // SDL_SetRenderDrawColor设置后续绘制的颜色（R, G, B, Alpha）
    switch (tank->type) {
        case TANK_TYPE_AI:
            // 蓝色（AI训练坦克）- RGB(0, 100, 255)
            SDL_SetRenderDrawColor(r->renderer, 0, 100, 255, 255);
            break;
        case TANK_TYPE_ENEMY:
            // 红色（敌人追踪AI）- RGB(255, 50, 50)
            SDL_SetRenderDrawColor(r->renderer, 255, 50, 50, 255);
            break;
        case TANK_TYPE_SELF_PLAY:
            // 浅红色（历史版本自我对弈）- RGB(255, 150, 150)
            SDL_SetRenderDrawColor(r->renderer, 255, 150, 150, 255);
            break;
        case TANK_TYPE_PLAYER:
            // 绿色（玩家1）- RGB(50, 255, 50)
            SDL_SetRenderDrawColor(r->renderer, 50, 255, 50, 255);
            break;
        case TANK_TYPE_PLAYER2:
            // 深绿色（玩家2）- RGB(0, 150, 50)
            SDL_SetRenderDrawColor(r->renderer, 0, 150, 50, 255);
            break;
    }

    // 填充矩形（坦克主体）
    SDL_RenderFillRect(r->renderer, &rect);

    // ========== 2. 绘制炮筒（显示坦克朝向）==========

    // 炮筒设计：
    // - 一部分在坦克内（8px）
    // - 一部分在坦克外（12px）
    // - 总长度20px，宽度6px
    // - 颜色：黄色（容易识别）

    SDL_Rect barrel_rect;
    int barrel_total_length = 20;  // 炮筒总长度
    int barrel_width = 6;          // 炮筒宽度
    int barrel_inside = 8;         // 炮筒在坦克内的长度
    int barrel_outside = barrel_total_length - barrel_inside;  // 炮筒在坦克外的长度（12px）

    // 根据坦克朝向计算炮筒位置
    // 炮筒始终从坦克中心伸出
    switch (tank->direction) {
        case DIR_UP:
            // 向上：炮筒垂直，上端突出坦克
            barrel_rect = (SDL_Rect){
                (int)tank->x + (tank->width - barrel_width) / 2,   // 水平居中
                (int)tank->y - barrel_outside,                      // 向上突出12px
                barrel_width,                                       // 宽6px
                barrel_total_length                                 // 高20px
            };
            break;
        case DIR_DOWN:
            // 向下：炮筒垂直，下端突出坦克
            barrel_rect = (SDL_Rect){
                (int)tank->x + (tank->width - barrel_width) / 2,   // 水平居中
                (int)tank->y + tank->height - barrel_inside,        // 从下方伸出
                barrel_width,
                barrel_total_length
            };
            break;
        case DIR_LEFT:
            // 向左：炮筒水平，左端突出坦克
            barrel_rect = (SDL_Rect){
                (int)tank->x - barrel_outside,                      // 向左突出12px
                (int)tank->y + (tank->height - barrel_width) / 2,  // 垂直居中
                barrel_total_length,                                // 宽20px
                barrel_width                                        // 高6px
            };
            break;
        case DIR_RIGHT:
            // 向右：炮筒水平，右端突出坦克
            barrel_rect = (SDL_Rect){
                (int)tank->x + tank->width - barrel_inside,         // 从右侧伸出
                (int)tank->y + (tank->height - barrel_width) / 2,  // 垂直居中
                barrel_total_length,
                barrel_width
            };
            break;
    }

    // 绘制黄色炮筒
    SDL_SetRenderDrawColor(r->renderer, 255, 255, 0, 255);  // 黄色RGBA
    SDL_RenderFillRect(r->renderer, &barrel_rect);

    // ========== 3. 渲染血量条（坦克上方）==========

    // 血量条设计：
    // - 位置：坦克上方8px
    // - 宽度：与坦克宽度一致（32px）
    // - 高度：4px
    // - 颜色：背景红色，前景绿色
    int bar_width = tank->width;        // 血量条宽度=坦克宽度
    int bar_height = 4;                 // 血量条高度固定4px
    int bar_x = (int)tank->x;           // 左对齐坦克
    int bar_y = (int)tank->y - 8;       // 坦克上方8px

    // 绘制背景（深红色）
    SDL_SetRenderDrawColor(r->renderer, 100, 0, 0, 255);
    SDL_Rect bg_rect = {bar_x, bar_y, bar_width, bar_height};
    SDL_RenderFillRect(r->renderer, &bg_rect);

    // 绘制血量（绿色，根据当前血量缩放宽度）
    // health_width = (当前血量 / 最大血量) * 总宽度
    // 例如：血量2/3时，blood条宽度=32 * 2/3 ≈ 21px
    int health_width = (bar_width * tank->health) / TANK_MAX_HEALTH;
    SDL_SetRenderDrawColor(r->renderer, 0, 255, 0, 255);
    SDL_Rect health_rect = {bar_x, bar_y, health_width, bar_height};
    SDL_RenderFillRect(r->renderer, &health_rect);

    // ========== 4. 渲染模型版本号（仅自我对弈坦克）==========

    // 自我对弈坦克会显示其使用的模型版本号
    // 例如："v200"表示第200回合保存的模型
    if (tank->type == TANK_TYPE_SELF_PLAY && r->small_font) {
        char version_text[16];
        snprintf(version_text, sizeof(version_text), "v%d", tank->model_version);
        SDL_Color white = {255, 255, 255, 255};  // 白色文本
        // 渲染在血量条上方（y-20）
        render_text(r, version_text, (int)tank->x, (int)tank->y - 20, white, r->small_font);
    }
}

/**
 * 渲染子弹（内部辅助函数）
 *
 * 功能：绘制8x8黄色方块代表子弹
 *
 * 参数：
 *   r      - 渲染器指针
 *   bullet - 要渲染的子弹指针
 *
 * 设计：
 *   - 尺寸：8x8像素（比坦克小，易于识别）
 *   - 颜色：黄色（与炮筒颜色一致）
 *   - 未激活的子弹不渲染
 */
static void render_bullet(Renderer* r, Bullet* bullet) {
    if (!bullet->active) return;  // 未激活的子弹不渲染

    // 定义子弹矩形（8x8黄色方块）
    SDL_Rect rect = {
        (int)bullet->x,       // 子弹X坐标
        (int)bullet->y,       // 子弹Y坐标
        bullet->width,        // 宽度（8px）
        bullet->height        // 高度（8px）
    };

    // 设置黄色并填充
    SDL_SetRenderDrawColor(r->renderer, 255, 255, 0, 255);  // 黄色RGBA
    SDL_RenderFillRect(r->renderer, &rect);
}

/**
 * 渲染游戏（主渲染函数）
 *
 * 功能：绘制完整的游戏画面
 *
 * 参数：
 *   r    - 渲染器指针
 *   game - 游戏状态指针
 *
 * 渲染顺序（从底层到顶层）：
 *   1. 清屏（深灰色背景）
 *   2. 网格线（50px间隔）
 *   3. 所有子弹
 *   4. 所有坦克（含血量条）
 *   5. 游戏结束文本
 *   6. 呈现到屏幕
 *
 * 性能优化：
 *   - 使用SDL_RenderPresent一次性刷新（避免撕裂）
 *   - 垂直同步（VSYNC）限制帧率在60fps
 */
void renderer_render_game(Renderer* r, GameState* game) {
    if (!r->initialized) return;  // 未初始化则跳过

    // ========== 步骤1：清屏（深灰色背景）==========
    // RGB(20, 20, 20)：接近黑色，但不是纯黑（避免过于刺眼）
    SDL_SetRenderDrawColor(r->renderer, 20, 20, 20, 255);
    SDL_RenderClear(r->renderer);  // 清空整个画布

    // ========== 步骤2：绘制网格线（辅助定位）==========
    // 网格间隔50px，便于玩家估算距离
    SDL_SetRenderDrawColor(r->renderer, 40, 40, 40, 255);  // 深灰色网格线

    // 绘制垂直线（每隔50px）
    for (int x = 0; x < game->map_width; x += 50) {
        SDL_RenderDrawLine(r->renderer, x, 0, x, game->map_height);
    }

    // 绘制水平线（每隔50px）
    for (int y = 0; y < game->map_height; y += 50) {
        SDL_RenderDrawLine(r->renderer, 0, y, game->map_width, y);
    }

    // ========== 步骤3：渲染所有子弹 ==========
    // 子弹在坦克下层，先渲染避免被坦克遮挡
    for (int i = 0; i < MAX_BULLETS; i++) {
        render_bullet(r, &game->bullets[i]);
    }

    // ========== 步骤4：渲染所有坦克 ==========
    // 坦克在最上层，包含血量条和炮筒
    for (int i = 0; i < game->tank_count; i++) {
        render_tank(r, &game->tanks[i]);
    }

    // ========== 步骤5：渲染游戏结束信息 ==========
    if (game->game_over && r->font) {
        SDL_Color color = {255, 255, 255, 255};  // 默认白色
        const char* text;

        // 根据胜者设置文本和颜色
        if (game->winner == 0) {
            // AI获胜（或玩家1获胜）
            text = "AI WINS!";
            color = (SDL_Color){0, 255, 0, 255};  // 绿色
        } else if (game->winner == 1) {
            // 敌人获胜（或玩家2获胜）
            text = "ENEMIES WIN!";
            color = (SDL_Color){255, 0, 0, 255};  // 红色
        } else {
            // 平局（超时）
            text = "DRAW!";
            color = (SDL_Color){255, 255, 0, 255};  // 黄色
        }

        // 居中渲染（手动计算偏移）
        // 假设文本宽度约160px，高度约40px
        render_text(r, text, game->map_width / 2 - 80, game->map_height / 2 - 20,
                   color, r->font);
    }

    // ========== 步骤6：呈现到屏幕 ==========
    // 将后台缓冲区的内容显示到窗口
    // 使用双缓冲技术，避免画面撕裂
    SDL_RenderPresent(r->renderer);
}

/**
 * 渲染训练信息（训练可视化模式）
 *
 * 功能：在游戏画面上叠加训练统计信息
 *
 * 参数：
 *   r               - 渲染器指针
 *   episode         - 当前回合数
 *   step            - 当前回合步数
 *   reward          - 累计奖励
 *   wins            - 胜利次数
 *   losses          - 失败次数
 *   current_enemies - 当前敌人数量
 *   status          - 状态文本（可选）
 *
 * 显示位置：
 *   - 屏幕左上角
 *   - 每行间隔30px
 *   - 白色文本（状态为绿色）
 *
 * 用途：
 *   - 仅在训练可视化模式（make train-viz）中使用
 *   - 帮助监控训练进度
 */
void renderer_render_training_info(Renderer* r, int episode, int step,
                                   float reward, int wins, int losses,
                                   int current_enemies, const char* status) {
    if (!r->initialized || !r->font) return;  // 未初始化或字体未加载则跳过

    char text[256];                           // 文本缓冲区
    SDL_Color white = {255, 255, 255, 255};   // 白色文本
    SDL_Color green = {0, 255, 0, 255};       // 绿色状态文本

    int y = 10;  // 起始Y坐标（距离顶部10px）

    // ========== 行1：回合数 ==========
    snprintf(text, sizeof(text), "Episode: %d", episode);
    render_text(r, text, 10, y, white, r->font);
    y += 30;  // 下一行

    // ========== 行2：当前回合步数 ==========
    snprintf(text, sizeof(text), "Steps: %d", step);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // ========== 行3：累计奖励 ==========
    // 保留2位小数，例如："Reward: 123.45"
    snprintf(text, sizeof(text), "Reward: %.2f", reward);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // ========== 行4：胜负统计 ==========
    // 格式："Wins: 10 | Losses: 5"
    snprintf(text, sizeof(text), "Wins: %d | Losses: %d", wins, losses);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // ========== 行5：敌人数量 ==========
    // 显示当前回合的敌人数（动态难度调整）
    snprintf(text, sizeof(text), "Enemies: %d", current_enemies);
    render_text(r, text, 10, y, white, r->font);
    y += 30;

    // ========== 行6：状态文本（可选）==========
    // 例如："Training..." 或 "Victory!" 等
    if (status) {
        render_text(r, status, 10, y, green, r->font);  // 绿色高亮
    }
}
