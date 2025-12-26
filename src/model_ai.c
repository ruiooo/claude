/*
 * model_ai.c - 基于Python训练模型的AI控制器实现
 *
 * 【架构说明】
 * 这个模块实现了C程序调用Python训练的DQN模型进行推理
 *
 * 技术栈：
 * - Python C API：嵌入Python解释器到C程序
 * - PyTorch：加载.pth模型文件
 * - NumPy/Tensor：C float数组 ↔ Python tensor转换
 *
 * 核心流程：
 * 1. 初始化Python解释器和虚拟环境
 * 2. 加载torch、model、config模块
 * 3. 创建DQNAgent实例并加载模型权重
 * 4. 每帧：C状态数组 → Python tensor → 模型推理 → 返回动作
 *
 * 关键挑战：
 * - Python路径配置（虚拟环境vs系统Python）
 * - 内存管理（Python对象引用计数）
 * - 错误处理（Python异常 → C错误码）
 * - 类型转换（C数组 ↔ Python列表 ↔ PyTorch tensor）
 */

#include "model_ai.h"
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <wchar.h>
#include <locale.h>
#include <unistd.h>
#include <limits.h>

// ========== 全局Python对象 ==========
// 这些对象在程序启动时初始化，全局共享，避免重复导入

static PyObject* g_torch_module = NULL;   // torch模块（用于创建tensor）
static PyObject* g_model_module = NULL;   // model模块（DQNAgent类定义）
static PyObject* g_config_module = NULL;  // config模块（超参数配置）
static bool g_system_initialized = false; // 系统初始化标志

/*
 * 初始化Python解释器和必要模块
 *
 * 【核心系统初始化】
 * 这是整个模型AI系统的入口，负责：
 * 1. 检测并配置Python虚拟环境
 * 2. 初始化Python解释器
 * 3. 配置Python模块搜索路径（sys.path）
 * 4. 导入必要的Python模块（torch, model, config）
 *
 * Python虚拟环境检测策略：
 * - 优先使用编译时指定的VENV_DIR（Makefile传入）
 * - 然后尝试常见名称：venv, .venv, env
 * - 最后回退到系统Python
 *
 * sys.path配置难点：
 * - 虚拟环境的site-packages必须在标准库之前
 * - 需要根据实际Python版本动态查找路径
 * - 必须包含基础stdlib（如_ctypes.so）
 *
 * 错误处理：
 * - 返回false表示初始化失败
 * - 调用方应该检查返回值并退出
 *
 * @return true=成功, false=失败
 */
bool model_ai_system_init(void) {
    // 防止重复初始化
    if (g_system_initialized) {
        return true;
    }

    // ========== 步骤1：检测虚拟环境 ==========

    struct stat st;
    // 虚拟环境候选目录列表（按优先级）
    const char* venv_candidates[] = {
#ifdef VENV_DIR
        "./" VENV_DIR,  // 编译时指定（Makefile传入）
#endif
        "./venv",       // 常见名称1
        "./.venv",      // 常见名称2
        "./env",        // 常见名称3
        NULL            // 结束标志
    };

    const char* found_venv = NULL;  // 找到的虚拟环境目录
    char venv_abs_path[PATH_MAX];   // 虚拟环境绝对路径

    // 遍历候选目录，检查是否存在python3可执行文件
    for (int i = 0; venv_candidates[i] != NULL; i++) {
        char python_path[512];
        snprintf(python_path, sizeof(python_path), "%s/bin/python3", venv_candidates[i]);

        // 检查文件是否存在
        if (stat(python_path, &st) == 0) {
            found_venv = venv_candidates[i];

            // 获取绝对路径（Python需要绝对路径）
            if (realpath(found_venv, venv_abs_path) == NULL) {
                fprintf(stderr, "获取虚拟环境绝对路径失败\n");
                return false;
            }
            break;  // 找到第一个就停止
        }
    }

    // ========== 步骤2：配置Python解释器 ==========

    if (found_venv) {
        // 使用虚拟环境的Python
        char python_exe[PATH_MAX];
        snprintf(python_exe, sizeof(python_exe), "%s/bin/python3", venv_abs_path);

        // 转换为宽字符（Python C API要求wchar_t*）
        wchar_t python_exe_wide[PATH_MAX];
        mbstowcs(python_exe_wide, python_exe, PATH_MAX);

        // 设置Python程序名（影响sys.executable和sys.prefix）
        Py_SetProgramName(python_exe_wide);
        printf("使用虚拟环境: %s\n", venv_abs_path);
    } else {
        // 未找到虚拟环境，使用系统Python
        printf("未找到虚拟环境，使用系统Python\n");
    }

    // ========== 步骤3：初始化Python解释器 ==========

    Py_Initialize();  // 启动Python运行时
    if (!Py_IsInitialized()) {
        fprintf(stderr, "Python解释器初始化失败\n");
        return false;
    }

    // ========== 步骤4：获取当前工作目录 ==========
    // 需要把项目python/目录加入sys.path

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        fprintf(stderr, "获取当前工作目录失败\n");
        Py_Finalize();
        return false;
    }

    // ========== 步骤5：配置Python模块搜索路径（sys.path）==========

    // 导入必要的Python模块（用于路径操作）
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("import os");
    PyRun_SimpleString("import glob");

    // 如果使用虚拟环境，需要重建sys.path
    // 原因：默认的sys.path可能不包含虚拟环境的site-packages
    // 策略：虚拟环境路径 > 基础标准库 > 其他路径
    if (found_venv) {
        // 这段Python代码动态构建sys.path：
        // 1. 从pyvenv.cfg读取基础Python位置
        // 2. 查找虚拟环境的site-packages
        // 3. 查找基础Python的标准库和lib-dynload
        // 4. 重新组织sys.path顺序
        char setup_venv_cmd[PATH_MAX * 5];
        snprintf(setup_venv_cmd, sizeof(setup_venv_cmd),
                 "venv_root = '%s'\n"
                 "venv_lib = os.path.join(venv_root, 'lib')\n"
                 "venv_paths = []\n"
                 "base_stdlib = []\n"
                 "other_paths = []\n"
                 "actual_version = f'python{sys.version_info.major}.{sys.version_info.minor}'\n"
                 "if os.path.exists(venv_lib):\n"
                 "    venv_site_packages = os.path.join(venv_lib, actual_version, 'site-packages')\n"
                 "    if os.path.exists(venv_site_packages):\n"
                 "        venv_paths.append(venv_site_packages)\n"
                 "pyvenv_cfg = os.path.join(venv_root, 'pyvenv.cfg')\n"
                 "python_home = None\n"
                 "if os.path.exists(pyvenv_cfg):\n"
                 "    with open(pyvenv_cfg, 'r') as f:\n"
                 "        for line in f:\n"
                 "            if line.startswith('home = '):\n"
                 "                python_home = line.split('=', 1)[1].strip()\n"
                 "                break\n"
                 "if python_home and os.path.exists(python_home):\n"
                 "    python_home_parent = os.path.dirname(python_home)\n"
                 "    for lib_dir in [os.path.join(python_home_parent, 'lib'), os.path.join(python_home, '..', 'lib')]:\n"
                 "        lib_dir = os.path.abspath(lib_dir)\n"
                 "        if os.path.exists(lib_dir):\n"
                 "            base_py_dir = os.path.join(lib_dir, actual_version)\n"
                 "            if os.path.exists(base_py_dir):\n"
                 "                base_stdlib.append(base_py_dir)\n"
                 "                base_dynload = os.path.join(base_py_dir, 'lib-dynload')\n"
                 "                if os.path.exists(base_dynload):\n"
                 "                    base_stdlib.append(base_dynload)\n"
                 "                break\n"
                 "if not base_stdlib:\n"
                 "    base_prefix = sys.base_prefix\n"
                 "    base_lib = os.path.join(base_prefix, 'lib')\n"
                 "    base_py_dir = os.path.join(base_lib, actual_version)\n"
                 "    if os.path.exists(base_py_dir):\n"
                 "        base_stdlib.append(base_py_dir)\n"
                 "        base_dynload = os.path.join(base_py_dir, 'lib-dynload')\n"
                 "        if os.path.exists(base_dynload):\n"
                 "            base_stdlib.append(base_dynload)\n"
                 "for p in sys.path:\n"
                 "    if 'site-packages' in p or 'lib/python' in p:\n"
                 "        continue\n"
                 "    else:\n"
                 "        other_paths.append(p)\n"
                 "sys.path = other_paths + venv_paths + base_stdlib\n",
                 venv_abs_path);
        PyRun_SimpleString(setup_venv_cmd);
    }

    // ========== 步骤6：添加项目目录到Python路径 ==========

    // 添加项目python/目录（包含model.py, config.py等）
    char add_python_dir_cmd[PATH_MAX];
    snprintf(add_python_dir_cmd, sizeof(add_python_dir_cmd),
             "if '%s/python' not in sys.path: sys.path.insert(0, '%s/python')",
             cwd, cwd);
    PyRun_SimpleString(add_python_dir_cmd);

    // 添加项目根目录（作为后备）
    char add_cwd_cmd[PATH_MAX];
    snprintf(add_cwd_cmd, sizeof(add_cwd_cmd),
             "if '%s' not in sys.path: sys.path.insert(0, '%s')",
             cwd, cwd);
    PyRun_SimpleString(add_cwd_cmd);

    // ========== 步骤7：打印诊断信息（调试用）==========

    printf("Python sys.path:\n");
    PyRun_SimpleString("for p in sys.path[:5]: print('  -', p)");

    // 诊断信息：检查Python环境和关键依赖
    printf("\n诊断信息：\n");
    PyRun_SimpleString(
        "import sys, os\n"
        "print('Python版本:', sys.version)\n"
        "print('sys.prefix:', sys.prefix)\n"
        "print('sys.base_prefix:', sys.base_prefix)\n"
        "print('sys.executable:', sys.executable)\n"
        "print('\\n检查_ctypes.so位置:')\n"
        "for path in sys.path:\n"
        "    if 'lib-dynload' in path:\n"
        "        ctypes_so = os.path.join(path, '_ctypes.cpython-' + str(sys.version_info.major) + str(sys.version_info.minor) + '-x86_64-linux-gnu.so')\n"
        "        exists = os.path.exists(ctypes_so)\n"
        "        print(f'  {path}: {\"存在\" if exists else \"不存在\"}')\n"
        "        if exists:\n"
        "            print(f'    -> {ctypes_so}')\n"
    );

    // ========== 步骤8：导入Python模块 ==========

    // 导入torch模块（PyTorch深度学习框架）
    g_torch_module = PyImport_ImportModule("torch");
    if (!g_torch_module) {
        fprintf(stderr, "导入torch模块失败\n");
        if (found_venv) {
            fprintf(stderr, "请确保torch已安装: %s/bin/pip install torch\n", venv_abs_path);
        } else {
            fprintf(stderr, "请确保torch已安装: pip install torch\n");
        }
        PyErr_Print();  // 打印Python异常堆栈
        Py_Finalize();
        return false;
    }

    // 导入model模块（包含DQNAgent类定义）
    g_model_module = PyImport_ImportModule("model");
    if (!g_model_module) {
        fprintf(stderr, "导入model模块失败\n");
        PyErr_Print();
        Py_XDECREF(g_torch_module);  // 清理已导入的模块
        Py_Finalize();
        return false;
    }

    // 导入config模块（包含超参数配置）
    g_config_module = PyImport_ImportModule("config");
    if (!g_config_module) {
        fprintf(stderr, "导入config模块失败\n");
        PyErr_Print();
        Py_XDECREF(g_torch_module);
        Py_XDECREF(g_model_module);
        Py_Finalize();
        return false;
    }

    g_system_initialized = true;
    printf("✓ Python模型AI系统初始化成功\n");
    return true;
}

// 清理Python解释器
void model_ai_system_cleanup(void) {
    if (!g_system_initialized) {
        return;
    }

    Py_XDECREF(g_config_module);
    Py_XDECREF(g_model_module);
    Py_XDECREF(g_torch_module);

    Py_Finalize();
    g_system_initialized = false;
}

// 初始化单个模型AI实例
bool model_ai_init(ModelAI* ai, const char* model_path) {
    if (!g_system_initialized) {
        fprintf(stderr, "请先调用 model_ai_system_init()\n");
        return false;
    }

    ai->initialized = false;
    ai->py_agent = NULL;
    ai->py_module = NULL;
    strncpy(ai->model_path, model_path, sizeof(ai->model_path) - 1);

    // 获取DQNAgent类
    PyObject* agent_class = PyObject_GetAttrString(g_model_module, "DQNAgent");
    if (!agent_class) {
        fprintf(stderr, "获取DQNAgent类失败\n");
        PyErr_Print();
        return false;
    }

    // 从config获取配置
    PyObject* model_config = PyObject_GetAttrString(g_config_module, "MODEL_CONFIG");
    PyObject* training_config = PyObject_GetAttrString(g_config_module, "TRAINING_CONFIG");

    if (!model_config || !training_config) {
        fprintf(stderr, "获取配置失败\n");
        PyErr_Print();
        Py_DECREF(agent_class);
        return false;
    }

    // 获取state_dim和action_dim
    PyObject* state_dim_obj = PyDict_GetItemString(model_config, "state_dim");
    PyObject* action_dim_obj = PyDict_GetItemString(model_config, "action_dim");
    PyObject* device_obj = PyDict_GetItemString(training_config, "device");

    if (!state_dim_obj || !action_dim_obj || !device_obj) {
        fprintf(stderr, "获取配置参数失败\n");
        Py_DECREF(agent_class);
        Py_XDECREF(model_config);
        Py_XDECREF(training_config);
        return false;
    }

    // 创建DQNAgent实例
    PyObject* args = PyTuple_Pack(4, state_dim_obj, action_dim_obj,
                                  model_config, device_obj);
    ai->py_agent = PyObject_CallObject(agent_class, args);

    Py_DECREF(args);
    Py_DECREF(agent_class);
    Py_XDECREF(model_config);
    Py_XDECREF(training_config);

    if (!ai->py_agent) {
        fprintf(stderr, "创建DQNAgent实例失败\n");
        PyErr_Print();
        return false;
    }

    // 加载模型
    PyObject* load_method = PyObject_GetAttrString((PyObject*)ai->py_agent, "load");
    if (!load_method) {
        fprintf(stderr, "获取load方法失败\n");
        PyErr_Print();
        Py_DECREF((PyObject*)ai->py_agent);
        return false;
    }

    PyObject* load_args = PyTuple_Pack(1, PyUnicode_FromString(model_path));
    PyObject* load_result = PyObject_CallObject(load_method, load_args);

    Py_DECREF(load_args);
    Py_DECREF(load_method);

    if (!load_result) {
        fprintf(stderr, "加载模型失败: %s\n", model_path);
        PyErr_Print();
        Py_DECREF((PyObject*)ai->py_agent);
        return false;
    }
    Py_DECREF(load_result);

    // 设置为评估模式
    PyObject* eval_method = PyObject_GetAttrString((PyObject*)ai->py_agent, "eval");
    if (eval_method) {
        PyObject* eval_result = PyObject_CallObject(eval_method, NULL);
        Py_XDECREF(eval_result);
        Py_DECREF(eval_method);
    }

    ai->initialized = true;
    printf("✓ 已加载AI模型: %s\n", model_path);

    // ✅ 打印模型加载详细信息
    PyObject* epsilon_attr = PyObject_GetAttrString((PyObject*)ai->py_agent, "epsilon");
    PyObject* train_step_attr = PyObject_GetAttrString((PyObject*)ai->py_agent, "train_step");

    if (epsilon_attr && train_step_attr) {
        double original_epsilon = PyFloat_AsDouble(epsilon_attr);
        long train_step = PyLong_AsLong(train_step_attr);
        printf("  - 训练步数: %ld\n", train_step);
        printf("  - 原始Epsilon值: %.4f ", original_epsilon);
        if (original_epsilon > 0.5) {
            printf("⚠️ (探索率过高)\n");
        } else if (original_epsilon > 0.2) {
            printf("(正常探索率)\n");
        } else {
            printf("(低探索率)\n");
        }

        // ✅ 【关键修复】推理模式下强制设置epsilon=0，完全利用策略
        // 训练时需要探索，但推理时应该100%使用最优策略
        printf("  - 推理模式：强制设置 Epsilon = 0.0 (纯利用模式)\n");
        PyObject* zero = PyFloat_FromDouble(0.0);
        PyObject_SetAttrString((PyObject*)ai->py_agent, "epsilon", zero);
        Py_DECREF(zero);

        Py_DECREF(epsilon_attr);
        Py_DECREF(train_step_attr);
    }

    return true;
}

// 清理模型AI实例
void model_ai_cleanup(ModelAI* ai) {
    if (ai->py_agent) {
        Py_DECREF((PyObject*)ai->py_agent);
        ai->py_agent = NULL;
    }
    ai->initialized = false;
}

/*
 * 将游戏状态转换为观察向量 v2.0（推理时使用）
 *
 * 【关键一致性要求】
 * 此函数必须与 game.c:game_get_observation() 完全一致！
 *
 * 状态维度：47维（v2.0升级）
 * - 6维：AI坦克状态
 * - 25维：5个敌人状态（每个5维）
 * - 12维：3个子弹状态（每个4维）
 * - 4维：战略信息（新增）
 *
 * 参数：
 * @param game     - 游戏状态
 * @param tank_id  - AI坦克ID
 * @param obs      - 输出：观察向量数组
 * @param obs_size - 输出：观察向量维度（=47）
 */
static void game_state_to_observation(const GameState* game, int tank_id,
                                      float* obs, int* obs_size) {
    int idx = 0;
    const Tank* ai_tank = &game->tanks[tank_id];

    // ========== 第1部分：AI坦克状态 (6维) ==========

    obs[idx++] = ai_tank->x / (float)game->map_width;
    obs[idx++] = ai_tank->y / (float)game->map_height;
    obs[idx++] = ai_tank->vx / TANK_SPEED;
    obs[idx++] = ai_tank->vy / TANK_SPEED;
    obs[idx++] = (float)ai_tank->health / (float)TANK_MAX_HEALTH;
    // 射击冷却改为连续值（与game.c一致）
    obs[idx++] = (float)ai_tank->shoot_cooldown / (float)SHOOT_COOLDOWN;

    // ========== 第2部分：最近5个敌人状态 (25维) ==========

    // 收集所有敌人并按距离排序
    typedef struct {
        float dx, dy, vx, vy, health, dist;
    } EnemyData;
    EnemyData all_enemies[32];
    int total_enemies = 0;
    float min_enemy_dist = 9999.0f;

    for (int i = 0; i < game->tank_count; i++) {
        if (i != tank_id && game->tanks[i].alive) {
            float dx = game->tanks[i].x - ai_tank->x;
            float dy = game->tanks[i].y - ai_tank->y;
            float dist = sqrtf(dx*dx + dy*dy);

            all_enemies[total_enemies].dx = dx;
            all_enemies[total_enemies].dy = dy;
            all_enemies[total_enemies].vx = game->tanks[i].vx;
            all_enemies[total_enemies].vy = game->tanks[i].vy;
            all_enemies[total_enemies].health = (float)game->tanks[i].health;
            all_enemies[total_enemies].dist = dist;

            if (dist < min_enemy_dist) {
                min_enemy_dist = dist;
            }
            total_enemies++;
        }
    }

    // 按距离排序
    for (int i = 0; i < total_enemies - 1; i++) {
        for (int j = 0; j < total_enemies - i - 1; j++) {
            if (all_enemies[j].dist > all_enemies[j+1].dist) {
                EnemyData temp = all_enemies[j];
                all_enemies[j] = all_enemies[j+1];
                all_enemies[j+1] = temp;
            }
        }
    }

    // 取最近的5个敌人
    float enemy_info[5][5] = {0};
    for (int i = 0; i < 5 && i < total_enemies; i++) {
        enemy_info[i][0] = all_enemies[i].dx / (float)game->map_width;
        enemy_info[i][1] = all_enemies[i].dy / (float)game->map_height;
        enemy_info[i][2] = all_enemies[i].vx / TANK_SPEED;
        enemy_info[i][3] = all_enemies[i].vy / TANK_SPEED;
        enemy_info[i][4] = all_enemies[i].health / (float)TANK_MAX_HEALTH;
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            obs[idx++] = enemy_info[i][j];
        }
    }

    // ========== 第3部分：最近3个子弹状态 (12维) ==========

    typedef struct {
        float dx, dy, vx, vy, dist;
    } BulletData;
    BulletData all_bullets[MAX_BULLETS];
    int total_bullets = 0;

    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active && game->bullets[i].owner_id != ai_tank->id) {
            float dx = game->bullets[i].x - ai_tank->x;
            float dy = game->bullets[i].y - ai_tank->y;
            float dist = sqrtf(dx*dx + dy*dy);

            all_bullets[total_bullets].dx = dx;
            all_bullets[total_bullets].dy = dy;
            all_bullets[total_bullets].vx = game->bullets[i].vx;
            all_bullets[total_bullets].vy = game->bullets[i].vy;
            all_bullets[total_bullets].dist = dist;
            total_bullets++;
        }
    }

    // 按距离排序
    for (int i = 0; i < total_bullets - 1; i++) {
        for (int j = 0; j < total_bullets - i - 1; j++) {
            if (all_bullets[j].dist > all_bullets[j+1].dist) {
                BulletData temp = all_bullets[j];
                all_bullets[j] = all_bullets[j+1];
                all_bullets[j+1] = temp;
            }
        }
    }

    float bullet_info[3][4] = {0};
    for (int i = 0; i < 3 && i < total_bullets; i++) {
        bullet_info[i][0] = all_bullets[i].dx / (float)game->map_width;
        bullet_info[i][1] = all_bullets[i].dy / (float)game->map_height;
        bullet_info[i][2] = all_bullets[i].vx / BULLET_SPEED;
        bullet_info[i][3] = all_bullets[i].vy / BULLET_SPEED;
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            obs[idx++] = bullet_info[i][j];
        }
    }

    // ========== 第4部分：战略信息 (4维) 【新增】 ==========

    // [43] 存活敌人数量
    obs[idx++] = (float)total_enemies / 10.0f;

    // [44] 最近敌人距离
    float normalized_min_dist = (min_enemy_dist < 9999.0f) ?
                                 min_enemy_dist / 800.0f : 1.0f;
    if (normalized_min_dist > 1.0f) normalized_min_dist = 1.0f;
    obs[idx++] = normalized_min_dist;

    // [45] 水平墙壁距离
    float dist_to_left = ai_tank->x;
    float dist_to_right = game->map_width - ai_tank->x;
    float min_horizontal_wall = (dist_to_left < dist_to_right) ?
                                 dist_to_left : dist_to_right;
    obs[idx++] = min_horizontal_wall / (float)game->map_width;

    // [46] 垂直墙壁距离
    float dist_to_top = ai_tank->y;
    float dist_to_bottom = game->map_height - ai_tank->y;
    float min_vertical_wall = (dist_to_top < dist_to_bottom) ?
                               dist_to_top : dist_to_bottom;
    obs[idx++] = min_vertical_wall / (float)game->map_height;

    *obs_size = idx;  // 6 + 25 + 12 + 4 = 47 维
}

/*
 * 获取模型AI的决策（推理入口）
 *
 * 【核心推理流程】
 * 这是每帧调用的函数，实现：C状态 → Python模型 → C动作
 *
 * 数据流：
 * 1. C: 游戏状态 → float数组(43维)
 * 2. C→Py: float数组 → Python列表 → PyTorch tensor
 * 3. Py: tensor → DQN网络 → 动作索引(0-8)
 * 4. Py→C: 动作索引 → TankAction枚举
 *
 * 内存管理：
 * - 所有PyObject*必须Py_DECREF释放
 * - 遵循Python引用计数规则
 * - 失败时清理已分配的对象
 *
 * 参数：
 * @param ai      - 模型AI实例
 * @param game    - 游戏状态
 * @param tank_id - AI坦克ID
 * @return 推荐的坦克动作
 */
TankAction model_ai_get_action(ModelAI* ai, const GameState* game, int tank_id) {
    // 前置检查
    if (!ai->initialized || !ai->py_agent) {
        return ACTION_IDLE;
    }

    // ========== 步骤1：获取游戏状态（C数组）==========

    float obs[128];  // 观察向量缓冲区（43维足够）
    int obs_size;
    game_state_to_observation(game, tank_id, obs, &obs_size);

    // ========== 步骤2：C数组 → Python列表 ==========

    PyObject* obs_list = PyList_New(obs_size);
    for (int i = 0; i < obs_size; i++) {
        // 创建Python float对象并加入列表
        // PyList_SetItem会"偷取"引用，不需要DECREF
        PyList_SetItem(obs_list, i, PyFloat_FromDouble(obs[i]));
    }

    // ========== 步骤3：Python列表 → PyTorch tensor ==========

    // 获取torch.tensor函数
    PyObject* torch_tensor_func = PyObject_GetAttrString(g_torch_module, "tensor");

    // 创建参数元组（torch.tensor需要一个参数：数据）
    PyObject* tensor_args = PyTuple_Pack(1, obs_list);

    // 调用torch.tensor([obs_list])
    PyObject* state_tensor = PyObject_CallObject(torch_tensor_func, tensor_args);

    // 清理中间对象
    Py_DECREF(obs_list);
    Py_DECREF(tensor_args);
    Py_DECREF(torch_tensor_func);

    if (!state_tensor) {
        fprintf(stderr, "创建state tensor失败\n");
        PyErr_Print();
        return ACTION_IDLE;
    }

    // ========== 步骤4：调用模型推理 ==========

    // 获取DQNAgent的select_action方法
    PyObject* select_action_method = PyObject_GetAttrString((PyObject*)ai->py_agent,
                                                            "select_action");
    if (!select_action_method) {
        fprintf(stderr, "获取select_action方法失败\n");
        PyErr_Print();
        Py_DECREF(state_tensor);
        return ACTION_IDLE;
    }

    // 创建位置参数元组：(state,)
    PyObject* action_args = PyTuple_Pack(1, state_tensor);

    // 创建关键字参数字典：{training: False}
    PyObject* action_kwargs = PyDict_New();
    PyDict_SetItemString(action_kwargs, "training", Py_False);

    // 调用agent.select_action(state, training=False)
    PyObject* action_result = PyObject_Call(select_action_method, action_args, action_kwargs);

    // 清理
    Py_DECREF(state_tensor);
    Py_DECREF(select_action_method);
    Py_DECREF(action_args);
    Py_DECREF(action_kwargs);

    if (!action_result) {
        fprintf(stderr, "调用select_action失败\n");
        PyErr_Print();
        return ACTION_IDLE;
    }

    // ========== 步骤5：Python整数 → C动作枚举 ==========

    // 将Python int转换为C long
    long action = PyLong_AsLong(action_result);
    Py_DECREF(action_result);

    // ✅ 调试：打印前5次观察值和动作
    static int debug_count = 0;
    if (debug_count < 5) {
        printf("\n[调试 %d] 推理详情：\n", debug_count);
        printf("  观察值(前10维): ");
        for (int i = 0; i < 10 && i < obs_size; i++) {
            printf("%.3f ", obs[i]);
        }
        printf("\n  选择动作: %ld (IDLE=0, 移动=1-4, 射击=5-8)\n", action);

        // 获取并打印epsilon值
        PyObject* epsilon_attr = PyObject_GetAttrString((PyObject*)ai->py_agent, "epsilon");
        if (epsilon_attr) {
            double epsilon = PyFloat_AsDouble(epsilon_attr);
            printf("  当前Epsilon: %.4f (探索率)\n", epsilon);
            Py_DECREF(epsilon_attr);
        }

        debug_count++;
    }

    // 验证动作范围（0-8对应ACTION_IDLE到ACTION_SHOOT_RIGHT）
    if (action >= 0 && action <= ACTION_SHOOT_RIGHT) {
        return (TankAction)action;
    }

    // 无效动作，返回默认
    fprintf(stderr, "⚠️ 无效动作: %ld，使用IDLE\n", action);
    return ACTION_IDLE;
}

// 模型文件信息
typedef struct {
    char path[256];
    time_t mtime;
} ModelFile;

// 比较函数：按修改时间降序排序
static int compare_models_by_time(const void* a, const void* b) {
    const ModelFile* ma = (const ModelFile*)a;
    const ModelFile* mb = (const ModelFile*)b;
    return (mb->mtime > ma->mtime) - (mb->mtime < ma->mtime);
}

// 列出可用的模型文件（优先显示：最优模型、最新模型、Top3训练版本）
int model_ai_list_models(char models[][256], int max_count) {
    ModelFile checkpoint_files[100];
    int checkpoint_count = 0;
    int result_count = 0;

    // 优先级模型路径
    const char* priority_models[] = {
        "saved_models/final_model.pth",
        "saved_models/latest_model.pth",
    };

    // 1. 添加优先级模型（final_model 和 latest_model）
    for (int i = 0; i < 2 && result_count < max_count; i++) {
        struct stat st;
        if (stat(priority_models[i], &st) == 0) {
            strncpy(models[result_count], priority_models[i], 256);
            result_count++;
        }
    }

    // 2. 收集所有 checkpoint 文件
    const char* checkpoint_dirs[] = {
        "saved_models/checkpoints",
        "checkpoints",
        NULL
    };

    for (int d = 0; checkpoint_dirs[d] != NULL && checkpoint_count < 100; d++) {
        DIR* dir = opendir(checkpoint_dirs[d]);
        if (!dir) continue;

        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL && checkpoint_count < 100) {
            // 只收集 checkpoint_ep*.pth 文件
            if (strstr(entry->d_name, "checkpoint_ep") &&
                (strstr(entry->d_name, ".pth") || strstr(entry->d_name, ".pt"))) {
                snprintf(checkpoint_files[checkpoint_count].path, 256,
                        "%s/%s", checkpoint_dirs[d], entry->d_name);

                // 获取文件修改时间
                struct stat st;
                if (stat(checkpoint_files[checkpoint_count].path, &st) == 0) {
                    checkpoint_files[checkpoint_count].mtime = st.st_mtime;
                    checkpoint_count++;
                }
            }
        }
        closedir(dir);
    }

    // 3. 如果有 checkpoint 文件，按时间排序并添加前3个
    if (checkpoint_count > 0) {
        qsort(checkpoint_files, checkpoint_count, sizeof(ModelFile), compare_models_by_time);

        // 添加最新的3个 checkpoint
        int checkpoints_to_add = (checkpoint_count < 3) ? checkpoint_count : 3;
        for (int i = 0; i < checkpoints_to_add && result_count < max_count; i++) {
            strncpy(models[result_count], checkpoint_files[i].path, 256);
            result_count++;
        }
    }

    // 4. 如果仍然没有找到任何模型，搜索其他目录
    if (result_count == 0) {
        const char* fallback_dirs[] = {
            "saved_models",
            "models",
            NULL
        };

        ModelFile fallback_files[50];
        int fallback_count = 0;

        for (int d = 0; fallback_dirs[d] != NULL && fallback_count < 50; d++) {
            DIR* dir = opendir(fallback_dirs[d]);
            if (!dir) continue;

            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL && fallback_count < 50) {
                if (strstr(entry->d_name, ".pth") || strstr(entry->d_name, ".pt")) {
                    snprintf(fallback_files[fallback_count].path, 256,
                            "%s/%s", fallback_dirs[d], entry->d_name);

                    struct stat st;
                    if (stat(fallback_files[fallback_count].path, &st) == 0) {
                        fallback_files[fallback_count].mtime = st.st_mtime;
                        fallback_count++;
                    }
                }
            }
            closedir(dir);
        }

        if (fallback_count > 0) {
            qsort(fallback_files, fallback_count, sizeof(ModelFile), compare_models_by_time);
            int to_add = (fallback_count < max_count) ? fallback_count : max_count;
            for (int i = 0; i < to_add; i++) {
                strncpy(models[result_count], fallback_files[i].path, 256);
                result_count++;
            }
        }
    }

    return result_count;
}
