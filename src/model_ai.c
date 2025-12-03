/*
 * model_ai.c - 基于Python训练模型的AI控制器实现
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

// 全局Python对象
static PyObject* g_torch_module = NULL;
static PyObject* g_model_module = NULL;
static PyObject* g_config_module = NULL;
static bool g_system_initialized = false;

// 初始化Python解释器和必要模块
bool model_ai_system_init(void) {
    if (g_system_initialized) {
        return true;
    }

    // 检查虚拟环境（优先使用编译时配置，然后检查常见名称）
    struct stat st;
    const char* venv_candidates[] = {
#ifdef VENV_DIR
        "./" VENV_DIR,
#endif
        "./venv",
        "./.venv",
        "./env",
        NULL
    };

    const char* found_venv = NULL;
    char venv_abs_path[PATH_MAX];
    for (int i = 0; venv_candidates[i] != NULL; i++) {
        char python_path[512];
        snprintf(python_path, sizeof(python_path), "%s/bin/python3", venv_candidates[i]);
        if (stat(python_path, &st) == 0) {
            found_venv = venv_candidates[i];
            // 获取绝对路径
            if (realpath(found_venv, venv_abs_path) == NULL) {
                fprintf(stderr, "获取虚拟环境绝对路径失败\n");
                return false;
            }
            break;
        }
    }

    if (found_venv) {
        // 设置Python程序路径为虚拟环境的Python可执行文件（使用绝对路径）
        char python_exe[PATH_MAX];
        snprintf(python_exe, sizeof(python_exe), "%s/bin/python3", venv_abs_path);
        wchar_t python_exe_wide[PATH_MAX];
        mbstowcs(python_exe_wide, python_exe, PATH_MAX);
        Py_SetProgramName(python_exe_wide);
        printf("使用虚拟环境: %s\n", venv_abs_path);
    } else {
        printf("未找到虚拟环境，使用系统Python\n");
    }

    // 初始化Python解释器
    Py_Initialize();
    if (!Py_IsInitialized()) {
        fprintf(stderr, "Python解释器初始化失败\n");
        return false;
    }

    // 获取当前工作目录
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        fprintf(stderr, "获取当前工作目录失败\n");
        Py_Finalize();
        return false;
    }

    // 配置Python路径
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("import os");
    PyRun_SimpleString("import glob");

    // 如果使用虚拟环境，重建sys.path：基于实际运行的Python版本
    if (found_venv) {
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

    // 添加项目python目录到Python路径
    char add_python_dir_cmd[PATH_MAX];
    snprintf(add_python_dir_cmd, sizeof(add_python_dir_cmd),
             "if '%s/python' not in sys.path: sys.path.insert(0, '%s/python')",
             cwd, cwd);
    PyRun_SimpleString(add_python_dir_cmd);

    char add_cwd_cmd[PATH_MAX];
    snprintf(add_cwd_cmd, sizeof(add_cwd_cmd),
             "if '%s' not in sys.path: sys.path.insert(0, '%s')",
             cwd, cwd);
    PyRun_SimpleString(add_cwd_cmd);

    // 打印Python路径和诊断信息
    printf("Python sys.path:\n");
    PyRun_SimpleString("for p in sys.path[:5]: print('  -', p)");

    // 诊断信息：检查_ctypes模块
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

    // 导入torch
    g_torch_module = PyImport_ImportModule("torch");
    if (!g_torch_module) {
        fprintf(stderr, "导入torch模块失败\n");
        if (found_venv) {
            fprintf(stderr, "请确保torch已安装: %s/bin/pip install torch\n", venv_abs_path);
        } else {
            fprintf(stderr, "请确保torch已安装: pip install torch\n");
        }
        PyErr_Print();
        Py_Finalize();
        return false;
    }

    // 导入model模块
    g_model_module = PyImport_ImportModule("model");
    if (!g_model_module) {
        fprintf(stderr, "导入model模块失败\n");
        PyErr_Print();
        Py_XDECREF(g_torch_module);
        Py_Finalize();
        return false;
    }

    // 导入config模块
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

// 将游戏状态转换为观察向量（简化版）
static void game_state_to_observation(const GameState* game, int tank_id,
                                      float* obs, int* obs_size) {
    // 这是一个简化的状态表示
    // 实际应该与训练时的状态表示一致
    int idx = 0;

    const Tank* self_tank = &game->tanks[tank_id];

    // 自身坦克信息 (8维)
    obs[idx++] = self_tank->x / (float)game->map_width;
    obs[idx++] = self_tank->y / (float)game->map_height;
    obs[idx++] = self_tank->vx / 5.0f;
    obs[idx++] = self_tank->vy / 5.0f;
    obs[idx++] = (float)self_tank->direction / 3.0f;
    obs[idx++] = self_tank->health / 100.0f;
    obs[idx++] = (self_tank->shoot_cooldown == 0) ? 1.0f : 0.0f;  // 是否可以射击
    obs[idx++] = self_tank->alive ? 1.0f : 0.0f;

    // 敌人信息（简化：只考虑最近的敌人）
    float min_dist = 1e9;
    const Tank* nearest_enemy = NULL;
    for (int i = 0; i < game->tank_count; i++) {
        if (i == tank_id || !game->tanks[i].alive) continue;

        float dx = game->tanks[i].x - self_tank->x;
        float dy = game->tanks[i].y - self_tank->y;
        float dist = dx * dx + dy * dy;

        if (dist < min_dist) {
            min_dist = dist;
            nearest_enemy = &game->tanks[i];
        }
    }

    if (nearest_enemy) {
        obs[idx++] = nearest_enemy->x / (float)game->map_width;
        obs[idx++] = nearest_enemy->y / (float)game->map_height;
        obs[idx++] = nearest_enemy->vx / 5.0f;
        obs[idx++] = nearest_enemy->vy / 5.0f;
        obs[idx++] = (float)nearest_enemy->direction / 3.0f;
        obs[idx++] = nearest_enemy->health / 100.0f;
    } else {
        // 没有敌人时填充0
        for (int i = 0; i < 6; i++) obs[idx++] = 0.0f;
    }

    // 子弹信息（简化：只考虑最近的子弹）
    min_dist = 1e9;
    const Bullet* nearest_bullet = NULL;
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!game->bullets[i].active) continue;

        float dx = game->bullets[i].x - self_tank->x;
        float dy = game->bullets[i].y - self_tank->y;
        float dist = dx * dx + dy * dy;

        if (dist < min_dist) {
            min_dist = dist;
            nearest_bullet = &game->bullets[i];
        }
    }

    if (nearest_bullet) {
        obs[idx++] = nearest_bullet->x / (float)game->map_width;
        obs[idx++] = nearest_bullet->y / (float)game->map_height;
        obs[idx++] = nearest_bullet->vx / 10.0f;
        obs[idx++] = nearest_bullet->vy / 10.0f;
    } else {
        for (int i = 0; i < 4; i++) obs[idx++] = 0.0f;
    }

    *obs_size = idx;
}

// 获取模型AI的决策
TankAction model_ai_get_action(ModelAI* ai, const GameState* game, int tank_id) {
    if (!ai->initialized || !ai->py_agent) {
        return ACTION_IDLE;
    }

    // 获取游戏状态
    float obs[128];
    int obs_size;
    game_state_to_observation(game, tank_id, obs, &obs_size);

    // 创建numpy数组（通过Python列表）
    PyObject* obs_list = PyList_New(obs_size);
    for (int i = 0; i < obs_size; i++) {
        PyList_SetItem(obs_list, i, PyFloat_FromDouble(obs[i]));
    }

    // 转换为tensor
    PyObject* torch_tensor_func = PyObject_GetAttrString(g_torch_module, "tensor");
    PyObject* tensor_args = PyTuple_Pack(1, obs_list);
    PyObject* state_tensor = PyObject_CallObject(torch_tensor_func, tensor_args);

    Py_DECREF(obs_list);
    Py_DECREF(tensor_args);
    Py_DECREF(torch_tensor_func);

    if (!state_tensor) {
        fprintf(stderr, "创建state tensor失败\n");
        PyErr_Print();
        return ACTION_IDLE;
    }

    // 调用select_action方法
    PyObject* select_action_method = PyObject_GetAttrString((PyObject*)ai->py_agent,
                                                            "select_action");
    if (!select_action_method) {
        fprintf(stderr, "获取select_action方法失败\n");
        PyErr_Print();
        Py_DECREF(state_tensor);
        return ACTION_IDLE;
    }

    // 创建参数：(state, training=False)
    PyObject* action_args = PyTuple_Pack(1, state_tensor);
    PyObject* action_kwargs = PyDict_New();
    PyDict_SetItemString(action_kwargs, "training", Py_False);

    PyObject* action_result = PyObject_Call(select_action_method, action_args, action_kwargs);

    Py_DECREF(state_tensor);
    Py_DECREF(select_action_method);
    Py_DECREF(action_args);
    Py_DECREF(action_kwargs);

    if (!action_result) {
        fprintf(stderr, "调用select_action失败\n");
        PyErr_Print();
        return ACTION_IDLE;
    }

    // 获取动作值
    long action = PyLong_AsLong(action_result);
    Py_DECREF(action_result);

    // 映射到TankAction
    if (action >= 0 && action <= ACTION_SHOOT_RIGHT) {
        return (TankAction)action;
    }

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

// 列出可用的模型文件（按时间排序，最新的在前，最多返回5个）
int model_ai_list_models(char models[][256], int max_count) {
    ModelFile model_files[100];
    int count = 0;

    // 搜索的目录列表
    const char* search_dirs[] = {
        "models",
        "checkpoints",
        "saved_models/checkpoints",
        NULL
    };

    // 遍历所有搜索目录
    for (int d = 0; search_dirs[d] != NULL && count < 100; d++) {
        DIR* dir = opendir(search_dirs[d]);
        if (!dir) continue;

        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL && count < 100) {
            if (strstr(entry->d_name, ".pth") || strstr(entry->d_name, ".pt")) {
                // 构建完整路径
                snprintf(model_files[count].path, 256, "%s/%s", search_dirs[d], entry->d_name);

                // 获取文件修改时间
                struct stat st;
                if (stat(model_files[count].path, &st) == 0) {
                    model_files[count].mtime = st.st_mtime;
                    count++;
                }
            }
        }
        closedir(dir);
    }

    if (count == 0) {
        return 0;
    }

    // 按修改时间排序（最新的在前）
    qsort(model_files, count, sizeof(ModelFile), compare_models_by_time);

    // 只返回最多max_count个（通常是5个）
    int return_count = (count < max_count) ? count : max_count;
    for (int i = 0; i < return_count; i++) {
        strncpy(models[i], model_files[i].path, 256);
    }

    return return_count;
}
