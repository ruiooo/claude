# Makefile for Tank Battle AI Training System

# 编译器设置
CC = gcc

# 虚拟环境配置（可修改）
VENV_DIR ?= tank

# 检测虚拟环境并获取Python版本
ifneq (,$(wildcard ./$(VENV_DIR)/pyvenv.cfg))
    # 从pyvenv.cfg读取Python版本
    VENV_VERSION := $(shell grep "^version = " ./$(VENV_DIR)/pyvenv.cfg | cut -d'=' -f2 | tr -d ' ' | cut -d'.' -f1,2)
    PYTHON_CONFIG := python$(VENV_VERSION)-config
    $(info 使用虚拟环境Python $(VENV_VERSION)编译)
else
    PYTHON_CONFIG = python3-config
    $(info 未找到虚拟环境，使用系统Python)
endif

PYTHON_INCLUDES = $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags --embed 2>/dev/null || $(PYTHON_CONFIG) --ldflags)
CFLAGS = -Wall -O3 -fPIC -std=c11 $(PYTHON_INCLUDES) -DVENV_DIR=\"$(VENV_DIR)\"
LDFLAGS = -lSDL2 -lSDL2_ttf -lm -shared

# 目录
SRC_DIR = src
BUILD_DIR = build
LIB_DIR = .

# 源文件
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# 排除main.c、multiplayer_main.c和model_ai.c用于共享库
LIB_SOURCES = $(filter-out $(SRC_DIR)/main.c $(SRC_DIR)/multiplayer_main.c $(SRC_DIR)/model_ai.c, $(SOURCES))
LIB_OBJECTS = $(LIB_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# 玩家程序需要的对象文件（包含model_ai.o）
PLAYER_OBJECTS = $(filter-out $(BUILD_DIR)/multiplayer_main.o, $(OBJECTS))

# 多人对战程序需要的对象文件（排除main.o和model_ai.o）
MULTIPLAYER_OBJECTS = $(filter-out $(BUILD_DIR)/main.o $(BUILD_DIR)/model_ai.o, $(OBJECTS))

# 输出文件
SHARED_LIB = $(LIB_DIR)/libtankbattle.so
PLAYER_EXEC = tank_battle_player
MULTIPLAYER_EXEC = tank_battle_multiplayer

# 默认目标
all: $(SHARED_LIB) $(PLAYER_EXEC) $(MULTIPLAYER_EXEC)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译对象文件
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# 创建共享库（用于Python训练）
$(SHARED_LIB): $(LIB_OBJECTS)
	$(CC) -shared -o $@ $^ -lSDL2 -lSDL2_ttf -lm
	@echo "共享库已创建: $(SHARED_LIB)"

# 创建玩家可执行文件（需要Python支持）
$(PLAYER_EXEC): $(PLAYER_OBJECTS)
	$(CC) -o $@ $^ -lSDL2 -lSDL2_ttf -lm $(PYTHON_LDFLAGS)
	@echo "玩家程序已创建: $(PLAYER_EXEC)"

# 创建多人对战可执行文件（不需要Python支持）
$(MULTIPLAYER_EXEC): $(MULTIPLAYER_OBJECTS)
	$(CC) -o $@ $^ -lSDL2 -lSDL2_ttf -lm
	@echo "多人对战程序已创建: $(MULTIPLAYER_EXEC)"

# 清理
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(SHARED_LIB)
	rm -f $(PLAYER_EXEC)
	rm -f $(MULTIPLAYER_EXEC)
	@echo "清理完成"

# 重新编译
rebuild: clean all

# 安装依赖（Ubuntu/Debian）
install-deps:
	sudo apt-get update
	sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev gcc make
	@echo "C依赖已安装"

# 创建虚拟环境
venv:
	python3 -m venv $(VENV_DIR)
	./$(VENV_DIR)/bin/pip install --upgrade pip
	@echo "✓ 虚拟环境已创建: ./$(VENV_DIR)"
	@echo "  激活: source $(VENV_DIR)/bin/activate"
	@echo "  安装依赖: make install-python-deps"
	@echo ""
	@echo "提示: 如需自定义虚拟环境名称，使用: make venv VENV_DIR=your_env_name"

# 安装Python依赖（自动检测venv）
install-python-deps:
	@if [ -d "./$(VENV_DIR)" ]; then \
		echo "安装到虚拟环境 $(VENV_DIR)/"; \
		./$(VENV_DIR)/bin/pip install -r requirements.txt; \
	else \
		echo "安装到系统Python"; \
		pip install -r requirements.txt; \
	fi
	@echo "✓ Python依赖已安装"

# 运行玩家模式
run-player: $(PLAYER_EXEC)
	./$(PLAYER_EXEC)

# 运行多人对战模式 - 服务器
run-multiplayer-server: $(MULTIPLAYER_EXEC)
	./$(MULTIPLAYER_EXEC) server

# 运行多人对战模式 - 客户端
run-multiplayer-client: $(MULTIPLAYER_EXEC)
	@echo "请输入服务器IP地址，然后运行："
	@echo "  ./$(MULTIPLAYER_EXEC) client <服务器IP>"

# 运行训练（使用Ray并行训练，AlphaGo风格分阶段）⚡⚡⚡
train:
	@echo "🚀 启动Ray并行训练（AlphaGo风格分阶段）..."
	@if [ -d "./$(VENV_DIR)" ]; then \
		./$(VENV_DIR)/bin/python python/train_ray.py; \
	else \
		python3 python/train_ray.py; \
	fi

# 诊断人类数据质量
diagnose-human-data:
	@echo "🔍 诊断人类数据质量..."
	@if [ -d "./$(VENV_DIR)" ]; then \
		./$(VENV_DIR)/bin/python python/diagnose_human_data.py; \
	else \
		python3 python/diagnose_human_data.py; \
	fi

# 评估人类数据质量（详细分析）
eval-human-data:
	@echo "📊 评估人类数据质量..."
	@if [ -d "./$(VENV_DIR)" ]; then \
		./$(VENV_DIR)/bin/python python/evaluate_human_data.py; \
	else \
		python3 python/evaluate_human_data.py; \
	fi

# 清理低质量人类数据（质量评分<30，模拟运行）
clean-human-data-dry:
	@echo "🗑️  模拟清理低质量人类数据（质量评分<30）..."
	@if [ -d "./$(VENV_DIR)" ]; then \
		./$(VENV_DIR)/bin/python python/evaluate_human_data.py --clean-threshold 30 --dry-run; \
	else \
		python3 python/evaluate_human_data.py --clean-threshold 30 --dry-run; \
	fi

# 清理低质量人类数据（质量评分<30，实际执行）
clean-human-data:
	@echo "🗑️  清理低质量人类数据（质量评分<30）..."
	@echo "⚠️  此操作将移动低质量文件到 human_data/low_quality_backup/"
	@read -p "确认继续？(y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	@if [ -d "./$(VENV_DIR)" ]; then \
		./$(VENV_DIR)/bin/python python/evaluate_human_data.py --clean-threshold 30; \
	else \
		python3 python/evaluate_human_data.py --clean-threshold 30; \
	fi

# 严格清理人类数据（质量评分<20）
clean-human-data-strict:
	@echo "🗑️  严格清理低质量人类数据（质量评分<20）..."
	@echo "⚠️  此操作将移动低质量文件到 human_data/low_quality_backup/"
	@read -p "确认继续？(y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	@if [ -d "./$(VENV_DIR)" ]; then \
		./$(VENV_DIR)/bin/python python/evaluate_human_data.py --clean-threshold 20; \
	else \
		python3 python/evaluate_human_data.py --clean-threshold 20; \
	fi

# 安装Ray依赖
install-ray:
	@echo "安装Ray Core..."
	@if [ -d "./$(VENV_DIR)" ]; then \
		./$(VENV_DIR)/bin/pip install ray; \
	else \
		pip install ray; \
	fi
	@echo "✓ Ray Core已安装（训练将自动使用4-6倍加速）⚡"

# 帮助
help:
	@echo "Tank Battle AI Training System - Makefile"
	@echo ""
	@echo "编译命令:"
	@echo "  make                  - 编译所有目标"
	@echo "  make all              - 编译所有目标"
	@echo "  make clean            - 清理构建文件"
	@echo "  make rebuild          - 重新编译"
	@echo ""
	@echo "环境管理:"
	@echo "  make venv             - 创建Python虚拟环境"
	@echo "  make install-deps     - 安装C依赖（需要sudo）"
	@echo "  make install-python-deps - 安装Python依赖（自动检测venv）"
	@echo "  make install-ray      - 安装Ray依赖（训练加速必需）⚡"
	@echo ""
	@echo "运行模式:"
	@echo "  make run-player       - 运行玩家对战模式"
	@echo "  make run-multiplayer-server  - 运行多人对战（服务器）"
	@echo "  make run-multiplayer-client  - 运行多人对战（客户端）"
	@echo ""
	@echo "训练模式:"
	@echo "  make train            - Ray并行训练（极简模式，3倍慢epsilon衰减）⚡⚡⚡"
	@echo ""
	@echo "人类数据管理:"
	@echo "  make eval-human-data       - 评估人类数据质量（详细分析）"
	@echo "  make clean-human-data-dry  - 模拟清理低质量数据（质量<30）"
	@echo "  make clean-human-data      - 清理低质量数据（质量<30，需确认）"
	@echo "  make clean-human-data-strict - 严格清理数据（质量<20，需确认）"
	@echo ""
	@echo "虚拟环境配置:"
	@echo "  VENV_DIR=tank         - 默认虚拟环境名称"
	@echo "  make venv VENV_DIR=myenv   - 创建自定义名称的虚拟环境"
	@echo "  make VENV_DIR=myenv        - 使用自定义虚拟环境编译"
	@echo ""
	@echo "注意:"
	@echo "  - 程序会自动检测虚拟环境（tank, venv, .venv, env）"
	@echo "  - 训练前需要先运行 make install-ray 安装Ray依赖"
	@echo "  - Ray并行训练提供8个worker并行采样，训练速度提升4-6倍"
	@echo ""

.PHONY: all clean rebuild install-deps install-python-deps install-ray run-player run-multiplayer-server run-multiplayer-client train diagnose-human-data eval-human-data clean-human-data-dry clean-human-data clean-human-data-strict help
