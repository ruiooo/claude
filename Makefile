# Makefile for Tank Battle AI Training System

# 编译器设置
CC = gcc
PYTHON_INCLUDES = $(shell python3-config --includes)
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags)
CFLAGS = -Wall -O3 -fPIC -std=c11 $(PYTHON_INCLUDES)
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

# 安装Python依赖
install-python-deps:
	pip install -r requirements.txt
	@echo "Python依赖已安装"

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

# 运行训练（纯文本模式）
train:
	python3 python/train.py --no-visualize

# 运行训练（可视化模式）
train-vis:
	python3 python/train.py --visualize

# 帮助
help:
	@echo "Tank Battle AI Training System - Makefile"
	@echo ""
	@echo "可用命令:"
	@echo "  make                  - 编译所有目标"
	@echo "  make all              - 编译所有目标"
	@echo "  make clean            - 清理构建文件"
	@echo "  make rebuild          - 重新编译"
	@echo "  make install-deps     - 安装C依赖（需要sudo）"
	@echo "  make install-python-deps - 安装Python依赖"
	@echo "  make run-player       - 运行玩家对战模式"
	@echo "  make run-multiplayer-server  - 运行多人对战（服务器）"
	@echo "  make run-multiplayer-client  - 运行多人对战（客户端）"
	@echo "  make train            - 运行训练（纯文本模式，自动提示选择模型）"
	@echo "  make train-vis        - 运行训练（可视化模式，自动提示选择模型）"
	@echo ""
	@echo "注意: 运行玩家对战模式时，会自动提示选择AI模型版本"
	@echo ""

.PHONY: all clean rebuild install-deps install-python-deps run-player run-multiplayer-server run-multiplayer-client train train-vis help
