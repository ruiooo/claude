#!/bin/bash

# 坦克大战AI训练系统 - 一键安装脚本

set -e

echo "========================================"
echo "  坦克大战 AI 训练系统 - 安装脚本"
echo "========================================"
echo ""

# 检测操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ 检测到Linux系统"
else
    echo "✗ 警告: 此脚本仅在Linux系统上测试过"
fi

# 检查Python版本
echo ""
echo "检查Python版本..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "✓ Python版本: $(python3 --version)"
else
    echo "✗ 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查CUDA
echo ""
echo "检查CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | cut -d',' -f1)
    echo "✓ CUDA版本: $CUDA_VERSION"
    HAS_CUDA=true
else
    echo "⚠ 未检测到CUDA，将使用CPU训练（速度较慢）"
    HAS_CUDA=false
fi

# 检查GPU
if $HAS_CUDA; then
    echo ""
    echo "检查GPU..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader
    fi
fi

# 检查并安装C依赖
echo ""
echo "检查C依赖 (SDL2)..."

# 检查SDL2是否已安装
SDL2_INSTALLED=false
if pkg-config --exists sdl2 sdl2_ttf 2>/dev/null; then
    SDL2_VERSION=$(pkg-config --modversion sdl2)
    SDL2_TTF_VERSION=$(pkg-config --modversion sdl2_ttf)
    echo "✓ SDL2已安装: $SDL2_VERSION"
    echo "✓ SDL2_ttf已安装: $SDL2_TTF_VERSION"
    SDL2_INSTALLED=true
elif command -v sdl2-config &> /dev/null; then
    SDL2_VERSION=$(sdl2-config --version)
    echo "✓ SDL2已安装: $SDL2_VERSION"
    SDL2_INSTALLED=true
fi

# 检查GCC和Make
if command -v gcc &> /dev/null && command -v make &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -n1)
    echo "✓ GCC已安装: $GCC_VERSION"
else
    SDL2_INSTALLED=false
fi

# 如果未安装，则安装
if ! $SDL2_INSTALLED; then
    echo "安装C依赖..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev gcc make
        echo "✓ C依赖安装完成"
    else
        echo "⚠ 无法自动安装SDL2，请手动安装"
    fi
else
    echo "✓ 所有C依赖已就绪，跳过安装"
fi

# 安装Python依赖
echo ""
echo "安装Python依赖..."

# 安装PyTorch
if $HAS_CUDA; then
    echo "安装PyTorch (CUDA版本)..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "安装PyTorch (CPU版本)..."
    pip3 install torch torchvision
fi

# 安装其他依赖
echo "安装其他Python包..."
pip3 install numpy

echo "✓ Python依赖安装完成"

# 编译C代码
echo ""
echo "编译游戏引擎..."
make clean
make all

if [ -f "libtankbattle.so" ] && [ -f "tank_battle_player" ]; then
    echo "✓ 编译成功"
else
    echo "✗ 编译失败"
    exit 1
fi

# 创建必要目录
echo ""
echo "创建目录..."
mkdir -p saved_models/checkpoints
mkdir -p saved_models/history
mkdir -p logs
echo "✓ 目录创建完成"

# 测试
echo ""
echo "运行测试..."
python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

# 完成
echo ""
echo "========================================"
echo "  安装完成!"
echo "========================================"
echo ""
echo "快速开始:"
echo "  1. 训练AI (纯文本模式):        make train"
echo "  2. 训练AI (可视化模式):        make train-vis"
echo "  3. 玩家对战简单AI:             make run-player"
echo "  4. 玩家对战训练后的AI:         make run-player-vs-ai"
echo ""
echo "注意: 玩家对战AI模式需要安装pygame:"
echo "  pip install pygame"
echo ""
echo "更多命令请运行: make help"
echo "详细文档请参阅 README.md"
echo ""
