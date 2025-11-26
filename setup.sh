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

# 安装C依赖
echo ""
echo "检查C依赖 (SDL2)..."
NEED_INSTALL=false

# 检查SDL2库
if ! pkg-config --exists sdl2 2>/dev/null; then
    echo "  - SDL2未安装"
    NEED_INSTALL=true
else
    echo "  ✓ SDL2已安装: $(pkg-config --modversion sdl2)"
fi

# 检查SDL2_ttf库
if ! pkg-config --exists SDL2_ttf 2>/dev/null; then
    echo "  - SDL2_ttf未安装"
    NEED_INSTALL=true
else
    echo "  ✓ SDL2_ttf已安装: $(pkg-config --modversion SDL2_ttf)"
fi

# 检查gcc
if ! command -v gcc &> /dev/null; then
    echo "  - gcc未安装"
    NEED_INSTALL=true
else
    echo "  ✓ gcc已安装: $(gcc --version | head -n1)"
fi

# 检查make
if ! command -v make &> /dev/null; then
    echo "  - make未安装"
    NEED_INSTALL=true
else
    echo "  ✓ make已安装: $(make --version | head -n1)"
fi

# 如果需要安装
if $NEED_INSTALL; then
    echo ""
    echo "安装缺失的C依赖..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev gcc make
        echo "✓ C依赖安装完成"
    else
        echo "⚠ 无法自动安装SDL2，请手动安装"
    fi
else
    echo "✓ 所有C依赖已安装"
fi

# 安装Python依赖
echo ""
echo "检查Python依赖..."

# 检查PyTorch
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "  ✓ PyTorch已安装: $TORCH_VERSION"
    TORCH_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$TORCH_CUDA" = "True" ]; then
        echo "    CUDA支持: 是"
    else
        echo "    CUDA支持: 否"
    fi
else
    echo "  - PyTorch未安装"
    # 安装PyTorch
    if $HAS_CUDA; then
        echo ""
        echo "安装PyTorch (CUDA版本)..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo ""
        echo "安装PyTorch (CPU版本)..."
        pip3 install torch torchvision
    fi
fi

# 检查NumPy
if python3 -c "import numpy" 2>/dev/null; then
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    echo "  ✓ NumPy已安装: $NUMPY_VERSION"
else
    echo "  - NumPy未安装"
    echo ""
    echo "安装NumPy..."
    pip3 install numpy
fi

echo "✓ Python依赖检查完成"

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
echo "  1. 训练AI (纯文本模式):   make train"
echo "  2. 训练AI (可视化模式):   make train-vis"
echo "  3. 玩家对战模式:          make run-player"
echo ""
echo "详细文档请参阅 README.md"
echo ""
