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
if pkg-config --exists sdl2 2>/dev/null; then
    SDL2_VERSION=$(pkg-config --modversion sdl2)
    echo "✓ SDL2已安装: $SDL2_VERSION"
    SDL2_INSTALLED=true
fi

# 检查SDL2_ttf是否已安装
SDL2_TTF_INSTALLED=false
if pkg-config --exists SDL2_ttf 2>/dev/null; then
    SDL2_TTF_VERSION=$(pkg-config --modversion SDL2_ttf)
    echo "✓ SDL2_ttf已安装: $SDL2_TTF_VERSION"
    SDL2_TTF_INSTALLED=true
fi

# 检查gcc是否已安装
GCC_INSTALLED=false
if command -v gcc &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -n1)
    echo "✓ GCC已安装: $GCC_VERSION"
    GCC_INSTALLED=true
fi

# 检查make是否已安装
MAKE_INSTALLED=false
if command -v make &> /dev/null; then
    MAKE_VERSION=$(make --version | head -n1)
    echo "✓ Make已安装: $MAKE_VERSION"
    MAKE_INSTALLED=true
fi

# 安装缺少的依赖
if ! $SDL2_INSTALLED || ! $SDL2_TTF_INSTALLED || ! $GCC_INSTALLED || ! $MAKE_INSTALLED; then
    if command -v apt-get &> /dev/null; then
        echo "安装缺少的C依赖..."
        PACKAGES=""
        if ! $SDL2_INSTALLED; then
            PACKAGES="$PACKAGES libsdl2-dev"
        fi
        if ! $SDL2_TTF_INSTALLED; then
            PACKAGES="$PACKAGES libsdl2-ttf-dev"
        fi
        if ! $GCC_INSTALLED; then
            PACKAGES="$PACKAGES gcc"
        fi
        if ! $MAKE_INSTALLED; then
            PACKAGES="$PACKAGES make"
        fi

        sudo apt-get update
        sudo apt-get install -y $PACKAGES
        echo "✓ C依赖安装完成"
    else
        echo "⚠ 无法自动安装SDL2，请手动安装"
    fi
else
    echo "✓ 所有C依赖已安装"
fi

# 创建和配置虚拟环境
echo ""
echo "配置Python虚拟环境..."

VENV_DIR="tank"

if [ -d "$VENV_DIR" ]; then
    echo "✓ 虚拟环境已存在: $VENV_DIR"
else
    echo "创建虚拟环境: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    echo "✓ 虚拟环境创建完成"
fi

# 使用虚拟环境的pip
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python3"

# 升级pip
echo "升级pip..."
$PIP install --upgrade pip

# 检查并安装Python依赖
echo ""
echo "检查Python依赖（虚拟环境）..."

# 检查PyTorch是否已安装
if $PYTHON -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$($PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "✓ PyTorch已安装: $TORCH_VERSION"
else
    echo "安装PyTorch到虚拟环境..."
    if $HAS_CUDA; then
        echo "安装PyTorch (CUDA版本)..."
        $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "安装PyTorch (CPU版本)..."
        $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    echo "✓ PyTorch安装完成"
fi

# 检查NumPy是否已安装
if $PYTHON -c "import numpy" 2>/dev/null; then
    NUMPY_VERSION=$($PYTHON -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    echo "✓ NumPy已安装: $NUMPY_VERSION"
else
    echo "安装NumPy到虚拟环境..."
    $PIP install numpy
    echo "✓ NumPy安装完成"
fi

# 检查tqdm是否已安装
if $PYTHON -c "import tqdm" 2>/dev/null; then
    TQDM_VERSION=$($PYTHON -c "import tqdm; print(tqdm.__version__)" 2>/dev/null)
    echo "✓ tqdm已安装: $TQDM_VERSION"
else
    echo "安装tqdm到虚拟环境..."
    $PIP install tqdm
    echo "✓ tqdm安装完成"
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
$PYTHON -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
echo ""
echo "虚拟环境信息:"
echo "  Python: $($PYTHON --version)"
echo "  位置: $(pwd)/$VENV_DIR"

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
