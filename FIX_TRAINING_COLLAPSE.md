# 训练崩溃修复方案

## 问题诊断结果

### 核心问题：训练崩溃（Training Collapse）

**症状：**
- ✅ 训练累计胜率：78.9%
- ❌ 评估胜率：7.8%（最近100回合）
- ❌ 推理时一直选择IDLE动作
- ❌ 所有动作Q值几乎相同（70.60~70.73，标准差0.043）

**根本原因：**
1. **存活奖励过高**（0.02/帧）→ Q值膨胀到+70
2. **模型无法区分动作价值** → Q值差异<0.12
3. **后期训练崩溃** → 前期有效，后期退化

---

## 修复方案

### 方案A：调整奖励函数（推荐，治本）

#### 步骤1：降低存活奖励

编辑 `src/ai_interface.c` 第92行：

```c
// 原代码
reward += 0.02f;  // ❌ 过高，导致Q值膨胀

// 修改为
reward += 0.005f;  // ✅ 降低到1/4，减少Q值膨胀
```

**原理：**
- 原来：60fps × 0.02 = 1.2分/秒 → 存活250秒 = 300分（等于胜利奖励）
- 现在：60fps × 0.005 = 0.3分/秒 → 存活1000秒才300分（更合理）

#### 步骤2：增加动作奖励（鼓励行动）

在 `src/ai_interface.c` 第92行后添加：

```c
reward += 0.005f;  // 存活奖励

// ✅ 新增：鼓励主动行动，惩罚IDLE
// 获取上一帧动作（需要在函数参数中添加）
if (prev_action == ACTION_IDLE) {
    reward -= 0.02f;  // IDLE惩罚
} else if (prev_action >= ACTION_SHOOT_UP && prev_action <= ACTION_SHOOT_RIGHT) {
    reward += 0.03f;  // 射击奖励（鼓励进攻）
} else {
    reward += 0.01f;  // 移动奖励（鼓励探索）
}
```

**注意：** 此修改需要在 `ai_step` 函数中记录上一帧动作。

#### 步骤3：减少奖励塑形（可选）

在 `src/ai_interface.c` 第125-200行，将所有奖励塑形减半：

```c
// 距离奖励：0.5 → 0.25
if (min_enemy_dist < prev_min_dist) {
    reward += 0.25f;  // 原 0.5f
}

// 瞄准奖励：0.3 → 0.15
reward += 0.15f * align_score;  // 原 0.3f

// 射击奖励：0.5 → 0.25
reward += 0.25f;  // 原 0.5f
```

#### 步骤4：重新编译和训练

```bash
make clean && make
make train
# 选择 [0] 创建新模型（必须从头训练！）
```

**预期效果：**
- Q值范围：-10 ~ +30（更合理）
- 动作Q值差异：>1.0（能明显区分）
- 训练胜率稳定提升：不会后期崩溃

---

### 方案B：使用早期模型（临时，治标）

#### 检查历史模型

```bash
ls -lt saved_models/checkpoints/checkpoint_ep*.pth | head -10
```

#### 测试早期模型

```bash
make run-player
# 选择早期的checkpoint（如 checkpoint_ep50000.pth）
```

**原理：** 前期训练可能有效，后期崩溃。早期模型可能更好。

#### 找到最佳模型

创建测试脚本 `python/find_best_checkpoint.py`：

```python
#!/usr/bin/env python3
"""测试所有checkpoint找到最佳模型"""
import os
import glob
from diagnose_model import analyze_q_values

checkpoints = sorted(glob.glob("saved_models/checkpoints/checkpoint_ep*.pth"))

print("测试所有checkpoint...")
for ckpt in checkpoints:
    ep = int(ckpt.split('_ep')[-1].replace('.pth', ''))
    print(f"\n{'='*60}")
    print(f"测试: {ckpt} (回合 {ep})")
    analyze_q_values(ckpt)
```

运行：
```bash
./tank/bin/python3 python/find_best_checkpoint.py > best_model_report.txt
```

---

### 方案C：调整训练配置（重新训练）

#### 修改 `python/config.py`：

```python
MODEL_CONFIG = {
    # 1. 降低网络容量（避免过拟合）
    'hidden_dims': [256, 256, 128],  # 原 [512, 512, 256, 128]

    # 2. 更慢的epsilon衰减（保持探索）
    'epsilon_decay': 0.99995,  # 原 0.9999
    'epsilon_end': 0.2,  # 原 0.1（保持20%探索）

    # 3. 更小的学习率（稳定训练）
    'learning_rate': 0.0001,  # 原 0.0003

    # 4. 更频繁更新目标网络
    'target_update_freq': 50,  # 原 100
}

ENV_CONFIG = {
    # 从1个敌人开始（降低难度）
    'initial_enemies': 1,  # 保持
}
```

---

### 方案D：诊断工具（监控训练）

#### 使用已创建的诊断工具：

```bash
# 1. 检查当前模型Q值
./tank/bin/python3 python/diagnose_model.py

# 2. 检查特定checkpoint
./tank/bin/python3 python/diagnose_model.py --model saved_models/checkpoints/checkpoint_ep100000.pth
```

#### 训练中定期检查：

训练时，每5000回合执行：
```bash
./tank/bin/python3 python/diagnose_model.py --model saved_models/latest_model.pth
```

如果发现Q值标准差<0.5，立即停止训练（已经崩溃）。

---

## 推荐操作流程

### 快速修复（临时解决）

```bash
# 1. 测试早期模型
ls -lt saved_models/checkpoints/*.pth | head -5

# 2. 用早期模型对战
make run-player
# 选择 checkpoint_ep50000.pth 之类的早期模型
```

### 彻底修复（长期解决）

```bash
# 1. 备份当前模型
cp -r saved_models saved_models_backup_$(date +%Y%m%d)

# 2. 修改奖励函数
# 编辑 src/ai_interface.c，按方案A修改

# 3. 修改训练配置
# 编辑 python/config.py，按方案C修改

# 4. 重新编译
make clean && make

# 5. 开始新训练
make train
# 选择 [0] 创建新模型

# 6. 定期监控（每5000回合）
./tank/bin/python3 python/diagnose_model.py --model saved_models/latest_model.pth
```

---

## 预期效果对比

| 指标 | 当前（崩溃） | 修复后 |
|------|-------------|--------|
| Q值范围 | 70.60~70.73 | -10~+30 |
| Q值标准差 | 0.043 | >1.0 |
| 评估胜率 | 7.8% | 40-60% |
| 推理行为 | 一直IDLE | 主动进攻 |
| 训练稳定性 | 后期崩溃 | 持续改进 |

---

## 常见问题

**Q: 必须从头训练吗？**
A: 是的。奖励函数变化后，旧模型的Q值不再适用。

**Q: 训练需要多久？**
A: 预计5000-10000回合看到效果（纯文本模式约3-5小时）。

**Q: 如何判断训练正常？**
A: 定期运行诊断工具，确保：
- Q值标准差 > 0.5
- 评估胜率持续上升
- 推理时有多样化动作

**Q: 可以只修改配置不改代码吗？**
A: 可以尝试方案C，但效果可能有限。最好配合方案A。

---

## 技术原理

### 为什么会训练崩溃？

1. **Q值过高估计**
   - 存活奖励累积 → Q值膨胀到+70
   - 网络学会"预测高Q值" → 所有动作Q值都高

2. **Q值坍缩**
   - 所有动作Q值接近 → 无法区分优劣
   - epsilon降到0.1 → 停止探索 → 固化错误策略

3. **奖励函数缺陷**
   - 存活奖励主导 → 模型学会"不动就能得分"
   - 缺少动作奖励 → IDLE最安全（不会触发惩罚）

### 修复原理

1. **降低存活奖励**
   - 减少Q值膨胀
   - 让击杀/胜利奖励更重要

2. **增加动作奖励**
   - 惩罚IDLE
   - 鼓励射击和移动
   - 打破"不动就能得分"的错误策略

3. **保持探索**
   - 更慢的epsilon衰减
   - 更高的epsilon_end（0.2 vs 0.1）
   - 后期仍能探索新策略

---

**最后更新：** 2025-12-18
**相关文件：**
- 诊断工具：`python/diagnose_model.py`
- 奖励函数：`src/ai_interface.c`
- 训练配置：`python/config.py`
