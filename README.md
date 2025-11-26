<div align="center">

# 🎮 点格棋 AlphaZero

**基于 AlphaZero 算法的点格棋人工智能**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](doc/LICENSE)

</div>

---

## 📖 项目简介

本项目实现了一个轻量级的 **AlphaZero** 算法，用于训练点格棋（Dots and Boxes）AI。项目基于 **OpenSpiel** 游戏引擎和 **PyTorch** 深度学习框架，可在单张 GPU 上高效训练。

### ✨ 核心特性

- 🧠 **完整的 AlphaZero 算法**: 自我对弈 + MCTS + 神经网络训练
- ⚡ **并行训练加速**: 多进程自我对弈，支持 GPU 批量推理
- 🏟️ **Arena 模型评估**: 新模型必须以 55%+ 胜率击败旧模型才能被接受
- 🎯 **双架构支持**: ResNet（稳定）和 CNN+Transformer 混合架构（实验性）
- 📊 **TensorBoard 集成**: 实时训练可视化和监控

---

## 🎯 点格棋规则

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Dots-and-boxes.svg" width="200" alt="点格棋"/>
</div>

- **棋盘**: 6×6 点阵（形成 5×5 个格子，共 25 个格子）
- **目标**: 通过连接相邻的点来"占领"格子，最终占领更多格子的玩家获胜
- **规则**:
  1. 双方轮流用线段连接横向或竖向相邻的两个点
  2. 当某个格子的四条边都被占满时，最后占边者获得该格子
  3. 占领格子后，该玩家可以继续下棋（连续行动）
  4. 所有格子被占领后，游戏结束，占领格子多者获胜

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      AlphaZero 训练循环                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │   自我对弈   │───▶│  神经网络   │───▶│   Arena     │        │
│   │  (并行执行)  │    │   训练     │    │  模型评估    │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │    MCTS     │    │   ResNet/   │    │   胜率      │        │
│   │  + 神经网络  │    │ Transformer │    │   ≥ 55%?   │        │
│   │    指导      │    │   网络      │    │             │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                │                │
│                                    ┌───────────┴───────────┐   │
│                                    ▼                       ▼   │
│                              ✅ 接受新模型           ❌ 保留旧模型 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **游戏环境** | `model/game.py` | OpenSpiel 封装，9 通道观察张量 (9×6×6) |
| **神经网络** | `model/model.py` | ResNet + SE 注意力 + GroupNorm，约 1775 万参数 |
| **MCTS** | `model/mcts.py` | AlphaZero 风格的蒙特卡洛树搜索，支持虚拟损失 |
| **训练教练** | `model/base_coach.py` | 训练循环、经验回放、Arena 集成 |
| **并行引擎** | `model/coach_parallel.py` | 多进程自我对弈，GPU 批量推理 |
| **模型评估** | `model/arena.py` | 新旧模型对战，胜率 ≥55% 才接受更新 |

---

## 🧠 算法详解

### AlphaZero 训练流程

```
每次迭代 (共 N 次):
    
    1️⃣ 自我对弈（并行）
       ├─ 使用当前最佳模型
       ├─ MCTS 搜索 (1200 次模拟/步)
       ├─ Dirichlet 噪声增加探索
       └─ 收集 (状态, 策略, 价值) 训练数据
    
    2️⃣ 神经网络训练
       ├─ 使用经验回放池（保留 60,000 样本）
       ├─ SGD + Momentum + Nesterov 优化器
       ├─ 策略损失 (交叉熵) + 价值损失 (均方误差)
       └─ 梯度裁剪防止梯度爆炸
    
    3️⃣ Arena 模型对战验证
       ├─ 新模型 vs 旧模型
       ├─ 对战 100 局（交替先后手保证公平）
       └─ 胜率 ≥ 55% → 接受新模型，否则保留旧模型
```

### 神经网络架构

**ResNet（默认）**:
- 输入: 9×6×6（9 通道状态表示）
- 特征提取: 128 滤波器卷积层 + 10 个残差块
- SE 注意力: Squeeze-and-Excitation 通道注意力
- 策略头: 输出 60 个动作的概率分布
- 价值头: 输出 [-1, 1] 的局面评估值

```python
# 模型架构概览
DotsAndBoxesNet(
    input_channels=9,      # 观察通道数
    num_filters=128,       # 卷积滤波器数
    num_res_blocks=10,     # 残差块数量
    dropout=0.3,           # Dropout 比率
    use_se=True            # SE 注意力
)
```

### MCTS 搜索

```python
# 关键超参数
num_simulations = 1200    # 每步搜索次数
cpuct = 1.25              # UCB 探索常数
dirichlet_alpha = 0.3     # 噪声参数
dirichlet_epsilon = 0.25  # 噪声比例
```

UCB 公式选择动作:
$$UCB(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}$$

---

## 📁 项目结构

```
DotsAndBoxes_AlphaZero/
├── 📂 model/                       # 核心算法模块
│   ├── game.py                     # OpenSpiel 游戏封装
│   ├── model.py                    # ResNet 神经网络（默认）
│   ├── model_transformer.py        # CNN+Transformer 变体
│   ├── mcts.py                     # MCTS 搜索算法
│   ├── base_coach.py               # 训练基类
│   ├── coach.py                    # 单进程训练
│   ├── coach_parallel.py           # 并行训练调度
│   ├── arena.py                    # 模型对战评估
│   ├── mcts_full_parallel_gpu.py   # 独立 GPU 并行引擎
│   ├── mcts_multiprocess_gpu.py    # 共享 GPU 并行引擎
│   └── mcts_concurrent_gpu.py      # 单进程并发引擎
│
├── 📂 cli/                         # 命令行工具
│   ├── train_parallel.py           # 🚀 主训练脚本
│   ├── evaluate_model.py           # 模型评估
│   ├── play_ultimate.py            # 终端人机对战
│   ├── play_gui.py                 # GUI 图形界面对战
│   ├── ai_interface.py             # AI 接口封装
│   └── quick_test.py               # 快速测试
│
├── 📂 config/                      # 配置文件
│   ├── config.yaml                 # 主配置
│   ├── game/dots_and_boxes.yaml    # 游戏配置
│   ├── model/resnet.yaml           # 模型配置
│   └── trainer/alphazero.yaml      # 训练配置
│
├── 📂 doc/                         # 文档
│   ├── ALPHAZERO_EXPLAINED.md      # 算法详解
│   ├── HOW_TO_TRAIN.md             # 训练指南
│   └── PLAY_GUIDE.md               # 对战指南
│
├── 📂 results/                     # 训练结果
│   ├── checkpoints/                # 模型检查点
│   └── logs/                       # TensorBoard 日志
│
└── 📂 tests/                       # 测试脚本
    └── test_system.py              # 系统测试
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+（需要 CUDA 支持）
- NVIDIA GPU（推荐 RTX 3080 或更高）

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/ironhxs/DotsAndBoxes_AlphaZero.git
cd DotsAndBoxes_AlphaZero

# 创建虚拟环境
conda create -n alphazero python=3.10
conda activate alphazero

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open_spiel numpy tqdm pyyaml tensorboard

# 验证安装
python tests/test_system.py
```

### 开始训练

```bash
# 使用默认配置开始训练
python cli/train_parallel.py

# 使用自定义配置
python cli/train_parallel.py --config config/config.yaml
```

### 评估模型

```bash
# 评估最佳模型
python cli/evaluate_model.py

# 人机对战
python cli/play_ultimate.py
```

---

## ⚙️ 配置说明

主要配置项（`config/config.yaml`）:

```yaml
# 训练配置
num_iterations: 200           # 总迭代次数
num_self_play_games: 200      # 每次迭代自我对弈局数
num_parallel_games: 30        # 并行游戏数

# MCTS 配置
num_simulations: 1200         # 每步 MCTS 模拟次数
cpuct: 1.25                   # 探索常数
dirichlet_alpha: 0.3          # Dirichlet 噪声

# 神经网络训练
batch_size: 4096              # 批大小
learning_rate: 0.001          # 学习率
train_epochs: 200             # 每次迭代训练轮数
replay_buffer_size: 60000     # 经验池大小

# Arena 评估
arena_compare: 100            # Arena 对战局数
update_threshold: 0.55        # 模型更新阈值
```

---

## 📊 训练效果

训练过程中可以使用 TensorBoard 监控:

```bash
tensorboard --logdir=results/logs
```

### 预期效果

| 训练阶段 | 迭代次数 | vs 随机策略 | vs 贪心策略 |
|----------|----------|-------------|-------------|
| 初期 | 1-10 | ~60% | ~40% |
| 中期 | 50-100 | ~90% | ~70% |
| 后期 | 150-200 | ~98% | ~85% |

---

## 📚 参考文献

- Silver, D., et al. (2017). **Mastering the game of Go without human knowledge.** *Nature*, 550(7676), 354-359.
- Silver, D., et al. (2018). **A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.** *Science*, 362(6419), 1140-1144.
- [OpenSpiel: A Framework for Reinforcement Learning in Games](https://github.com/deepmind/open_spiel)

---

## 📝 开源协议

本项目基于 MIT 协议开源，详见 [LICENSE](doc/LICENSE) 文件。

---

## 🙏 致谢

- [DeepMind OpenSpiel](https://github.com/deepmind/open_spiel) - 游戏环境框架
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) - AlphaZero 参考实现

---

<div align="center">

**⭐ 如果这个项目对你有帮助，欢迎 Star！⭐**

</div>
