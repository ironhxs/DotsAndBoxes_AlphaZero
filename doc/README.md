# 点格棋 AlphaZero# Dots and Boxes - AlphaZero



基于 **OpenSpiel** 的点格棋 AI 训练项目，使用 **AlphaZero** 算法，复用 GMD 项目的 conda 环境。基于 AlphaZero 算法的点格棋（Dots and Boxes）AI训练项目。



## 🎮 游戏规则（安徽省大赛）## 项目简介



- **棋盘**: 6×6 点阵（形成 5×5 格子）本项目实现了一个轻量级的 AlphaZero 算法，用于训练点格棋AI。项目使用 OpenSpiel 作为游戏环境，PyTorch 作为深度学习框架，可在单张 RTX 4090 显卡上高效训练。

- **玩法**: 

  1. 双方轮流连接相邻的点形成边### 点格棋规则

  2. 完成一个格子的四条边时，该玩家获得这个格子并继续行动

  3. 所有格子被占领后游戏结束- **棋盘**: 6×6 点阵（5×5 个格子）

  4. 占领格子多的玩家获胜- **目标**: 通过连接相邻的点来占领格子，占领更多格子的玩家获胜

- **规则**:

## 📦 环境配置  1. 双方轮流用线段连接横向或竖向相邻的两点

  2. 当一个格子的四条边都被占满时，最后一个占边者获得该格子

本项目复用 **GMD（Gradient-guided-Modality-Decoupling）** 的 conda 环境：  3. 占领格子后，该玩家继续下棋

  4. 游戏结束时，占领格子更多的玩家获胜

```bash

# 激活 GMD 环境## 项目结构

conda activate gmd

```

# 只需额外安装 OpenSpielDotsAndBoxes_AlphaZero/

pip install open_spiel├── __init__.py          # 包初始化

```├── config.py            # 配置文件（所有超参数）

├── game.py              # 游戏环境封装

GMD 环境已包含：PyTorch 1.10.1, CUDA 11.1, numpy, tqdm 等。├── model.py             # 神经网络模型

├── mcts.py              # 蒙特卡洛树搜索

## 📁 项目结构├── train.py             # 训练主循环

├── self_play.py         # 自我对弈

```├── replay_buffer.py     # 经验回放池

DotsAndBoxes_AlphaZero/├── evaluate.py          # 模型评估

├── game.py          # OpenSpiel 游戏封装├── play.py              # 人机对战

├── model.py         # 神经网络（ResNet）├── visualize.py         # 可视化工具

├── mcts.py          # 蒙特卡洛树搜索├── utils.py             # 工具函数

├── coach.py         # AlphaZero 训练教练├── main.py              # 训练入口

├── main.py          # 训练主程序├── requirements.txt     # 依赖列表

├── play.py          # 对战程序（人机/AI对战）├── checkpoints/         # 模型检查点目录

└── README.md        # 本文件├── logs/                # TensorBoard日志目录

```└── eval/                # 评估结果目录

```

## 🚀 快速开始

## 技术特点

### 1. 安装 OpenSpiel

### 1. 改进的 AlphaZero 算法

```bash

conda activate gmd- **蒙特卡洛树搜索（MCTS）**: 结合 UCB 公式和神经网络指导的前瞻性搜索

cd /HFUT_002/DotsAndBoxes_AlphaZero- **策略-价值网络**: ResNet 架构，同时输出策略和价值评估

pip install open_spiel- **自我对弈训练**: 通过自我对弈生成训练数据

```- **经验回放**: 使用大容量经验池提高样本利用率



### 2. 训练模型### 2. 优化策略



```bash- **混合精度训练（AMP）**: 加速训练，减少显存占用

python main.py- **梯度裁剪**: 稳定训练过程

```- **学习率调度**: 自适应调整学习率

- **数据增强**: 利用棋盘对称性扩充训练数据

训练参数（在 `main.py` 中可修改）：- **温度采样**: 前期探索，后期贪心

- `num_simulations`: 100（MCTS 模拟次数）

- `num_episodes`: 100（每次迭代的自我对弈局数）### 3. 工程实现

- `num_iterations`: 1000（总迭代次数）

- `batch_size`: 256- **模块化设计**: 清晰的代码结构，易于理解和修改

- **完善的日志**: TensorBoard 可视化训练过程

### 3. 人机对战- **检查点管理**: 自动保存和清理模型检查点

- **命令行接口**: 灵活的训练和对战选项

```bash

python play.py## 快速开始

```

### 1. 安装依赖

### 4. 观看 AI 自我对战

```bash

```bashcd DotsAndBoxes_AlphaZero

python play.py aipip install -r requirements.txt

``````



## 🧠 模型架构**注意**: 安装 OpenSpiel 可能需要额外步骤，请参考 [OpenSpiel 官方文档](https://github.com/deepmind/open_spiel)。



**输入**: (3, 6, 6) 张量### 2. 开始训练

- Channel 0: 水平边占据情况

- Channel 1: 垂直边占据情况```bash

- Channel 2: 格子归属（1=己方，-1=对方，0=未占）# 从头开始训练

python -m DotsAndBoxes_AlphaZero.main

**网络结构**: ResNet

- 初始卷积层# 从检查点恢复训练

- 5 个残差块（每块 2 个卷积层）python -m DotsAndBoxes_AlphaZero.main --resume

- 64 个卷积核

# 自定义参数

**输出**: python -m DotsAndBoxes_AlphaZero.main --iterations 500 --batch-size 256 --lr 0.001

- **Policy Head**: 60 个动作的概率分布```

- **Value Head**: 局面评估值 (-1 到 1)

### 3. 监控训练

## 🔄 AlphaZero 训练流程

在另一个终端启动 TensorBoard：

1. **自我对弈**: AI 自己和自己下棋，生成训练数据

   - 使用 MCTS 搜索提升决策质量```bash

   - 添加 Dirichlet 噪声增加探索tensorboard --logdir=DotsAndBoxes_AlphaZero/logs

```

2. **数据收集**: 保存 (状态, MCTS策略, 最终结果)

然后在浏览器中访问 `http://localhost:6006`

3. **神经网络训练**: 

   - 策略损失：交叉熵### 4. 人机对战

   - 价值损失：均方误差

```bash

4. **迭代优化**: 重复上述过程 1000 次# 使用训练好的模型对战

python -m DotsAndBoxes_AlphaZero.play --checkpoint latest.pth

## ⚙️ 关键参数说明

# 人类先手

| 参数 | 默认值 | 说明 |python -m DotsAndBoxes_AlphaZero.play --human-first

|------|--------|------|

| `num_simulations` | 100 | MCTS每步模拟次数，越大越强但越慢 |# 调整 AI 强度（减少 MCTS 模拟次数）

| `cpuct` | 1.0 | UCB探索常数，控制探索vs利用 |python -m DotsAndBoxes_AlphaZero.play --simulations 50

| `temp_threshold` | 15 | 前N步使用温度采样，后面贪心 |```

| `dirichlet_alpha` | 0.3 | Dirichlet噪声参数 |

| `num_episodes` | 100 | 每次迭代的对局数 |### 5. 可视化结果

| `batch_size` | 256 | 训练批次大小 |

| `lr` | 0.001 | 学习率 |```bash

# 绘制训练曲线

## 💻 硬件需求python -m DotsAndBoxes_AlphaZero.visualize



- **GPU**: NVIDIA 4090（或其他支持 CUDA 11.1+ 的 GPU）# 只显示统计信息

- **显存**: 至少 8GBpython -m DotsAndBoxes_AlphaZero.visualize --stats

- **预计训练时间**: ```

  - 100 次迭代：约 2-3 小时

  - 1000 次迭代：约 24-48 小时## 配置说明



## 📊 监控训练所有超参数都在 `config.py` 中定义，主要配置项：



检查点保存在 `checkpoints/` 目录：### 游戏配置

- `latest.pth`: 最新模型- `BOARD_SIZE`: 棋盘大小（默认 6×6 点阵）

- `checkpoint_N.pth`: 第N次迭代的模型

### MCTS 配置

可以随时中断训练，下次加载 `latest.pth` 继续。- `NUM_SIMULATIONS`: 每步 MCTS 模拟次数（默认 200）

- `CPUCT`: 探索常数（默认 1.5）

## 📚 相关论文- `TEMP_THRESHOLD`: 温度采样阈值（默认前 15 步）



- [Mastering the game of Go without human knowledge (Nature 2017)](https://www.nature.com/articles/nature24270)### 神经网络配置

- [A general reinforcement learning algorithm (Science 2018)](https://science.sciencemag.org/content/362/6419/1140)- `NUM_RESIDUAL_BLOCKS`: 残差块数量（默认 6）

- `NUM_FILTERS`: 卷积核数量（默认 128）

## 🙏 致谢- `DROPOUT`: Dropout 比例（默认 0.3）



- [OpenSpiel](https://github.com/deepmind/open_spiel) - DeepMind 开源游戏框架### 训练配置

- GMD 项目 - 提供 conda 环境基础- `NUM_ITERATIONS`: 总训练迭代次数（默认 1000）

- `NUM_SELF_PLAY_GAMES`: 每次迭代的自我对弈局数（默认 50）

## 📝 License- `BATCH_SIZE`: 批次大小（默认 512）

- `LEARNING_RATE`: 学习率（默认 0.001）

MIT License- `REPLAY_BUFFER_SIZE`: 经验池大小（默认 50000）


## 性能优化建议

### 单卡 4090 优化配置

```python
# 在 config.py 中调整
NUM_SIMULATIONS = 150          # 减少模拟次数加速训练
BATCH_SIZE = 512               # 根据显存调整
NUM_SELF_PLAY_GAMES = 40       # 减少对弈局数
TRAIN_STEPS_PER_ITERATION = 80 # 减少训练步数
USE_AMP = True                 # 开启混合精度
```

### 多卡训练（待实现）

目前暂不支持多卡训练，但可以通过以下方式扩展：
- 使用 `torch.nn.DataParallel` 进行数据并行
- 使用 `torch.distributed` 进行分布式训练

## 项目亮点

1. **完整的 AlphaZero 实现**: 包含所有核心组件（MCTS、神经网络、自我对弈）
2. **高度优化**: 混合精度训练、梯度裁剪、学习率调度等
3. **易于使用**: 清晰的代码结构和完善的文档
4. **可扩展性**: 模块化设计，易于添加新功能
5. **实用工具**: 评估、对战、可视化等完整的工具链

## 常见问题

### Q: 训练需要多长时间？

A: 在 RTX 4090 上，默认配置下大约需要 24-48 小时训练 1000 次迭代。可以通过减少 `NUM_SIMULATIONS` 和 `NUM_SELF_PLAY_GAMES` 来加速。

### Q: 显存不足怎么办？

A: 减小 `BATCH_SIZE` 和 `NUM_FILTERS`，或者关闭混合精度训练（`USE_AMP = False`）。

### Q: 如何评估模型性能？

A: 使用 `evaluate.py` 让新旧模型对战，或者使用 `play.py` 与人类玩家对战。

### Q: 可以用于其他游戏吗？

A: 可以！只需修改 `game.py` 来适配 OpenSpiel 中的其他游戏即可。

## 未来改进方向

- [ ] 支持多卡并行训练
- [ ] 实现棋盘的图形化可视化
- [ ] 添加更多的对称性数据增强
- [ ] 实现在线评估和模型选择
- [ ] 支持不同棋盘大小
- [ ] 添加 Web 界面进行在线对战

## 参考文献

- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) - AlphaGo Zero 论文
- [A general reinforcement learning algorithm that masters chess, shogi, and Go](https://science.sciencemag.org/content/362/6419/1140) - AlphaZero 论文
- [OpenSpiel](https://github.com/deepmind/open_spiel) - 游戏环境库

## 许可证

MIT License

## 作者

ironhxs

---

祝训练顺利！如有问题，欢迎提 Issue。
