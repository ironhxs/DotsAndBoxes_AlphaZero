# 训练改进完整指南

## 已完成的三大改进

### ✅ 1. Resume训练功能

**实现位置:** `model/base_coach.py` - `learn()`方法开头

**使用方法:**
```yaml
# config/config.yaml
resume: true                # 开启恢复训练
resume_file: "latest.pth"   # 或 "checkpoint_30.pth"
```

**工作流程:**
```
启动训练 → 检测resume=true → 加载checkpoint → 解析迭代号 → 从中断处继续
```

**示例:**
```bash
# 训练到第46次迭代时中断
^C

# 修改配置
vim config/config.yaml
# resume: true
# resume_file: "checkpoint_30.pth"  # 或 "latest.pth"

# 重启训练,自动从第31次(或47次)继续
/root/miniconda3/envs/gmd/bin/python cli/train_parallel.py
```

---

### ✅ 2. 完整的TensorBoard监控

**新增指标:**

#### 损失类 (Loss/*)
- `Loss/Policy` - 策略损失
- `Loss/Value` - 价值损失
- `Loss/Total` - 总损失
- `Loss/BestLoss` - 历史最佳损失

#### 训练类 (Training/*)
- `Training/LearningRate` - 实时学习率
- `Training/Epoch` - 当前epoch
- `Training/GradNorm` - 梯度范数(监控梯度爆炸/消失)

#### 早停类 (EarlyStopping/*)
- `EarlyStopping/PatienceCounter` - 耐心计数器
- `EarlyStopping/ImprovementNeeded` - 需要改进的阈值

#### Arena类 (Arena/*)
- `Arena/WinRate` - 新模型胜率
- `Arena/Accepted` - 是否接受新模型
- `Arena/Threshold` - 接受阈值
- `Arena/Improvement` - 相对随机策略的改进

**启动TensorBoard:**
```bash
tensorboard --logdir=results/logs --port=6006
# 浏览器访问: http://localhost:6006
```

**重点关注:**
- 如果`Loss/Total`平稳 → 可能收敛,检查`Arena/WinRate`
- 如果`Training/GradNorm`爆炸(>100) → 降低学习率
- 如果`Training/GradNorm`消失(<1e-5) → 检查模型架构

---

### ✅ 3. 改进的学习率调度

**新机制: Warmup + Cosine Annealing**

```python
# 前10%迭代: Warmup (线性增长)
Iteration 1-100:  lr = 0.001 * (iter / 100)
  - 第1次:  lr = 0.00001
  - 第50次: lr = 0.0005
  - 第100次:lr = 0.001

# 之后: Cosine退火 (每个iteration内)
Epoch 1-300: lr从当前值cosine衰减到0.01*lr
```

**为什么有效:**
- **Warmup**: 训练初期使用小学习率,避免破坏预训练权重
- **Cosine**: 平滑衰减,在收敛阶段更精细地调整

---

## Loss无法下降的原因分析

### 可能原因1: 学习率过大
**症状:** Loss在1.02附近震荡,波动>0.01  
**解决:** 降低学习率
```yaml
learning_rate: 0.0005  # 从0.001降低
```

### 可能原因2: Batch size过大
**症状:** Loss平稳但胜率不提升  
**解决:** 减小batch提高数据多样性
```yaml
batch_size: 4096  # 从7200降低到4096
```

### 可能原因3: MCTS质量不足
**症状:** 策略损失pi>0.9,价值损失v<0.05  
**解决:** 增加MCTS模拟次数
```yaml
num_simulations: 1200  # 从800提升到1200
```

### 可能原因4: 经验池数据过旧
**症状:** 训练后期loss不降反升  
**解决:** 减小replay buffer保留更新数据
```yaml
replay_buffer_size: 40000  # 从60000减少,只保留最近数据
```

### 可能原因5: 已经收敛
**症状:** Loss停在1.02,但Arena胜率持续>55%  
**判断:** 这是正常现象!Loss不是唯一指标,棋力才是

---

## 运行诊断工具

```bash
# 诊断当前训练状态
/root/miniconda3/envs/gmd/bin/python tests/diagnose_training.py
```

**输出包括:**
1. 梯度流检查(是否有梯度消失/爆炸)
2. 学习率合理性
3. 数据分布统计
4. 个性化改进建议

---

## 推荐的配置调整

### 场景1: Loss停滞,想快速突破
```yaml
learning_rate: 0.0005        # 降低学习率
batch_size: 4096             # 减小batch
num_simulations: 1200        # 增加MCTS质量
replay_buffer_size: 100000   # 更多数据多样性
patience: 15                 # 给更多尝试机会
```

### 场景2: 训练太慢,想加速
```yaml
train_epochs: 200            # 从300减少到200
patience: 5                  # 更激进的早停
num_simulations: 600         # 降低MCTS(牺牲质量换速度)
batch_size: 8192             # 增大batch(更快但可能欠拟合)
```

### 场景3: 已经很强,想冲击极限
```yaml
learning_rate: 0.0001        # 超低学习率精调
num_simulations: 2000        # 超高MCTS质量
replay_buffer_size: 200000   # 大容量经验池
patience: 30                 # 极度耐心
```

---

## 监控训练健康度的清单

| 指标 | 健康范围 | 异常表现 | 处理方法 |
|------|---------|---------|---------|
| Loss/Total | 0.5-1.5 | >2.0 | 降低学习率 |
| Loss/Policy | 0.3-1.0 | >1.5 | 增加MCTS模拟 |
| Loss/Value | 0.01-0.5 | >1.0 | 检查价值标签 |
| GradNorm | 0.1-10 | >100 | 梯度裁剪/降低lr |
| Arena/WinRate | >0.52 | <0.48 | 模型退化,回滚 |
| LearningRate | 1e-4~1e-3 | >0.01 | 过大,减小 |

---

## 快速FAQ

### Q1: Loss=1.02算高吗?
**A:** 对于Dots&Boxes,这是正常水平。关键看Arena胜率,不要只盯Loss。

### Q2: 为什么策略损失(pi)远大于价值损失(v)?
**A:** 正常!策略有60个动作,价值只有1个标量,天然损失尺度不同。

### Q3: Early Stopping会不会过早停止?
**A:** `patience=10`意味着连续10个epoch无改进才停,很保守了。

### Q4: Resume会丢失训练历史吗?
**A:** 不会,TensorBoard日志会累积。但replay buffer会重新开始积累。

### Q5: 什么时候应该停止训练?
**A:** 
- Arena胜率连续5次<50% → 模型崩溃,停止
- Arena胜率稳定在55-60% → 已收敛,可停止
- 训练100次迭代后loss无变化 → 收敛,停止

---

## 立即应用这些改进

**当前训练继续运行,从下次迭代自动生效**

1. ✅ 早停已启用 - 下次训练会在~30 epochs停止
2. ✅ TensorBoard增强 - 打开tensorboard可看到新指标
3. ✅ Resume已就绪 - 下次中断设置`resume: true`即可

**建议的下一步:**
```bash
# 1. 打开TensorBoard观察
tensorboard --logdir=results/logs --bind_all

# 2. 运行诊断工具
/root/miniconda3/envs/gmd/bin/python tests/diagnose_training.py

# 3. 根据诊断结果调整config.yaml
vim config/config.yaml

# 4. 让当前训练自然结束,新配置下次迭代生效
```
