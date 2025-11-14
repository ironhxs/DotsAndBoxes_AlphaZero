# Arena 对决逻辑说明

## 对决双方
- **新模型**: `self.nnet` (当前训练的模型)
- **旧模型**: `self.previous_nnet` (上一次被接受的baseline模型)

## 对决流程

### 1. 初始状态
```python
# 训练开始时
self.nnet = DotsAndBoxesNet(...)           # 当前模型
self.previous_nnet = deepcopy(self.nnet)   # baseline = 当前模型
```

### 2. 每次迭代
```
1. 自我对弈: 用self.nnet生成训练数据
2. 训练: 更新self.nnet权重
3. Arena对决: self.nnet vs self.previous_nnet
```

### 3. Arena决策

**胜率 ≥ 55%** (接受新模型):
```python
self.previous_nnet = deepcopy(self.nnet)  # 更新baseline
```
- ✅ 新模型成为新的baseline
- ✅ 下次Arena将与这个新baseline比较

**胜率 < 55%** (拒绝新模型):
```python
# 什么都不做
pass
```
- ❌ `self.previous_nnet`保持不变 (旧baseline仍是标准)
- ✅ `self.nnet`继续训练 (不回滚权重!)
- ⚠️ 下次Arena仍然与旧baseline比较

### 4. AlphaZero原理
引用自 *Science 2018*:
> "If the new player won by a margin of 55%, then it replaced the best player; 
> otherwise, it was discarded."

**"discarded"的真实含义**:
- ❌ 不是回滚权重
- ✅ 是不接受为新baseline
- ✅ 但继续从当前模型训练

## 为什么不回滚?

1. **探索价值**: 即使当前模型不够好，它探索的方向可能有价值
2. **训练连续性**: 梯度下降需要连续更新，回滚会破坏优化轨迹
3. **AlphaZero标准**: 原论文也不回滚，只是不更新baseline

## Resume时的状态

Resume必须保存/恢复:
```python
checkpoint = {
    'state_dict': self.nnet.state_dict(),              # 当前模型
    'previous_state_dict': self.previous_nnet.state_dict(),  # baseline
    'train_examples_history': [...],                   # 经验池
    'optimizer_state': self.global_optimizer.state_dict(),
    'scheduler_state': self.global_scheduler.state_dict(),
}
```

如果只保存`state_dict`，Resume后:
- ✅ `self.nnet`恢复正确
- ❌ `self.previous_nnet`会重新初始化为`self.nnet`的副本
- ❌ Arena对决变成自己打自己! (胜率永远50%)

## 验证Arena是否正常

正常情况下的胜率分布:
- 初期: 新模型稍强，胜率55%-65%
- 中期: 偶尔失败(胜率<55%)，保持旧baseline
- 后期: 收敛后，胜率接近50% (新旧模型接近)

**异常信号**:
- 胜率一直是50%或接近50% → 可能是自己打自己
- 胜率一直>95% → 可能baseline是随机/初始权重
- 胜率一直<20% → 训练可能崩了
