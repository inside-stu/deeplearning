# 第 18 周：剪枝与结构化压缩框架

## 目标

掌握更接近落地的剪枝和结构化压缩工具。

## 学习框架

- torch-pruning
- NNI model compression
- NNCF
- SparseML

## 学习内容

- dependency graph
- channel pruning
- structured pruning
- sparsity schedule
- pruning + fine-tuning
- 非结构化稀疏为什么不一定加速

## 代码入口

- `optimization/pruning.py`
- `framework_reference/pruning_frameworks.md`

## 实战任务

1. 使用 torch-pruning 压缩一个 CNN 或检测模型。
2. 剪枝后重新训练。
3. 记录 mAP/Acc、FPS、FLOPs、模型大小。
4. 对比 PyTorch pruning API 和 torch-pruning 的差异。

## 验收标准

- 能解释 dependency graph 的作用。
- 能说明结构化剪枝为什么更可能带来真实加速。
- 能设计剪枝比例和微调策略。

## 答案闭环

<details>
<summary>先自己做</summary>

先用 PyTorch pruning API 做非结构化剪枝，再读 torch-pruning 的 dependency graph，理解结构化剪枝为什么更复杂。

</details>

<details>
<summary>卡住时看提示</summary>

剪掉一个 Conv 的输出通道，会影响后面 Conv、BN、残差连接、concat 等层；dependency graph 就是用来同步这些结构变化的。

</details>

<details>
<summary>参考答案</summary>

torch-pruning 的典型流程：

```python
import torch
import torch_pruning as tp

example_inputs = torch.randn(1, 3, 224, 224)
ignored_layers = [model.head]
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=tp.importance.MagnitudeImportance(p=2),
    pruning_ratio=0.3,
    ignored_layers=ignored_layers,
)
pruner.step()
```

</details>

<details>
<summary>为什么这样做</summary>

非结构化剪枝容易理解，但不一定加速；结构化剪枝会真的改变通道数和模型结构，更可能降低延迟，但也更容易破坏模型，需要依赖图分析和剪枝后微调。

</details>

<details>
<summary>自检标准</summary>

- 剪枝后模型能正常 forward。
- 参数量、FLOPs 或延迟至少有一项下降。
- 剪枝后进行微调并记录精度恢复情况。

</details>
