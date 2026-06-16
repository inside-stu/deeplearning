# 第 9 周：手写剪枝与压缩

## 目标

理解剪枝 mask、稀疏率、结构化剪枝和真实加速之间的关系。

## 学习内容

- 非结构化剪枝
- 结构化剪枝
- L1 norm 通道重要性
- pruning mask
- pruning re-parametrize
- 剪枝后微调
- 参数量、FLOPs、实际延迟

## 代码入口

- `optimization/pruning.py`
- `optimization/analysis.py`

## 实战任务

1. 对 `TinyClassifier` 做 global L1 pruning。
2. 统计剪枝前后稀疏率。
3. 执行 `remove_pruning_reparam`，理解 mask 如何变成真实权重。
4. 剪枝后微调，并记录精度恢复情况。

## 验收标准

- 能解释为什么非结构化剪枝不一定加速。
- 能说明剪枝后微调的必要性。
- 能区分参数量减少、FLOPs 减少、延迟降低。

## 答案闭环

<details>
<summary>先自己做</summary>

对 `TinyClassifier` 做一次 global L1 pruning，统计剪枝前后的稀疏率，再执行 `remove_pruning_reparam`。

</details>

<details>
<summary>卡住时看提示</summary>

PyTorch pruning 默认不是直接改掉 `weight`，而是加入 `weight_orig` 和 `weight_mask`，forward 时用 mask 后的权重。

</details>

<details>
<summary>参考答案</summary>

```python
from core.models import TinyClassifier
from optimization.analysis import sparsity
from optimization.pruning import apply_global_l1_pruning, remove_pruning_reparam

model = TinyClassifier(num_classes=3)
print("before:", sparsity(model))
apply_global_l1_pruning(model, amount=0.3)
print("after pruning:", sparsity(model))
remove_pruning_reparam(model)
print("after remove:", sparsity(model))
```

</details>

<details>
<summary>为什么这样做</summary>

非结构化剪枝把很多单个权重变成 0，但矩阵形状没变，普通硬件不一定更快。结构化剪枝删除通道或层，更容易改变真实计算量，但需要处理上下游层依赖。

</details>

<details>
<summary>自检标准</summary>

- 剪枝后稀疏率上升。
- `remove_pruning_reparam` 后模型仍能 forward。
- 你能解释为什么剪枝后通常要微调恢复精度。

</details>
