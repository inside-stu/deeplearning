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
- `core/models.py` 中的 `TinyClassifier`
- `core/train_loop.py`

## 学习要点与代码对应

| 学习要点 | 代码位置 | 实战中用在哪里 |
|---|---|---|
| 可剪枝层选择 | `conv_linear_modules` | 找出 `TinyClassifier` 里的 `Conv2d` 和 `Linear` |
| 全局 L1 非结构化剪枝 | `apply_global_l1_pruning` | 按权重绝对值大小，在所有可剪枝层里统一剪掉一部分权重 |
| pruning mask | PyTorch pruning 自动生成 `weight_mask` | 决定哪些权重保留，哪些权重变成 0 |
| pruning re-parametrize | PyTorch pruning 自动生成 `weight_orig` | 原始权重不直接消失，forward 时使用 `weight_orig * weight_mask` |
| remove reparam | `remove_pruning_reparam` | 把 mask 结果固化回真实 `weight` |
| 稀疏率统计 | `optimization.analysis.sparsity` / `module_sparsity` | 观察权重里有多少比例是 0 |
| 通道重要性 | `channel_l1_importance` | 为结构化剪枝理解“哪些输出通道更不重要” |
| 剪枝后微调 | `train_one_epoch` / `evaluate_classifier` | 剪枝后继续训练，让精度恢复 |

## 基于第 1-2 周模型的剪枝对象

第 9 周继续使用前面训练过的 `TinyClassifier`。它的结构大致是：

```text
TinyClassifier
  ├─ backbone
  │   ├─ stem.conv    Conv2d(3 -> 32)
  │   ├─ stage1.conv  Conv2d(32 -> 64)
  │   └─ stage2.conv  Conv2d(64 -> 128)
  ├─ pool             AdaptiveAvgPool2d
  └─ head             Linear(128 -> num_classes)
```

`conv_linear_modules(model)` 默认只选：

```text
Conv2d
Linear
```

也就是说，这里会剪：

```text
stem.conv.weight
stage1.conv.weight
stage2.conv.weight
head.weight
```

不会剪：

```text
BatchNorm 参数
SiLU 激活
pool 层
```

原因是剪枝通常针对有大量权重的层。BN、激活、池化层不是主要参数来源。

## 实战任务

1. 对 `TinyClassifier` 做 global L1 pruning。
2. 统计剪枝前后稀疏率。
3. 执行 `remove_pruning_reparam`，理解 mask 如何变成真实权重。
4. 剪枝后微调，并记录精度恢复情况。
5. 对比剪枝前后参数量、稀疏率、验证精度和推理延迟。

## 一个关键区别

非结构化剪枝回答的是：

```text
哪些单个权重可以变成 0？
```

结构化剪枝回答的是：

```text
哪些通道、卷积核或层可以真正删除？
```

第 9 周的 `apply_global_l1_pruning` 是非结构化剪枝。它会让很多权重变成 0，但张量形状不变，所以参数文件、FLOPs 和普通 CPU/GPU 推理速度不一定明显下降。

这和量化类似：你不能只看“理论上压缩了”，还要看：

```text
accuracy 有没有掉？
sparsity 有没有升？
state_dict size 有没有变？
p50/p95 latency 有没有变？
```

## 验收标准

- 能解释为什么非结构化剪枝不一定加速。
- 能说明剪枝后微调的必要性。
- 能区分参数量减少、FLOPs 减少、延迟降低。

## 答案闭环入口

<details>
<summary>先自己做</summary>

对 `TinyClassifier` 做一次 global L1 pruning，统计剪枝前后的稀疏率，再执行 `remove_pruning_reparam`。

</details>

## 按任务拆解的答案闭环

这一部分按第 1-2 周的方式来学：每个任务都先自己做，卡住再看提示，再看参考答案、为什么这样做和自检标准。

### 任务 1：找出 `TinyClassifier` 里哪些层会被剪

<details>
<summary>先自己做</summary>

打开 `core/models.py` 和 `optimization/pruning.py`，找出 `TinyClassifier` 里哪些层属于 `Conv2d` 或 `Linear`。

</details>

<details>
<summary>卡住时看提示</summary>

搜索：

```text
class TinyClassifier
class ConvBlock
def conv_linear_modules
```

`conv_linear_modules` 只收集 `nn.Conv2d` 和 `nn.Linear`。

</details>

<details>
<summary>参考答案</summary>

`optimization/pruning.py` 里的可剪枝层收集逻辑：

```python
def conv_linear_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    modules: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules.append((module, "weight"))
    return modules
```

你可以用下面的代码打印可剪枝层：

```python
from core.models import TinyClassifier
from optimization.pruning import conv_linear_modules

model = TinyClassifier(num_classes=3)

for module, parameter_name in conv_linear_modules(model):
    weight = getattr(module, parameter_name)
    print(type(module).__name__, tuple(weight.shape))
```

典型输出会包含：

```text
Conv2d (32, 3, 3, 3)
Conv2d (64, 32, 3, 3)
Conv2d (128, 64, 3, 3)
Linear (3, 128)
```

</details>

<details>
<summary>为什么这样做</summary>

剪枝前必须先知道自己剪的是哪些层。这里先剪 Conv 和 Linear，因为它们包含主要权重。BN、SiLU、pool 不在剪枝列表里，因为它们不是主要计算权重，或者没有类似卷积核/线性权重可以直接剪。

</details>

<details>
<summary>自检标准</summary>

- 能说出 `TinyClassifier` 中 3 个 Conv 和 1 个 Linear 会被剪。
- 能解释为什么 BatchNorm 和 SiLU 不在当前剪枝列表里。
- 能看懂 Conv2d 权重形状 `[out_channels, in_channels, kH, kW]`。

</details>

### 任务 2：执行 global L1 非结构化剪枝

<details>
<summary>先自己做</summary>

对 `TinyClassifier` 执行一次 `amount=0.3` 的 global L1 pruning，观察剪枝前后稀疏率。

</details>

<details>
<summary>卡住时看提示</summary>

`amount=0.3` 表示在所有参与剪枝的权重里，按 L1 绝对值大小，全局剪掉约 30%。

</details>

<details>
<summary>参考答案</summary>

```python
from core.models import TinyClassifier
from optimization.analysis import sparsity
from optimization.pruning import apply_global_l1_pruning

model = TinyClassifier(num_classes=3)

print("before_sparsity:", sparsity(model))
apply_global_l1_pruning(model, amount=0.3)
print("after_pruning_sparsity:", sparsity(model))
```

注意：PyTorch pruning 刚执行完时，模块里会出现 `weight_orig` 和 `weight_mask`。如果稀疏率统计函数只遍历 `model.parameters()`，它看到的是没有被 mask 的 `weight_orig`，就会出现“剪枝前后稀疏率不变”的现象。这里的 `sparsity(model)` 已经按剪枝后的有效权重统计，所以能看到稀疏率上升。

`apply_global_l1_pruning` 的核心代码：

```python
prune.global_unstructured(
    conv_linear_modules(model),
    pruning_method=prune.L1Unstructured,
    amount=amount,
)
```

</details>

<details>
<summary>为什么这样做</summary>

L1 pruning 的直觉是：绝对值越小的权重，对输出影响可能越小，所以优先剪掉。global pruning 不是每层各剪 30%，而是在所有 Conv/Linear 权重放在一起比较后，整体剪掉 30%。

所以某些层可能被剪得多，某些层可能被剪得少。这比每层固定比例更灵活，但也需要观察是否伤害关键层。

</details>

<details>
<summary>自检标准</summary>

- 剪枝后 `sparsity(model)` 明显上升。
- 能解释 `amount=0.3` 的含义。
- 能说明 global pruning 和 layer-wise pruning 的区别。

</details>

### 任务 3：理解 `weight_orig` 和 `weight_mask`

<details>
<summary>先自己做</summary>

剪枝后打印某个卷积层的参数名和 buffer 名，观察 PyTorch pruning 插入了什么。

</details>

<details>
<summary>卡住时看提示</summary>

PyTorch pruning 默认不是直接删除 `weight`，而是做 re-parametrize：

```text
weight = weight_orig * weight_mask
```

</details>

<details>
<summary>参考答案</summary>

```python
from core.models import TinyClassifier
from optimization.pruning import apply_global_l1_pruning

model = TinyClassifier(num_classes=3)
conv = model.backbone.stem.conv

print("before parameters:", list(dict(conv.named_parameters()).keys()))
print("before buffers:", list(dict(conv.named_buffers()).keys()))

apply_global_l1_pruning(model, amount=0.3)

print("after parameters:", list(dict(conv.named_parameters()).keys()))
print("after buffers:", list(dict(conv.named_buffers()).keys()))
print("has weight_orig:", hasattr(conv, "weight_orig"))
print("has weight_mask:", hasattr(conv, "weight_mask"))
print("weight shape:", tuple(conv.weight.shape))
```

剪枝前常见是：

```text
parameters: ['weight']
buffers: []
```

剪枝后常见是：

```text
parameters: ['weight_orig']
buffers: ['weight_mask']
has weight_orig: True
has weight_mask: True
```

</details>

<details>
<summary>为什么这样做</summary>

PyTorch pruning 先保留原始权重 `weight_orig`，再新增一个二值 mask：`weight_mask`。forward 时用 mask 后的权重参与计算。mask 中为 0 的位置表示这个权重被剪掉，为 1 的位置表示保留。

这一步叫 re-parametrize。它方便你继续训练，因为被保留的权重仍然可以更新，而被 mask 掉的位置保持为 0。

</details>

<details>
<summary>自检标准</summary>

- 能说出 `weight_orig` 是原始可训练权重。
- 能说出 `weight_mask` 是 0/1 掩码。
- 能写出 `weight = weight_orig * weight_mask`。

</details>

### 任务 4：执行 `remove_pruning_reparam`

<details>
<summary>先自己做</summary>

剪枝后执行 `remove_pruning_reparam(model)`，再观察 `weight_orig` 和 `weight_mask` 是否还存在。

</details>

<details>
<summary>卡住时看提示</summary>

`remove` 不是撤销剪枝，而是把当前 mask 后的结果固化成真正的 `weight`。

</details>

<details>
<summary>参考答案</summary>

```python
from core.models import TinyClassifier
from optimization.analysis import sparsity
from optimization.pruning import apply_global_l1_pruning, remove_pruning_reparam

model = TinyClassifier(num_classes=3)
conv = model.backbone.stem.conv

apply_global_l1_pruning(model, amount=0.3)
print("before_remove_has_mask:", hasattr(conv, "weight_mask"))
print("before_remove_sparsity:", sparsity(model))

remove_pruning_reparam(model)
print("after_remove_has_mask:", hasattr(conv, "weight_mask"))
print("after_remove_has_weight_orig:", hasattr(conv, "weight_orig"))
print("after_remove_sparsity:", sparsity(model))
```

预期现象：

```text
remove 前：有 weight_orig / weight_mask
remove 后：没有 weight_orig / weight_mask，重新变成普通 weight
稀疏率仍然保持较高
```

</details>

<details>
<summary>为什么这样做</summary>

训练或实验阶段保留 mask 很方便，但部署或保存最终模型时，通常希望模型结构更普通。`prune.remove` 会把：

```text
weight_orig * weight_mask
```

固化成新的 `weight`。注意，它不会让被剪掉的权重恢复，而是让 0 真的写进权重 tensor。

</details>

<details>
<summary>自检标准</summary>

- 能说明 `remove` 不是恢复权重。
- 能看到 `weight_mask` 消失。
- 能确认 remove 后模型仍然可以 forward。

</details>

### 任务 5：验证剪枝后模型还能 forward

<details>
<summary>先自己做</summary>

剪枝并 remove 后，用随机输入跑一次 `TinyClassifier`，确认输出 shape 正常。

</details>

<details>
<summary>卡住时看提示</summary>

分类模型输入是 `[B, 3, 32, 32]`，输出是 `[B, num_classes]`。

</details>

<details>
<summary>参考答案</summary>

```python
import torch

from core.models import TinyClassifier
from optimization.pruning import apply_global_l1_pruning, remove_pruning_reparam

model = TinyClassifier(num_classes=3)
apply_global_l1_pruning(model, amount=0.3)
remove_pruning_reparam(model)

model.eval()
x = torch.randn(2, 3, 32, 32)

with torch.no_grad():
    logits = model(x)

print("logits_shape:", tuple(logits.shape))
```

预期输出：

```text
logits_shape: (2, 3)
```

</details>

<details>
<summary>为什么这样做</summary>

剪枝后第一件事不是看速度，而是确认模型还能正常 forward。非结构化剪枝只是把部分权重变成 0，张量形状不变，所以输出 shape 应该不变。

</details>

<details>
<summary>自检标准</summary>

- 能解释输入 shape `[2, 3, 32, 32]` 的含义。
- 能解释输出 shape `[2, 3]` 的含义。
- 剪枝后 forward 不报错。

</details>

### 任务 6：剪枝后微调并观察精度恢复

<details>
<summary>先自己做</summary>

复用第 1-2 周的训练循环：先训练一个 baseline，再剪枝，然后继续微调若干 epoch，对比剪枝前后验证集指标。

</details>

<details>
<summary>卡住时看提示</summary>

剪枝会突然把一部分权重变成 0，模型输出可能变差。微调的目标是让剩余权重重新适应这个稀疏结构。

</details>

<details>
<summary>参考答案</summary>

```python
import torch
from torch import nn

from core.models import TinyClassifier
from core.train_loop import build_dataloaders, evaluate_classifier, train_one_epoch
from optimization.analysis import sparsity
from optimization.pruning import apply_global_l1_pruning, remove_pruning_reparam

device = torch.device("cpu")
train_loader, val_loader = build_dataloaders(
    batch_size=32,
    num_classes=3,
    image_size=32,
    dataset="synthetic",
)

model = TinyClassifier(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(3):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

before_metrics = evaluate_classifier(model, val_loader, criterion, device, num_classes=3)
print("before_pruning_acc:", before_metrics["accuracy"])
print("before_pruning_sparsity:", sparsity(model))

apply_global_l1_pruning(model, amount=0.3)
remove_pruning_reparam(model)

after_prune_metrics = evaluate_classifier(model, val_loader, criterion, device, num_classes=3)
print("after_pruning_acc:", after_prune_metrics["accuracy"])
print("after_pruning_sparsity:", sparsity(model))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for epoch in range(2):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    metrics = evaluate_classifier(model, val_loader, criterion, device, num_classes=3)
    print(f"finetune_epoch={epoch + 1} train_loss={train_loss:.4f} val_acc={metrics['accuracy']:.4f}")
```

</details>

<details>
<summary>为什么这样做</summary>

剪枝相当于突然改变模型参数。即使剪的是 L1 较小的权重，也可能破坏一些已经学到的组合。剪枝后微调能让剩余权重重新调整，尽量恢复精度。

剪枝实验不能只看“剪完稀疏率变高”，还要看：

```text
剪枝前 accuracy
剪枝后 accuracy
微调后 accuracy
```

</details>

<details>
<summary>自检标准</summary>

- 能跑出剪枝前、剪枝后、微调后的验证指标。
- 能说明剪枝后精度可能下降。
- 能解释微调为什么通常必要。

</details>

### 任务 7：理解为什么非结构化剪枝不一定加速

<details>
<summary>先自己做</summary>

对比剪枝前后的参数量、稀疏率和推理延迟，观察它们是否同步变化。

</details>

<details>
<summary>卡住时看提示</summary>

非结构化剪枝把单个权重变成 0，但 Conv/Linear 的张量形状没有改变。

</details>

<details>
<summary>参考答案</summary>

可以复用 `optimization.analysis`：

```python
import torch

from core.models import TinyClassifier
from optimization.analysis import benchmark_forward, parameter_count, sparsity, state_dict_size_mb
from optimization.pruning import apply_global_l1_pruning, remove_pruning_reparam

model = TinyClassifier(num_classes=3)
sample = torch.randn(1, 3, 32, 32)

print("before_params:", parameter_count(model))
print("before_size_mb:", state_dict_size_mb(model))
print("before_sparsity:", sparsity(model))
print("before_latency:", benchmark_forward(model, sample, warmup=5, repeats=20))

apply_global_l1_pruning(model, amount=0.3)
remove_pruning_reparam(model)

print("after_params:", parameter_count(model))
print("after_size_mb:", state_dict_size_mb(model))
print("after_sparsity:", sparsity(model))
print("after_latency:", benchmark_forward(model, sample, warmup=5, repeats=20))
```

常见现象：

```text
sparsity 上升
parameter_count 基本不变
state_dict_size_mb 不一定明显变小
latency 不一定下降
```

</details>

<details>
<summary>为什么这样做</summary>

普通 Conv/Linear kernel 通常还是按原来的 dense tensor 计算。即使里面有很多 0，如果硬件和算子没有专门利用稀疏性，计算量不会自动减少。

这就是为什么“参数变成 0”和“真实加速”不是一回事。要更可能加速，通常需要结构化剪枝，例如删通道、删卷积核，让张量形状真的变小。

</details>

<details>
<summary>自检标准</summary>

- 能解释为什么 `parameter_count` 不一定下降。
- 能解释为什么 `latency` 不一定下降。
- 能说出结构化剪枝比非结构化剪枝更可能加速的原因。

</details>

### 任务 8：理解通道 L1 重要性

<details>
<summary>先自己做</summary>

选择一个卷积层，计算每个输出通道的 L1 importance，找出最小的几个通道。

</details>

<details>
<summary>卡住时看提示</summary>

Conv2d 权重形状是：

```text
[out_channels, in_channels, kernel_h, kernel_w]
```

对 `(in_channels, kernel_h, kernel_w)` 求绝对值和，就得到每个输出通道的重要性。

</details>

<details>
<summary>参考答案</summary>

```python
from core.models import TinyClassifier
from optimization.pruning import channel_l1_importance, lowest_importance_channels

model = TinyClassifier(num_classes=3)
conv = model.backbone.stage2.conv

importance = channel_l1_importance(conv)
lowest = lowest_importance_channels(conv, amount=5)

print("importance_shape:", tuple(importance.shape))
print("lowest_channels:", lowest.tolist())
print("lowest_importance:", importance[lowest].tolist())
```

`channel_l1_importance` 的核心代码：

```python
return conv.weight.detach().abs().sum(dim=(1, 2, 3))
```

</details>

<details>
<summary>为什么这样做</summary>

非结构化剪枝看的是单个权重。结构化通道剪枝看的是整个输出通道是否重要。一个输出通道对应下一层输入的一整组特征图，如果删除它，就可能真的减少后续计算。

但结构化剪枝更复杂，因为删掉一个通道后，下一层对应的输入通道也要一起处理。这就是后面 torch-pruning 需要 dependency graph 的原因。

</details>

<details>
<summary>自检标准</summary>

- 能说出 `importance_shape` 等于输出通道数。
- 能解释为什么对 `dim=(1, 2, 3)` 求和。
- 能说明通道剪枝为什么涉及上下游层依赖。

</details>

## 剪枝实验记录模板

建议每次剪枝都记录：

| 实验 | pruning amount | val_acc before | val_acc after prune | val_acc after finetune | sparsity | params | size MB | p50 latency | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| global L1 | 0.3 |  |  |  |  |  |  |  |  |

结论写法示例：

```text
本次 global L1 非结构化剪枝将稀疏率提升到约 30%，剪枝后精度轻微下降，微调后部分恢复。参数量和延迟没有明显下降，说明非结构化剪枝主要制造稀疏性，不等于真实加速。后续如果追求速度，应尝试结构化通道剪枝。
```

## 快速答案补充

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
