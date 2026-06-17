# 第 3-4 周：手写微调、冻结与 LoRA

## 目标

真正理解“冻结层”和“参数高效微调”，而不是只会使用命令参数。

## 学习内容

- `requires_grad=False`
- optimizer 只接收可训练参数
- backbone/head 参数划分
- BatchNorm 是否冻结
- 全量微调、只训 head、分组学习率
- 小样本过拟合判断
- LoRA：冻结原权重，只训练低秩增量矩阵

## 代码入口

- `core/finetune.py`
- `core/models.py` 中的 `LoRALinear`
- `core/lora_walkthrough.py`

## 实战任务

1. 跑 `python -m core.finetune`，观察冻结前后可训练参数数量。
2. 修改 `build_finetune_param_groups` 的学习率，理解 head/backbone 分组。
3. 把 `TinyClassifier.head` 替换成 `LoRALinear`。
4. 跑 `python -m core.lora_walkthrough`，观察 LoRA 的 base 分支、update 分支和 merge 后权重。
5. 记录全量微调、冻结 backbone、LoRA head 三种方案的参数量和指标。

## 验收标准

- 能解释 `--freeze` 背后对应的 `requires_grad` 和 optimizer 变化。
- 能说明 BatchNorm 冻结和参数冻结不是同一件事。
- 能手写一个简化版 LoRA Linear，并解释 rank 和 alpha。
- 能说明为什么 LoRA 里仍然保留冻结的 `self.weight`。
- 能解释 `lora_a`、`lora_b` 的形状，以及 `B @ A` 为什么能形成和原权重同形状的增量。

## 答案闭环

<details>
<summary>先自己做</summary>

先运行 `core.finetune` 和 `core.lora_walkthrough`，再打开源码找出四个关键点：哪里设置 `requires_grad=False`，哪里过滤可训练参数，哪里替换成 `LoRALinear`，LoRA 的 base/update 两条分支在哪里相加。

</details>

<details>
<summary>卡住时看提示</summary>

冻结不是删除层，而是让参数不再产生梯度；optimizer 也应该只接收 `parameter.requires_grad=True` 的参数。

理解 LoRA 时先不要急着看框架。先把普通 Linear 写成：

```text
y = xW^T + b
```

LoRA 做的是在冻结的 `W` 旁边加一个低秩增量：

```text
y = xW^T + x(BA)^T * alpha/r + b
```

所以 LoRA 不是替代原权重，而是在原权重上加一个可训练的轻量修正。

</details>

<details>
<summary>参考答案</summary>

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.finetune
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.lora_walkthrough
```

核心代码位置：

```python
for parameter in module.parameters():
    parameter.requires_grad = False

if not parameter.requires_grad:
    continue
```

LoRA 的最小形式在 `core/models.py`：

```python
base = F.linear(x, self.weight, self.bias)
update = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
return base + update
```

对应公式：

```text
base   = xW^T + b
update = x(BA)^T * alpha/r
output = base + update
```

形状对应关系：

| 名称 | 代码参数 | 形状 | 是否训练 | 含义 |
|---|---|---|---|---|
| 原始权重 | `self.weight` | `[out_features, in_features]` | 否 | 保留原模型能力的冻结权重 W |
| LoRA A | `self.lora_a` | `[rank, in_features]` | 是 | 先把输入从 in 维降到 rank 维 |
| LoRA B | `self.lora_b` | `[out_features, rank]` | 是 | 再从 rank 维升回 out 维 |
| LoRA 增量 | `self.lora_b @ self.lora_a` | `[out_features, in_features]` | 由 A/B 得到 | 和 W 同形状，所以可以加到 W 上 |
| 缩放系数 | `self.scaling` | 标量 | 否 | `alpha / rank`，控制 LoRA 增量强度 |

`replace_classifier_with_lora` 逐行看：

```python
old_head = model.head
```

拿到原来的分类头。这里的 `old_head` 是一个普通 `nn.Linear`，里面有已经初始化或训练过的 `weight/bias`。

```python
lora_head = LoRALinear(
    old_head.in_features,
    old_head.out_features,
    rank=rank,
    alpha=alpha,
    bias=old_head.bias is not None,
)
```

创建一个同输入维度、同输出维度的 LoRA 版分类头。维度必须相同，否则 backbone 输出接不上新 head，类别数也会不一致。

```python
lora_head.weight.copy_(old_head.weight)
```

把旧分类头的权重复制到 LoRA 的冻结 base 分支，也就是把原来的 `W` 保留下来。

```python
if old_head.bias is not None and lora_head.bias is not None:
    lora_head.bias.copy_(old_head.bias)
```

如果原 head 有 bias，也复制过去。这样替换前后 base 分支完全一致。

```python
model.head = lora_head
```

真正把模型的分类头换成 LoRA 分类头。替换后 forward 会变成 `base + update`。

为什么替换后初始输出应该几乎一致？因为 `LoRALinear` 里 `lora_b` 初始化为 0：

```python
nn.init.zeros_(self.lora_b)
```

所以一开始：

```text
B = 0
B @ A = 0
update = 0
output = base
```

也就是说，刚替换成 LoRA 时模型行为不应该突然改变；后续训练只让 A/B 学一个增量修正。

</details>

<details>
<summary>为什么这样做</summary>

微调本质是控制哪些参数参与训练。冻结 backbone 后，backbone 仍参与 forward，但不会更新权重；LoRA 则进一步冻结原始权重，只训练低秩矩阵 A/B，让参数更新成本更低。

为什么 LoRA 还需要 `self.weight`？因为 LoRA 的目标不是从零训练一个新 Linear，而是保留原模型已经学到的能力。`self.weight` 就是原来的 W，它负责稳定的基础输出；`lora_a/lora_b` 只负责学习新任务需要的小修正。这样训练参数量少，初始行为稳定，也方便最后用：

```python
merged_weight = self.weight + (self.lora_b @ self.lora_a) * self.scaling
```

把 LoRA 增量合并回一个普通 Linear 权重。

</details>

<details>
<summary>自检标准</summary>

- 冻结后可训练参数数量明显下降。
- optimizer 参数组里不包含已冻结参数。
- 你能解释 BatchNorm 的 `eval()` 和 `requires_grad=False` 分别控制什么。
- 你能解释普通 Linear 的 `y = xW^T + b` 和 LoRA 的 `y = xW^T + x(BA)^T * alpha/r + b`。
- 你能说明 `self.weight` 为什么冻结但仍参与 forward。
- 你能通过 `core.lora_walkthrough` 的输出看到：初始 `update=0`，训练一步后 `self.weight` 不变而 `lora_b` 改变。
- 你能解释 `merge_weight()` 为什么和 LoRA forward 等价。

</details>
