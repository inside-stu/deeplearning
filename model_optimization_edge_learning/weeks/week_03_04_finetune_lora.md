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

## 实战任务

1. 跑 `python -m core.finetune`，观察冻结前后可训练参数数量。
2. 修改 `build_finetune_param_groups` 的学习率，理解 head/backbone 分组。
3. 把 `TinyClassifier.head` 替换成 `LoRALinear`。
4. 记录全量微调、冻结 backbone、LoRA head 三种方案的参数量和指标。

## 验收标准

- 能解释 `--freeze` 背后对应的 `requires_grad` 和 optimizer 变化。
- 能说明 BatchNorm 冻结和参数冻结不是同一件事。
- 能手写一个简化版 LoRA Linear，并解释 rank 和 alpha。

## 答案闭环

<details>
<summary>先自己做</summary>

先运行 `core.finetune`，再打开源码找出三个关键点：哪里设置 `requires_grad=False`，哪里过滤可训练参数，哪里替换成 `LoRALinear`。

</details>

<details>
<summary>卡住时看提示</summary>

冻结不是删除层，而是让参数不再产生梯度；optimizer 也应该只接收 `parameter.requires_grad=True` 的参数。

</details>

<details>
<summary>参考答案</summary>

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.finetune
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

</details>

<details>
<summary>为什么这样做</summary>

微调本质是控制哪些参数参与训练。冻结 backbone 后，backbone 仍参与 forward，但不会更新权重；LoRA 则进一步冻结原始权重，只训练低秩矩阵 A/B，让参数更新成本更低。

</details>

<details>
<summary>自检标准</summary>

- 冻结后可训练参数数量明显下降。
- optimizer 参数组里不包含已冻结参数。
- 你能解释 BatchNorm 的 `eval()` 和 `requires_grad=False` 分别控制什么。

</details>
