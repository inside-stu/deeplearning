# 第 13-14 周：高效微调框架

## 目标

把第 3-4 周手写 LoRA 的理解迁移到工程框架中，掌握 Hugging Face Transformers、PEFT、bitsandbytes、TRL、Optimum 这些工具在高效微调中的位置。

这两周不是只学会复制 `LoraConfig`，而是要回答清楚：

- PEFT 到底把 LoRA 插到了模型的哪些层。
- `r`、`lora_alpha`、`target_modules` 和手写 `LoRALinear` 里的 `rank`、`alpha`、`lora_a/lora_b` 如何对应。
- 为什么 LoRA 只训练少量参数，却能适配新任务。
- LoRA merge 前后模型结构和推理方式有什么变化。
- QLoRA 和普通 LoRA 的区别在哪里。
- 什么时候用全量微调，什么时候用 LoRA/QLoRA。

## 和阶段一的关系

阶段一你已经手写过一个最小 LoRA：

```python
base = F.linear(x, self.weight, self.bias)
update = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
return base + update
```

PEFT 做的事情，本质上就是把这个思想自动插入到大模型的指定模块里。

| 阶段一手写概念 | PEFT 框架参数 | 含义 |
|---|---|---|
| `rank` | `r` | 低秩矩阵的秩，越大可训练参数越多，表达能力越强 |
| `alpha` | `lora_alpha` | LoRA 更新分支的缩放强度 |
| `self.lora_a` | 框架内部 LoRA A | 把输入从 `in_features` 降到 `r` |
| `self.lora_b` | 框架内部 LoRA B | 把低秩表示从 `r` 升回 `out_features` |
| `self.weight.requires_grad=False` | 冻结 base model | 原模型大部分权重不训练 |
| `replace_classifier_with_lora` | `get_peft_model` | 把普通层替换/包装成带 LoRA 的层 |
| `merge_weight()` | `merge_and_unload()` | 把 LoRA 增量合并回原权重，用于推理导出 |

## 学习框架

- `Transformers`：加载预训练模型、tokenizer、Trainer、generation。
- `PEFT`：给模型插入 LoRA、Adapter、Prefix tuning 等参数高效微调模块。
- `bitsandbytes`：4bit/8bit 加载大模型，常用于 QLoRA。
- `TRL`：面向 SFT、DPO、RLHF 等 LLM 训练流程。
- `Optimum`：和 ONNX Runtime、OpenVINO 等推理优化生态衔接。

## 学习要点与实战对应

| 学习要点 | 代码/框架位置 | 实战中用在哪里 | 对应任务 |
|---|---|---|---|
| LoRA 公式 | `core/models.py` 的 `LoRALinear` | 理解 PEFT 不是魔法，只是自动插入低秩分支 | 任务 1 |
| 模型模块名 | `model.named_modules()` | 找到能写进 `target_modules` 的层名 | 任务 2 |
| `LoraConfig` | `peft.LoraConfig` | 配置 rank、alpha、dropout、target_modules | 任务 3 |
| 可训练参数比例 | `model.print_trainable_parameters()` | 判断是否真的只训练 LoRA 参数 | 任务 3 |
| 最小微调循环 | `Trainer` 或手写训练循环 | 让 LoRA 参数真正参与训练 | 任务 4 |
| LoRA merge | `merge_and_unload()` | 部署前把 LoRA 增量合并回 base 权重 | 任务 5 |
| QLoRA | `BitsAndBytesConfig(load_in_4bit=True)` | base model 低比特加载，LoRA 参数保持可训练 | 任务 6 |
| 全量微调对比 | 参数量、显存、保存文件大小 | 选择合适工程方案 | 任务 7 |

## 代码入口

- `core/models.py` 中的 `LoRALinear`
- `core/lora_walkthrough.py`
- `framework_reference/peft_lora.md`
- `framework_reference/peft_lora_demo.py`

## 环境说明

PEFT/Transformers 示例通常需要安装：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m pip install transformers peft accelerate
```

如果要做 QLoRA，通常还需要：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m pip install bitsandbytes
```

注意：大模型加载可能需要联网下载模型权重。如果当前环境不能联网，可以先只读代码和运行 `core/lora_walkthrough.py`，等有本地缓存模型或网络环境后再跑 PEFT 示例。

## 完整主线示例

后面的参考答案不要当成彼此独立的碎片代码。真正的完整答案放在：

```text
framework_reference/peft_lora_demo.py
```

它按一条完整流程执行：

```text
1. 加载 Transformers base model
2. 打印候选模块名，帮助你选择 target_modules
3. 用 LoraConfig 和 get_peft_model 插入 LoRA
4. 打印可训练参数比例
5. 用几条 toy text 训练几步
6. 检查 LoRA 参数变化、冻结参数不变
7. merge_and_unload 合并 LoRA
8. 比较 merge 前后输出差异
9. 可选保存 adapter 和 merged model
```

推荐先运行或至少完整阅读这个脚本，再看下面的任务拆解。

最小运行命令：

```powershell
cd D:\项目\诗兰姆\越南扎扣件\deeplearning\model_optimization_edge_learning
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe framework_reference\peft_lora_demo.py --device cpu --steps 3
```

如果模型已经下载到本地缓存，想避免联网：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe framework_reference\peft_lora_demo.py --device cpu --steps 3 --local-files-only
```

如果你使用的不是 GPT-2 类模型，自动推断的 `target_modules` 不合适，可以手动指定：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe framework_reference\peft_lora_demo.py --model-name your-local-model --target-modules q_proj v_proj --device cpu
```

你应该重点看这些输出：

```text
stage=2 inspect_target_modules
candidate_module=...
selected_target_modules=...

stage=3 insert_lora
trainable params: ...
manual_parameter_count trainable=... total=... ratio=...

stage=4 train_lora_for_a_few_steps
step=1 loss=...
largest_trainable_change name=... max_abs_diff=...
largest_frozen_change max_abs_diff=0.00000000

stage=5 compare_before_after_merge
merge_max_abs_diff=...
merge_allclose=True
```

这些输出分别回答：

| 输出 | 你要理解的问题 |
|---|---|
| `candidate_module` | 模型里真实有哪些模块名 |
| `selected_target_modules` | LoRA 被插到了哪些模块 |
| `trainable params` | 是否真的只训练少量参数 |
| `largest_trainable_change` | LoRA 参数是否真的被训练更新 |
| `largest_frozen_change` | base model 是否保持冻结 |
| `merge_allclose` | LoRA 合并前后输出是否基本一致 |

## 实战任务

1. 复习手写 `LoRALinear`，把公式和 PEFT 参数对应起来。
2. 打印一个 Transformers 模型的模块名，找出 `target_modules` 应该写什么。
3. 用 PEFT 给模型插入 LoRA，并打印可训练参数比例。
4. 跑一个最小 LoRA 微调流程，确认只有 LoRA 参数更新。
5. 执行 `merge_and_unload()`，验证 merge 前后输出接近一致。
6. 阅读并理解 QLoRA 的 4bit 加载配置。
7. 对比全量微调、LoRA、QLoRA 的成本和适用场景。

## 验收标准

- 能解释 PEFT 的 `target_modules` 对应模型里的哪些层。
- 能把 `r/lora_alpha/target_modules` 和手写 `LoRALinear` 对应起来。
- 能用 `print_trainable_parameters()` 判断 LoRA 是否插入成功。
- 能说明 LoRA merge 前后推理差异。
- 能解释 QLoRA 为什么既涉及量化，又仍然属于高效微调。
- 能判断什么时候该全量微调，什么时候该 LoRA，什么时候该 QLoRA。

## 答案闭环

下面每个任务都是对 `framework_reference/peft_lora_demo.py` 的拆解。学习时建议顺序是：

```text
先完整跑脚本 -> 再看单个任务解释 -> 最后回到脚本里调参数
```

不要把下面的代码片段当成互不相关的小脚本。它们分别来自完整流程中的不同阶段。

### 任务 1：复习手写 LoRA，并对应到 PEFT

<details>
<summary>先自己做</summary>

打开 `core/models.py`，阅读 `LoRALinear`，写出普通 Linear 和 LoRA Linear 的公式。

要求你自己写出：

```text
普通 Linear:
y = ?

LoRA Linear:
y = ?
```

</details>

<details>
<summary>卡住时看提示</summary>

普通 Linear 只有一个主分支：

```text
y = xW^T + b
```

LoRA 在冻结原权重的基础上，加一个低秩更新分支：

```text
y = xW^T + x(BA)^T * alpha/r + b
```

</details>

<details>
<summary>参考答案</summary>

手写实现里的核心代码是：

```python
base = F.linear(x, self.weight, self.bias)
update = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
return base + update
```

其中：

```text
self.weight  -> 原始冻结权重 W
self.lora_a  -> 低秩矩阵 A，形状 [rank, in_features]
self.lora_b  -> 低秩矩阵 B，形状 [out_features, rank]
self.scaling -> alpha / rank
```

PEFT 里的对应关系：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,              # 对应手写 rank
    lora_alpha=16,    # 对应手写 alpha
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
```

</details>

<details>
<summary>为什么这样做</summary>

全量微调会更新原模型的大量参数。LoRA 的思路是：保留原模型已经学到的通用能力，只学习一个低秩增量矩阵来适配新任务。

所以 LoRA 不是重新训练一个模型，而是在原模型权重旁边加一个很小的可训练修正量。

</details>

<details>
<summary>自检标准</summary>

- 能写出 `y = xW^T + b`。
- 能写出 `y = xW^T + x(BA)^T * alpha/r + b`。
- 能说出 PEFT 的 `r` 对应手写 `rank`。
- 能说出 PEFT 的 `lora_alpha` 对应手写 `alpha`。

</details>

### 任务 2：打印模型模块名，理解 `target_modules`

<details>
<summary>先自己做</summary>

加载一个小的 Transformers 模型，打印它的模块名，找出哪些层可能适合插入 LoRA。

重点观察类似这些名字：

```text
q_proj
v_proj
k_proj
o_proj
c_attn
c_proj
```

</details>

<details>
<summary>卡住时看提示</summary>

`target_modules` 不是随便写的。它必须匹配模型内部真实存在的模块名。

不同模型的命名不一样：

| 模型类型 | 常见 target_modules |
|---|---|
| LLaMA/Qwen 类 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| GPT-2 类 | `c_attn`, `c_proj` |
| BERT 类 | `query`, `key`, `value`, `dense` |
| ViT 类 | `query`, `key`, `value`, `dense` |

</details>

<details>
<summary>参考答案</summary>

这一步对应完整脚本中的：

```text
stage=2 inspect_target_modules
```

如果环境能联网或已有缓存模型，可以运行：

```python
from transformers import AutoModelForCausalLM

model_name = "sshleifer/tiny-gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)

for name, module in model.named_modules():
    if "attn" in name or "proj" in name or "c_" in name:
        print(name, "->", type(module).__name__)
```

GPT-2 类模型常见输出会包含：

```text
transformer.h.0.attn.c_attn -> Conv1D
transformer.h.0.attn.c_proj -> Conv1D
transformer.h.0.mlp.c_proj -> Conv1D
```

所以对 GPT-2 类模型，`target_modules` 可以先尝试：

```python
target_modules=["c_attn", "c_proj"]
```

如果是 LLaMA/Qwen 类模型，通常会看到：

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
```

这时可以写：

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```

</details>

<details>
<summary>为什么这样做</summary>

PEFT 需要知道“把 LoRA 插到哪里”。`target_modules` 就是告诉 PEFT：遇到名字匹配的模块，就给它加 LoRA 分支。

如果 `target_modules` 写错，可能出现两种情况：

- 直接报错：找不到目标模块。
- 没有真正插入 LoRA：可训练参数比例不符合预期。

所以工程里不要盲抄 `target_modules`，要先打印模型结构。

</details>

<details>
<summary>自检标准</summary>

- 能打印模型的 `named_modules()`。
- 能解释为什么 GPT-2 和 LLaMA 的 `target_modules` 不一样。
- 能说出 `target_modules` 写错会导致什么问题。

</details>

### 任务 3：用 PEFT 插入 LoRA，并打印可训练参数比例

<details>
<summary>先自己做</summary>

用 `LoraConfig` 和 `get_peft_model` 包装一个 Transformers 模型，然后打印可训练参数比例。

</details>

<details>
<summary>卡住时看提示</summary>

核心流程只有三步：

```text
1. 加载 base model
2. 写 LoraConfig
3. get_peft_model(model, config)
```

</details>

<details>
<summary>参考答案</summary>

这一步对应完整脚本中的：

```text
stage=3 insert_lora
```

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model_name = "sshleifer/tiny-gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

你也可以手动统计：

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable={trainable}")
print(f"total={total}")
print(f"ratio={trainable / total:.6f}")
```

</details>

<details>
<summary>为什么这样做</summary>

`get_peft_model` 会冻结 base model 的大部分参数，并在指定模块上插入 LoRA 参数。

所以判断 LoRA 是否生效，不能只看代码有没有运行成功，还要看：

```text
可训练参数比例是否明显下降
```

如果你看到可训练参数还是接近 100%，说明你可能没有真正冻结 base model，或者没有使用 PEFT 包装后的模型。

</details>

<details>
<summary>自检标准</summary>

- 能看到 `trainable params` 明显少于 `all params`。
- 能解释 `bias="none"` 表示不额外训练 bias。
- 能解释 `task_type="CAUSAL_LM"` 表示当前任务是因果语言模型微调。

</details>

### 任务 4：跑一个最小 LoRA 微调流程

<details>
<summary>先自己做</summary>

构造几条极小文本样本，让 LoRA 模型做一次或几次训练，确认 loss 可以反向传播，并观察只有 LoRA 参数在更新。

</details>

<details>
<summary>卡住时看提示</summary>

你不一定一开始就用复杂的 `Trainer`。为了理解机制，可以先写一个非常小的 PyTorch 训练循环。

LLM 的输入通常包括：

```text
input_ids
attention_mask
labels
```

在因果语言模型里，`labels=input_ids` 是最小可运行做法。

</details>

<details>
<summary>参考答案</summary>

这一步对应完整脚本中的：

```text
stage=4 train_lora_for_a_few_steps
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, config).to(device)
model.train()

texts = [
    "model compression means making models smaller.",
    "LoRA trains a small low-rank update.",
    "quantization changes numerical precision.",
]

batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
batch = {key: value.to(device) for key, value in batch.items()}
batch["labels"] = batch["input_ids"].clone()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,
)

for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    loss = output.loss
    loss.backward()
    optimizer.step()
    print(f"step={step} loss={loss.item():.4f}")
```

检查只有 LoRA 参数可训练：

```python
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name, tuple(parameter.shape))
```

输出里通常会看到包含 `lora_A`、`lora_B` 的参数名。

</details>

<details>
<summary>为什么这样做</summary>

这个最小训练循环的目的不是得到一个好模型，而是看清楚 PEFT 的训练机制：

- base model 的大多数参数被冻结。
- optimizer 只接收 `requires_grad=True` 的 LoRA 参数。
- loss.backward 后，梯度主要流向 LoRA A/B。
- optimizer.step 只更新 LoRA 参数。

这和第 3-4 周手写 `LoRALinear` 的思想完全一致。

</details>

<details>
<summary>自检标准</summary>

- 训练循环能跑通并打印 loss。
- `named_parameters()` 中可训练参数名包含 `lora_A` 或 `lora_B`。
- 能解释为什么 optimizer 要过滤 `requires_grad=True` 的参数。

</details>

### 任务 5：合并 LoRA 权重，并验证输出差异

<details>
<summary>先自己做</summary>

训练或加载一个 LoRA 模型后，执行 `merge_and_unload()`，比较 merge 前后同一输入的 logits 是否接近。

</details>

<details>
<summary>卡住时看提示</summary>

merge 的含义和手写 `merge_weight()` 一样：

```text
W_merged = W + B @ A * alpha/r
```

merge 前：

```text
base 分支 + LoRA update 分支
```

merge 后：

```text
只有合并后的普通权重
```

</details>

<details>
<summary>参考答案</summary>

这一步对应完整脚本中的：

```text
stage=5 compare_before_after_merge
```

```python
import torch

model.eval()
inputs = tokenizer("LoRA is useful for", return_tensors="pt").to(model.device)

with torch.no_grad():
    logits_before = model(**inputs).logits

merged_model = model.merge_and_unload()
merged_model.eval()

with torch.no_grad():
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before - logits_after).abs().max().item()
print("max_abs_diff:", max_abs_diff)
print("allclose:", torch.allclose(logits_before, logits_after, atol=1e-4, rtol=1e-4))
```

保存合并后的模型：

```python
merged_model.save_pretrained("outputs/merged_lora_model")
tokenizer.save_pretrained("outputs/merged_lora_model")
```

只保存 LoRA adapter：

```python
model.save_pretrained("outputs/lora_adapter")
```

</details>

<details>
<summary>为什么这样做</summary>

训练阶段保留 LoRA 分支，是为了只更新少量参数。

部署阶段通常希望模型结构简单，所以可以把 LoRA 增量合并回原始权重。合并后推理不再需要额外 LoRA 分支，加载方式也更接近普通 Transformers 模型。

但要注意：

- 只保存 adapter 文件小，适合多任务切换。
- 保存 merged model 文件大，适合直接部署推理。

</details>

<details>
<summary>自检标准</summary>

- 能解释 `merge_and_unload()` 对应手写 `merge_weight()`。
- merge 前后同一输入的输出差异很小。
- 能区分保存 adapter 和保存 merged model 的区别。

</details>

### 任务 6：理解 QLoRA 的 4bit 加载

<details>
<summary>先自己做</summary>

阅读下面的 QLoRA 配置，回答：哪些部分是量化，哪些部分是 LoRA 微调？

</details>

<details>
<summary>卡住时看提示</summary>

QLoRA 可以理解成：

```text
base model 用 4bit 加载，节省显存
LoRA A/B 仍然以可训练参数形式更新
```

所以 QLoRA 不是“把 LoRA 参数也全部变成 4bit 训练”，而是把大块的 base model 压低精度，腾出显存训练 LoRA。

</details>

<details>
<summary>参考答案</summary>

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "your-local-or-hf-model",
    quantization_config=quant_config,
    device_map="auto",
)

base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

这里：

```text
BitsAndBytesConfig -> 负责 4bit 加载 base model
prepare_model_for_kbit_training -> 做 k-bit 训练前准备
LoraConfig -> 决定 LoRA 插到哪里、rank 多大
get_peft_model -> 真正插入 LoRA 参数
```

</details>

<details>
<summary>为什么这样做</summary>

大模型全量 FP16/FP32 加载会占用大量显存。QLoRA 先把 base model 以 4bit 方式加载，显著降低显存占用；同时只训练 LoRA 小参数，所以普通单卡更容易做微调实验。

这也解释了为什么 QLoRA 同时属于：

```text
量化技术 + 参数高效微调技术
```

</details>

<details>
<summary>自检标准</summary>

- 能说出 `load_in_4bit=True` 是在量化加载 base model。
- 能说出 LoRA 参数仍然是训练对象。
- 能解释 QLoRA 为什么比普通 LoRA 更省显存。

</details>

### 任务 7：对比全量微调、LoRA、QLoRA

<details>
<summary>先自己做</summary>

整理一张表，比较全量微调、LoRA、QLoRA 在训练参数量、显存、保存文件、部署复杂度上的区别。

</details>

<details>
<summary>卡住时看提示</summary>

不要只问“哪个最好”，要看当前约束：

```text
数据量
显存
任务数量
部署方式
是否需要最高精度
是否需要频繁切换 adapter
```

</details>

<details>
<summary>参考答案</summary>

| 方法 | 训练哪些参数 | 显存压力 | 保存文件 | 优点 | 风险 |
|---|---|---|---|---|---|
| 全量微调 | 所有参数 | 最高 | 完整模型 | 上限高，表达能力最强 | 显存大，训练慢，容易过拟合 |
| LoRA | 冻结 base，只训 LoRA A/B | 中低 | adapter 或 merged model | 省显存，适合多任务适配 | target_modules 选错会效果差 |
| QLoRA | 4bit base + LoRA A/B | 最低 | adapter 或 merged model | 更省显存，适合大模型单卡实验 | 训练和部署链路更复杂 |

选择建议：

```text
资源充足 + 追求最高上限 -> 全量微调
资源有限 + 快速适配任务 -> LoRA
模型很大 + 显存很紧张 -> QLoRA
需要多个客户/多个场景切换 -> 保存多个 LoRA adapter
需要单个部署包简单推理 -> merge 后保存完整模型
```

</details>

<details>
<summary>为什么这样做</summary>

工程选择不是只看精度。LoRA/QLoRA 的价值在于用更少显存、更小保存文件、更快实验速度换取可接受的任务适配效果。

但如果任务和原模型差异非常大，或者你有充足数据和算力，全量微调仍然可能更强。

</details>

<details>
<summary>自检标准</summary>

- 能说出三种方法分别训练哪些参数。
- 能解释 adapter 文件为什么比完整模型小。
- 能说明什么情况下应该 merge，什么情况下保留 adapter。
- 能结合自己的项目场景选择方案，而不是固定只用 LoRA。

</details>

## 实验记录模板

建议每次 PEFT 实验都记录：

| 实验 | base model | 方法 | r | alpha | target_modules | trainable ratio | train loss | eval metric | adapter size | 是否 merge | 结论 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| lora_demo | tiny-gpt2 | LoRA | 8 | 16 | c_attn,c_proj |  |  |  |  |  |  |

结论写法示例：

```text
本次 LoRA 实验只训练了很小比例的参数，训练能够正常反向传播。merge 前后同一输入 logits 差异很小，说明 LoRA 增量可以正确合并。后续需要在真实任务数据上比较全量微调、LoRA 和 QLoRA 的指标差异。
```

## 常见问题

### `target_modules` 应该怎么选？

先打印 `model.named_modules()`。不要直接照抄别人的配置。不同模型模块名不同，GPT-2、LLaMA、BERT、ViT 的命名都可能不一样。

### LoRA 参数量为什么这么少？

普通 Linear 的权重是：

```text
W: [out_features, in_features]
```

LoRA 只训练：

```text
A: [r, in_features]
B: [out_features, r]
```

当 `r` 很小时，`A+B` 的参数量远小于完整 `W`。

### merge 后还需要 PEFT 吗？

如果已经 `merge_and_unload()` 并保存完整模型，推理时可以按普通 Transformers 模型加载。
如果只保存 adapter，推理时仍需要先加载 base model，再加载 adapter。

### QLoRA 是量化还是微调？

两者都是。它用 4bit 量化方式加载 base model，同时用 LoRA 做参数高效微调。
