# 第 13-14 周：高效微调框架

## 目标

学习 LoRA、QLoRA、Adapter 等主流参数高效微调方法。

## 学习框架

- Hugging Face Transformers
- PEFT
- TRL
- bitsandbytes
- Optimum

## 学习内容

- LoRA、QLoRA、Adapter、Prefix tuning
- rank、alpha、target_modules
- 4bit/8bit 加载
- instruction tuning 基础
- ViT 或 LLM 上的参数高效微调

## 代码入口

- `framework_reference/peft_lora.md`
- `core/models.py` 中的 `LoRALinear`

## 实战任务

1. 用 PEFT 跑一个 LoRA 微调 demo。
2. 打印可训练参数比例。
3. 合并 LoRA 权重并导出。
4. 对比全量微调与 LoRA 微调成本。

## 验收标准

- 能解释 PEFT 的 target_modules 对应哪些层。
- 能说明 LoRA 合并前后推理差异。
- 能判断什么时候该全量微调，什么时候该 LoRA。

## 答案闭环

<details>
<summary>先自己做</summary>

先读 `core.models.LoRALinear`，再看 PEFT 的 `LoraConfig`，把 rank、alpha、target_modules 和手写实现对应起来。

</details>

<details>
<summary>卡住时看提示</summary>

PEFT 里的 `r` 就是低秩矩阵的 rank，`lora_alpha` 对应缩放，`target_modules` 决定把 LoRA 插到哪些 Linear/Conv 层。

</details>

<details>
<summary>参考答案</summary>

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

阶段一对应代码：

```python
update = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
```

</details>

<details>
<summary>为什么这样做</summary>

LoRA 的工程价值在于冻结大部分参数，只训练很小的增量矩阵。全量微调适合数据和资源充足时追求上限，LoRA 适合显存有限、多任务适配、快速实验。

</details>

<details>
<summary>自检标准</summary>

- `print_trainable_parameters()` 显示可训练参数比例很低。
- 能解释 `target_modules` 写错会导致 LoRA 没插进去。
- 能说明 LoRA merge 后推理为什么不再需要额外分支。

</details>
