# PEFT, LoRA, QLoRA, And Adapters

## 核心框架

- Hugging Face Transformers
- PEFT
- TRL
- bitsandbytes
- Optimum

## 对照阶段一

- `core/models.py` 中的 `LoRALinear` 展示了 LoRA 的核心公式：
  - 原权重冻结。
  - 只训练低秩矩阵 A 和 B。
  - 推理时可以把低秩更新合并回原权重。

## 学习重点

- `rank`：低秩矩阵大小，影响可训练参数量和表达能力。
- `alpha`：缩放系数，影响 LoRA 更新强度。
- `target_modules`：决定 LoRA 插入哪些层。
- `modules_to_save`：除 LoRA 外额外保存哪些模块。
- QLoRA：基础模型 4bit 加载，LoRA 参数仍可训练。

## 实战任务

1. 打印 PEFT 模型的 trainable parameters。
2. 对比 full finetune 与 LoRA 的显存占用。
3. 修改 rank，观察效果和参数量。
4. 合并 LoRA 权重，验证合并前后输出差异。

