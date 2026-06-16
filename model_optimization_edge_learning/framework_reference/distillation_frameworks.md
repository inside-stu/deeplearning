# Distillation Frameworks And Engineering Patterns

## 常见方案

- logits distillation：学生学习 teacher 的 soft label。
- feature distillation：学生中间特征对齐 teacher。
- relation distillation：学习样本间或目标间关系。
- pseudo-label distillation：teacher 生成伪标签，student 当普通标注训练。

## 对照阶段一

- `optimization/distillation.py` 展示了 hard loss、soft KL loss、temperature 和 alpha。

## 工程重点

- teacher 通常固定为 eval 模式。
- teacher 推理可缓存，避免每个 epoch 重复推理。
- 检测蒸馏要考虑框匹配、类别置信度、特征尺度。
- teacher 过强但数据噪声大时，伪标签也会放大错误。

## 实战任务

1. teacher 生成训练集伪标签。
2. student 使用真实标签训练。
3. student 使用伪标签或 soft label 训练。
4. 对比普通 student 和蒸馏 student。

