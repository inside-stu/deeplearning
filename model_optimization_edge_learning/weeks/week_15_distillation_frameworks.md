# 第 15 周：蒸馏框架与工程方案

## 目标

从手写蒸馏过渡到工程蒸馏。

## 学习框架/工具

- Hugging Face distillation 示例
- MMDetection distillation 相关方案
- 自定义 PyTorch distillation trainer

## 学习内容

- logits distillation
- feature distillation
- detection distillation
- teacher 推理缓存
- 蒸馏数据选择

## 代码入口

- `optimization/distillation.py`
- `framework_reference/distillation_frameworks.md`

## 实战任务

1. 用 teacher 模型生成 soft label 或伪标签。
2. 训练 student 模型。
3. 对比 teacher、student、distilled student。
4. 记录蒸馏是否提升了速度/精度综合收益。

## 验收标准

- 能解释分类蒸馏和检测蒸馏差异。
- 能判断伪标签质量是否影响 student。
- 能设计 teacher 缓存方案减少训练成本。

## 答案闭环

<details>
<summary>先自己做</summary>

先用 teacher 对训练集生成预测结果，再让 student 学习真实标签和 teacher 输出，对比普通 student 与蒸馏 student。

</details>

<details>
<summary>卡住时看提示</summary>

工程蒸馏常见两种做法：在线蒸馏每次训练都跑 teacher；离线蒸馏先缓存 teacher 预测，训练 student 时直接读取缓存。

</details>

<details>
<summary>参考答案</summary>

```python
teacher.eval()
student.train()

with torch.no_grad():
    teacher_logits = teacher(images)

student_logits = student(images)
loss, parts = distillation_loss(student_logits, teacher_logits, labels)
loss.backward()
optimizer.step()
```

检测任务可以先从伪标签蒸馏做起：teacher 生成 boxes/classes/scores，过滤低置信度结果，再作为 student 训练数据。

</details>

<details>
<summary>为什么这样做</summary>

框架蒸馏的难点不只是 loss，还包括 teacher 推理成本、伪标签质量、检测框匹配、多尺度特征对齐。先做离线伪标签，最容易形成闭环。

</details>

<details>
<summary>自检标准</summary>

- teacher 不参与梯度更新。
- student 的指标和速度都被记录。
- 能说明蒸馏失败时应先查 teacher 质量还是 loss 权重。

</details>
