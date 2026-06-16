# 第 6 周：手写知识蒸馏

## 目标

理解 teacher-student 训练过程，知道 soft label 为什么有价值。

## 学习内容

- hard label 与 soft label
- temperature
- KLDivLoss
- feature distillation
- 分类蒸馏与检测蒸馏差异

## 代码入口

- `optimization/distillation.py`

## 实战任务

1. 训练一个较大的 teacher 和一个较小的 student。
2. 用 `distillation_loss` 训练 student。
3. 对比普通 student 和 distilled student。
4. 改变 temperature 和 alpha，记录指标变化。

## 验收标准

- 能写出蒸馏 loss 的组成。
- 能解释 teacher 为什么要 `eval()` 和 `no_grad()`。
- 能判断蒸馏是否真的改善了 student。

## 答案闭环

<details>
<summary>先自己做</summary>

阅读 `optimization/distillation.py`，找出 hard loss、soft loss、temperature 和 alpha 分别在哪里。

</details>

<details>
<summary>卡住时看提示</summary>

teacher 不更新参数，只提供更平滑的类别分布；student 同时学习真实标签和 teacher 的 soft label。

</details>

<details>
<summary>参考答案</summary>

```python
with torch.no_grad():
    teacher_logits = teacher(images)

student_logits = student(images)
loss, parts = distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=4.0,
    alpha=0.7,
)
```

`alpha` 越大，student 越依赖 teacher；`temperature` 越大，soft label 越平滑。

</details>

<details>
<summary>为什么这样做</summary>

真实标签只告诉模型“正确类别是谁”，teacher 的 soft label 还隐含类别间相似度。例如某张图 teacher 认为 A=0.7、B=0.25、C=0.05，student 能学到 B 比 C 更像 A。

</details>

<details>
<summary>自检标准</summary>

- teacher 处于 `eval()` 模式。
- teacher forward 包在 `torch.no_grad()` 中。
- student 的 loss 同时包含 CE 和 KL 两部分。

</details>
