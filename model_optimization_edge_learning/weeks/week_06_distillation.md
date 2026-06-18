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
5. 打开 `feature_distillation_loss`，理解 feature 蒸馏和 logits 蒸馏的区别。
6. 修改 `--feature-weight`，观察 feature loss 是否影响 student 指标。

## 完整执行例子

第 6 周现在有一个完整可运行 demo，不再只是函数片段。默认使用 synthetic 分类数据，不需要额外准备图片：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 3 --device cpu
```

快速 smoke test 可以只跑 1 个 epoch：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu
```

调参实验：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu --feature-weight 0
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu --temperature 2 --alpha 0.5
```

你会看到三段训练结果：

```text
teacher epoch=...
normal_student epoch=...
distilled_student epoch=... hard_loss=... soft_loss=... distill_loss=... feature_loss=... val_acc=...
final_comparison:
  teacher_val_acc=...
  normal_student_val_acc=...
  distilled_student_val_acc=...
```

这里的判断重点不是“蒸馏 student 一定每次都更高”，而是你要能看懂：teacher 质量如何、普通 student 表现如何、蒸馏 student 有没有从 soft label 或 feature 对齐里得到收益。

## 蒸馏数据流

```text
images
  ├─ teacher.eval() + no_grad()
  │    ├─ teacher_logits
  │    └─ teacher_features
  └─ student.train()
       ├─ student_logits
       └─ student_features

distillation_loss = CE(student_logits, labels) + KL(student_logits, teacher_logits)
feature_distillation_loss = MSE(projector(student_features), teacher_features)
total_loss = distillation_loss + feature_weight * feature_distillation_loss
```

`distillation_loss` 用在最终分类输出上，也叫 logits distillation。它让 student 同时学习两件事：

- hard label：真实标签，例如这张图就是类别 1。
- soft label：teacher 给出的类别概率分布，例如类别 1 很像类别 2，但不像类别 0。

`feature_distillation_loss` 用在中间特征上。它不直接看最终分类对不对，而是让 student 的 backbone 表示尽量接近 teacher 的 backbone 表示。如果 student 和 teacher 的通道数不同，就用一个 projector 把 student feature 映射到 teacher feature 的维度。

## Loss 公式拆解

`distillation_loss` 由两部分组成：

```python
hard_loss = F.cross_entropy(student_logits, labels)
soft_loss = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=1),
    F.softmax(teacher_logits / temperature, dim=1),
    reduction="batchmean",
) * (temperature * temperature)
loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
```

- `hard_loss`：student 学真实标签。
- `soft_loss`：student 学 teacher 的概率分布。
- `temperature`：把概率分布变平滑，让非最大类别也带有信息。
- `temperature * temperature`：补偿除以 temperature 后梯度尺度变小的问题。
- `alpha`：控制更相信 teacher 还是更相信真实标签。

`KLDivLoss` 的输入要注意：PyTorch 的 `F.kl_div` 默认要求 input 是 log probability，target 是 probability，所以 student 用 `log_softmax`，teacher 用 `softmax`。

## 验收标准

- 能写出蒸馏 loss 的组成。
- 能解释 teacher 为什么要 `eval()` 和 `no_grad()`。
- 能判断蒸馏是否真的改善了 student。
- 能解释 `distillation_loss` 和 `feature_distillation_loss` 分别用在哪里。
- 能读懂 teacher、normal student、distilled student 的对比输出。

## 答案闭环入口

<details>
<summary>先自己做</summary>

阅读 `optimization/distillation.py`，找出 hard loss、soft loss、temperature 和 alpha 分别在哪里。

</details>

## 按任务拆解的答案闭环

这一部分按第 1-2 周的方式来学：每个任务都先自己做，卡住再看提示，再看参考答案、原理解释和自检标准。蒸馏不要一上来背公式，先把它当成一个训练流程：

```text
先训练一个 teacher
再训练一个普通 student
最后训练一个 distilled student
对比三者结果
```

### 任务 1：跑通完整蒸馏 demo

<details>
<summary>先自己做</summary>

先不要改代码，只运行一次完整 demo，观察输出里是否有 teacher、normal_student、distilled_student 三段。

</details>

<details>
<summary>卡住时看提示</summary>

这个 demo 默认使用 synthetic 数据，所以不需要准备图片路径。它复用了第 1-2 周的 `build_dataloaders`、`train_one_epoch` 和 `evaluate_classifier`。

</details>

<details>
<summary>参考答案</summary>

```powershell
cd D:\项目\诗兰姆\越南扎扣件\deeplearning\model_optimization_edge_learning
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu
```

你应该重点看这些输出：

```text
distillation_setup:
  teacher_width=32 teacher_params=...
  student_width=16 student_params=...

teacher epoch=...
normal_student epoch=...
distilled_student epoch=... hard_loss=... soft_loss=... distill_loss=... feature_loss=...

final_comparison:
  teacher_val_acc=...
  normal_student_val_acc=...
  distilled_student_val_acc=...
```

</details>

<details>
<summary>为什么这样做</summary>

蒸馏不是单独一个 loss 函数就能学明白。你必须先看到三个角色：

- teacher：较大模型，先被训练好，用来提供“参考答案”。
- normal_student：较小模型，只用真实标签训练，作为 baseline。
- distilled_student：较小模型，同时学真实标签和 teacher 输出。

只有把 normal_student 和 distilled_student 放在一起比较，才能判断蒸馏有没有带来收益。

</details>

<details>
<summary>自检标准</summary>

- 输出里能看到 teacher、normal_student、distilled_student 三段。
- 能说出 teacher 参数量为什么比 student 大。
- 能说出 normal_student 是蒸馏实验的对照组。
- 能找到 `final_comparison`。

</details>

### 任务 2：理解 teacher 和 student 分别做什么

<details>
<summary>先自己做</summary>

打开 `optimization/distillation.py`，找到 teacher 和 student 的创建位置，观察它们的 `width` 参数有什么不同。

</details>

<details>
<summary>卡住时看提示</summary>

搜索 `teacher_width`、`student_width`、`TinyClassifier`。

</details>

<details>
<summary>参考答案</summary>

```python
teacher = TinyClassifier(num_classes=args.num_classes, width=args.teacher_width).to(device)
normal_student = TinyClassifier(num_classes=args.num_classes, width=args.student_width).to(device)
distilled_student = TinyClassifier(num_classes=args.num_classes, width=args.student_width).to(device)
```

默认参数：

```text
teacher_width=32
student_width=16
```

`width` 越大，backbone 通道数越多，参数量通常越大。demo 输出里会打印：

```text
teacher_params=...
student_params=...
```

</details>

<details>
<summary>为什么这样做</summary>

蒸馏通常假设 teacher 更强、更大，student 更小、更快。目标不是让 student 复制 teacher 的结构，而是让 student 学 teacher 的输出分布或中间特征。

teacher 先通过普通监督训练学会任务。之后训练 student 时，teacher 固定不动，只负责给 student 提供额外监督。

</details>

<details>
<summary>自检标准</summary>

- 能说出 teacher 和 student 的默认宽度。
- 能解释为什么 teacher 通常比 student 大。
- 能说明 teacher 在 student 蒸馏训练阶段不更新参数。

</details>

### 任务 3：理解 hard label 和 soft label

<details>
<summary>先自己做</summary>

打开 `distillation_loss`，找出哪一行对应真实标签，哪一行对应 teacher 的 soft label。

</details>

<details>
<summary>卡住时看提示</summary>

真实标签来自 `labels`，teacher 的 soft label 来自 `teacher_logits` 经过 `softmax` 后得到的概率分布。

</details>

<details>
<summary>参考答案</summary>

hard label 部分：

```python
hard_loss = F.cross_entropy(student_logits, labels)
```

soft label 部分：

```python
student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
```

如果真实标签是类别 1，hard label 只告诉 student：

```text
正确答案是 1
```

soft label 会告诉 student teacher 的完整判断：

```text
类别 0: 0.10
类别 1: 0.70
类别 2: 0.20
```

这比单个类别编号包含更多信息。

</details>

<details>
<summary>为什么这样做</summary>

真实标签是“硬”的：它只告诉模型哪个类别对。teacher 的输出是“软”的：它还告诉模型其它类别有多像。

例如一个样本真实类别是 1，teacher 认为类别 2 也有一点像，这个信息对 student 有帮助。student 不只是学“选 1”，还学“类别 2 比类别 0 更接近类别 1”。

</details>

<details>
<summary>自检标准</summary>

- 能指出 `hard_loss` 使用的是 `labels`。
- 能指出 `soft_loss` 使用的是 `teacher_logits`。
- 能解释 soft label 为什么比 hard label 多了类别相似度信息。

</details>

### 任务 4：逐行理解 `distillation_loss`

<details>
<summary>先自己做</summary>

逐行阅读 `distillation_loss`，写出最终 loss 是怎么由 hard loss 和 soft loss 混合得到的。

</details>

<details>
<summary>卡住时看提示</summary>

重点看三个参数：`temperature`、`alpha`、`temperature * temperature`。

</details>

<details>
<summary>参考答案</summary>

```python
hard_loss = F.cross_entropy(student_logits, labels)
```

这一步让 student 学真实标签。

```python
student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
```

这一步把 teacher 和 student 的 logits 都除以 temperature，再变成概率分布。temperature 越大，分布越平滑，非最大类别的信息越明显。

```python
soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)
```

这一步衡量 student 的概率分布和 teacher 的概率分布差多少。乘上 `temperature * temperature` 是为了补偿 temperature 带来的梯度尺度变化。

```python
distill_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
```

这一步混合两种监督信号：

```text
alpha 越大：越相信 teacher
alpha 越小：越相信真实标签
```

</details>

<details>
<summary>为什么这样做</summary>

如果只用 hard loss，student 只能学真实标签。如果只用 soft loss，student 可能继承 teacher 的错误。混合两者是为了让 student 同时看真实答案和 teacher 的概率判断。

`F.kl_div` 这里不是用来判断“类别是否相等”，而是用来判断两个概率分布是否接近。

</details>

<details>
<summary>自检标准</summary>

- 能写出 `distill_loss = alpha * soft_loss + (1 - alpha) * hard_loss`。
- 能解释 `alpha=0.7` 表示 soft loss 权重更大。
- 能解释 `temperature` 为什么会让 soft label 更平滑。
- 能说明 `F.kl_div` 比较的是两个概率分布。

</details>

### 任务 5：理解 `feature_distillation_loss`

<details>
<summary>先自己做</summary>

打开 `classifier_logits_and_features` 和 `feature_distillation_loss`，找出 student feature 和 teacher feature 是从哪里来的。

</details>

<details>
<summary>卡住时看提示</summary>

logits 是最终分类输出，features 是 backbone 后、head 前的中间表示。

</details>

<details>
<summary>参考答案</summary>

提取 logits 和 features：

```python
features = model.backbone(images)
pooled = model.pool(features).flatten(1)
logits = model.head(pooled)
return logits, pooled
```

feature 蒸馏：

```python
if projector is not None:
    student_features = projector(student_features)
return F.mse_loss(student_features, teacher_features.detach())
```

如果 student feature 维度和 teacher feature 维度不同，就需要 projector：

```python
projector = nn.Linear(student_channels, teacher_channels)
```

</details>

<details>
<summary>为什么这样做</summary>

logits 蒸馏约束的是最终答案，feature 蒸馏约束的是中间表示。可以把它理解成：

```text
logits 蒸馏：学 teacher 最后的答案
feature 蒸馏：学 teacher 解题过程中的中间表达
```

teacher 和 student 结构不同，feature 维度可能不同，所以需要 projector 做维度对齐。这里的 MSE loss 表示让两个 feature tensor 的数值尽量接近。

</details>

<details>
<summary>自检标准</summary>

- 能说出 logits 和 features 的区别。
- 能解释 projector 为什么存在。
- 能说明 `feature_weight=0` 时 feature 蒸馏被关闭。

</details>

### 任务 6：理解蒸馏训练循环

<details>
<summary>先自己做</summary>

打开 `train_student_one_epoch`，按顺序标出 teacher forward、student forward、loss 计算、backward、optimizer.step。

</details>

<details>
<summary>卡住时看提示</summary>

teacher 只 forward，不 backward；student 才 backward 和更新参数。

</details>

<details>
<summary>参考答案</summary>

teacher 分支：

```python
teacher.eval()
with torch.no_grad():
    teacher_logits, teacher_features = classifier_logits_and_features(teacher, images)
```

student 分支：

```python
student.train()
student_logits, student_features = classifier_logits_and_features(student, images)
```

loss 和更新：

```python
logit_loss, parts = distillation_loss(student_logits, teacher_logits, labels)
feature_loss = feature_distillation_loss(student_features, teacher_features, projector=projector)
loss = logit_loss + feature_weight * feature_loss
loss.backward()
optimizer.step()
```

</details>

<details>
<summary>为什么这样做</summary>

teacher 在这里不是被训练的模型，而是提供监督信号的模型。它必须稳定，所以用 `eval()`；它不需要梯度，所以用 `torch.no_grad()`。

student 才是要被训练的模型，所以 student 要 `train()`，并且 loss 的梯度只更新 student 和 projector。

</details>

<details>
<summary>自检标准</summary>

- 能指出 teacher forward 在 `torch.no_grad()` 里。
- 能指出只有 student 的 loss 调用了 `backward()`。
- 能解释 optimizer 为什么不应该更新 teacher 参数。

</details>

### 任务 7：判断蒸馏是否有效

<details>
<summary>先自己做</summary>

分别运行默认蒸馏、关闭 feature 蒸馏、修改 temperature/alpha 三组命令，记录 `final_comparison`。

</details>

<details>
<summary>卡住时看提示</summary>

不要只看 distilled student 的单次结果，要和 normal student 比。normal student 是判断蒸馏是否有收益的 baseline。

</details>

<details>
<summary>参考答案</summary>

默认蒸馏：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu
```

关闭 feature 蒸馏：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu --feature-weight 0
```

调整 temperature 和 alpha：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu --temperature 2 --alpha 0.5
```

记录表格：

| 实验 | teacher val_acc | normal student val_acc | distilled student val_acc | 结论 |
|---|---:|---:|---:|---|
| 默认参数 |  |  |  |  |
| `feature_weight=0` |  |  |  |  |
| `temperature=2 alpha=0.5` |  |  |  |  |

</details>

<details>
<summary>为什么这样做</summary>

蒸馏不是一定提升。它依赖几个条件：

- teacher 要足够好，否则 student 学到的是错误分布。
- student 要有一定容量，否则学不动 teacher。
- 训练轮数太少时，student 可能还没体现收益。
- `alpha`、`temperature`、`feature_weight` 不合适时，蒸馏信号可能太弱或太强。

所以判断蒸馏是否有效，一定要和普通 student baseline 对比。

</details>

<details>
<summary>自检标准</summary>

- 能说明为什么要有 normal student baseline。
- 能解释蒸馏 student 不一定每次都超过普通 student。
- 能根据 `hard_loss`、`soft_loss`、`feature_loss` 判断当前训练用了哪些监督信号。
- 能写出一行自己的结论，例如“当前 1 epoch 下 teacher 很强，但 distilled student 还没超过 normal student，需要增加 epoch 或调整权重”。

</details>

## 快速答案补充

<details>
<summary>卡住时看提示</summary>

teacher 不更新参数，只提供更平滑的类别分布；student 同时学习真实标签和 teacher 的 soft label。

</details>

<details>
<summary>参考答案</summary>

完整 demo 命令：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu
```

核心训练片段：

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

如果开启 feature 蒸馏，还会多出一项：

```python
feature_loss = feature_distillation_loss(
    student_features,
    teacher_features,
    projector=projector,
)
total_loss = distill_loss + feature_weight * feature_loss
```

`feature_weight=0` 表示只做 logits 蒸馏；`feature_weight>0` 表示同时让 student 的中间特征向 teacher 对齐。

</details>

<details>
<summary>为什么这样做</summary>

真实标签只告诉模型“正确类别是谁”，teacher 的 soft label 还隐含类别间相似度。例如某张图 teacher 认为 A=0.7、B=0.25、C=0.05，student 能学到 B 比 C 更像 A。

feature 蒸馏补充的是“中间思考过程”。logits 蒸馏只约束最终答案，feature 蒸馏会让 student 的中间表示更接近 teacher。小模型容量有限时，这不一定总能提升指标，但它能让你理解工程蒸馏里常见的 feature alignment 思路。

teacher 要 `eval()`，是因为 teacher 只负责提供稳定目标，不应该被 Dropout、BatchNorm 训练态扰动。teacher forward 放在 `torch.no_grad()` 里，是因为 teacher 不参与反向传播，不需要保存计算图，也不该更新参数。

</details>

<details>
<summary>自检标准</summary>

- teacher 处于 `eval()` 模式。
- teacher forward 包在 `torch.no_grad()` 中。
- student 的 loss 同时包含 CE 和 KL 两部分。
- 能运行 `python -m optimization.distillation --epochs 1 --device cpu`。
- 能在输出中找到 `hard_loss`、`soft_loss`、`distill_loss`、`feature_loss`。
- 能说明 `feature_weight=0` 时为什么 feature 蒸馏关闭。
- 能用 `final_comparison` 对比 teacher、普通 student 和蒸馏 student。

</details>
