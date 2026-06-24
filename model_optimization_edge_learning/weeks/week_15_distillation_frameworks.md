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

## 补充：和第 6 周手写蒸馏的关系

第 6 周你已经看过手写蒸馏的核心流程：

```text
teacher.eval() + torch.no_grad()
student.train()
hard_loss = CE(student_logits, labels)
soft_loss = KL(student_logits / T, teacher_logits / T) * T^2
loss = alpha * soft_loss + (1 - alpha) * hard_loss
```

第 15 周要做的事情，是把这个最小流程迁移到更工程化的训练方案里。也就是说，不只是知道 `distillation_loss` 怎么写，还要能回答：

- teacher 输出是在线实时算，还是提前缓存。
- student 学的是 logits、features、伪标签，还是多种信号组合。
- 分类任务和检测任务的蒸馏差异在哪里。
- teacher 质量、伪标签阈值、loss 权重如何影响 student。
- 最后是否真的提升了速度/精度综合收益。

| 第 6 周手写概念 | 第 15 周工程问题 | 说明 |
|---|---|---|
| `teacher.eval()` | 固定 teacher | teacher 只提供监督信号，不参与训练 |
| `torch.no_grad()` | 降低 teacher 推理成本 | 不保存 teacher 计算图，减少显存 |
| `teacher_logits` | soft label 缓存 | 可以在线算，也可以离线保存 |
| `KLDivLoss` | logits distillation | student 学 teacher 的类别概率分布 |
| `feature_distillation_loss` | feature alignment | student 中间特征对齐 teacher 中间特征 |
| `alpha` | hard/soft loss 权重 | 控制更相信真实标签还是 teacher |
| `temperature` | soft label 平滑程度 | 控制非最大类别的信息强度 |
| 普通 student baseline | 工程收益判断 | 没有 baseline 就无法判断蒸馏是否有效 |

## 学习要点与实战对应

| 学习要点 | 代码/框架位置 | 实战中用在哪里 | 对应任务 |
|---|---|---|---|
| logits 蒸馏 | `distillation_loss` | 分类 student 学 teacher soft label | 任务 1 |
| teacher 固定 | `teacher.eval()`、`torch.no_grad()` | 保证 teacher 稳定且不更新 | 任务 2 |
| feature 蒸馏 | `feature_distillation_loss` | 对齐 teacher/student 中间特征 | 任务 3 |
| 在线蒸馏 | 每个 batch 调 teacher | 省存储，但训练慢 | 任务 4 |
| 离线缓存 | 保存 teacher logits/features/pseudo labels | 训练快，但占磁盘且要管理版本 | 任务 4 |
| 检测蒸馏 | boxes/classes/scores/features | 需要置信度过滤和框匹配 | 任务 5 |
| 结果对比 | teacher/student/distilled student | 判断蒸馏是否有收益 | 任务 6 |
| 工程选择 | 指标、速度、存储、部署 | 决定是否值得蒸馏 | 任务 7 |

## 环境说明

当前仓库里的最小可运行蒸馏 demo 主要依赖 PyTorch，不需要额外安装复杂框架：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu
```

如果要迁移到 Hugging Face、MMDetection 或其它工程框架，需要额外关注：

```text
1. teacher/student 的模型加载方式
2. dataset 和 collator 是否同时能喂给 teacher 与 student
3. loss 里 logits/features/pseudo labels 的 shape 是否对齐
4. teacher 输出缓存格式是否稳定
5. 最终评估脚本是否同时记录精度、速度和模型大小
```

## 完整主线示例

不要把后面的参考答案看成互不相关的小片段。真正的完整蒸馏闭环应该按这条线走：

```text
1. 准备 teacher 和 student
2. 训练或加载一个可靠 teacher
3. 训练普通 student，作为 baseline
4. 选择蒸馏信号：logits、features、伪标签
5. 选择 teacher 方案：在线推理或离线缓存
6. 训练 distilled student
7. 对比 teacher、normal student、distilled student
8. 记录精度、速度、参数量、推理延迟和结论
```

当前仓库中的最小运行命令：

```powershell
cd D:\project\deeplearning\model_optimization_edge_learning
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu
```

你应该重点看这些输出：

```text
distillation_setup:
  teacher_width=...
  student_width=...
  temperature=... alpha=... feature_weight=...

teacher epoch=...
normal_student epoch=...
distilled_student epoch=... hard_loss=... soft_loss=... distill_loss=... feature_loss=...

final_comparison:
  teacher_val_acc=...
  normal_student_val_acc=...
  distilled_student_val_acc=...
```

这些输出分别回答：

| 输出 | 你要理解的问题 |
|---|---|
| `teacher_width/student_width` | teacher 是否比 student 更大 |
| `temperature/alpha/feature_weight` | 蒸馏 loss 的权重如何设置 |
| `hard_loss` | student 学真实标签的损失 |
| `soft_loss` | student 学 teacher soft label 的损失 |
| `feature_loss` | student 中间特征是否对齐 teacher |
| `normal_student_val_acc` | 没有蒸馏时 student 的 baseline |
| `distilled_student_val_acc` | 蒸馏后 student 是否改善 |

## 扩展版答案闭环

下面每个任务都按“先自己做 -> 卡住时看提示 -> 参考答案 -> 为什么这样做 -> 自检标准”的顺序拆解。学习时建议先完整跑一次 `optimization.distillation`，再回到单个任务逐步理解。

### 任务 1：理解 logits distillation

<details>
<summary>先自己做</summary>

打开 `optimization/distillation.py`，找到 `distillation_loss`，标出 hard loss、soft loss、temperature 和 alpha。

</details>

<details>
<summary>卡住时看提示</summary>

真实标签来自 `labels`，teacher 的 soft label 来自 `teacher_logits`。student 最终要同时接近真实标签和 teacher 分布。

</details>

<details>
<summary>参考答案</summary>

核心代码是：

```python
hard_loss = F.cross_entropy(student_logits, labels)

student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
soft_loss = F.kl_div(
    student_log_probs,
    teacher_probs,
    reduction="batchmean",
) * (temperature * temperature)

distill_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
```

各部分含义：

```text
hard_loss  -> student 学真实标签
soft_loss  -> student 学 teacher 的概率分布
temperature -> 让 soft label 更平滑
alpha -> 控制 teacher 信号和真实标签的比例
```

</details>

<details>
<summary>为什么这样做</summary>

真实标签只告诉 student “正确类别是谁”。teacher 的 soft label 还会告诉 student 类别之间的相似度。例如 teacher 认为类别 A 是 0.70，类别 B 是 0.25，类别 C 是 0.05，student 就能学到 B 比 C 更接近 A。

`teacher_logits.detach()` 很重要，它表示 teacher 只提供目标分布，不被 student 的 loss 反向更新。

</details>

<details>
<summary>自检标准</summary>

- 能写出 `distill_loss = alpha * soft_loss + (1 - alpha) * hard_loss`。
- 能解释 `temperature` 为什么会让概率分布更平滑。
- 能说明 `alpha` 越大，student 越依赖 teacher。
- 能说明为什么 teacher logits 要 `detach()`。

</details>

### 任务 2：确认 teacher 不参与训练

<details>
<summary>先自己做</summary>

打开 `train_student_one_epoch`，找出 teacher forward 是在哪里执行的，确认它有没有参与 `backward()`。

</details>

<details>
<summary>卡住时看提示</summary>

teacher 应该处于 `eval()` 模式，并且 forward 放在 `torch.no_grad()` 中。optimizer 不应该接收 teacher 参数。

</details>

<details>
<summary>参考答案</summary>

teacher 分支：

```python
student.train()
teacher.eval()

with torch.no_grad():
    teacher_logits, teacher_features = classifier_logits_and_features(teacher, images)
```

student 分支：

```python
student_logits, student_features = classifier_logits_and_features(student, images)
logit_loss, parts = distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=temperature,
    alpha=alpha,
)
loss.backward()
optimizer.step()
```

optimizer 只接收 student 和 projector：

```python
optimizer = torch.optim.AdamW(trainable_parameters(student, projector), lr=lr)
```

</details>

<details>
<summary>为什么这样做</summary>

teacher 是已经训练好的参考模型。蒸馏阶段要改变的是 student，不是 teacher。`eval()` 保证 Dropout、BatchNorm 等行为稳定；`torch.no_grad()` 避免保存 teacher 计算图，节省显存和计算。

</details>

<details>
<summary>自检标准</summary>

- 能指出 teacher forward 在 `torch.no_grad()` 内。
- 能指出 optimizer 没有 teacher 参数。
- 能解释为什么 teacher 要 `eval()`。
- 能解释为什么只有 student loss 调用 `backward()`。

</details>

### 任务 3：理解 feature distillation

<details>
<summary>先自己做</summary>

打开 `classifier_logits_and_features`、`feature_distillation_loss` 和 `create_feature_projector`，说明 feature 是从哪里来的，projector 为什么存在。

</details>

<details>
<summary>卡住时看提示</summary>

logits 是最终分类输出，features 是中间表示。teacher 和 student 的通道数可能不同，所以需要 projector 做维度对齐。

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

创建 projector：

```python
student_channels = student.backbone.out_channels
teacher_channels = teacher.backbone.out_channels
if student_channels == teacher_channels:
    return None
return nn.Linear(student_channels, teacher_channels)
```

最终 loss：

```python
loss = logit_loss + feature_weight * feature_loss
```

</details>

<details>
<summary>为什么这样做</summary>

logits 蒸馏让 student 学 teacher 的最终答案；feature 蒸馏让 student 学 teacher 的中间表示。检测、分割、多尺度视觉任务里，feature 蒸馏经常比单纯 logits 蒸馏更重要，因为中间特征直接影响定位和细节表达。

但是 feature 蒸馏不是越强越好。`feature_weight` 太大时，student 可能被迫模仿自己容量无法表达的 teacher feature，反而影响分类 loss。

</details>

<details>
<summary>自检标准</summary>

- 能说出 logits 和 features 的区别。
- 能说明 projector 解决的是维度不一致问题。
- 能解释 `feature_weight=0` 表示关闭 feature 蒸馏。
- 能说明 feature 蒸馏不一定每次提升指标。

</details>

### 任务 4：设计在线蒸馏和离线缓存方案

<details>
<summary>先自己做</summary>

分别写出在线蒸馏和离线缓存蒸馏的数据流，说明它们各自的优缺点。

</details>

<details>
<summary>卡住时看提示</summary>

在线蒸馏每个 batch 都跑 teacher。离线缓存先把 teacher 输出保存下来，训练 student 时直接读缓存。

</details>

<details>
<summary>参考答案</summary>

在线蒸馏：

```python
for images, labels in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(images)

    student_logits = student(images)
    loss, parts = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
```

离线缓存第一阶段，生成 teacher 输出：

```python
teacher.eval()
cache = []

with torch.no_grad():
    for sample_id, images, labels in train_loader:
        logits = teacher(images)
        cache.append({
            "sample_id": sample_id,
            "labels": labels.cpu(),
            "teacher_logits": logits.cpu(),
        })

torch.save(cache, "outputs/distillation/teacher_logits.pt")
```

离线缓存第二阶段，训练 student：

```python
cache = torch.load("outputs/distillation/teacher_logits.pt")

for batch in cache_loader:
    images = batch["images"].to(device)
    labels = batch["labels"].to(device)
    teacher_logits = batch["teacher_logits"].to(device)

    student_logits = student(images)
    loss, parts = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
```

对比：

| 方案 | 优点 | 风险 |
|---|---|---|
| 在线蒸馏 | 实现直接，teacher 输出总是和当前增强一致 | 每个 epoch 都跑 teacher，训练慢 |
| 离线缓存 | student 训练快，可重复实验 | 占磁盘，缓存要和数据增强/模型版本对齐 |

</details>

<details>
<summary>为什么这样做</summary>

工程蒸馏里，teacher 往往比 student 大很多。如果每个 epoch 都跑 teacher，训练成本会很高。离线缓存能把 teacher 推理成本前置，后续调 `alpha`、`temperature`、student 结构时复用同一份 teacher 输出。

但离线缓存也有代价：如果训练时用了强随机增强，缓存 logits 可能和增强后的图片不匹配；如果 teacher 或数据版本变了，缓存也必须重新生成。

</details>

<details>
<summary>自检标准</summary>

- 能说出在线蒸馏为什么慢。
- 能说出离线缓存为什么要记录 `sample_id`。
- 能说明随机数据增强会影响 teacher 缓存。
- 能判断什么时候该在线，什么时候该离线。

</details>

### 任务 5：理解检测任务蒸馏

<details>
<summary>先自己做</summary>

把分类蒸馏和检测蒸馏做对比：分类任务只有类别 logits，检测任务还要处理哪些输出？

</details>

<details>
<summary>卡住时看提示</summary>

检测输出通常包括 boxes、classes、scores，有时还包括 FPN 多尺度 features。检测蒸馏难点是框匹配和置信度过滤。

</details>

<details>
<summary>参考答案</summary>

分类蒸馏：

```text
输入 image
teacher 输出 class logits
student 输出 class logits
loss = CE + KL
```

检测伪标签蒸馏：

```python
teacher.eval()
pseudo_targets = []

with torch.no_grad():
    detections = teacher(images)

for det in detections:
    keep = det["scores"] > score_threshold
    pseudo_targets.append({
        "boxes": det["boxes"][keep],
        "labels": det["labels"][keep],
        "scores": det["scores"][keep],
    })

student_loss = detector_train_step(student, images, pseudo_targets)
```

检测 feature 蒸馏：

```text
teacher FPN features: P3, P4, P5
student FPN features: P3, P4, P5
对每个尺度做 MSE/KL/attention distillation
```

分类蒸馏和检测蒸馏的区别：

| 项目 | 分类蒸馏 | 检测蒸馏 |
|---|---|---|
| teacher 输出 | class logits | boxes/classes/scores/features |
| 对齐难点 | 类别分布对齐 | 框匹配、尺度对齐、前景背景不平衡 |
| 常见做法 | CE + KL | 伪标签、feature 蒸馏、logit 蒸馏 |
| 质量控制 | teacher acc | score 阈值、NMS、框质量 |

</details>

<details>
<summary>为什么这样做</summary>

检测任务不只是判断“这张图是什么类别”，还要判断“目标在哪里”。teacher 生成的低质量框会直接误导 student，所以检测伪标签一般要过滤低置信度结果，并保留 score、box、class 等信息。

如果做 feature 蒸馏，还要考虑多尺度特征是否一一对应。比如 FPN 的 P3/P4/P5 尺度不同，不能随便拿一个 feature 去对齐另一个 feature。

</details>

<details>
<summary>自检标准</summary>

- 能说出检测蒸馏比分类蒸馏多了 boxes/scores。
- 能解释为什么要过滤低置信度伪标签。
- 能说明多尺度 feature 蒸馏为什么要按尺度对齐。
- 能说出检测蒸馏失败时要先检查 teacher 检测质量。

</details>

### 任务 6：对比 teacher、普通 student、蒸馏 student

<details>
<summary>先自己做</summary>

运行一次默认蒸馏，再运行一次关闭 feature 蒸馏，记录 `final_comparison`。

</details>

<details>
<summary>卡住时看提示</summary>

判断蒸馏是否有效，不能只看 distilled student 的绝对指标，要和 normal student baseline 对比。

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

调整 soft/hard 权重：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.distillation --epochs 1 --device cpu --temperature 2 --alpha 0.5
```

实验记录表：

| 实验 | teacher val_acc | normal student val_acc | distilled student val_acc | 速度/参数变化 | 结论 |
|---|---:|---:|---:|---|---|
| 默认参数 |  |  |  |  |  |
| `feature_weight=0` |  |  |  |  |  |
| `temperature=2 alpha=0.5` |  |  |  |  |  |

结论写法示例：

```text
本次实验中 teacher 参数量大于 student，normal student 是判断蒸馏收益的 baseline。
如果 distilled student 高于 normal student，说明 teacher soft label 或 feature 对齐带来了收益。
如果没有提升，优先检查 teacher 质量、训练 epoch、alpha/temperature/feature_weight 是否合适。
```

</details>

<details>
<summary>为什么这样做</summary>

蒸馏的目标不是超过 teacher，而是在更小、更快的 student 上尽量保留 teacher 的效果。真正要比较的是：

```text
normal student -> distilled student 是否提升
distilled student -> teacher 差距是否缩小
student 推理速度/参数量是否满足部署目标
```

</details>

<details>
<summary>自检标准</summary>

- 能解释 normal student 为什么必须存在。
- 能说明 distilled student 不一定每次都超过 normal student。
- 能根据 `hard_loss/soft_loss/feature_loss` 判断当前用了哪些监督信号。
- 能写出自己的实验结论，而不是只复制指标。

</details>

### 任务 7：判断蒸馏工程方案是否值得做

<details>
<summary>先自己做</summary>

从精度、速度、训练成本、缓存成本、部署复杂度五个角度，判断当前任务是否适合蒸馏。

</details>

<details>
<summary>卡住时看提示</summary>

蒸馏不是默认一定值得。如果 student 已经足够好，或者 teacher 本身质量不高，蒸馏可能浪费时间。

</details>

<details>
<summary>参考答案</summary>

| 场景 | 推荐方案 | 原因 |
|---|---|---|
| teacher 很强，student 明显更快 | logits + feature 蒸馏 | 有机会把 teacher 能力迁移到小模型 |
| 数据无标签但 teacher 可靠 | 伪标签蒸馏 | teacher 可以扩充训练监督 |
| teacher 推理很慢，实验要反复调参 | 离线缓存 | 避免每次训练重复跑 teacher |
| 数据增强强依赖随机图像变换 | 在线蒸馏 | teacher 输出和当前增强样本一致 |
| 检测任务中 teacher 框质量一般 | 谨慎使用伪标签 | 低质量 boxes 会误导 student |
| student 容量太小 | 先增大 student 或减弱蒸馏权重 | 太小的 student 可能学不动 |

工程决策表：

| 指标 | 需要记录什么 |
|---|---|
| 精度 | teacher、normal student、distilled student 的 val/test metric |
| 速度 | FPS、latency、batch 推理时间 |
| 模型大小 | 参数量、权重文件大小 |
| 训练成本 | 是否每轮跑 teacher、是否需要缓存 |
| 部署复杂度 | student 是否能独立部署，不依赖 teacher |

</details>

<details>
<summary>为什么这样做</summary>

蒸馏最终服务的是部署目标。一个 distilled student 即使精度只提升一点，如果推理速度大幅提升，也可能是好方案。反过来，如果蒸馏训练很复杂、指标也没有超过普通 student，就应该优先检查 teacher 或换更简单的训练策略。

</details>

<details>
<summary>自检标准</summary>

- 能说出什么时候用在线蒸馏，什么时候用离线缓存。
- 能解释为什么 teacher 质量是蒸馏上限。
- 能说明检测伪标签为什么要设置 score threshold。
- 能用精度和速度一起判断蒸馏是否值得。

</details>

## 实验记录模板

建议每次蒸馏实验都记录：

| 实验 | teacher | student | 蒸馏信号 | temperature | alpha | feature_weight | teacher metric | normal student metric | distilled student metric | latency/FPS | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| distill_demo | TinyClassifier-32 | TinyClassifier-16 | logits+feature | 4.0 | 0.7 | 0.1 |  |  |  |  |  |

结论写法示例：

```text
本次实验中 teacher 参数量更大，normal student 作为 baseline。
蒸馏 student 使用 logits 和 feature 两种监督信号。
如果 distilled student 指标高于 normal student，说明 teacher 信号有效；
如果没有提升，需要继续检查 teacher 质量、训练轮数、temperature、alpha 和 feature_weight。
最终是否采用蒸馏，还要结合 student 的推理速度和部署目标判断。
```

## 常见问题

### teacher 为什么不能一起训练？

蒸馏里的 teacher 是监督来源。如果 teacher 在 student 训练时也不断变化，student 学到的目标会不稳定。通常先训练或加载一个固定 teacher，再训练 student。

### `temperature` 越大越好吗？

不是。temperature 太小，soft label 接近 one-hot，信息不够丰富；temperature 太大，分布过平，teacher 的判断差异被冲淡。常见做法是从 `2`、`4`、`8` 做小范围实验。

### `alpha` 应该怎么选？

`alpha` 控制 soft loss 权重。teacher 可靠时可以适当增大；teacher 不稳定或数据标签很可靠时，可以降低。比如先试：

```text
alpha=0.5
alpha=0.7
alpha=0.9
```

### feature 蒸馏一定要开吗？

不一定。feature 蒸馏会增加约束，也可能增加训练难度。分类任务可以先只做 logits 蒸馏；检测、分割等任务再逐步加入 feature 蒸馏。

### 伪标签蒸馏和 logits 蒸馏有什么区别？

logits 蒸馏直接学习 teacher 的概率分布。伪标签蒸馏先把 teacher 输出转成类似真实标注的结果，例如 boxes/classes/scores，再让 student 当作普通标注训练。

### 蒸馏失败先查什么？

优先按这个顺序排查：

```text
1. teacher 自己是否足够好
2. normal student baseline 是否合理
3. hard_loss/soft_loss/feature_loss 数值是否异常
4. alpha、temperature、feature_weight 是否过强或过弱
5. 检测伪标签的 score threshold 是否合适
6. student 容量是否太小
```
