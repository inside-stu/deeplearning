# 第 1-2 周：手写 PyTorch 训练基础

## 目标

不依赖 YOLO、Lightning、Trainer 等训练框架，完整理解一个最小训练系统：数据从哪里来，batch 怎么组成，模型如何 forward，loss 如何 backward，参数如何更新，checkpoint 如何保存和恢复，以及训练出来的模型如何做验证集评估和样本推理。

## 本周数据路线

第 1 周不要求你一开始准备真实数据集，训练脚本默认使用合成数据。

| 路线 | 命令参数 | 数据来源 | 学习目的 |
|---|---|---|---|
| 合成数据 | `--dataset synthetic` | `SyntheticClassificationDataset` 在内存中生成随机图片和标签 | 先跑通训练循环，不被数据准备卡住 |
| CSV 图片 | `--dataset csv` | 本地图片 + `train.csv` / `val.csv` | 学习真实 Dataset、图片路径、label、transform、collate_fn |

默认命令没有数据集目录参数，是因为默认路线是 `synthetic`，数据不在磁盘上，而是在 `core/datasets.py` 里即时生成。

## 学习要点与实战对应

| 学习要点 | 代码位置 | 实战中用在哪里 | 对应任务 |
|---|---|---|---|
| Dataset | `core/datasets.py` | 决定一个样本怎么读取，返回 `(image, label)` | 任务 1、任务 2 |
| DataLoader | `core/train_loop.py` 的 `build_dataloaders` | 把多个样本组成 batch，并控制 shuffle | 任务 1、任务 2 |
| transform | `image_to_tensor` / `image_size` | 把 PIL 图片转成 NCHW tensor，并 resize 到统一尺寸 | 任务 2 |
| collate_fn | `classification_collate` | 把多个 `(image, label)` 堆叠成 batch tensor | 任务 2 |
| train/eval | `train_one_epoch` / `evaluate_classifier` | 切换训练模式和验证模式 | 任务 3、任务 6 |
| forward/loss/backward/step | `train_one_epoch` | 完成一次参数更新 | 任务 3 |
| scheduler | `main` 中的 `CosineAnnealingLR` | 每个 epoch 调整学习率 | 任务 4 |
| checkpoint | `save_checkpoint` / `load_checkpoint` | 保存和恢复模型、优化器、epoch、metrics | 任务 5 |
| inference | `core/evaluate_checkpoint.py` | 加载训练好的模型，对验证集和样本做推理 | 任务 6 |
| metrics | `core/metrics.py` | 计算 accuracy、precision、recall、confusion matrix | 任务 7 |

## 代码入口

- `core/datasets.py`
- `core/models.py`
- `core/train_loop.py`
- `core/metrics.py`
- `core/create_toy_image_dataset.py`
- `core/checkpoint_walkthrough.py`
- `core/evaluate_checkpoint.py`

## 实战任务

1. 用默认合成数据跑通训练循环。
2. 生成本地玩具图片数据集，并用 CSV 路线训练。
3. 阅读 `train_one_epoch`，逐行解释梯度清零、forward、loss、backward、参数更新。
4. 修改 batch size、学习率、scheduler，观察验证指标变化。
5. 保存 checkpoint，再手动加载并验证模型输出一致。
6. 加载训练好的 checkpoint，对验证集做评估，并打印若干样本的预测结果。
7. 阅读验证指标，理解 accuracy、precision、recall 和 confusion matrix。

## 一个关键区别

`checkpoint_walkthrough.py` 只回答：“模型保存和加载后，权重有没有保持一致？”

`evaluate_checkpoint.py` 才回答：“这个训练好的模型在验证集上表现好不好？具体样本预测对不对？”

两者都重要，但不能互相替代。

## 答案闭环

### 任务 1：用默认合成数据跑通训练循环

<details>
<summary>先自己做</summary>

直接运行训练脚本，不传任何真实数据集路径，观察脚本是否能训练并保存 checkpoint。

</details>

<details>
<summary>卡住时看提示</summary>

默认 `--dataset synthetic`，所以没有 `--data-root`、`--train-csv`、`--val-csv` 也能运行。

</details>

<details>
<summary>参考答案</summary>

```powershell
cd D:\项目\诗兰姆\越南扎扣件\model_optimization_edge_learning
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.train_loop --dataset synthetic --epochs 3 --batch-size 32 --device cpu --checkpoint models/handwritten_tiny_classifier.pt
```

</details>

<details>
<summary>为什么这样做</summary>

合成数据让你不用准备图片就能先理解训练流程。`SyntheticClassificationDataset` 在内存中生成形状为 `[3, image_size, image_size]` 的 tensor，并生成 label。它不是业务数据，只是第一个训练循环的脚手架。

</details>

<details>
<summary>自检标准</summary>

- 命令能正常结束。
- 输出包含 `epoch=... train_loss=... val_loss=... val_acc=...`。
- checkpoint 文件被创建。
- 你能指出数据来自 `SyntheticClassificationDataset`，不是来自某个图片目录。

</details>

### 任务 2：生成本地玩具图片数据集，并用 CSV 路线训练

<details>
<summary>先自己做</summary>

先生成图片和 CSV，再使用 `--dataset csv` 训练。重点观察 CSV 里的相对路径如何通过 `--data-root` 找到真实图片。

</details>

<details>
<summary>卡住时看提示</summary>

CSV 里只写相对路径，例如 `images/train/red_square_000.png`。真实图片路径等于 `data_root / image`。

</details>

<details>
<summary>参考答案</summary>

```powershell
cd D:\项目\诗兰姆\越南扎扣件\model_optimization_edge_learning
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.create_toy_image_dataset --output data/classification_toy

C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.train_loop --dataset csv --train-csv data/classification_toy/train.csv --val-csv data/classification_toy/val.csv --data-root data/classification_toy --epochs 3 --batch-size 32 --device cpu --checkpoint models/csv_toy_classifier.pt
```

CSV 内容类似：

```csv
image,label
images/train/red_square_000.png,0
images/train/green_circle_000.png,1
images/train/blue_triangle_000.png,2
```

</details>

<details>
<summary>为什么这样做</summary>

真实项目里的数据由路径、标签、读取逻辑共同决定。`CsvImageClassificationDataset.__getitem__` 会读取 CSV 的一行，打开图片，resize，转 tensor，再返回 `(image_tensor, label)`。

</details>

<details>
<summary>自检标准</summary>

- `train.csv` 和 `val.csv` 存在。
- 图片目录下有 png 图片。
- CSV 路线训练能正常跑完。
- 你能解释 `--data-root`、`--train-csv`、`--val-csv` 分别控制什么。

</details>

### 任务 3：阅读一次参数更新如何发生

<details>
<summary>先自己做</summary>

打开 `core/train_loop.py`，按顺序标出：取 batch、搬到 device、清空梯度、forward、loss、backward、step。

</details>

<details>
<summary>卡住时看提示</summary>

核心都在 `train_one_epoch` 里。搜索这些关键词：`optimizer.zero_grad`、`logits = model(images)`、`loss = criterion`、`loss.backward`、`optimizer.step`。

</details>

<details>
<summary>参考答案</summary>

```python
images = images.to(device)
labels = labels.to(device)
optimizer.zero_grad(set_to_none=True)

logits = model(images)
loss = criterion(logits, labels)

loss.backward()
optimizer.step()
```

</details>

<details>
<summary>为什么这样做</summary>

PyTorch 默认会累积梯度，所以每个 batch 开始前要清空梯度。`loss.backward()` 根据 loss 反向计算梯度，`optimizer.step()` 根据梯度更新参数。

</details>

<details>
<summary>自检标准</summary>

- 能说出 `zero_grad` 为什么在 backward 前。
- 能说出 `loss.backward()` 不会直接更新参数。
- 能说出真正更新参数的是 `optimizer.step()`。

</details>

### 任务 4：修改 batch size、学习率和 scheduler

<details>
<summary>先自己做</summary>

分别修改 `--batch-size` 和 `--lr` 运行两次，比较训练 loss、验证 accuracy 和学习率输出。

</details>

<details>
<summary>卡住时看提示</summary>

命令行参数会进入 `parse_args()`，再传给 DataLoader、optimizer 和 scheduler。

</details>

<details>
<summary>参考答案</summary>

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.train_loop --dataset synthetic --epochs 3 --batch-size 16 --lr 0.001 --device cpu
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.train_loop --dataset synthetic --epochs 3 --batch-size 64 --lr 0.0003 --device cpu
```

如果要改 scheduler，可以在 `main()` 中把：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
```

替换成你想实验的 scheduler。

</details>

<details>
<summary>为什么这样做</summary>

batch size 会影响每次参数更新看到多少样本；学习率控制每次更新步子大小；scheduler 控制学习率随 epoch 如何变化。这些都是训练稳定性的核心旋钮。

</details>

<details>
<summary>自检标准</summary>

- 能看到不同命令的 `lr=` 输出变化。
- 能解释 batch size 变大时每个 epoch 的 iteration 数会变少。
- 能说出学习率过大可能 loss 不稳定，过小可能收敛慢。

</details>

### 任务 5：保存 checkpoint，再手动加载并验证输出一致

<details>
<summary>先自己做</summary>

先训练并保存 checkpoint，然后创建两个同结构模型，加载同一份权重，用同一个输入 tensor 推理，比较输出是否一致。

</details>

<details>
<summary>卡住时看提示</summary>

验证一致需要同时满足：同结构模型、同一份权重、同一个输入、都处于 `eval()` 模式、推理时使用 `torch.no_grad()`。

</details>

<details>
<summary>参考答案</summary>

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.checkpoint_walkthrough --checkpoint models/handwritten_tiny_classifier.pt --device cpu
```

手写代码版本：

```python
import torch

from core.models import TinyClassifier
from core.train_loop import load_checkpoint

model_a = TinyClassifier(num_classes=3)
checkpoint = load_checkpoint("models/handwritten_tiny_classifier.pt", model_a)
model_a.eval()

model_b = TinyClassifier(num_classes=3)
model_b.load_state_dict(checkpoint["model"])
model_b.eval()

x = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    y_a = model_a(x)
    y_b = model_b(x)

print(torch.allclose(y_a, y_b, atol=1e-6))
print((y_a - y_b).abs().max())
```

</details>

<details>
<summary>为什么这样做</summary>

`torch.save()` 保存的是字典。`model.state_dict()` 是模型权重，`optimizer.state_dict()` 是优化器状态，`epoch` 和 `metrics` 是训练进度。`torch.load()` 只是读回字典，真正把权重放进模型的是 `model.load_state_dict()`。

这一步只验证保存加载是否正确，不评价模型好坏。

</details>

<details>
<summary>自检标准</summary>

- `loaded_models_allclose=True`。
- `loaded_models_max_abs_diff=0.0` 或非常接近 0。
- 随机初始化模型和加载权重模型的输出差异不是 0。

</details>

### 任务 6：加载 checkpoint 做验证集评估和样本推理

<details>
<summary>先自己做</summary>

训练得到 checkpoint 后，加载它，对验证集计算指标，并打印若干样本的真实标签、预测标签和 top-k 概率。

</details>

<details>
<summary>卡住时看提示</summary>

模型好不好至少看两层：

- 整体验证指标：`val_loss`、`val_accuracy_micro`、all-class macro precision/recall、present-class macro precision/recall。
- 具体样本预测：target 是什么，pred 是什么，top-k 概率是否合理。
- 类别分布：`present_classes` 和 `class_support` 能告诉你验证集中实际有哪些类别。

</details>

<details>
<summary>参考答案</summary>

合成数据 checkpoint：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.evaluate_checkpoint --checkpoint models/handwritten_tiny_classifier.pt --dataset synthetic --num-classes 3 --device cpu --sample-count 8
```

如果想随机抽样查看验证样本，加上 `--sample-mode random --sample-seed 123`。同一个 seed 会得到同一组样本，方便复现实验：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.evaluate_checkpoint --checkpoint models/handwritten_tiny_classifier.pt --dataset synthetic --num-classes 3 --device cpu --sample-count 8 --sample-mode random --sample-seed 123
```

你的 MNIST 100 类 CSV checkpoint：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m core.evaluate_checkpoint --checkpoint models/mnist_100class_tiny_classifier.pt --dataset csv --train-csv data/extracted_images/train.csv --val-csv data/extracted_images/val.csv --data-root data/extracted_images --num-classes 100 --image-size 32 --device cpu --max-val-samples 512 --sample-count 12 --topk 5 --sample-mode random --sample-seed 123
```

输出会包含：

```text
val_loss=...
val_accuracy_micro=...
macro_precision_all_classes=...
macro_recall_all_classes=...
macro_precision_present_classes=...
macro_recall_present_classes=...
present_classes=[...]
class_support={...}
sample_predictions:
  000 dataset_index=... path=... target=... pred=... correct=True top5=[...]
```

</details>

<details>
<summary>为什么这样做</summary>

训练时打印的 `val_acc` 是每个 epoch 的验证结果，但训练结束后你还需要能独立加载某个 checkpoint 来复查模型。这是工程里非常常见的动作：训练、保存、加载、评估、抽样看预测。只有 checkpoint 能被加载并且验证集指标和样本预测都合理，才说明模型真的可用。

`val_accuracy_micro` 是按样本统计的整体准确率：验证集中有多少样本预测对了。macro precision/recall 是先分别计算每个类别的 precision/recall，再对类别求平均，所以它让每个类别权重相同，适合观察类别不均衡问题。

这里故意同时输出两个 macro 口径：

- `macro_precision_all_classes` / `macro_recall_all_classes`：对所有类别求平均。即使某个类别在当前验证集中没有出现，也会参与平均，所以小验证集或类别缺失时可能很低。
- `macro_precision_present_classes` / `macro_recall_present_classes`：只对验证集中实际出现的类别求平均，更适合快速判断“当前这批验证样本上的表现”。

例如 3 分类任务里，如果验证集刚好只有类别 1，模型也全部预测对了，那么 `val_accuracy_micro=1.0000`。但 all-class macro 会近似是 `(0 + 1 + 0) / 3 = 0.3333`，因为类别 0 和类别 2 在这批验证集中没有有效表现。这不是模型输出矛盾，而是指标平均口径不同。

默认 `--sample-mode first` 会打印验证集中的前 N 个样本，不是随机抽样；使用 `--sample-mode random --sample-seed 123` 才是可复现的随机抽样。

</details>

<details>
<summary>自检标准</summary>

- 评估脚本能成功加载 checkpoint。
- 输出验证集指标，并能解释 micro accuracy、all-class macro、present-class macro 的区别。
- 能根据 `present_classes` 和 `class_support` 判断验证集是否缺少某些类别。
- 输出若干样本的 `target`、`pred`、`correct` 和 top-k 概率。
- 能说明默认样本输出是前 N 个样本，随机抽样需要显式使用 `--sample-mode random`。
- 如果 `val_accuracy_micro` 很低，你能说明它可能是训练轮数太少、类别太多、模型太小、数据预处理不合适或学习率不合适。

</details>

### 任务 7：阅读验证指标

<details>
<summary>先自己做</summary>

打开 `core/metrics.py`，阅读 `accuracy`、`confusion_matrix` 和 `precision_recall_from_confusion`。

</details>

<details>
<summary>卡住时看提示</summary>

accuracy 看整体对了多少；precision 看预测为某类时有多少是真的；recall 看某类真实样本有多少被找出来。

</details>

<details>
<summary>参考答案</summary>

验证逻辑在 `evaluate_classifier`：

```python
logits = model(images)
loss = criterion(logits, labels)
total_accuracy += accuracy(logits, labels) * images.shape[0]
matrix += confusion_matrix(logits, labels, num_classes=num_classes)
precision, recall = precision_recall_from_confusion(matrix)
```

</details>

<details>
<summary>为什么这样做</summary>

单看 loss 不够。分类任务至少要看 accuracy 和混淆矩阵；工业视觉里还要关注某些关键类别的 recall，因为漏检可能比误检代价更高。

</details>

<details>
<summary>自检标准</summary>

- 能解释混淆矩阵的行和列分别代表什么。
- 能说出 precision 和 recall 的区别。
- 能说明为什么验证时要 `model.eval()` 和 `torch.no_grad()`。

</details>

## 验收标准

- 能说清楚第 1 周默认数据来自 `SyntheticClassificationDataset`，不是某个隐藏目录。
- 能生成并训练 CSV 图片玩具数据集。
- 能解释 Dataset、DataLoader、transform、collate_fn 在训练中分别做什么。
- 能手写或复述一次参数更新流程。
- 能保存 checkpoint、加载 checkpoint，并验证同权重模型输出一致。
- 能加载 checkpoint 做验证集评估和样本推理，判断模型好不好。
