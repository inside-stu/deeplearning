# 第 11-12 周：视觉训练框架

## 目标

把阶段一手写的训练、微调、检测指标映射到成熟视觉框架。

## 学习框架

- Ultralytics YOLO
- MMDetection
- TorchVision detection

## 学习内容

- 配置文件体系
- dataset 注册
- backbone/head/loss 配置
- pretrained weight 加载
- freeze 参数内部实现
- evaluator 如何计算 mAP

## 代码入口

- `scripts/train_yolo.py`
- `scripts/export_yolo.py`
- `framework_reference/vision_frameworks.md`

## 实战任务

1. 用 Ultralytics 跑一个检测任务。
2. 用 MMDetection 跑一个同类任务。
3. 找到框架中冻结参数的位置。
4. 写一份对照：框架封装了阶段一哪些代码。

## 验收标准

- 能看懂配置文件如何映射到模型组件。
- 能解释框架训练循环和手写训练循环的对应关系。
- 不再把 `--freeze` 当成黑盒。

## 答案闭环

<details>
<summary>先自己做</summary>

用 Ultralytics 跑一个最小训练命令，然后在框架源码或日志里找：数据加载、模型构建、冻结层、optimizer、evaluator。

</details>

<details>
<summary>卡住时看提示</summary>

把阶段一的手写代码当地图：Dataset/DataLoader 对应数据配置，`requires_grad=False` 对应 freeze，`optimizer.step()` 对应框架 trainer 的训练步。

</details>

<details>
<summary>参考答案</summary>

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe scripts/train_yolo.py --model yolo11n.pt --data configs/dataset.yaml --epochs 3 --imgsz 640 --device 0 --freeze 10 --name framework_smoke
```

对照关系：

| 框架配置 | 手写实现 |
|---|---|
| dataset yaml | `Dataset` / `DataLoader` |
| `freeze` | `parameter.requires_grad=False` |
| optimizer config | `torch.optim.AdamW(...)` |
| evaluator | `core/metrics.py` 中的指标思想 |

</details>

<details>
<summary>为什么这样做</summary>

工程框架不是替代原理，而是把数据、模型、loss、优化器、评估、导出组织成稳定流程。学习时要能把每个参数映射回手写实现。

</details>

<details>
<summary>自检标准</summary>

- 能说清楚 `freeze=10` 对哪些参数产生影响。
- 能找到训练日志里的 loss 和 evaluator 结果。
- 能解释框架训练和 `core/train_loop.py` 的共同结构。

</details>
