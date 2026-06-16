# Vision Training Frameworks

## Ultralytics YOLO

重点学习：

- dataset yaml 如何映射到 Dataset/DataLoader。
- `freeze` 如何改变 `requires_grad`。
- optimizer 如何构建参数组。
- augment 配置如何影响训练样本。
- export 如何调用 ONNX/TensorRT。

对照阶段一：

- `core/train_loop.py`：训练循环。
- `core/finetune.py`：冻结与分组学习率。
- `core/metrics.py`：IoU、NMS、AP。

## MMDetection

重点学习：

- config inheritance。
- dataset registry。
- model backbone/neck/head/loss 配置。
- train pipeline 和 test pipeline。
- evaluator 和 mAP。

建议任务：

- 找到 `backbone.frozen_stages` 或类似冻结配置。
- 找到 optimizer wrapper。
- 找到 bbox coder、assigner、sampler、NMS 配置。

## TorchVision Detection

重点学习：

- Faster R-CNN / RetinaNet / SSD 的模型接口。
- target dict 格式。
- transforms v2。
- COCO evaluator。

