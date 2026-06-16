# 最终项目报告

## 1. 项目目标

- 任务：
- 数据集：
- 部署目标：
- 约束：精度、速度、模型大小、显存、延迟

## 2. 阶段一：手写原理验证

| 模块 | 代码入口 | 是否完成 | 关键结论 |
|---|---|---|---|
| 手写训练循环 | `core/train_loop.py` |  |  |
| 冻结/微调 | `core/finetune.py` |  |  |
| LoRA | `core/models.py` |  |  |
| 蒸馏 | `optimization/distillation.py` |  |  |
| 量化 | `optimization/quantization_manual.py` |  |  |
| 剪枝 | `optimization/pruning.py` |  |  |
| ONNX 推理 | `deployment/infer_onnxruntime.py` |  |  |

## 3. 阶段二：框架工程实验

| 模型 | 方法 | 框架 | 格式 | mAP/Acc | FPS | P50 延迟 | P95 延迟 | 模型大小 | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| baseline |  |  |  |  |  |  |  |  |  |
| finetuned |  |  |  |  |  |  |  |  |  |
| distilled |  |  |  |  |  |  |  |  |  |
| quantized |  |  |  |  |  |  |  |  |  |
| pruned |  |  |  |  |  |  |  |  |  |
| deployed |  |  |  |  |  |  |  |  |  |

## 4. 关键取舍

| 方案 | 收益 | 代价 | 是否推荐 |
|---|---|---|---|
| 全量微调 |  |  |  |
| LoRA/Adapter |  |  |  |
| 蒸馏 |  |  |  |
| FP16 |  |  |  |
| INT8 |  |  |  |
| 剪枝 |  |  |  |

## 5. 部署链路

- PyTorch 权重：
- ONNX 模型：
- TensorRT engine：
- 推理脚本：
- 输入输出格式：

## 6. 结论

- 推荐模型：
- 推荐部署格式：
- 推荐阈值：
- 主要风险：
- 下一步优化：

