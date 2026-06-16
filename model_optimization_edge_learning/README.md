# Model Finetuning, Compression, And Edge Deployment Learning

这是一个面向 **模型微调、压缩、量化、剪枝、蒸馏与端侧部署** 的双阶段学习工作区。

核心原则：

- 第一阶段先手写：看清机制，不依赖高级框架封装。
- 第二阶段再用框架：理解工程应用为什么使用成熟工具。
- 主线以视觉检测和工业端侧部署为核心，同时补充 LoRA/QLoRA 等参数高效微调能力。

## 学习阶段

### 阶段一：手写原理阶段

约 10 周。目标是理解每个技术点到底在代码中发生了什么。

- 手写 PyTorch 训练循环
- 手写冻结、微调、分组学习率
- 手写简化 LoRA Linear
- 手写 IoU、NMS、简化检测头
- 手写 teacher-student 蒸馏
- 手写量化/反量化，并使用 PyTorch PTQ/QAT
- 使用 PyTorch pruning API 做剪枝
- 手写 PyTorch -> ONNX -> ONNX Runtime 推理链路

### 阶段二：主流框架阶段

约 10-12 周。目标是把阶段一理解到的机制映射到真实工程框架。

- 视觉训练框架：Ultralytics YOLO、MMDetection、TorchVision detection
- 高效微调框架：Transformers、PEFT、TRL、bitsandbytes、Optimum
- 蒸馏工程方案：teacher 伪标签、logits/feature distillation
- 量化框架：PyTorch Quantization、ONNX Runtime Quantization、TensorRT INT8、OpenVINO/NNCF
- 剪枝框架：torch-pruning、NNI、NNCF、SparseML
- 部署框架：ONNX Runtime、TensorRT、OpenVINO、NCNN/MNN、Triton

## 目录

```text
model_optimization_edge_learning/
  core/                    # 阶段一：手写训练、模型、指标、微调
  optimization/            # 阶段一：手写蒸馏、量化、剪枝、分析
  deployment/              # 阶段一：手写 ONNX 导出与 ONNX Runtime 推理
  framework_reference/     # 阶段二：主流框架学习说明
  weeks/                   # 20 周课程任务书
  scripts/                 # 阶段二框架参考脚本，保留 YOLO/Ultralytics 工程入口
  templates/               # 实验记录、周复盘、误检漏检分析模板
  reports/                 # 指标表和最终报告模板
```

## 推荐学习顺序

1. 先读 `weeks/week_01_02_handwritten_training.md`，跑通手写训练循环。
2. 再读 `core/finetune.py`，理解冻结层和分组学习率。
3. 进入 `optimization/`，分别学习蒸馏、量化、剪枝。
4. 进入 `deployment/`，学习 ONNX 导出和 ONNX Runtime 推理。
5. 最后进入 `framework_reference/` 和 `scripts/`，把手写机制对应到工程框架。

每个周任务都提供折叠式答案块：先自己做，卡住再看提示和参考答案。目标是形成“实操 -> 卡住 -> 看答案 -> 继续实操”的学习闭环。

## 快速入口

手写训练基础：

```powershell
python -m core.train_loop --epochs 3 --batch-size 32
```

Checkpoint 保存与加载讲解：

```powershell
python -m core.checkpoint_walkthrough --checkpoint models/handwritten_tiny_classifier.pt --device cpu
```

手写微调/冻结机制阅读：

```powershell
python -m core.finetune
```

手写量化/反量化演示：

```powershell
python -m optimization.quantization_manual
```

ONNX 导出示例：

```powershell
python -m deployment.export_onnx_manual --checkpoint models/baseline.pt --output models/baseline.onnx
```

框架参考脚本仍保留，例如：

```powershell
python scripts/train_yolo.py --model yolo11n.pt --data configs/dataset.yaml --epochs 80 --imgsz 640 --device 0
```

这类命令用于阶段二工程应用，不作为阶段一学习主线。

## 指标记录

所有实验结果统一记录到 `reports/metrics_table.csv`。至少记录：

- 模型和方法
- 格式：PyTorch、ONNX、TensorRT、INT8 等
- mAP 或 Accuracy
- FPS
- P50/P95 延迟
- 模型大小
- 参数量
- 显存
- 结论

学习时不要只记录成功实验。失败实验、精度下降、部署不兼容，都是后面真正排查问题时最值钱的材料。
