# Quantization Frameworks

## PyTorch Quantization

适合学习 PTQ/QAT 原理。

对照代码：

- `optimization/quantization_manual.py`
- `optimization/quantization_ptq.py`
- `optimization/quantization_qat.py`

重点：

- observer 如何统计激活范围。
- calibration 数据如何影响 scale。
- QAT 中 fake quant 如何模拟部署误差。

## ONNX Runtime Quantization

适合部署前量化 ONNX 模型。

重点：

- dynamic quantization。
- static quantization。
- calibration data reader。
- QDQ 格式和 QOperator 格式。

## TensorRT INT8

适合 NVIDIA GPU 工业部署。

重点：

- FP16 通常是第一优先级。
- INT8 需要校准或 QAT 友好模型。
- engine 与 GPU、TensorRT 版本强绑定。

## LLM 量化

重点概念：

- bitsandbytes 8bit/4bit 加载。
- GPTQ：后训练权重量化。
- AWQ：激活感知权重量化。
- QLoRA：量化基础模型 + LoRA 训练。

