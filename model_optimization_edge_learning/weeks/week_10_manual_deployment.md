# 第 10 周：手写导出与推理部署

## 目标

理解 PyTorch -> ONNX -> ONNX Runtime 部署链路的数据流。

## 学习内容

- `torch.onnx.export`
- ONNX graph、opset、dynamic axes
- ONNX Runtime provider
- 输入预处理、输出后处理
- latency、throughput、P50/P95
- 模型耗时与端到端耗时

## 代码入口

- `deployment/export_onnx_manual.py`
- `deployment/infer_onnxruntime.py`
- `deployment/preprocess_postprocess.py`

## 实战任务

1. 导出 `TinyClassifier` 到 ONNX。
2. 使用 ONNX Runtime 跑随机输入 benchmark。
3. 补充真实图像预处理和分类后处理。
4. 记录 PyTorch 与 ONNX Runtime 延迟差异。

## 验收标准

- 能解释 ONNX 输入输出 tensor shape。
- 能说明 dynamic axes 的意义。
- 能区分模型 forward latency 和端到端 latency。

## 答案闭环

<details>
<summary>先自己做</summary>

先导出 ONNX，再用 ONNX Runtime 对随机输入做 benchmark，记录 provider、P50/P95 和 FPS。

</details>

<details>
<summary>卡住时看提示</summary>

ONNX 导出需要：同结构 PyTorch 模型、一个 dummy input、输入输出名字、opset 和 dynamic axes。

</details>

<details>
<summary>参考答案</summary>

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m deployment.export_onnx_manual --checkpoint models/handwritten_tiny_classifier.pt --output models/tiny_classifier.onnx
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m deployment.infer_onnxruntime --model models/tiny_classifier.onnx --provider cpu
```

</details>

<details>
<summary>为什么这样做</summary>

`torch.onnx.export` 把 PyTorch 的计算图导出成中间表示。ONNX Runtime 根据 provider 选择 CPU、CUDA 等执行后端。模型 forward latency 只统计模型计算，端到端 latency 还包括读图、resize、归一化、NMS、绘制等步骤。

</details>

<details>
<summary>自检标准</summary>

- ONNX 文件成功生成。
- ONNX Runtime 能加载并输出延迟。
- 你能说出 `images` 输入的 shape 是 NCHW。

</details>
