# 第 19-20 周：端侧部署框架

## 目标

形成工业部署闭环。

## 学习框架

- ONNX Runtime
- TensorRT
- OpenVINO
- NCNN/MNN 了解
- Triton Inference Server 了解

## 学习内容

- ONNX simplifier
- TensorRT engine 构建
- FP16/INT8 engine
- dynamic shape profile
- C++/Python 推理接口
- batch、stream、显存、延迟分析

## 代码入口

- `deployment/`
- `deployment/trtexec_notes.md`
- `framework_reference/deployment_frameworks.md`

## 实战任务

1. 完成 PyTorch -> ONNX -> TensorRT 链路。
2. 做单图、文件夹、视频流 benchmark。
3. 对比 ONNX Runtime、TensorRT、OpenVINO。
4. 输出最终部署报告。

## 验收标准

- 能构建 TensorRT engine。
- 能解释 dynamic shape profile。
- 能给出延迟、吞吐、显存、模型大小的综合结论。

## 答案闭环

<details>
<summary>先自己做</summary>

先导出 ONNX，再用 ONNX Runtime 验证输出，最后用 TensorRT 构建 engine 并做 benchmark。

</details>

<details>
<summary>卡住时看提示</summary>

部署链路排查顺序：先保证 PyTorch 输出正确，再保证 ONNX 输出一致，最后看 TensorRT 构建和性能。不要一开始就排查 engine。

</details>

<details>
<summary>参考答案</summary>

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m deployment.export_onnx_manual --checkpoint models/handwritten_tiny_classifier.pt --output models/tiny_classifier.onnx
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m deployment.infer_onnxruntime --model models/tiny_classifier.onnx --provider cpu
trtexec --onnx=models/model.onnx --saveEngine=models/model_fp16.engine --fp16 --shapes=images:1x3x640x640
```

</details>

<details>
<summary>为什么这样做</summary>

ONNX 是跨框架中间表示，TensorRT engine 是针对具体 NVIDIA GPU 和 TensorRT 版本优化后的产物。dynamic shape profile 告诉 TensorRT 最小、最佳、最大输入尺寸，性能通常围绕 opt shape 最好。

</details>

<details>
<summary>自检标准</summary>

- ONNX Runtime 能跑通。
- TensorRT engine 构建日志保存。
- 最终报告包含硬件、CUDA、TensorRT、输入尺寸、精度模式和延迟。

</details>
