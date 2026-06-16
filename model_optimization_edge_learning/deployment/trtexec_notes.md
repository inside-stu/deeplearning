# TensorRT trtexec Notes

`trtexec` 是学习 TensorRT 最直接的工具。它能把 ONNX 构建成 TensorRT engine，并输出构建耗时、推理延迟、吞吐、显存等信息。

## FP16 engine

```powershell
trtexec --onnx=models/model.onnx --saveEngine=models/model_fp16.engine --fp16 --shapes=images:1x3x640x640
```

重点理解：

- `--onnx`：输入 ONNX 模型。
- `--saveEngine`：保存 TensorRT engine。
- `--fp16`：允许 TensorRT 使用 FP16 kernel。
- `--shapes`：指定动态输入模型的实际 shape。

## INT8 engine

```powershell
trtexec --onnx=models/model.onnx --saveEngine=models/model_int8.engine --int8 --shapes=images:1x3x640x640
```

INT8 需要关注 calibration。框架导出 INT8 engine 时通常会替你准备校准流程，但学习时要知道本质是统计激活范围，再决定 INT8 scale。

## Dynamic shape

```powershell
trtexec --onnx=models/model.onnx --saveEngine=models/model_dynamic.engine --fp16 --minShapes=images:1x3x320x320 --optShapes=images:1x3x640x640 --maxShapes=images:4x3x960x960
```

重点理解：

- `minShapes`：允许的最小输入。
- `optShapes`：优化目标，性能通常围绕这个 shape 最好。
- `maxShapes`：允许的最大输入。

## 记录指标

每次构建 engine 后记录：

- GPU 型号
- CUDA 版本
- TensorRT 版本
- precision：FP32、FP16、INT8
- input shape
- latency mean / P50 / P95
- throughput
- engine size
- 是否包含 NMS 或后处理
