# Edge Deployment Frameworks

## ONNX Runtime

重点：

- provider：CUDA、CPU、TensorRT、OpenVINO。
- 输入输出 tensor 名称和 shape。
- 图优化等级。
- 模型耗时和端到端耗时分开统计。

对照代码：

- `deployment/infer_onnxruntime.py`

## TensorRT

重点：

- ONNX parser。
- engine。
- FP16/INT8。
- dynamic shape profile。
- workspace。
- plugin。

对照文档：

- `deployment/trtexec_notes.md`

## OpenVINO

重点：

- Intel CPU/iGPU/NPU 部署。
- model optimizer。
- NNCF 量化。
- latency/throughput mode。

## NCNN/MNN

重点：

- 移动端/嵌入式部署。
- 算子支持。
- 模型转换。
- 后处理是否要手写。

## Triton Inference Server

重点：

- 多模型服务。
- dynamic batching。
- model repository。
- ensemble pipeline。

