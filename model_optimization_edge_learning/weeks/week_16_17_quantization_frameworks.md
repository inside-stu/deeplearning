# 第 16-17 周：量化框架

## 目标

掌握实际部署常用量化工具，并能把框架行为对应到阶段一的量化原理。

## 学习框架

- PyTorch Quantization
- ONNX Runtime Quantization
- TensorRT INT8
- OpenVINO POT/NNCF
- bitsandbytes
- GPTQ/AWQ 基础概念

## 学习内容

- PTQ/QAT 工程流程
- calibration dataset 选择
- INT8 TensorRT engine
- LLM 4bit/8bit 量化
- 精度下降排查

## 代码入口

- `optimization/quantization_manual.py`
- `optimization/quantization_ptq.py`
- `optimization/quantization_qat.py`
- `framework_reference/quantization_frameworks.md`

## 实战任务

1. 做 ONNX Runtime INT8 量化。
2. 做 TensorRT FP16/INT8 对比。
3. 跑一个 LLM 8bit/4bit 加载 demo。
4. 输出量化误差分析报告。

## 验收标准

- 能解释 calibration 失败会带来什么现象。
- 能说明 FP16 和 INT8 的工程优先级。
- 能判断量化精度下降时先查数据、算子还是模型结构。

## 答案闭环

<details>
<summary>先自己做</summary>

先完成 ONNX Runtime INT8 量化或 TensorRT FP16 导出，再比较 FP32、FP16、INT8 的精度和延迟。

</details>

<details>
<summary>卡住时看提示</summary>

FP16 通常不需要校准，风险较低；INT8 需要校准集或 QAT 支持，精度下降时先查 calibration 是否覆盖真实场景。

</details>

<details>
<summary>参考答案</summary>

ONNX Runtime 静态量化的核心流程是：

```python
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

quantize_static(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    calibration_data_reader=my_reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)
```

TensorRT FP16 参考：

```powershell
trtexec --onnx=models/model.onnx --saveEngine=models/model_fp16.engine --fp16 --shapes=images:1x3x640x640
```

</details>

<details>
<summary>为什么这样做</summary>

框架量化做的事情和阶段一一致：统计范围、确定 scale/zero point、插入量化/反量化节点或选择低精度 kernel。差别是框架会处理大量算子兼容和图优化细节。

</details>

<details>
<summary>自检标准</summary>

- 有 FP32 与低精度模型的指标对比。
- 能解释 INT8 校准集如何选择。
- 能区分权重量化、激活量化和 LLM 4bit 加载。

</details>
