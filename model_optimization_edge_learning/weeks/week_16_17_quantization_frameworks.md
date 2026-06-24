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

## 补充：和第 7-8 周手写量化的关系

第 7-8 周你已经手写过量化最核心的公式：

```text
q = round(x / scale + zero_point)
x_hat = (q - zero_point) * scale
```

第 16-17 周要做的是把这个公式迁移到真实框架里。框架会帮你处理模型图、算子替换、observer、calibration、backend kernel、导出格式等细节，但底层问题仍然是：

```text
1. 数值范围怎么统计？
2. scale / zero_point 怎么确定？
3. 哪些权重和激活要量化？
4. 哪些算子支持低精度 kernel？
5. 精度下降后如何定位问题？
```

| 第 7-8 周手写概念 | 第 16-17 周框架概念 | 说明 |
|---|---|---|
| `calculate_qparams` | observer / calibration | 统计激活或权重范围 |
| `quantize_tensor` | quantize / convert | 把 FP32 映射到 INT8/低比特表示 |
| `dequantize_tensor` | DeQuantizeLinear / DQ node | 把整数结果还原到浮点计算域 |
| `fake_quantize` | QAT fake quant module | 训练时模拟量化误差 |
| symmetric/asymmetric | QuantType / calibration method | 决定 zero point 和整数范围 |
| quantization error | accuracy drop / layer sensitivity | 分析量化后精度下降原因 |

## 学习要点与实战对应

| 学习要点 | 框架位置 | 实战中用在哪里 | 对应任务 |
|---|---|---|---|
| PyTorch PTQ | `optimization/quantization_ptq.py` | 训练后快速量化小模型 | 任务 1 |
| PyTorch QAT | `optimization/quantization_qat.py` | PTQ 掉点明显时恢复精度 | 任务 2 |
| ONNX Runtime dynamic quant | `onnxruntime.quantization.quantize_dynamic` | 主要量化权重，简单快速 | 任务 3 |
| ONNX Runtime static quant | `quantize_static` + `CalibrationDataReader` | 量化权重和激活，部署前 INT8 | 任务 4 |
| TensorRT FP16 | `trtexec --fp16` | NVIDIA GPU 上的低风险加速首选 | 任务 5 |
| TensorRT INT8 | `trtexec --int8` + calibration | 更激进加速，需要校准 | 任务 6 |
| OpenVINO / NNCF | POT/NNCF | Intel CPU/iGPU/NPU 部署优化 | 任务 7 |
| LLM 量化 | bitsandbytes/GPTQ/AWQ | 大模型 8bit/4bit 加载或压缩 | 任务 8 |
| 精度排查 | calibration、算子、数据分布 | 找量化掉点根因 | 任务 9 |

## 环境说明

本仓库可直接运行的量化学习入口主要是 PyTorch 部分：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe weeks/code/wk07_08/quantization_manual_demo.py
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe weeks/code/wk07_08/ptq_api_demo.py
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe weeks/code/wk07_08/qat_api_demo.py
```

如果要做 ONNX Runtime 量化，需要安装并准备 ONNX 模型：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m pip install onnx onnxruntime onnxruntime-tools
```

TensorRT、OpenVINO、bitsandbytes、GPTQ/AWQ 依赖硬件和系统环境，不建议在没有对应 GPU/驱动/运行时前强行跑。学习时先读流程，等部署环境准备好再实验。

## 完整主线示例

框架量化不要从某一个 API 开始背，而是按这个工程顺序走：

```text
1. 先保留 FP32 baseline
2. 记录 FP32 accuracy / latency / model size
3. 尝试 FP16，判断是否已经满足需求
4. 尝试 PTQ INT8，准备 calibration dataset
5. 对比 INT8 accuracy / latency / model size
6. 如果 PTQ 掉点明显，再考虑 QAT 或敏感层回退
7. 导出最终部署格式，例如 ONNX / TensorRT engine / OpenVINO IR
8. 用真实输入做端到端评估
```

快速 PyTorch smoke test：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.quantization_experiment --epochs 1 --qat-epochs 1 --device cpu --warmup 5 --repeats 20
```

输出重点：

```text
quantization_comparison:
| model | accuracy | val_loss | state_dict_size_mb | params | p50_ms | p95_ms | fps | conclusion |
| FP32 | ... |
| PTQ_INT8 | ... |
| QAT_INT8 | ... |
```

读表时不要只看一个指标：

| 输出 | 你要理解的问题 |
|---|---|
| `accuracy` | 量化是否掉点 |
| `state_dict_size_mb` | 权重存储是否变小 |
| `params` | 量化不会像剪枝一样减少逻辑参数量 |
| `p50_ms / p95_ms` | 推理延迟是否真的改善 |
| `fps` | 吞吐是否改善 |
| `conclusion` | 相对 FP32 的综合收益 |

## 按任务拆解的答案闭环

这一部分按第 1-2 周的格式来学。每个任务都先自己做，卡住再看提示，再看参考答案、为什么这样做和自检标准。

### 任务 1：用 PyTorch PTQ 跑通训练后量化

<details>
<summary>先自己做</summary>

打开 `optimization/quantization_ptq.py`，按顺序找出 `prepare_ptq_model`、`calibrate`、`convert_ptq_model`。尝试运行 `weeks/code/wk07_08/ptq_api_demo.py`。

</details>

<details>
<summary>卡住时看提示</summary>

PTQ 的核心顺序是：

```text
model.eval()
prepare -> calibration forward -> convert
```

PTQ 不重新训练模型，只用校准数据统计范围。

</details>

<details>
<summary>参考答案</summary>

运行命令：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe weeks/code/wk07_08/ptq_api_demo.py
```

核心流程：

```python
model.eval()
torch.backends.quantized.engine = backend
model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
prepared = torch.ao.quantization.prepare(model, inplace=False)

with torch.no_grad():
    for images, _ in calibration_loader:
        prepared(images)

quantized_model = torch.ao.quantization.convert(prepared, inplace=False)
```

如果用仓库里的封装函数：

```python
prepared = prepare_ptq_model(model, backend="fbgemm")
calibrate(prepared, calibration_loader, device=torch.device("cpu"))
quantized_model = convert_ptq_model(prepared)
```

</details>

<details>
<summary>为什么这样做</summary>

`prepare` 会插入 observer。observer 在 calibration forward 中记录激活范围。`convert` 根据 observer 统计结果决定 scale/zero point，并把支持的模块转换成量化形式。

这和第 7-8 周手写 `calculate_qparams` 是同一件事，只是框架帮你对模型里的多个 tensor 自动统计范围。

</details>

<details>
<summary>自检标准</summary>

- 能说出 PTQ 不重新训练模型。
- 能说明 calibration 只是 forward，不需要 backward。
- 能解释 observer 和 scale/zero point 的关系。
- 能说明为什么 calibration dataset 要覆盖真实场景。

</details>

### 任务 2：用 PyTorch QAT 理解训练时感知量化

<details>
<summary>先自己做</summary>

打开 `optimization/quantization_qat.py`，按顺序找出 `prepare_qat_model`、`qat_train_one_epoch`、`convert_qat_model`。尝试运行 `weeks/code/wk07_08/qat_api_demo.py`。

</details>

<details>
<summary>卡住时看提示</summary>

QAT 的关键区别是：训练时已经插入 fake quant，模型在 forward 中感受到量化误差。

</details>

<details>
<summary>参考答案</summary>

运行命令：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe weeks/code/wk07_08/qat_api_demo.py
```

核心流程：

```python
model.train()
torch.backends.quantized.engine = backend
model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
prepared_model = torch.ao.quantization.prepare_qat(model, inplace=False)

prepared_model.train()
logits = prepared_model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()

prepared_model.eval()
quantized_model = torch.ao.quantization.convert(prepared_model, inplace=False)
```

完整实验命令：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.quantization_experiment --epochs 1 --qat-epochs 1 --device cpu --warmup 5 --repeats 20
```

</details>

<details>
<summary>为什么这样做</summary>

PTQ 是训练完成后再压缩，模型训练时不知道自己会被量化。QAT 是训练时就模拟量化误差，让参数主动适应低精度部署。因此当 PTQ 精度下降明显时，QAT 往往更稳，但训练成本更高。

</details>

<details>
<summary>自检标准</summary>

- 能说明 QAT 训练时为什么要 `model.train()`。
- 能解释 fake quant 的输出仍是浮点数，但包含量化误差。
- 能说明 QAT 为什么比 PTQ 成本更高。
- 能判断什么时候值得从 PTQ 升级到 QAT。

</details>

### 任务 3：理解 ONNX Runtime dynamic quantization

<details>
<summary>先自己做</summary>

阅读 ONNX Runtime dynamic quantization 的 API，回答：它是否需要 calibration dataset？它主要量化权重还是激活？

</details>

<details>
<summary>卡住时看提示</summary>

dynamic quantization 通常不需要校准集，主要对权重做量化，激活范围可以在运行时动态处理。它简单，但对 CNN/检测模型未必是最优。

</details>

<details>
<summary>参考答案</summary>

核心 API：

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_dynamic_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

适用理解：

```text
优点：不需要 calibration dataset，流程简单。
缺点：通常主要量化权重，激活不一定静态 INT8，性能收益依模型和后端而定。
常见场景：Transformer/Linear-heavy 模型可先试。
```

</details>

<details>
<summary>为什么这样做</summary>

dynamic quantization 的工程门槛低，适合快速试探量化收益。但如果目标是更充分的 INT8 部署，尤其是视觉模型或需要激活量化的场景，通常还要看 static quantization。

</details>

<details>
<summary>自检标准</summary>

- 能说出 dynamic quantization 通常不需要 calibration dataset。
- 能说明它和 static quantization 的区别。
- 能解释为什么它不一定带来最佳加速。

</details>

### 任务 4：理解 ONNX Runtime static quantization

<details>
<summary>先自己做</summary>

阅读 `quantize_static` 的参数，重点理解 `CalibrationDataReader`、`QuantFormat.QDQ`、`activation_type`、`weight_type`。

</details>

<details>
<summary>卡住时看提示</summary>

static quantization 需要 calibration dataset。它会根据校准样本统计激活范围，因此更接近部署时的全 INT8 流程。

</details>

<details>
<summary>参考答案</summary>

最小结构：

```python
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, batches):
        self.batches = iter(batches)

    def get_next(self):
        try:
            images = next(self.batches)
        except StopIteration:
            return None
        return {"images": images}


reader = ImageCalibrationReader(calibration_batches)

quantize_static(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)
```

关键点：

```text
CalibrationDataReader -> 按 ONNX 输入名喂校准数据
QDQ -> 图中插入 QuantizeLinear / DeQuantizeLinear 节点
activation_type -> 激活量化类型
weight_type -> 权重量化类型
```

</details>

<details>
<summary>为什么这样做</summary>

static quantization 要量化激活，因此必须知道激活在真实输入下的数值范围。校准数据太少或分布不对，会导致 scale/zero_point 不合适，部署时可能出现明显掉点。

QDQ 格式通常更直观，因为图里能看到量化和反量化节点；QOperator 格式会把某些算子替换成量化算子。

</details>

<details>
<summary>自检标准</summary>

- 能说明 `CalibrationDataReader.get_next()` 返回的是 ONNX 输入名到 numpy array 的映射。
- 能解释 QDQ 格式是什么。
- 能说明 static quantization 为什么比 dynamic quantization 更依赖数据。
- 能判断校准集是否覆盖真实亮度、尺寸、类别和场景。

</details>

### 任务 5：优先尝试 TensorRT FP16

<details>
<summary>先自己做</summary>

如果有 NVIDIA GPU 和 TensorRT，先用同一个 ONNX 模型构建 FP16 engine，并记录 latency。

</details>

<details>
<summary>卡住时看提示</summary>

FP16 通常不需要 calibration，风险比 INT8 低，是 NVIDIA GPU 部署中常见的第一优先级。

</details>

<details>
<summary>参考答案</summary>

示例命令：

```powershell
trtexec --onnx=models/model.onnx --saveEngine=models/model_fp16.engine --fp16 --shapes=images:1x3x640x640
```

如果是动态 batch：

```powershell
trtexec --onnx=models/model.onnx --saveEngine=models/model_fp16.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
```

记录：

```text
FP32 ONNX latency
TensorRT FP16 latency
engine size
输出精度/指标是否接近
```

</details>

<details>
<summary>为什么这样做</summary>

FP16 不像 INT8 那样需要校准激活范围，通常精度风险较低。很多部署场景先做 FP16 就能获得足够速度提升。如果 FP16 已经满足延迟目标，就不一定要冒 INT8 掉点风险。

</details>

<details>
<summary>自检标准</summary>

- 能说明 FP16 为什么通常比 INT8 风险低。
- 能解释 TensorRT engine 与 GPU/TensorRT 版本强绑定。
- 能记录 FP32 和 FP16 的 latency 对比。

</details>

### 任务 6：理解 TensorRT INT8 calibration

<details>
<summary>先自己做</summary>

在理解 FP16 后，再阅读 TensorRT INT8 calibration 流程。回答：INT8 为什么需要校准？calibration cache 有什么作用？

</details>

<details>
<summary>卡住时看提示</summary>

TensorRT INT8 需要知道每层激活范围，calibrator 会用代表性数据跑网络并生成量化尺度。

</details>

<details>
<summary>参考答案</summary>

命令层面示例：

```powershell
trtexec --onnx=models/model.onnx --saveEngine=models/model_int8.engine --int8 --shapes=images:1x3x640x640
```

真实工程中通常需要：

```text
1. 准备 representative calibration dataset
2. 实现或配置 TensorRT calibrator
3. 构建 INT8 engine
4. 保存 calibration cache
5. 对比 FP32/FP16/INT8 指标和 latency
```

记录表：

| 模型 | 精度 | latency | engine size | 结论 |
|---|---:|---:|---:|---|
| FP32 ONNX |  |  |  | baseline |
| TensorRT FP16 |  |  |  | 低风险加速 |
| TensorRT INT8 |  |  |  | 看是否掉点 |

</details>

<details>
<summary>为什么这样做</summary>

INT8 的速度收益来自低精度 kernel，但代价是数值表示范围变窄。calibration 决定每层激活的 scale。如果 calibration 数据不覆盖真实部署输入，某些激活会被截断或刻度过粗，从而导致精度下降。

</details>

<details>
<summary>自检标准</summary>

- 能说明 TensorRT INT8 为什么需要代表性校准数据。
- 能解释 calibration cache 的作用是复用校准结果。
- 能说明 INT8 不应该只看 latency，也要看精度掉点。

</details>

### 任务 7：理解 OpenVINO / NNCF 量化路线

<details>
<summary>先自己做</summary>

阅读 OpenVINO POT/NNCF 的基本流程，回答它和 PyTorch/ONNX Runtime 量化有什么共同点。

</details>

<details>
<summary>卡住时看提示</summary>

OpenVINO 面向 Intel CPU/iGPU/NPU 部署。POT 更偏训练后优化，NNCF 支持训练时压缩和量化感知训练。

</details>

<details>
<summary>参考答案</summary>

通用流程：

```text
1. 准备 FP32 模型
2. 转成 OpenVINO IR 或直接读取支持格式
3. 准备 calibration dataset
4. 配置 INT8 quantization preset
5. 运行量化
6. 用 OpenVINO Runtime 评估精度和 latency
```

可以这样理解：

| 工具 | 类比 |
|---|---|
| PyTorch PTQ | 训练后在 PyTorch 内量化 |
| ONNX Runtime static quant | 对 ONNX 图做校准和 QDQ/QOperator |
| OpenVINO POT | 面向 OpenVINO 部署图做训练后量化 |
| NNCF | 更完整的压缩/量化训练工具 |

</details>

<details>
<summary>为什么这样做</summary>

不同部署后端有自己的图格式和低精度 kernel。OpenVINO 的优势在 Intel 设备部署，TensorRT 的优势在 NVIDIA GPU 部署。选择工具时先看部署硬件，而不是先看 API 名字。

</details>

<details>
<summary>自检标准</summary>

- 能说明 OpenVINO 主要面向 Intel 部署生态。
- 能区分 POT 和 NNCF 的大致定位。
- 能说明为什么同一个模型在不同后端量化收益不同。

</details>

### 任务 8：理解 LLM 8bit/4bit、GPTQ、AWQ

<details>
<summary>先自己做</summary>

阅读 bitsandbytes、GPTQ、AWQ、QLoRA 的概念，回答它们分别解决什么问题。

</details>

<details>
<summary>卡住时看提示</summary>

LLM 量化常常先解决“能不能加载进显存”。这和 CNN 部署里的 TensorRT INT8 加速不是完全同一个问题。

</details>

<details>
<summary>参考答案</summary>

bitsandbytes 8bit/4bit 加载示例：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "your-model",
    quantization_config=quant_config,
    device_map="auto",
)
```

4bit / QLoRA 常见配置：

```python
import torch
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

概念对比：

| 方法 | 主要用途 | 常见理解 |
|---|---|---|
| bitsandbytes 8bit/4bit | 低显存加载和微调 | 运行时加载更省显存 |
| QLoRA | 4bit base + LoRA 训练 | 量化基础模型，训练 LoRA adapter |
| GPTQ | 后训练权重量化 | 常用于离线压缩 LLM 权重 |
| AWQ | 激活感知权重量化 | 保护重要通道，降低量化损失 |

</details>

<details>
<summary>为什么这样做</summary>

大语言模型参数量极大，很多时候 FP16 都放不进单卡。4bit/8bit 加载首先降低显存门槛。GPTQ/AWQ 更偏向把模型离线量化成可部署权重。QLoRA 则把低比特加载和参数高效微调结合起来。

</details>

<details>
<summary>自检标准</summary>

- 能说明 bitsandbytes 缺失时为什么 `BitsAndBytesConfig(load_in_4bit=True)` 会报错。
- 能区分 QLoRA 和普通 LoRA。
- 能说明 GPTQ/AWQ 更偏后训练权重量化。
- 能解释 LLM 量化常常优先解决显存问题。

</details>

### 任务 9：量化精度下降排查

<details>
<summary>先自己做</summary>

假设 INT8 后 accuracy 明显下降，列出你会按什么顺序排查。

</details>

<details>
<summary>卡住时看提示</summary>

先查 calibration 数据，再查输入预处理和算子支持，最后再考虑 QAT 或混合精度回退。

</details>

<details>
<summary>参考答案</summary>

推荐排查顺序：

```text
1. 确认 FP32 baseline 指标正常
2. 确认量化模型输入预处理完全一致
3. 检查 calibration dataset 是否覆盖真实场景
4. 增大 calibration 样本数，重新量化
5. 比较每层输出误差，定位敏感层
6. 将首层、末层、检测头等敏感层保留 FP16/FP32
7. 尝试 per-channel weight quantization
8. 尝试 QAT
9. 检查后端是否有不支持或 fallback 的算子
```

实验记录表：

| 实验 | calibration 数据 | 方法 | accuracy | latency | 现象 | 下一步 |
|---|---|---|---:|---:|---|---|
| baseline | - | FP32 |  |  | 正常 | 作为对照 |
| int8_v1 | 100 张随机图 | PTQ INT8 |  |  | 掉点 | 检查校准集 |
| int8_v2 | 真实场景 500 张 | PTQ INT8 |  |  |  | 对比 v1 |
| qat_v1 | 真实训练集 | QAT INT8 |  |  |  | 判断是否恢复 |

</details>

<details>
<summary>为什么这样做</summary>

量化精度下降通常不是一个单点原因。最常见问题是 calibration 数据不代表真实分布，导致激活 scale 不合适。其次是某些层对量化非常敏感，例如检测头、输出层、归一化附近的算子。工程上常用混合精度或敏感层回退来解决。

</details>

<details>
<summary>自检标准</summary>

- 能说明为什么先查 FP32 baseline。
- 能解释 calibration 数据太少或分布错误会导致什么现象。
- 能说出敏感层回退是什么意思。
- 能判断什么时候该尝试 QAT。

</details>

## 实验记录模板

建议每次框架量化实验都记录：

| 实验 | 模型 | 框架 | 方法 | calibration 数据 | precision | accuracy | p50 latency | p95 latency | 模型大小 | 结论 |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| ptq_demo | TinyClassifier | PyTorch | PTQ | synthetic val | INT8 |  |  |  |  |  |
| onnx_int8 | model.onnx | ONNX Runtime | static quant | real val 500 | INT8 |  |  |  |  |  |
| trt_fp16 | model.onnx | TensorRT | FP16 | 不需要 | FP16 |  |  |  |  |  |
| trt_int8 | model.onnx | TensorRT | INT8 | real val 500 | INT8 |  |  |  |  |  |

结论写法示例：

```text
本次实验中 FP16 相比 FP32 延迟下降且精度基本不变，因此可以作为低风险部署方案。
PTQ INT8 在当前 calibration 数据下出现明显掉点，下一步应扩大校准集并检查敏感层。
如果扩大校准集后仍无法恢复精度，再考虑 QAT 或将首尾层保留 FP16。
```

## 常见问题

### FP16 和 INT8 应该先做哪个？

通常先做 FP16。FP16 不需要校准，精度风险低，TensorRT 等 GPU 后端支持成熟。如果 FP16 已经满足延迟目标，就不一定要继续做 INT8。

### calibration dataset 怎么选？

选择能代表真实部署输入的数据。它不一定要很大，但要覆盖真实场景的亮度、尺寸、类别、背景、设备来源和异常情况。校准集太单一会让激活范围统计不准。

### 为什么 INT8 模型不一定更快？

可能原因包括：模型太小，量化/反量化开销抵消收益；后端没有对应 INT8 kernel；某些算子 fallback 到 FP32；batch size 或输入 shape 不适合；benchmark 没有区分模型 forward 和端到端耗时。

### 为什么参数量没有减少？

量化改变的是参数表示精度，不是删除参数。`params` 仍然表示逻辑参数个数。模型文件大小可能减少，但参数个数通常不变。

### QDQ 和 QOperator 有什么区别？

QDQ 会在 ONNX 图里显式插入 `QuantizeLinear` / `DeQuantizeLinear` 节点，量化边界更清楚。QOperator 会把部分算子替换成量化算子。实际选择取决于后端支持。

### LLM 4bit 加载和部署 INT8 是一回事吗？

不是完全一回事。LLM 4bit/8bit 加载常常首先是为了解决显存问题；TensorRT/ONNX/OpenVINO INT8 更偏部署图优化和低精度 kernel 加速。两者都属于量化，但工程目标和工具链不同。
