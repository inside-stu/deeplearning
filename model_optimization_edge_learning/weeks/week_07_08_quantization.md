# 第 7-8 周：手写量化原理

## 目标

理解 INT8 量化里 scale、zero point、calibration、fake quant 到底在做什么。

## 学习内容

- FP32、FP16、INT8
- scale、zero point
- per-tensor / per-channel
- symmetric / asymmetric
- calibration
- observer
- fake quantization
- PTQ 与 QAT

## 代码入口

- `optimization/quantization_manual.py`
- `optimization/quantization_ptq.py`
- `optimization/quantization_qat.py`
- `optimization/quantization_experiment.py`

## 学习要点与代码对应

| 学习要点 | 代码位置 | 实战中用在哪里 |
|---|---|---|
| scale / zero point | `calculate_qparams` | 把 FP32 数值范围映射到 INT8/UINT8 整数范围 |
| quantize | `quantize_tensor` | 将浮点 tensor 转成整数 tensor |
| dequantize | `dequantize_tensor` | 将整数 tensor 还原成近似浮点 tensor |
| fake quantization | `fake_quantize` | 训练时模拟量化误差，但 tensor 仍以浮点形式参与计算 |
| 量化误差 | `quantization_error` | 对比原始值和反量化值，观察精度损失 |
| PTQ prepare | `prepare_ptq_model` | 给模型插入 observer，用于统计激活范围 |
| calibration | `calibrate` | 用校准数据跑 forward，收集 min/max 或直方图统计 |
| PTQ convert | `convert_ptq_model` | 根据 observer 统计结果生成量化模型 |
| QAT prepare | `prepare_qat_model` | 插入 fake quant，让训练阶段感知量化误差 |
| QAT train | `qat_train_one_epoch` | 训练模型适应量化误差 |
| 完整实验 | `quantization_experiment.py` | 同一模型上比较 FP32、PTQ INT8、QAT INT8 的精度、大小和速度 |

## 实战任务

1. 跑 `python -m optimization.quantization_manual`。
2. 修改输入 tensor 范围，观察 scale 和误差变化。
3. 用 PyTorch PTQ 做一次训练后量化。
4. 用 QAT 训练一个小模型，再 convert。
5. 对比 FP32、PTQ、QAT 的精度和延迟。
6. 运行完整量化实验，输出 FP32 / PTQ / QAT 对比表。

## 完整量化实验

前面的 `quantization_manual.py` 只回答：“一个 tensor 如何量化和反量化？”

`quantization_experiment.py` 才回答：“一个训练好的小模型量化后，精度、模型大小、速度到底发生了什么变化？”

运行命令：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.quantization_experiment --epochs 3 --qat-epochs 2 --device cpu
```

快速 smoke test：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.quantization_experiment --epochs 1 --qat-epochs 1 --device cpu --warmup 5 --repeats 20
```

输出会包含：

```text
fp32_train epoch=...
qat_train epoch=...
quantization_comparison:
| model | accuracy | val_loss | state_dict_size_mb | params | p50_ms | p95_ms | fps | conclusion |
| FP32 | ... |
| PTQ_INT8 | ... |
| QAT_INT8 | ... |
```

读表时注意三件事：

- `state_dict_size_mb` 变小，说明权重存储和打包方式变了。
- `params` 不一定减少，因为量化不是剪枝，不会删除卷积核或线性层权重个数。
- `p50_ms/fps` 不一定总是更好，INT8 是否更快取决于 CPU 后端、模型大小、算子支持和 benchmark 设置。

判断模板：

```text
如果 PTQ accuracy 接近 FP32，且 size/latency 有收益，PTQ 就可能够用。
如果 PTQ 掉点明显，但 QAT 恢复了 accuracy，可以考虑 QAT。
如果 INT8 体积变小但速度没变快，要检查后端、算子融合、batch size 和端到端流程。
```

## 量化先看懂什么

量化的核心不是“把模型变成 int8”这么一句话，而是回答三个问题：

```text
1. 一个浮点数 x 怎么映射成整数 q？
2. 整数 q 怎么近似还原成浮点数 x_hat？
3. 这个近似会带来多大误差？
```

最基础公式：

```text
q = round(x / scale + zero_point)
x_hat = (q - zero_point) * scale
```

- `scale`：一个整数刻度代表多少浮点距离。
- `zero_point`：浮点 0 对应到整数空间里的哪个位置。
- `qmin/qmax`：整数能表示的最小/最大值，例如 UINT8 是 0 到 255，INT8 是 -128 到 127。

例如 asymmetric UINT8 量化里，tensor 范围是 `[-1.0, 2.0]`：

```text
qmin=0
qmax=255
scale = (2.0 - (-1.0)) / (255 - 0) = 0.0117647
zero_point = round(0 - (-1.0) / scale) = 85
```

所以：

```text
x=0.0 -> q=round(0 / 0.0117647 + 85)=85
x=2.0 -> q=255
x=-1.0 -> q=0
```

这就是为什么量化时必须先统计数据范围。范围越大，`scale` 越大，每个整数刻度越粗，精度损失通常越明显。

## PTQ 与 QAT 的区别

| 方法 | 全称 | 训练时是否感知量化 | 主要步骤 | 适合场景 |
|---|---|---|---|---|
| PTQ | Post-Training Quantization | 否 | 已训练模型 -> calibration -> convert | 快速部署、成本低 |
| QAT | Quantization-Aware Training | 是 | prepare_qat -> 训练中 fake quant -> convert | 精度要求高、PTQ 掉点明显 |

PTQ 像是“训练完以后再压缩”。它依赖校准数据统计激活范围，如果校准集不代表真实场景，部署时就容易出现精度下降。

QAT 像是“训练时就假装自己会被量化”。它用 fake quantization 在训练中模拟量化误差，让模型提前适应，所以通常比 PTQ 更稳，但训练成本更高。

## 验收标准

- 能手算简单 tensor 的 INT8 量化结果。
- 能解释 calibration 数据为什么要覆盖真实场景。
- 能说明 QAT 为什么通常比 PTQ 更稳。

## 答案闭环入口

<details>
<summary>先自己做</summary>

先运行手写量化 demo，再手动改 tensor 的最大值和最小值，观察 scale、zero point 和误差变化。

</details>

## 按任务拆解的答案闭环

这一部分按第 1-2 周和第 6 周的方式来学。量化不要先背很多名词，先跑通最小 tensor 例子，再过渡到 PTQ/QAT。

### 任务 1：跑通手写量化 demo

<details>
<summary>先自己做</summary>

运行 `optimization.quantization_manual`，观察原始 tensor、scale、zero_point、量化后的整数、反量化后的浮点值和误差。

</details>

<details>
<summary>卡住时看提示</summary>

这个 demo 不需要模型，也不需要数据集。它只演示一个 tensor 如何从 FP32 映射到整数，再映射回近似 FP32。

</details>

<details>
<summary>参考答案</summary>

```powershell
cd D:\项目\诗兰姆\越南扎扣件\deeplearning\model_optimization_edge_learning
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.quantization_manual
```

核心代码：

```python
tensor = torch.tensor([-1.0, -0.2, 0.0, 0.3, 1.2, 2.0])
scale, zero_point, qmin, qmax = calculate_qparams(tensor, symmetric=False)
qtensor = quantize_tensor(tensor, scale, zero_point, qmin, qmax)
restored = dequantize_tensor(qtensor, scale, zero_point)
error = quantization_error(tensor, restored)
```

你要关注的是：

```text
original: 原始浮点数
scale: 浮点刻度
zero_point: 浮点 0 映射到整数的位置
quantized: 整数量化结果
restored: 反量化后的近似浮点数
error: 量化误差
```

</details>

<details>
<summary>为什么这样做</summary>

量化的本质是近似。INT8/UINT8 能表示的整数数量有限，所以不能保留所有 FP32 小数细节。你先在一个小 tensor 上看懂误差怎么产生，再去看模型量化，脑子里就不会只剩“调用 API”。

</details>

<details>
<summary>自检标准</summary>

- 能说出 `scale` 和 `zero_point` 的含义。
- 能解释为什么 `restored` 不一定等于 `original`。
- 能指出量化误差来自 `round` 和整数范围限制。

</details>

### 任务 2：手算一次 scale、zero point 和 q

<details>
<summary>先自己做</summary>

用 demo 里的 tensor 范围 `[-1.0, 2.0]`，手算 asymmetric UINT8 的 `scale`、`zero_point`，再算 `x=0.0` 对应的整数 q。

</details>

<details>
<summary>卡住时看提示</summary>

asymmetric UINT8 的整数范围是：

```text
qmin=0
qmax=255
```

公式：

```text
scale = (max_val - min_val) / (qmax - qmin)
zero_point = round(qmin - min_val / scale)
q = round(x / scale + zero_point)
```

</details>

<details>
<summary>参考答案</summary>

已知：

```text
min_val=-1.0
max_val=2.0
qmin=0
qmax=255
```

计算 scale：

```text
scale = (2.0 - (-1.0)) / (255 - 0)
      = 3.0 / 255
      = 0.0117647
```

计算 zero point：

```text
zero_point = round(0 - (-1.0) / 0.0117647)
           = round(85)
           = 85
```

计算 `x=0.0`：

```text
q = round(0.0 / 0.0117647 + 85)
  = 85
```

反量化：

```text
x_hat = (85 - 85) * 0.0117647
      = 0.0
```

</details>

<details>
<summary>为什么这样做</summary>

`zero_point=85` 表示浮点 0 在整数空间里不是 0，而是 85。这样 UINT8 的 0 到 255 能覆盖负数到正数的范围。asymmetric 量化适合数据范围不围绕 0 对称的情况。

</details>

<details>
<summary>自检标准</summary>

- 能手算 `scale=0.0117647`。
- 能解释为什么 `zero_point` 不是 0。
- 能用公式算出 `x=0.0` 对应 `q=85`。

</details>

### 任务 3：修改输入范围，观察误差变化

<details>
<summary>先自己做</summary>

打开 `optimization/quantization_manual.py`，把 demo 里的 tensor 改成范围更大的值，例如 `[-10.0, -0.2, 0.0, 0.3, 1.2, 20.0]`，重新运行并观察误差。

</details>

<details>
<summary>卡住时看提示</summary>

范围变大时，仍然只有 256 个整数刻度可用，所以每个刻度代表的浮点距离会变大。

</details>

<details>
<summary>参考答案</summary>

原始范围：

```python
tensor = torch.tensor([-1.0, -0.2, 0.0, 0.3, 1.2, 2.0])
```

实验范围：

```python
tensor = torch.tensor([-10.0, -0.2, 0.0, 0.3, 1.2, 20.0])
```

观察输出：

```text
scale 变大
restored 和 original 的差距可能变大
mae / max_abs_error 可能变大
```

</details>

<details>
<summary>为什么这样做</summary>

量化不是单纯由 bit 数决定，也由数据范围决定。范围越大，同样 8 bit 要覆盖的浮点区间越宽，刻度越粗，小数值更容易被四舍五入到相邻整数。

这也是 calibration 很重要的原因：校准集统计到的范围如果过大，精度会变粗；如果过小，真实部署时超出范围的值会被 clamp 到 `qmin/qmax`。

</details>

<details>
<summary>自检标准</summary>

- 能说明范围变大为什么会导致 `scale` 变大。
- 能说明 `clamp(qmin, qmax)` 的作用。
- 能解释为什么校准数据范围不准确会影响部署效果。

</details>

### 任务 4：理解 symmetric 和 asymmetric

<details>
<summary>先自己做</summary>

在 `calculate_qparams` 里分别阅读 `symmetric=True` 和 `symmetric=False` 两个分支，比较它们的 `qmin/qmax/zero_point`。

</details>

<details>
<summary>卡住时看提示</summary>

symmetric 通常让 0 对应整数 0；asymmetric 会根据 min/max 计算 zero point，让非对称范围也能更充分利用整数区间。

</details>

<details>
<summary>参考答案</summary>

symmetric 分支：

```python
qmin = -(2 ** (num_bits - 1))
qmax = 2 ** (num_bits - 1) - 1
max_abs = tensor.abs().max().clamp_min(1e-8)
scale = max_abs / qmax
zero_point = torch.tensor(0.0, device=tensor.device)
```

asymmetric 分支：

```python
qmin = 0
qmax = 2**num_bits - 1
min_val = tensor.min()
max_val = tensor.max()
scale = ((max_val - min_val) / (qmax - qmin)).clamp_min(1e-8)
zero_point = torch.round(qmin - min_val / scale).clamp(qmin, qmax)
```

对比：

| 类型 | 整数范围 | zero_point | 适合情况 |
|---|---|---|---|
| symmetric | INT8: `[-128, 127]` | 通常是 0 | 权重、围绕 0 分布的数据 |
| asymmetric | UINT8: `[0, 255]` | 根据 min/max 计算 | 激活、非对称分布的数据 |

</details>

<details>
<summary>为什么这样做</summary>

权重通常有正有负，且比较接近围绕 0 分布，symmetric 更简单。激活值可能被 ReLU 截断成非负，也可能范围明显偏向一侧，asymmetric 能更充分利用整数区间。

</details>

<details>
<summary>自检标准</summary>

- 能说出 symmetric 的 zero point 为什么通常是 0。
- 能说出 asymmetric 为什么需要计算 zero point。
- 能判断权重和激活分别更常见哪种量化方式。

</details>

### 任务 5：理解 fake quantization

<details>
<summary>先自己做</summary>

阅读 `fake_quantize`，解释为什么它先 quantize，再 dequantize。

</details>

<details>
<summary>卡住时看提示</summary>

fake quant 的输出仍然是浮点 tensor，但数值已经被量化误差“污染”过。

</details>

<details>
<summary>参考答案</summary>

```python
def fake_quantize(tensor: torch.Tensor, num_bits: int = 8, symmetric: bool = False) -> torch.Tensor:
    scale, zero_point, qmin, qmax = calculate_qparams(tensor, num_bits=num_bits, symmetric=symmetric)
    qtensor = quantize_tensor(tensor, scale, zero_point, qmin, qmax)
    return dequantize_tensor(qtensor, scale, zero_point)
```

流程：

```text
FP32 tensor
  -> quantize 成整数刻度
  -> dequantize 回 FP32
  -> 得到带量化误差的 FP32 tensor
```

</details>

<details>
<summary>为什么这样做</summary>

训练时很多算子仍然需要浮点计算和反向传播，所以 QAT 不是真的全程用 INT8 训练，而是在 forward 中模拟量化误差。这样模型会逐渐适应“部署时会被量化”的事实。

</details>

<details>
<summary>自检标准</summary>

- 能说明 fake quant 输出为什么仍是浮点数。
- 能说明 fake quant 为什么能模拟量化误差。
- 能解释 QAT 为什么需要 fake quant。

</details>

### 任务 6：理解 PTQ 的 prepare、calibrate、convert

<details>
<summary>先自己做</summary>

打开 `optimization/quantization_ptq.py`，按顺序找出 `prepare_ptq_model`、`calibrate`、`convert_ptq_model`。

</details>

<details>
<summary>卡住时看提示</summary>

PTQ 是训练后量化。它不重新训练模型，只用 calibration 数据统计激活范围，再 convert 成量化模型。

</details>

<details>
<summary>参考答案</summary>

prepare：

```python
model.eval()
torch.backends.quantized.engine = backend
model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
prepared = torch.ao.quantization.prepare(model, inplace=False)
```

calibrate：

```python
prepared_model.eval()
for batch_index, (images, _) in enumerate(dataloader):
    if batch_index >= max_batches:
        break
    prepared_model(images.to(device))
```

convert：

```python
prepared_model.cpu()
prepared_model.eval()
quantized_model = torch.ao.quantization.convert(prepared_model, inplace=False)
```

完整流程：

```text
FP32 trained model
  -> prepare: 插入 observer
  -> calibrate: 用校准数据跑 forward，统计范围
  -> convert: 生成量化模型
```

</details>

<details>
<summary>为什么这样做</summary>

observer 的作用是观察 tensor 的数值范围，帮助决定 scale 和 zero point。calibration 数据越接近真实部署数据，统计到的范围越可靠。convert 才是真正把可量化模块转换成量化形式的步骤。

</details>

<details>
<summary>自检标准</summary>

- 能说出 PTQ 不会重新训练模型。
- 能解释 calibration 为什么只是 forward。
- 能说明 observer 统计范围是为了生成 scale/zero point。

</details>

### 任务 7：理解 QAT 的 prepare_qat、train、convert

<details>
<summary>先自己做</summary>

打开 `optimization/quantization_qat.py`，按顺序找出 `prepare_qat_model`、`qat_train_one_epoch`、`convert_qat_model`。

</details>

<details>
<summary>卡住时看提示</summary>

QAT 和 PTQ 最大区别是：QAT 在训练过程中就模拟量化误差。

</details>

<details>
<summary>参考答案</summary>

prepare QAT：

```python
model.train()
torch.backends.quantized.engine = backend
model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
prepared_model = torch.ao.quantization.prepare_qat(model, inplace=False)
```

训练：

```python
prepared_model.train()
logits = prepared_model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

convert：

```python
prepared_model.cpu()
prepared_model.eval()
quantized_model = torch.ao.quantization.convert(prepared_model, inplace=False)
```

完整流程：

```text
FP32 model
  -> prepare_qat: 插入 fake quant / observer
  -> train: 让模型适应量化误差
  -> convert: 生成量化模型
```

</details>

<details>
<summary>为什么这样做</summary>

PTQ 是事后压缩，模型训练时不知道自己会被量化。QAT 是训练时就把量化误差暴露给模型，让模型参数主动适应这种误差，所以当 PTQ 精度下降明显时，QAT 往往更稳。

</details>

<details>
<summary>自检标准</summary>

- 能说出 QAT 为什么需要 `model.train()`。
- 能说明 QAT 训练时 forward 里已经包含 fake quant。
- 能解释为什么 QAT 通常比 PTQ 成本更高。

</details>

### 任务 8：对比 FP32、PTQ、QAT

<details>
<summary>先自己做</summary>

设计一张实验表，记录 FP32、PTQ、QAT 的 accuracy、latency、模型大小和结论。

</details>

<details>
<summary>卡住时看提示</summary>

量化不是只看模型大小。部署里至少要同时看精度、速度、模型大小和是否容易落地。

</details>

<details>
<summary>参考答案</summary>

建议记录表：

| 模型 | 方法 | Acc | Latency P50 | Latency P95 | 模型大小 | 结论 |
|---|---|---:|---:|---:|---:|---|
| TinyClassifier | FP32 |  |  |  |  | baseline |
| TinyClassifier | PTQ INT8 |  |  |  |  | 观察是否掉点 |
| TinyClassifier | QAT INT8 |  |  |  |  | 对比 PTQ 是否更稳 |

判断示例：

```text
如果 PTQ 掉点很小，优先 PTQ，因为成本低。
如果 PTQ 掉点明显，但 QAT 能恢复精度，考虑 QAT。
如果 INT8 没有带来实际速度提升，要检查后端、算子支持和端到端流程。
```

</details>

<details>
<summary>为什么这样做</summary>

量化的目标不是“得到一个 INT8 文件”本身，而是在可接受精度损失下换取更低延迟、更小模型或更低资源占用。工程里最终要根据指标做取舍。

</details>

<details>
<summary>自检标准</summary>

- 能说明 FP32 是 baseline。
- 能说明 PTQ 和 QAT 都要和 FP32 比。
- 能解释“模型变小”和“真实推理变快”不是一回事。

</details>

### 任务 9：运行完整量化实验并解释结果

<details>
<summary>先自己做</summary>

运行 `optimization.quantization_experiment`，得到 FP32、PTQ INT8、QAT INT8 三行对比结果。不要只看 accuracy，也要看模型大小和推理速度。

</details>

<details>
<summary>卡住时看提示</summary>

这个实验会先训练 FP32 baseline，再复制同一份权重做 PTQ 和 QAT。这样三者比较才公平。

</details>

<details>
<summary>参考答案</summary>

完整命令：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.quantization_experiment --epochs 3 --qat-epochs 2 --device cpu
```

快速验证命令：

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.quantization_experiment --epochs 1 --qat-epochs 1 --device cpu --warmup 5 --repeats 20
```

输出表类似：

```text
| model | accuracy | val_loss | state_dict_size_mb | params | p50_ms | p95_ms | fps | conclusion |
| FP32 | ... |
| PTQ_INT8 | ... |
| QAT_INT8 | ... |
```

每一列含义：

| 列 | 含义 |
|---|---|
| `accuracy` | 验证集准确率，判断量化是否掉点 |
| `val_loss` | 验证集 loss，观察预测分布是否变差 |
| `state_dict_size_mb` | 保存权重后的大小，观察存储是否减少 |
| `params` | 逻辑参数量，量化通常不会减少这个数 |
| `p50_ms` | 单次 forward 延迟中位数 |
| `p95_ms` | 较慢尾部延迟 |
| `fps` | 根据平均延迟估算的吞吐 |
| `conclusion` | 相对 FP32 的精度变化、大小比例和速度变化 |

</details>

<details>
<summary>为什么这样做</summary>

量化的工程目标是综合取舍，不是只看一个指标。你需要同时回答：

```text
精度掉了多少？
模型文件小了多少？
推理速度快了多少？
如果 QAT 比 PTQ 好，是否值得多花训练成本？
```

`params` 不减少是正常的，因为量化不是剪枝。剪枝会删除或屏蔽参数，量化主要是把参数用更低 bit 的整数表示，或由后端打包成更紧凑的格式。

</details>

<details>
<summary>自检标准</summary>

- 能跑出 FP32、PTQ_INT8、QAT_INT8 三行结果。
- 能解释为什么 `state_dict_size_mb` 可能减少，但 `params` 不减少。
- 能判断 PTQ 是否已经够用。
- 能判断 QAT 是否比 PTQ 更稳。
- 能说明 INT8 没有明显加速时，应该检查 CPU 后端、算子融合和 benchmark 设置。

</details>

## 快速答案补充

<details>
<summary>卡住时看提示</summary>

量化的关键公式是：`q = round(x / scale + zero_point)`，反量化是：`x_hat = (q - zero_point) * scale`。

</details>

<details>
<summary>参考答案</summary>

```powershell
C:\Users\yuhaiyang\AppData\Local\miniforge3\envs\MultiADS\python.exe -m optimization.quantization_manual
```

核心代码：

```python
scale, zero_point, qmin, qmax = calculate_qparams(tensor)
qtensor = quantize_tensor(tensor, scale, zero_point, qmin, qmax)
restored = dequantize_tensor(qtensor, scale, zero_point)
```

</details>

<details>
<summary>为什么这样做</summary>

INT8 只能表示有限整数范围，所以必须用 scale 和 zero point 把浮点数映射到整数。PTQ 用校准数据统计范围；QAT 在训练时模拟量化误差，因此通常更能适应部署时的 INT8 误差。

</details>

<details>
<summary>自检标准</summary>

- 能解释 scale 变大时精度为什么会变粗。
- 能说出 symmetric 和 asymmetric zero point 的区别。
- 能说明校准集太简单会导致哪些部署问题。

</details>
