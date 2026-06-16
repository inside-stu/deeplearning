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

## 实战任务

1. 跑 `python -m optimization.quantization_manual`。
2. 修改输入 tensor 范围，观察 scale 和误差变化。
3. 用 PyTorch PTQ 做一次训练后量化。
4. 用 QAT 训练一个小模型，再 convert。
5. 对比 FP32、PTQ、QAT 的精度和延迟。

## 验收标准

- 能手算简单 tensor 的 INT8 量化结果。
- 能解释 calibration 数据为什么要覆盖真实场景。
- 能说明 QAT 为什么通常比 PTQ 更稳。

## 答案闭环

<details>
<summary>先自己做</summary>

先运行手写量化 demo，再手动改 tensor 的最大值和最小值，观察 scale、zero point 和误差变化。

</details>

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
