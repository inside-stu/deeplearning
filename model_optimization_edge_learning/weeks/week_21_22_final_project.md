# 第 21-22 周：最终综合项目

## 目标

选择一个视觉检测任务作为主项目，完成从手写理解到框架落地的完整闭环。

## 必做内容

- 训练 baseline。
- 手写理解对应核心原理。
- 用框架完成工程训练。
- 使用微调或 LoRA/Adapter 思想做迁移学习。
- 蒸馏 student 模型。
- FP16/INT8 量化。
- 剪枝或结构化压缩。
- ONNX/TensorRT 部署。
- 输出最终指标表和报告。

## 最终指标表

| 模型 | 方法 | 格式 | mAP/Acc | FPS | P50 延迟 | P95 延迟 | 模型大小 | 参数量 | 显存 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline |  |  |  |  |  |  |  |  |  |  |
| finetuned |  |  |  |  |  |  |  |  |  |  |
| distilled |  |  |  |  |  |  |  |  |  |  |
| quantized |  |  |  |  |  |  |  |  |  |  |
| pruned |  |  |  |  |  |  |  |  |  |  |
| deployed |  |  |  |  |  |  |  |  |  |  |

## 验收标准

- 能讲清楚每个优化手段的收益和代价。
- 能解释框架参数背后的手写实现。
- 能给出精度优先、速度优先、均衡方案三个推荐。

## 答案闭环

<details>
<summary>先自己做</summary>

选择一个视觉检测任务，按 baseline、微调、蒸馏、量化、剪枝、部署六个阶段推进，每个阶段都记录指标。

</details>

<details>
<summary>卡住时看提示</summary>

不要一开始追求所有技术都做到最好。先跑通 baseline，再一次只改一个变量：先微调，再 FP16，再 INT8，再剪枝或蒸馏。

</details>

<details>
<summary>参考答案</summary>

最终项目推荐拆分：

```text
1. baseline: 原始模型训练和验证
2. finetune: 冻结/全量微调对比
3. distillation: teacher/student 对比
4. quantization: FP32/FP16/INT8 对比
5. pruning: 剪枝前后对比
6. deployment: ONNX/TensorRT 推理对比
7. report: 写清楚取舍和推荐方案
```

最终结论至少给三种：

| 方案 | 适用场景 | 推荐依据 |
|---|---|---|
| 精度优先 | 漏检代价高 | mAP/Recall 最高 |
| 速度优先 | 产线节拍紧 | FPS/P95 最好 |
| 均衡方案 | 默认上线 | 精度损失小且速度提升明显 |

</details>

<details>
<summary>为什么这样做</summary>

综合项目的价值不是堆技术，而是建立工程判断：某个优化方法带来了多少速度收益，损失了多少精度，是否增加部署复杂度，是否值得上线。

</details>

<details>
<summary>自检标准</summary>

- 每个实验都有唯一变量和指标记录。
- 最终表格包含 mAP/Acc、FPS、P50/P95、模型大小、参数量、显存。
- 能把框架参数解释回阶段一的手写机制。

</details>
