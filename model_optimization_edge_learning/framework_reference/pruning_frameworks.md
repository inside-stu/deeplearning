# Pruning And Compression Frameworks

## torch-pruning

适合学习结构化剪枝。

重点：

- dependency graph。
- channel pruning。
- 剪掉一个通道时，下游层如何同步改变。

## NNI Model Compression

适合系统化实验剪枝、量化、蒸馏。

重点：

- pruner 配置。
- sparsity schedule。
- evaluator 和 fine-tuning。

## NNCF

适合 OpenVINO 生态下的压缩和量化。

重点：

- PTQ。
- QAT。
- structured pruning。

## SparseML

适合学习稀疏训练 schedule。

重点：

- recipe。
- 稀疏率曲线。
- 剪枝和微调组合。

## 对照阶段一

- `optimization/pruning.py` 展示了 mask、L1 importance、remove re-parametrize。
- `optimization/analysis.py` 展示了参数量、稀疏率和延迟统计。

