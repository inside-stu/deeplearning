# Framework Reference

这个目录用于阶段二。学习方式不是“背命令”，而是把框架参数映射回阶段一的手写实现。

推荐阅读顺序：

1. `vision_frameworks.md`
2. `peft_lora.md`
3. `distillation_frameworks.md`
4. `quantization_frameworks.md`
5. `pruning_frameworks.md`
6. `deployment_frameworks.md`

每学一个框架，都回答三个问题：

- 它封装了阶段一哪段代码？
- 哪些参数会改变训练/推理行为？
- 如果结果异常，应该从数据、模型、loss、优化器、量化校准还是部署后处理开始排查？

