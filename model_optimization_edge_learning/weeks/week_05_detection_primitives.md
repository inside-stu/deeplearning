# 第 5 周：手写检测核心概念

## 目标

理解目标检测核心机制，再回头看 YOLO/MMDetection 的封装。

## 学习内容

- bbox 表示：xyxy、xywh、cxcywh
- IoU、GIoU/DIoU/CIoU 基础
- anchor-based 与 anchor-free
- objectness、classification、box regression loss
- NMS 原理
- mAP 基本计算流程

## 代码入口

- `core/models.py` 中的 `TinyDetector`
- `core/metrics.py` 中的 IoU、NMS、AP 函数

## 实战任务

1. 手写几个 bbox，计算 IoU。
2. 构造重叠框和分数，运行 NMS。
3. 阅读 `TinyDetector` 的输出 shape。
4. 写一页笔记：YOLO 的 head 比这个极简 head 多了哪些工程设计。

## 验收标准

- 能解释 NMS 为什么会删除框。
- 能说明 mAP 不是单纯 accuracy。
- 能看懂检测模型输出的 box/objectness/class 三类张量。

## 答案闭环

<details>
<summary>先自己做</summary>

自己构造 3 个框和 3 个分数，调用 `box_iou_xyxy` 与 `nms_xyxy`，观察哪些框被保留。

</details>

<details>
<summary>卡住时看提示</summary>

NMS 的顺序是：按分数排序，保留最高分框，删除与它 IoU 超过阈值的低分框，然后继续处理剩余框。

</details>

<details>
<summary>参考答案</summary>

```python
import torch
from core.metrics import box_iou_xyxy, nms_xyxy

boxes = torch.tensor([
    [0.10, 0.10, 0.50, 0.50],
    [0.12, 0.12, 0.52, 0.52],
    [0.60, 0.60, 0.90, 0.90],
])
scores = torch.tensor([0.90, 0.80, 0.70])

print(box_iou_xyxy(boxes, boxes))
print(nms_xyxy(boxes, scores, iou_threshold=0.5))
```

</details>

<details>
<summary>为什么这样做</summary>

检测模型通常会对同一个目标输出多个相近框。NMS 用 IoU 判断框是否高度重叠，用 score 判断保留哪一个。mAP 则在不同置信度和召回率下衡量检测质量，不等同于分类 accuracy。

</details>

<details>
<summary>自检标准</summary>

- 高重叠的两个框只保留分数更高的那个。
- 距离较远的框不会被删除。
- 你能说出 xyxy 和 cxcywh 的字段含义。

</details>
