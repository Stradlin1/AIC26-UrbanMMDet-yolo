# Exp04：参考 EMOE 的动态模态专家融合

## 1. 目的

当前 Exp04 将 RGB、IR、Depth 直接拼成 `[R,G,B,IR,Depth]`，然后交给同一个 5 通道卷积。这个结构证明多模态有效，但它没有显式回答一个关键问题：**同一张样本里，当前到底应该更信 RGB、IR 还是 Depth？**

EMOE（CVPR 2025）的核心思路之一是：不要给所有样本使用固定模态比例，而是通过 Router 根据每个样本的多模态特征生成一组动态权重，再让不同模态专家参与融合。

本实验只借鉴这个通用思想，不直接复制 EMOE 的情感识别代码，也不在第一版同时加入其 Unimodal Distillation。这样可以把主要实验变量控制为“动态模态路由”。

## 2. 新结构

输入仍然是：

```text
[R, G, B, IR, Depth]
```

原 Exp04：

```text
5ch input -> one Conv -> BN -> SiLU -> YOLO11s
```

新 Exp04-EMOE：

```text
RGB   -> RGB expert ------------------------------+
IR    -> IR expert ---- * router(IR) -------------+--> shared BN -> SiLU -> YOLO11s
Depth -> D expert ----- * router(Depth) -----------+
                    ^
                    |
          sample-wise softmax router
          [w_rgb, w_ir, w_depth]
```

实际融合公式采用 RGB anchor：

```text
F = F_rgb + 3 * (w_ir * F_ir + w_depth * F_depth)
```

其中 Router 输出：

```text
[w_rgb, w_ir, w_depth] = softmax(router(x))
```

`w_rgb` 虽然不直接乘 RGB 主分支，但它会通过 softmax 归一化抑制两个辅助权重。当 Router 判断 RGB 更可靠时，`w_rgb` 上升，`w_ir` 和 `w_depth` 会同步下降。

这样做的原因是：RGB 是当前最稳定的语义主干，不希望动态 Router 在训练早期直接把 RGB 主路径压低。

## 3. 为什么从现有 Exp04 best.pt 初始化

该实现最重要的工程性质是：**可以把原 Exp04 的 5 通道卷积无损拆成 3 个 expert 卷积。**

原卷积权重：

```text
W[:, 0:3] -> RGB expert
W[:, 3:4] -> IR expert
W[:, 4:5] -> Depth expert
```

Router 最后一层初始化为全 0，所以开始时：

```text
softmax([0,0,0]) = [1/3,1/3,1/3]
```

又因为：

```text
auxiliary_scale = 3
```

因此初始融合：

```text
F = F_rgb + F_ir + F_depth
```

与原来的 5 通道卷积等价，仅可能因为浮点累加顺序产生极小误差。

这意味着新的动态模型不是“随机重做一个网络”，而是从已经验证过的 Exp04 detector 出发，让 Router 学习什么时候该加强/减弱 IR 和 Depth。

## 4. Router

完整 960×960 图像不能像 EMOE 原代码那样直接整体 flatten。这里先对 5 通道输入做 `4×4 AdaptiveAvgPool`，得到一个很小的空间描述，再经过两层 MLP：

```text
5 x 4 x 4 = 80 dims
        -> Linear(80, 32)
        -> SiLU
        -> Linear(32, 3)
        -> temperature
        -> Softmax
```

Router 参数量很小，几乎不会改变整体模型规模。

## 5. 第一版为什么不加 Unimodal Distillation

EMOE 还使用了单模态蒸馏来维持各模态自身的预测能力。但在目标检测里直接照搬需要增加独立 IR/Depth 检测头和额外检测损失，这会同时改变：

- 网络结构；
- loss；
- 参数量；
- 优化难度。

第一版先只验证：

```text
固定 5ch Early Fusion
        ->
Sample-wise Dynamic Expert Routing
```

是否有增益。如果 Router 版本有效，再做第二阶段的 auxiliary detection/distillation 才有实验解释性。

## 6. 新增文件

```text
scripts/exp04_emoe_dynamic_model.py
scripts/exp04_emoe_dynamic_trainer.py
scripts/test_exp04_emoe_dynamic_conversion.py
scripts/train_exp04_emoe_dynamic_yolo11s_960.py
```

## 7. 推荐执行顺序

先确认稳定 Exp04 权重存在：

```text
runs/exp04_rgbid_early5_yolo11s_960/weights/best.pt
```

先跑结构/转换 smoke test：

```bash
python scripts/test_exp04_emoe_dynamic_conversion.py
```

通过后再正式训练：

```bash
python scripts/train_exp04_emoe_dynamic_yolo11s_960.py
```

正式输出：

```text
runs/exp04_emoe_dynamic_experts_yolo11s_960/
```

## 8. 默认训练配置

这是从已训练 Exp04 checkpoint 出发的结构微调，不按 COCO 从头再跑 200 epoch：

```text
imgsz           = 960
batch           = 8
epochs          = 60
patience        = 15
optimizer       = AdamW
lr0             = 5e-4
lrf             = 0.10
weight_decay    = 5e-4
warmup_epochs   = 1
mosaic          = 1.0
close_mosaic    = 10
fliplr          = 0.5
seed            = 2026
```

Mosaic 等数据增强维持原 Exp04 思路，避免把“Router 改动”和“大幅数据增强改动”混在一起。

## 9. 判定标准

训练内部验证仍使用 `max_det=300`，不能直接和正式成绩下结论。训练结束后仍应使用固定 val400：

```text
imgsz   = 960
conf    = 0.001
iou     = 0.70
max_det = 100
Rect inference
```

至少比较：

```text
mAP50
mAP75
mAP50-95
per-class AP
```

同时应记录 Router 在验证集上的平均权重，以及按样本/场景的权重分布。真正有价值的结果不仅是 mAP 上升，还应该能观察到 Router 对 IR/Depth 的使用不是固定常数。

## 10. 后续实验

如果这一版有效，再按单变量原则继续：

1. Router temperature A/B；
2. IR/Depth modality dropout；
3. Depth quality signal（valid ratio）输入 Router；
4. 中层动态专家，而不是只在输入 stem 做专家；
5. 最后才考虑 detection-specific unimodal distillation。
