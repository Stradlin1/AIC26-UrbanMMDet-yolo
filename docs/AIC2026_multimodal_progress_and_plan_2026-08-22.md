# AIC2026 面向城市场景的视觉多模态目标检测
## 项目阶段总结与下一阶段规划

**更新时间：2026-08-22 17:27（UTC+8）**

---

# 0. 一句话状态

截至目前，项目已经完成了从单模态 baseline、RGB+IR+Depth 五通道 Early Fusion、固定 val400 评估、模态消融、Rect 推理优化、Leaderboard 验证，到 Exp04b No-Mosaic 微调失败验证的完整闭环。

**当前正式最优基线：**

```text
Exp04
RGB + IR + Depth
5-channel Early Fusion
YOLO11s @ 960
+
Rect inference

Local Rect val400 mAP50-95 = 0.380711
Leaderboard                = 45.614
```

**下一阶段不再继续微调 Exp04，而是正式进入 Exp05：RGB-dominant、Depth-strong、IR-light 的非对称 Gated Feature Fusion。**

---

# 1. 当前工程状态

工程根目录：

```text
aicomp/
├── datasets/
├── environment/
├── pretrained/
├── runs/
├── scripts/
├── splits/
│   └── aic2026_group_stratified_v2/
├── yolo_views/
└── ultralytics/
```

工程约定：

- 新训练实验使用独立 Python 脚本：
  - `scripts/train_xxx.py`
  - 使用 `python scripts/train_xxx.py` 启动
- 新验证/推理使用独立脚本：
  - `scripts/test_xxx.py`
  - `scripts/infer_xxx.py`
- 路径统一从：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

推导，不再硬编码 `/home/xhm/...`。
- `datasets/`、`runs/`、`pretrained/`、模型权重和提交 ZIP 不进入 Git。
- 项目本地使用 editable Ultralytics 源码：

```text
ultralytics commit:
64e29b080b3893bd9bb5ce095357c6da9662cce4
```

---

# 2. Git / GitHub 当前状态

`aicomp` 已经正式建立 Git 仓库，并连接远程：

```text
origin:
git@github.com:Stradlin1/AIC2026-multimodal-detection-Shu_Anlin.git
```

稳定主线最初提交：

```text
88c7093
chore: initialize AIC2026 multimodal detection project
```

当前重要 tag：

```text
exp04-rect-lb45.614
```

它代表：

```text
Exp04 5ch Early Fusion
+
Rect inference
+
Leaderboard = 45.614
```

Exp04b 实验使用独立分支：

```text
exp04b-nomosaic-ft
```

最新已推送 commit：

```text
015d5c5
test: add strict Exp04b v2 rect val400 comparison
```

## Git 原则

后续继续保持：

```text
main
= 已验证、可复现、当前稳定 baseline

experiment branch
= 新结构、新训练方法、新消融

tag
= 明确性能里程碑
```

不要把失败实验直接覆盖稳定 baseline。

---

# 3. 数据集与任务

任务输入为严格对齐的三模态数据：

```text
RGB
IR
Depth
```

Exp04 模型输入顺序固定：

```text
[R, G, B, IR, Depth]
```

共 5 channels。

比赛共 12 类：

```text
0  person
1  boat
2  animal
3  seat
4  sign
5  bicycle
6  car
7  ball
8  light
9  garbage_can
10 uav
11 tricycle
```

固定划分：

```text
总训练组：2000
train：1600
val：400
```

固定 split：

```text
splits/aic2026_group_stratified_v2/
```

---

# 4. Depth 表示

当前 Depth 使用已固定的 `depth8_v1` 表示。

原始 metric depth 为 uint16，转换规则：

```text
depth < 300     -> 0
depth > 20000   -> 0

有效区间：
300   -> 255
20000 -> 1

近处亮
远处暗
无效深度 = 0
```

该表示已经用于 Exp04 正式训练和后续全部验证，不应在 Exp05 中随意改变，否则会引入额外实验变量。

---

# 5. 已完成的单模态 baseline

当前已经完成：

```text
RGB-only
IR-only
Depth-only
```

固定 val400 上的主要参考：

```text
RGB   mAP50-95 ≈ 0.380843
IR    mAP50-95 ≈ 0.209259
Depth mAP50-95 ≈ 0.177121
```

公开 Leaderboard 中 RGB baseline：

```text
43.611
```

结论：

- RGB 是绝对主语义模态。
- IR 和 Depth 单独做目标检测能力明显弱于 RGB。
- 但单模态性能弱并不等价于“作为辅助模态没有价值”。

这点后来被 Exp04 模态消融进一步验证。

---

# 6. Exp04：五通道 Early Fusion

## 6.1 模型结构

Exp04 使用：

```text
YOLO11s @ 960
```

第一层从：

```text
3 channels
```

修改为：

```text
5 channels
[R,G,B,IR,Depth]
```

初始化：

```text
RGB   <- COCO pretrained weights
IR    <- zero
Depth <- zero
```

训练后 IR / Depth 第一层权重均已学习为非零。

## 6.2 Dataset 设计

五通道 Dataset 保证：

```text
RGB HSV augmentation   = ON
IR HSV augmentation    = OFF
Depth HSV augmentation = OFF
```

所有几何增强：

```text
resize
flip
mosaic
letterbox
...
```

必须对 RGB / IR / Depth 同步执行。

## 6.3 正式训练配置

核心配置：

```text
imgsz        = 960
batch        = 8
epochs       = 200
patience     = 50
seed         = 2026

mosaic       = 1.0
close_mosaic = 10

fliplr       = 0.5
```

正式训练提前 Early Stop，因此没有完整进入原计划末尾的 no-Mosaic 收敛阶段，这也是后来 Exp04b 的动机。

---

# 7. Exp04 正式验证结果

Official Validator：

```text
mAP50    = 0.628168
mAP75    = 0.388485
mAP50-95 = 0.379090
```

RGB baseline：

```text
mAP50-95 = 0.380843
```

仅从本地 Official Val 看，Exp04 并没有明显超过 RGB。

但是测试集 Leaderboard 给出了不同结论：

```text
RGB-only                = 43.611
Exp04 5ch Early Fusion  = 45.120
```

提升：

```text
+1.509 leaderboard points
```

因此确认：

> 多模态方向在真实测试集分布上有效，本地 val400 与测试集存在一定分布差异，不能只看单一 local mAP 做最终决策。

---

# 8. P1：Exp04 模态消融

在**同一个 Exp04 best.pt** 上做 test-time modality ablation。

协议：

```text
square 960
conf        = 0.001
NMS IoU     = 0.70
max_det     = 100
multi_label = True
101-point AP
IoU 0.50:0.05:0.95
```

结果：

| 模式 | 输入 | mAP50 | mAP75 | mAP50-95 | 相对 Full |
|---|---|---:|---:|---:|---:|
| Full | RGB + IR + Depth | 0.624089 | 0.401399 | **0.375390** | — |
| No IR | RGB + Depth | 0.620693 | 0.379228 | **0.372447** | -0.002943 |
| No Depth | RGB + IR | 0.603962 | 0.356414 | **0.354958** | -0.020433 |
| RGB-only-in-5ch | RGB only | 0.612682 | 0.343426 | **0.355652** | -0.019738 |

## 8.1 最重要结论

### RGB

```text
主语义模态
```

所有新模型必须保护 RGB pretrained representation。

### Depth

去掉 Depth：

```text
0.375390
→
0.354958

drop = 0.020433
```

这是一个非常大的下降。

因此：

> **Depth 是强辅助模态，尤其对几何定位和高 IoU 检测很重要。**

### IR

去掉 IR：

```text
0.375390
→
0.372447

drop = 0.002943
```

IR 有正贡献，但远弱于 Depth。

因此当前模态角色可以确定为：

```text
RGB   = 主干
Depth = 强几何辅助
IR    = 弱、条件性辅助
```

这直接决定 Exp05 不应该使用 IR / Depth 完全对称的融合结构。

---

# 9. P2：Rect inference

原 manual inference 使用 fixed square 960。

之后实现了更接近 Ultralytics Official Validator 的 Rect preprocessing：

```text
按 aspect ratio 排序
logical rect batch = 8
long-side resize
stride-aligned rectangular LetterBox
rect pad = 0.5
forward batch = 4
```

其余固定：

```text
conf        = 0.001
IoU         = 0.70
max_det     = 100
multi_label = True
FP32 / 255
```

固定 val400 A/B：

| 推理 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|
| Square | 0.624089 | 0.401399 | 0.375390 |
| Rect | **0.627469** | 0.390003 | **0.380711** |
| Official Validator | 0.628168 | 0.388485 | 0.379090 |

Rect 相对 Square：

```text
mAP50-95:
0.375390
→
0.380711

+0.005321
```

Rect manual 与 Official Validator 已基本对齐。

## 9.1 Leaderboard 验证

使用同一个 Exp04 best.pt，仅将测试集推理由 Square 改成 Rect：

```text
Square submission = 45.120
Rect submission   = 45.614
```

提升：

```text
+0.494 leaderboard points
```

与 RGB baseline：

```text
45.614 - 43.611 = +2.003
```

因此：

> **Rect inference 已经成为当前项目的正式生产推理协议。**

以后所有模型都必须优先使用同一套 Rect val400 / test inference 做最终比较。

---

# 10. 当前最优正式 baseline

目前必须冻结：

```text
Model:
Exp04 RGB+IR+Depth 5ch Early Fusion
YOLO11s @ 960

Weights:
runs/exp04_rgbid_early5_yolo11s_960/weights/best.pt

Inference:
Rect
conf        = 0.001
NMS IoU     = 0.70
max_det     = 100
multi_label = True
```

性能：

```text
Official val mAP50-95 = 0.379090
Manual Rect val400    = 0.380711
Leaderboard           = 45.614
```

Git milestone：

```text
exp04-rect-lb45.614
```

该状态不能被后续实验覆盖。

---

# 11. P3：Exp04b No-Mosaic Fine-tune

Exp04 正式训练提前 Early Stop，因此尝试从 Exp04 `best.pt` 继续做短周期 no-Mosaic fine-tune。

目标：

```text
补一次干净的 no-Mosaic box refinement
```

---

# 12. Exp04b-v1

配置：

```text
start      = Exp04 best.pt
mosaic     = 0
optimizer  = MuSGD
lr0        = 0.001
epochs     = 25
patience   = 10
```

但后来发现 v1 仍保留：

```text
warmup_epochs  = 3.0
warmup_bias_lr = 0.1
```

导致 fine-tune 前几个 epoch 的部分 bias/head 参数学习率实际达到：

```text
~0.067
~0.034
```

远高于计划中的 low-LR refinement。

因此 v1 判定为：

```text
配置受 warmup 干扰
不作为最终结论
```

---

# 13. Exp04b-v2

为了排除 warmup 干扰，建立 v2：

```text
start          = Exp04 best.pt
mosaic         = 0
close_mosaic   = 0

optimizer      = MuSGD
lr0            = 0.001
momentum       = 0.9

warmup_epochs  = 0.0
warmup_bias_lr = 0.0

epochs         = 25
patience       = 10
```

实际 LR：

```text
普通参数       ≈ 0.001
boosted head   ≈ 0.003
```

因此 v2 是一个有效、干净的 no-Mosaic / no-warmup low-LR 实验。

Internal validation 并没有产生明显提升。

为了避免不同 evaluator 造成误判，最终又做了严格的同协议 Rect A/B。

---

# 14. Exp04 vs Exp04b-v2 严格 Rect A/B

同一脚本、同一数据、同一 evaluator、同一 Rect preprocessing。

| 模型 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|
| **Exp04** | **0.627469** | **0.390003** | **0.380711** |
| Exp04b-v2 | 0.626315 | 0.370412 | 0.371336 |

变化：

```text
mAP50:
-0.001154

mAP75:
-0.019591

mAP50-95:
-0.009375
```

并且 Exp04 baseline 在此次 A/B 中重新复现：

```text
0.380711
```

与历史 Rect 结果一致，因此 evaluator parity 没有问题。

## 14.1 其他现象

预测框数量：

```text
Exp04      = 16620
Exp04b-v2  = 12865
```

约下降：

```text
22.6%
```

P / R：

```text
Precision:
0.73973 -> 0.74549

Recall:
0.60634 -> 0.58719
```

表现为：

```text
模型变得更保守
Precision 略升
Recall 明显下降
高 IoU 定位质量下降
```

尤其：

```text
mAP75 -0.019591
```

说明 no-Mosaic fine-tune 并没有实现预期的 box refinement，反而损害定位。

## 14.2 Exp04b 最终结论

```text
Exp04b-v1
= warmup 干扰，不能作为最终结论

Exp04b-v2
= 干净实验
= Rect mAP50-95 下降 0.009375
```

因此：

> **Exp04b No-Mosaic low-LR fine-tune 正式关闭。**

不再继续：

```text
lr0 sweep
AdamW sweep
延长 epoch
继续 no-Mosaic fine-tune
```

也**不提交 Exp04b-v2 到 Leaderboard**。

---

# 15. 当前阶段最关键的科研结论

目前已经得到四个比“继续堆训练”更有价值的结论。

## 15.1 多模态在真实测试集有效

```text
RGB LB   = 43.611
Exp04 LB = 45.614
```

最终提升：

```text
+2.003 leaderboard points
```

因此项目主方向正确。

## 15.2 Early Fusion 能工作，但不是理想最终结构

当前：

```text
[R,G,B,IR,Depth]
       ↓
同一个 Conv
```

虽然取得 45.614，但 RGB / IR / Depth 的统计分布和信息角色明显不同。

继续让三者在第一层无条件混合，缺乏模态质量控制。

## 15.3 Depth 的价值远大于 IR

消融：

```text
Depth contribution ≈ +0.020433 mAP50-95
IR contribution    ≈ +0.002943 mAP50-95
```

因此后续融合必须是：

```text
asymmetric
```

而不是：

```text
IR / Depth symmetric
```

## 15.4 继续优化 Exp04 训练末期的收益很有限

Exp04b-v2 已给出强否定证据。

下一阶段收益更可能来自：

```text
改变模态融合方式
```

而不是：

```text
对 Exp04 best.pt 做更多小修小补
```

---

# 16. 下一阶段：Exp05

## Exp05 总目标

从：

```text
5-channel Early Fusion
```

升级到：

```text
RGB-dominant
Depth-strong
IR-light
Asymmetric Gated Feature Fusion
```

原则：

> RGB 负责稳定主语义表示；Depth 作为强几何 residual；IR 作为较弱、条件性的 residual。辅助模态只能帮助 RGB，不应无条件覆盖 RGB。

---

# 17. Exp05 初步结构思想

概念结构：

```text
RGB ───── RGB Stem ───────────── F_rgb ────────────────┐
                                                       │
Depth ─── Depth Stem ─────────── F_depth ─ Gate_depth ─┤
                                                       ├── F_fused
IR ────── IR Stem ────────────── F_ir ───── Gate_ir ───┤
                                                       │
                                                      YOLO
```

核心：

```python
F_fused = (
    F_rgb
    + alpha_depth * F_depth
    + alpha_ir * F_ir
)
```

其中：

```text
alpha_depth
alpha_ir
```

应为可学习 gate。

初始先验可以设计为：

```text
Depth contribution > IR contribution
```

例如概念上：

```text
alpha_depth 初始约 0.2~0.3
alpha_ir    初始约 0.03~0.08
```

但**最终具体实现和初始化必须在确认 YOLO11s 实际层结构后再决定，不应直接硬写。**

---

# 18. 为什么不能马上写 Exp05 正式模型

必须先知道当前锁定 Ultralytics commit 下 YOLO11s 的真实 layer graph。

需要明确：

```text
model[0]
model[1]
model[2]
...
```

每层的：

```text
module type
from
input/output channels
stride
feature map resolution
parameter count
```

尤其要回答：

1. YOLO11s 第一阶段到底在哪里结束？
2. P2 / P3 feature 是在哪些 layer 形成？
3. Depth 应该在 early 还是 early-mid 融合？
4. IR 是否应该比 Depth 更晚融合？
5. 哪个融合位置能最大程度保留 COCO pretrained RGB path？

因此 Exp05 第一件事不是训练，而是 architecture audit。

---

# 19. 接下来严格执行的开发顺序

## Step 1：Git 收尾 Exp04b

如果尚未执行，建议：

```bash
git tag -a exp04b-closed \
  -m "Exp04b no-Mosaic fine-tuning closed: Rect val400 0.371336 vs Exp04 0.380711"

git push origin exp04b-closed
```

之后：

```bash
git checkout main
git pull
```

不要把失败的 Exp04b 分支直接作为 Exp05 起点。

---

## Step 2：创建 Exp05 分支

```bash
git checkout -b exp05-asymmetric-gated-fusion
```

Exp05 必须从稳定的 `main / exp04-rect-lb45.614` 状态出发。

---

## Step 3：审计 YOLO11s 实际结构

下一份脚本：

```text
scripts/inspect_exp05_yolo11s_architecture.py
```

要求输出：

```text
layer index
from
module type
parameters
input channels
output channels
stride
actual feature shape @ 960
```

并至少做一次：

```text
input = [1,3,960,960]
```

的 feature hook / forward audit。

### 这一阶段的交付物

明确写出：

```text
RGB stem ends at layer ?
Depth fusion at layer ?
IR fusion at layer ?
```

之后才进入模型实现。

---

## Step 4：实现 Exp05 模型模块

预计脚本：

```text
scripts/multimodal_gated_model.py
```

第一版原则：

```text
简单
可解释
容易做消融
不要一次堆多个 attention 模块
```

应优先验证：

```text
separate stem
+
learnable gated residual fusion
```

而不是一上来：

```text
cross-attention
transformer fusion
multi-stage complex attention
```

否则难以判断增益来自哪里。

---

## Step 5：模型单元测试

脚本：

```text
scripts/test_exp05_gated_model.py
```

必须检查：

### 结构

```text
输入确实为 5 channels
RGB / IR / Depth 分流正确
输出 shape 与 YOLO 后续兼容
```

### 初始化

```text
RGB pretrained weights 完整继承
Depth stem 初始化合理
IR stem 初始化合理
```

### Gate

```text
Depth gate 初始值
IR gate 初始值
gate 参数 requires_grad=True
```

### 训练

必须通过：

```text
forward
loss
backward
optimizer step
```

并确认：

```text
RGB gradient != 0
Depth gradient != 0
IR gradient != 0
gate_depth gradient != 0
gate_ir gradient != 0
```

---

## Step 6：Exp05 Smoke Training

脚本：

```text
scripts/train_exp05_gated_smoke_960.py
```

只跑：

```text
2~3 epochs
```

目的不是看 mAP，而是验证：

```text
Dataset
Model
Trainer
AMP
loss
validation
checkpoint
reload
Rect inference compatibility
```

全部正常。

---

## Step 7：Exp05 正式训练

Smoke PASS 后才建立：

```text
scripts/train_exp05_*.py
```

正式训练仍保持尽量公平：

```text
same train/val split
same imgsz=960
same basic augmentation philosophy
same YOLO11s scale
same seed where possible
```

尽量让主要实验变量只有：

```text
fusion architecture
```

---

# 20. Exp05 的验证顺序

每一个正式 Exp05 checkpoint 必须经过：

```text
Training
    ↓
Official Validation
    ↓
Rect val400
    ↓
Per-class analysis
    ↓
Modality ablation
    ↓
只有明确提升
    ↓
Test submission
```

禁止：

```text
训练结束
↓
直接 test submission
```

---

# 21. Exp05 的判定基线

固定基线：

```text
Exp04 Rect val400 = 0.380711
Leaderboard       = 45.614
```

建议：

### Local candidate

至少：

```text
Exp05 Rect mAP50-95
>= 0.381711
```

即：

```text
>= +0.001
```

才进入候选区。

### 更有说服力的 local 提升

优先希望：

```text
>= +0.002 ~ +0.003
```

再消耗一次 Leaderboard submission。

由于此前已经确认 local val 与 leaderboard 分布不完全一致，因此：

> local gain 只能决定“值不值得提交”，不能预测 leaderboard 精确提升幅度。

---

# 22. Exp05 必须重新做模态消融

如果 Exp05 取得提升，不能只报总 mAP。

必须重新测：

```text
Full
No IR
No Depth
RGB-only
```

理想的 Exp05 应表现为：

```text
Full > No Depth
Full > No IR

Depth drop > IR drop
```

同时相比 Exp04：

- Full 更高；
- 去掉某个辅助模态后模型不能完全崩；
- Gate 应降低无效模态的负迁移风险。

如果 Exp05 Full 提高，但：

```text
No IR > Full
```

说明 IR gate 仍然没有正确工作。

---

# 23. 后续 Exp06：暂时不要做

只有 Exp05 gated fusion 证明有效后，再考虑：

```text
Exp06
Quality-aware Gate
+
Modality Dropout
```

可能加入：

```python
depth_valid_ratio = (depth > 0).float().mean()
```

作为 Depth quality signal。

训练时可以尝试：

```text
随机 drop IR
随机 drop Depth
少量同时 drop IR + Depth
```

目标：

```text
RGB 正常       -> 保持稳定
IR 失效        -> gate 自动减弱
Depth 无效     -> gate 自动减弱
RGB 困难       -> IR / Depth 提供补充
```

但这些都属于下一阶段，不应提前叠到 Exp05-v1 中。

---

# 24. 当前暂时不应该做的事情

以下事项现在都不建议：

```text
继续 sweep Exp04b 学习率
继续训练 Exp04b
提交 Exp04b-v2
修改 Depth 编码
修改 split
更换 YOLO 模型尺寸
同时增加多种 attention
同时增加 modality dropout
同时调整大量 augmentation
```

原因：

> Exp05 的目标是验证“非对称 gated fusion”这一核心假设，必须控制实验变量。

---

# 25. 当前正式实验树

```text
RGB baseline
LB 43.611
    │
    ▼
Exp04
5ch Early Fusion
Square inference
LB 45.120
    │
    ├── Modality Ablation
    │     Depth = strong
    │     IR    = weak positive
    │
    ├── Rect inference
    │     local:
    │     0.375390 -> 0.380711
    │
    │     leaderboard:
    │     45.120 -> 45.614
    │
    ▼
Current Stable Baseline
Exp04 + Rect
LB 45.614
    │
    ├── Exp04b-v1
    │     no-Mosaic
    │     warmup interference
    │
    ├── Exp04b-v2
    │     no-Mosaic
    │     no warmup
    │
    │     Rect:
    │     0.371336
    │
    │     delta = -0.009375
    │
    └── CLOSED
          │
          ▼
        Exp05
   Asymmetric Gated Fusion
          │
          ▼
 architecture audit
          │
          ▼
 model implementation
          │
          ▼
 unit test
          │
          ▼
 smoke train
          │
          ▼
 formal train
          │
          ▼
 Rect val400
          │
          ├── no gain -> analyze / redesign
          │
          └── clear gain -> test submission
```

---

# 26. 近期任务清单

## P0 — Exp04b Git 结案

状态：

```text
基本完成
```

待做：

```text
[ ] 可选：创建 exp04b-closed tag
[ ] checkout main
[ ] 从 main 创建 exp05-asymmetric-gated-fusion
```

---

## P1 — YOLO11s architecture audit

下一任务：

```text
scripts/inspect_exp05_yolo11s_architecture.py
```

状态：

```text
NEXT
```

完成标准：

```text
[ ] 静态层表
[ ] forward feature shape
[ ] stride
[ ] channels
[ ] P2/P3/P4 对应关系
[ ] 确定 Depth / IR 融合层
```

---

## P2 — Exp05 model

预计：

```text
scripts/multimodal_gated_model.py
```

状态：

```text
WAITING FOR P1
```

---

## P3 — Exp05 unit test

预计：

```text
scripts/test_exp05_gated_model.py
```

状态：

```text
WAITING
```

---

## P4 — Exp05 smoke

预计：

```text
scripts/train_exp05_gated_smoke_960.py
```

状态：

```text
WAITING
```

---

## P5 — Exp05 formal training

状态：

```text
WAITING
```

最终判断：

```text
Rect mAP50-95 > 0.380711
```

并且最好：

```text
gain >= +0.001
```

才考虑下一步提交。

---

# 27. 推荐下一次直接开始的工作

**下一步只做一件事：**

```text
scripts/inspect_exp05_yolo11s_architecture.py
```

先把当前 YOLO11s 的实际 backbone / neck feature graph 查清楚。

完成后再根据真实 feature shape 决定：

```text
Depth 在哪里进入
IR 在哪里进入
Gate 如何初始化
是否用同层融合还是不同层融合
```

不要先凭经验写复杂 Exp05。

---

# 28. 当前记忆锚点

后续新对话如果需要快速恢复上下文，只需要记住下面这段：

> 截至 2026-08-22，AIC2026 多模态目标检测项目当前正式最好模型是 Exp04 YOLO11s@960 RGB+IR+Depth 五通道 Early Fusion。RGB baseline Leaderboard 为 43.611；Exp04 square submission 为 45.120；采用与 Official Validator 更一致的 Rect inference 后达到 45.614，成为当前主 baseline。固定 val400 中 Exp04 Rect mAP50-95=0.380711。Test-time 模态消融显示 Depth 是强辅助模态：去掉 Depth 从 0.375390 降到 0.354958，drop=0.020433；IR 是弱正贡献：去掉 IR 降到 0.372447，drop=0.002943。因此后续融合必须采用 RGB-dominant、Depth-strong、IR-light 的非对称设计。Exp04b No-Mosaic low-LR fine-tune 已完成 v1/v2 验证，其中干净的 v2 在同协议 Rect val400 上只有 0.371336，相对 Exp04 下降 0.009375，因此 Exp04b 正式关闭，不再 sweep，也不提交。工程已建立 Git/GitHub，稳定里程碑 tag 为 `exp04-rect-lb45.614`，Exp04b 实验在 `exp04b-nomosaic-ft` 分支。下一步从稳定 main 创建 `exp05-asymmetric-gated-fusion`，首先写 `scripts/inspect_exp05_yolo11s_architecture.py` 审计 YOLO11s 的真实 layer/feature shape，然后再实现 Separate Stem + Asymmetric Gated Residual Fusion，依次做 unit test、2~3 epoch smoke、正式训练、Official Val、Rect val400，只有 local Rect 相对 0.380711 有明确提升后才生成测试集提交。

---

# 29. 最终优先级

```text
1. 冻结 Exp04 + Rect = 45.614
2. Exp04b 正式 CLOSED
3. 完成 Exp05 Git 分支
4. 审计 YOLO11s architecture
5. 确定 Depth / IR 非对称融合位置
6. 实现最简 Gated Fusion
7. Unit Test
8. Smoke Train
9. Formal Train
10. Official + Rect Val
11. 有明确提升才提交
12. Exp05 成功后再考虑 Exp06
```

**当前下一步：`scripts/inspect_exp05_yolo11s_architecture.py`。**
