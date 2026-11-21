# ResNet-50 + SE-ResNet-50 复现 TODO List（总览）

> 和 VGGNet **同一套任务要求**（结构复现 → 官方权重 → 猫狗迁移学习 → TensorBoard + 实拍推理 → DDP），
> 额外加一条：**集成 SENet 并做消融**。框架仍是 **mmpretrain 1.2.0**。
>
> **本次范围**：只复现 **ResNet-50**，以及用于 SENet 集成的 **SE-ResNet-50**。
> ResNet-18/34/101/152 和 SE-ResNet-101/152 **不要求实现或对齐**；参考实现里保留这些分支只是为了代码完整。
>
> **环境与数据已就绪**：直接用 `Pratice_project/.venv`（torch 2.11.0+cu128 / mmpretrain 1.2.0 /
> mmengine 0.10.7 / mmcv 2.2.0 lite / tensorboard 2.21.0）；数据已在 `data/cats_dogs`
> （22,499 训练 / 2,498 验证），**不用重装、不用重下**。
>
> **本文件只做总览和索引**。原来的 TODO 已经拆成两份，**按顺序做：先代码、后训练**：
>
> | 文件 | 内容 | 目标 |
> | :--- | :--- | :--- |
> | **[01_代码部分.md](01_代码部分.md)** | Phase 1–6：环境自检 → 读懂 ResNet-50/SE 结构 → 项目骨架 → 手写 ResNet-50 并对齐 → 集成 SENet 并对齐 SE-ResNet-50 → 加载官方权重 | 1、2 + SE 集成 |
> | **[02_训练部分.md](02_训练部分.md)** | Phase 7–14：猫狗数据 → 两个训练配置（ResNet-50 / SE-ResNet-50）→ 迁移学习（>95%）→ TensorBoard → 实拍推理 → SE 消融 → 恒源云 DDP → 交付 | 3、4、5 + DDP |
>
> **五个目标（外加 DDP 支持）**：
>
> 1. **代码实现**：PyTorch 手写 **ResNet-50**，接进 MMPretrain，`state_dict` 与官方逐键一致、`max|diff| = 0.0`。
> 2. **加载预训练权重**：官方 ImageNet-1k（1000 类）ResNet-50 / SE-ResNet-50 权重灌进你的实现，2 类时正确替换 `head.fc`。
> 3. **迁移学习**：真实猫狗 24,997 张（微软版），验证集 **top-1 > 95%**。
> 4. **可视化 + 推理**：TensorBoard 记录 loss / acc / lr；手机实拍猫狗照片推理验证。
> 5. **SENet 集成 + 消融**：手写 `SELayer` / `SEBottleneck`，**SE-ResNet-50** 与官方 `SEResNet` 对齐；
>    同一数据/超参下对比 ResNet-50 vs SE-ResNet-50 的参数、精度、耗时，并可视化 SE 门控。
>
> **外加 DDP 多卡**：代码支持多卡，实际训练在恒源云 2 卡（Linux + NCCL）上跑；本机 Windows 只做单卡。
>
> **环境**：`mmcv-lite`；本机 Windows + RTX 5060 Laptop 8GB 只做单卡，多卡上恒源云。
> **版本基线**：python 3.12.10 / torch 2.11.0+cu128 / mmpretrain 1.2.0 / mmengine 0.10.7 / mmcv-lite 2.2.0。

---

## 一、进度追踪（2026-11-21 全部完成）

| Phase | 内容 | 所在文件 | 结果 |
| :-: | :--- | :--- | :--- |
| 1 | 环境冒烟 | [01](01_代码部分.md) | ✅ `25.56M` / `(2,1000)` / `feat (2,2048,7,7)` |
| 2 | 读懂 ResNet + SE 结构 | [01](01_代码部分.md) | ✅ 结构表/残差/SE 公式 |
| 3 | 项目骨架 | [01](01_代码部分.md) | ✅ `pip install -e .`、`import resnet` |
| 4 | 手写 ResNet-50 + 结构对齐 | [01](01_代码部分.md) | ✅ `318` 键 / `23.51M` / `diff = 0.0` |
| 5 | 集成 SENet + SE-ResNet-50 对齐 | [01](01_代码部分.md) | ✅ `382` 键 / `26.04M` / `diff = 0.0` |
| 6 | 加载官方权重 | [01](01_代码部分.md) | ✅ `max \|diff\| = 0.0`；2 类 `head.fc` 替换正确 |
| 7 | 猫狗数据 | [02](02_训练部分.md) | ✅ 22,499 / 2,498 张 |
| 8 | 两个训练配置 | [02](02_训练部分.md) | ✅ ResNet-50 / SE-ResNet-50 均可构建 |
| 9 | ResNet-50 迁移学习 | [02](02_训练部分.md) | ✅ 单卡 **99.16%**，DDP **99.16%** |
| 10 | TensorBoard | [02](02_训练部分.md) | ✅ loss / acc / lr 曲线 + 截图 |
| 11 | 实拍推理 | [02](02_训练部分.md) | ✅ ResNet-50 5/5；SE-ResNet-50 4/5 |
| 12 | SE 消融 + 门控可视化 | [02](02_训练部分.md) | ✅ ResNet 99.16 vs SE 99.32（+0.16） |
| 13 | DDP（恒源云 2×4090） | [02](02_训练部分.md) | ✅ ResNet 99.16；SE 99.00；6:50 / 7:55；FileZilla 带回 |
| 14 | README / 交付 | [02](02_训练部分.md) | ✅ README 已写 |

---

## 二、五条关键结论（先记住，能省掉几小时）

1. **ResNet 的分类器在 `head`，不在 backbone**：官方权重的分类键是 `head.fc.weight`，不是 VGG 的
   `backbone.classifier.6.weight`。backbone **没有** `num_classes` 参数；2 类信息写在 `head.num_classes` 和数据集 `classes` 里。
2. **官方已经注册了 `ResNet` / `SEResNet`**：自己的实现必须换名，本项目用 `MyResNet` / `MySEResNet`；
   官方 `SEResNet` 正好是 SE 结构对齐的基准（有 SE-ResNet-50 的 ImageNet 权重）。
3. **SE 的属主必须叫 `se_layer`**：官方键是 `layer1.0.se_layer.conv1.conv.weight`；写成 `self.se` 会让
   `state_dict` 键不一致，`diff` 永远对不上。普通 ResNet-50 权重灌进 SE-ResNet-50 时，**恰好缺 64 个 SE 键**。
4. **ResNet 的 `train()` 官方也没 `return self`**，且 `frozen_stages` 在 `__init__` 里就生效（VGG 是在 `train()` 里）。
   两个项目行为不同，别混。自己的实现记得 `return self`。
5. **本机只做单卡**：无 NCCL、gloo+CUDA 崩溃、`torchrun` 报 libuv，多卡一律上恒源云；
   **torch ≥2.6 读 mmengine 的 ckpt 会报 `UnpicklingError`**，必须加 `tools/compat.py` 补丁（01 的 6.5）。

---

## 三、和 VGGNet 的差异清单（迁移经验时最容易照搬错的地方）

| 项 | VGGNet | ResNet-50 + SE-ResNet-50 |
| :--- | :--- | :--- |
| 分类层位置 | `backbone.classifier`（3 层 FC） | `head.fc`（1 层 Linear） |
| `num_classes` 写在哪 | `backbone.num_classes` | `head.num_classes` |
| 2 类时尺寸不匹配的键 | `backbone.classifier.6.*` | `head.fc.weight` / `head.fc.bias` |
| 归一化层 | 可选（VGG-BN） | 每个卷积后都有 BN |
| 默认 lr（微调） | `1e-4`（无 BN 敏感） | `1e-2` 起步，抖就降到 `1e-3` |
| backbone 参数量 | 138.36 M | 23.51 M（ResNet-50）/ 26.04 M（SE） |
| 特征图 | 7×7×512 | 7×7×2048 |
| 额外结构 | 无 | 残差 shortcut + SE 通道注意力 |
| 官方注册名 | `VGG` | `ResNet` / `SEResNet`（所以自己叫 `MyResNet` / `MySEResNet`） |

---

> 本文件只保留总览、关键结论、差异清单和进度表；**具体步骤、代码、命令、验收标准都在 01/02 里**。
