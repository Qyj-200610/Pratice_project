# VGGNet 结构复现 TODO List（PyTorch × MMPretrain）

> **本次目标（已确定）**：**只做结构复现**——自己手写 VGG backbone，接进 MMPretrain，
> 证明它与官方实现**结构完全等价**（`state_dict` 逐键一致、加载官方权重后输出 `max|diff| = 0`）。
> ImageNet 指标复现**不在本次范围**，代码已挪到「附录 F」，以后想跑再看。
>
> **环境路线（已确定）**：**路线 A —— `mmcv-lite`**。不编译任何 CUDA 算子，纯分类任务够用。
>
> **使用方式**：从上往下执行。每个 Phase 末尾都有「验收标准」，**不达标不要进入下一个 Phase**。
> 卡住时先查「附录 A 报错速查」，再查官方 `vgg.py` 源码（见「附录 B 源码地图」）。
> 完整参考实现放在「附录 E」，但**请先自己写**，否则这个项目对你没有价值。

---

## 0. 开工前必读

### 0.1 本次范围：只做“结构复现”

| 层次 | 含义 | 判定标准 | 本次是否做 |
| :--- | :--- | :--- | :---: |
| **结构复现** | 你写的网络与论文/官方结构完全一致 | `state_dict` 键名与形状逐一致；加载官方权重后输出误差为 0 | ✅ 主线 |
| 指标复现 | 同数据、同 recipe 训出同精度 | ImageNet top-1 对得上（71.6 / 73.7） | ⏭️ 不做，见附录 F |

**为什么结构复现值得单独做**：它不需要 ImageNet、不需要 GPU 天数，半小时就能给出“对/错”的确定答案；
而绝大多数人卡在结构上却以为自己在调超参，白白浪费一周。

> 但要清楚：`max|diff| = 0` 只证明**数值路径一致**，不证明结构等于论文（官方实现本身也可能偏离论文）。
> 所以 Phase 2 的结构表 + Phase 4 的 8 变体对齐，两者都要做。
> 另外：对齐必须在 `eval()` 下进行，否则 dropout / BN 会让 diff 永远不为 0（详见 4.4 的警告）。

### 0.2 版本基线（本项目锁定）

| 组件 | 版本 | 说明 |
| :--- | :--- | :--- |
| Python | 3.10 / 3.11 / 3.12 | 本机为 3.12.10 |
| PyTorch | 2.1 ~ 2.11 | 本机为 2.11.0+cu128 |
| mmpretrain | **1.2.0** | 当前最新稳定版，VGG 实现在 `mmpretrain/models/backbones/vgg.py` |
| mmengine | **0.10.7** | `>=0.8.3,<1.0.0` |
| mmcv | **2.2.0** | 官方要求 `>=2.0.0,<2.4.0`；分类任务可用纯 Python 的 `mmcv-lite` |
| 系统 | Windows / Linux | 本机 Windows，已实测通过 |

> ⚠️ 版本不要自由发挥。OpenMMLab 系的报错 80% 来自“mmcv / mmengine / torch 三者版本不匹配”。

### 0.3 本机实测结论（重要，决定了 Phase 1 怎么选）

我在你当前这台机器（Windows + Python 3.12.10 + torch 2.11.0+cu128）上把整条链路跑通了，结论如下：

1. **`mmcv` 全量版在 torch 2.11 上没有预编译轮子**。`mim install mmcv` 会退化成源码编译，需要完整 CUDA Toolkit + MSVC，Windows 上极易失败。
2. **纯分类根本不需要全量 mmcv**。装 `mmcv-lite`（纯 Python）即可，实测 VGG16 前向、mini 数据上的训练/验证流程、`state_dict` 对齐全部通过，参数量 138.36M ✅。
3. **中文 Windows 必须先设 `PYTHONUTF8=1`**，否则 mmengine 收集环境信息时会对 `cl.exe` 输出按 GBK 解码并崩溃：
   `UnicodeDecodeError: 'gbk' codec can't decode byte 0x80`。这个坑很隐蔽，直接导致 `Runner` 起不来。

### 0.4 路线图与时间预算（结构复现 ≈ 3–4 天）

| Phase | 内容 | 预计耗时 | 产出 |
| :--- | :--- | :--- | :--- |
| 1 | 环境（路线 A） | 0.5 天 | 能 `import mmpretrain` 并跑通 VGG16 前向 |
| 2 | 读论文 + 设计 | 0.5 天 | 一张自己画的 VGG16 结构表 |
| 3 | 项目骨架 | 0.5 天 | `pip install -e .` 成功 |
| 4 | **手写 backbone + 对齐验收** | 1–2 天 | 8 个变体与官方 `state_dict` 逐位对齐 |
| 5 | 写 config（接线用） | 0.5 天 | 1 epoch mini 数据跑通 |
| 6 | 训练/评测设施验证 | 0.5 天 | 证明它能被框架正常训练、评测、推理 |
| 7 | 加分项（选做） | — | 多尺度特征、冻结、ONNX |

---

## Phase 1：环境（0.5 天）

### 1.1 路线已定：mmcv-lite

只做分类，VGG 用到的 mmcv 能力只有 `mmcv.cnn.ConvModule`，**纯 Python 的 `mmcv-lite` 完全覆盖**，
不需要编译任何 CUDA 算子。

> 为什么不用全量 `mmcv`：torch 2.11 上官方没有对应预编译轮子，`mim install mmcv` 会退化成源码编译
> （要完整 CUDA Toolkit + MSVC），Windows 上基本必失败。**本次不需要它。**
>
> 边界：`mmcv-lite` 不含 `mmcv.ops`。以后如果要接 mmdet/mmseg（可变形卷积、NMS 等），
> 再单独开一个环境装全量 mmcv，与本项目互不影响。

唯一副作用：启动时会出现一条 `Fail to import MultiScaleDeformableAttention` 的 **UserWarning——无害，忽略**。

### 1.2 安装

```powershell
# 在 VGGNet/ 下建独立环境（venv 或 conda 都行，别装进系统 Python）
python -m venv .venv
# 想复用系统里已装好的 torch，可改用：python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
# 全新 venv 里必须自己装一份 torch；按显卡选 cu 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 三个包必须一起指定，缺一个都跑不起来
pip install mmpretrain==1.2.0 mmengine==0.10.7 mmcv-lite==2.2.0
```

- [ ] 装完后 `pip list | Select-String "mmpretrain|mmengine|mmcv"` 能看到三行

> 国内网络可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
> **不要**用 `mim install mmpretrain`：它会顺带把全量 `mmcv` 拉下来编译，在 torch 2.11 上大概率卡死。

### 1.3 设置 UTF-8（Windows 必做）

```powershell
$env:PYTHONUTF8 = "1"     # 当前会话生效
# 想永久生效：系统环境变量里新建 PYTHONUTF8=1，然后重开终端
```

- [ ] 新开终端后 `echo $env:PYTHONUTF8` 输出 `1`

### 1.4 冒烟验证（必须全绿）

新建 `tools/check_env.py`：

```python
import torch
from mmengine import Config
from mmpretrain import get_model

print("torch      :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("mmpretrain :", __import__("mmpretrain").__version__)

cfg = Config(
    dict(
        model=dict(
            type="ImageClassifier",
            backbone=dict(type="VGG", depth=16, num_classes=1000),
            neck=None,
            head=dict(
                type="ClsHead",
                loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
                topk=(1, 5),
            ),
        )
    )
)
model = get_model(cfg)
out = model(torch.randn(2, 3, 224, 224))
print("logits     :", tuple(out.shape))
print("params(M)  : %.2f" % (sum(p.numel() for p in model.parameters()) / 1e6))
```

期望输出：

```
torch      : 2.11.0+cu128 | cuda: True
mmpretrain : 1.2.0
logits     : (2, 1000)
params(M)  : 138.36
```

- [ ] `params(M)` 是 **138.36**，不是就说明 backbone 没按 VGG16 建出来
- [ ] 只出现一条 `MultiScaleDeformableAttention` 警告，无其他报错

> **坑**：`get_model(dict(...))` 传普通 dict 会抛一个莫名其妙的
> `UnboundLocalError: cannot access local variable 'config'`（mmpretrain 1.2.0 的 bug）。
> 必须传 `Config`，且 **`Config` 里要有 `model` 键**（即完整训练配置的形态），否则报 `KeyError: 'model'`。

**验收**：上面 4 行输出全部符合预期。

---

## Phase 2：读论文 & 结构设计（0.5 天）

### 2.1 精读论文，只抓这 5 件事

论文：*Very Deep Convolutional Networks for Large-Scale Image Recognition*（arXiv:1409.1556）

- [ ] 2.1.1 为什么全用 **3×3 卷积**堆叠？（两个 3×3 的感受野 = 一个 5×5，但参数更少、非线性更多）
- [ ] 2.1.2 五个 stage 的通道规则：`64 → 128 → 256 → 512 → 512`，**每次 MaxPool 后通道翻倍，512 封顶**
- [ ] 2.1.3 三个 FC 层：`25088 → 4096 → 4096 → 1000`，**参数量 1.24 亿，占全网 89%**
- [ ] 2.1.4 训练 recipe（**本次只需知道，不训 ImageNet**）：SGD、lr 0.01、momentum 0.9、weight decay 5e-4（原版）、batch 256；论文约 74 epoch，mmpretrain 用 100 epoch + step [30,60,90] ×0.1
- [ ] 2.1.5 测试：single-crop 71.5%，dense/multi-crop 74.5%（VGG16）

### 2.2 手填结构表（写完再对照下表）

| stage | 配置 | 输出通道 | 输出尺寸(224 输入) | 该 stage 参数量 |
| :--- | :--- | :---: | :---: | :---: |
| 1 | 2×conv3-64 + pool | 64 | 112×112 | 0.039 M |
| 2 | 2×conv3-128 + pool | 128 | 56×56 | 0.221 M |
| 3 | 3×conv3-256 + pool | 256 | 28×28 | 1.475 M |
| 4 | 3×conv3-512 + pool | 512 | 14×14 | 5.900 M |
| 5 | 3×conv3-512 + pool | 512 | 7×7 | 7.079 M |
| FC | 4096-4096-1000 | — | 1000 | 123.64 M |
| **合计** | 13 conv + 3 fc | | | **138.36 M** |

- [ ] 2.2.1 不看答案，自己算一遍 stage 1 和 stage 4 的参数量，对上表
- [ ] 2.2.2 能回答：为什么 `512 * 7 * 7 = 25088`？（5 次 pool，224/32 = 7）

### 2.3 确定复现范围

- [ ] 2.3.1 **结构基线：VGG16（配置 D，无 BN）** —— 论文主角，13 conv + 3 fc
- [ ] 2.3.2 **结构变体：VGG16-BN** —— 只改 `norm_cfg=dict(type="BN")`，本次只验证结构（BN 层位置、参数、`state_dict` 键名），不追求精度
- [ ] 2.3.3 **整个结构族**：11 / 13 / 16 / 19 四个深度 × 有/无 BN，共 8 个变体 —— 这就是 Phase 4 的验收范围
- [ ] 2.3.4 明确不做什么：不训 ImageNet、不做 multi-crop / dense evaluation（那是测试技巧，不是结构）

### 2.4 搞懂 MMPretrain 怎么组装模型

```
ImageClassifier
├── data_preprocessor : ClsDataPreprocessor（归一化 + one-hot 标签）
├── backbone          : VGG —— 输出 tuple(feature maps)，或者 tuple(logits)
├── neck              : None（VGG 不用 GAP）
└── head              : ClsHead（只算 loss 和 accuracy，没有 fc）
```

关键设计（**必须理解，否则你的实现无法加载官方权重**）：

- 官方把 **3 层 FC 放在了 backbone 内部**，`num_classes > 0` 时 backbone 直接吐出 `(B, 1000)` 的 logits；
- `out_indices` 默认值随 `num_classes` 变：`num_classes > 0 → (5,)`（第 5 个 stage 输出 + 分类头），`num_classes <= 0 → (4,)`（只要特征图）；
- 所以 `head` 只配 loss，不带 `in_channels`。

- [ ] 2.4.1 能解释：为什么这种设计下 `ClsHead` 不需要 `num_classes`
- [ ] 2.4.2 想清楚另一种“教科书式”写法（backbone 只出特征 → GAP → LinearClsHead）为什么**加载不了**官方权重

### 2.5 先读官方实现（20 分钟，只读不抄）

- [ ] 打开 `mmpretrain/models/backbones/vgg.py`，重点看三处：`make_vgg_layer`、`range_sub_modules` 的构建、`forward` 的分 stage 循环

**验收**：合上论文，能默写出 VGG16 的 stage 表；能说清 `out_indices` 与 `num_classes` 的联动。

---

## Phase 3：项目骨架（0.5 天）

### 3.1 目录结构

```
VGGNet/
├── pyproject.toml
├── README.md
├── configs/
│   ├── vgg16net_8xb32_in1k.py   # 将来真训练用（100 epoch）
│   └── vgg16net_mini.py         # 本次冒烟用（1 epoch，16 张假图）
├── vggnet/                      # 你自己的包（不用改 mmpretrain 源码）
│   ├── __init__.py
│   └── backbones/
│       ├── __init__.py          # 导出 VGG
│       └── vgg.py               # ★ 你要写的核心文件
├── tools/
│   ├── check_env.py
│   ├── check_parity.py          # 单变体（VGG16）快速对齐
│   ├── check_all_variants.py    # ★ 8 变体总验收（Phase 4.5）
│   ├── train.py
│   └── make_fake_imagenet.py    # 造 16 张图，用于冒烟
└── docs/
    └── TODOLIST.md              # 本文件
```

- [ ] 3.1.1 建好目录（`vggnet/` 用包，不要用单个 .py 文件）

### 3.2 `pyproject.toml`（让 `import vggnet` 在任何目录都成立）

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vggnet"
version = "0.1.0"
requires-python = ">=3.8"

[tool.setuptools.packages.find]
include = ["vggnet*"]
```

```powershell
pip install -e . --no-build-isolation
```

- [ ] 3.2.1 `python -c "import vggnet; print(vggnet.__file__)"` 指向本项目目录

> 为什么要 editable 安装：config 里用 `custom_imports` 导入你的模块时，训练进程的工作目录不一定是项目根目录，
> 靠 `sys.path` 碰运气不如装一下。`--no-build-isolation` 可避免每次都去联网拉 setuptools。

### 3.3 注册名怎么起（易错）

mmpretrain 已经注册了 `'VGG'`。你如果再 `@MODELS.register_module()` 一个同名类，会直接抛
`KeyError: 'VGG' is already registered`。所以：

```python
@MODELS.register_module("VGGNet")   # 显式指定注册名
class VGG(BaseBackbone):
    ...
```

好处：config 里 `type='VGGNet'` 是你的实现，`type='VGG'` 是官方实现，**同一个脚本里可以并存、可以互相加载权重做对照**。

- [ ] 3.3.1 `from vggnet.backbones.vgg import VGG` 能导入

**验收**：`pip install -e .` 成功，`import vggnet` 成功。

---

## Phase 4：手写 Backbone（1–2 天，核心）

### 4.1 骨架（先抄签名，别抄实现）

`vggnet/backbones/vgg.py`：

```python
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS


def make_vgg_layer(in_channels, out_channels, num_blocks, conv_cfg=None,
                   norm_cfg=None, act_cfg=dict(type="ReLU"),
                   dilation=1, ceil_mode=False):
    """一个 stage：num_blocks 个 ConvModule(k=3) + 1 个 MaxPool2d。"""
    # TODO: 返回 list[nn.Module]
    raise NotImplementedError


@MODELS.register_module("VGGNet")
class VGG(BaseBackbone):
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4),
    }

    def __init__(self, depth, num_classes=-1, num_stages=5,
                 dilations=(1, 1, 1, 1, 1), out_indices=None,
                 frozen_stages=-1, conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type="ReLU"), norm_eval=False,
                 ceil_mode=False, with_last_pool=True,
                 init_cfg=[dict(type="Kaiming", layer=["Conv2d"]),
                           dict(type="Constant", val=1., layer=["_BatchNorm"]),
                           dict(type="Normal", std=0.01, layer=["Linear"])]):
        super().__init__(init_cfg)
        # TODO 1: 校验 depth / num_stages / dilations；决定 out_indices 默认值
        # TODO 2: 逐 stage 建层，同时记录 range_sub_modules（后面要用）
        # TODO 3: 处理 with_last_pool=False（去掉最后一层 pool）
        # TODO 4: num_classes > 0 时建 classifier
        raise NotImplementedError

    def forward(self, x):
        # TODO: 按 stage 前向；命中 out_indices 就 append；有 num_classes 时追加 logits
        raise NotImplementedError

    def _freeze_stages(self):
        # TODO: 冻结前 frozen_stages 个 stage（eval + requires_grad=False）
        ...

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        # TODO: mode=True 且 norm_eval=True 时，把 BN 切回 eval
        ...
        return self   # ★ 官方实现漏了这个 return，见 4.5
```

### 4.2 关键约束（对照论文检查）

| # | 约束 | 说明 |
| :--- | :--- | :--- |
| 1 | 卷积一律 `kernel_size=3, stride=1, padding=dilation` | 保持分辨率不变，尺寸只由 pool 改变 |
| 2 | **`bias=True`** | 无 BN 时必须有 bias；官方即使有 BN 也传 `True`（与已发布权重一致） |
| 3 | 通道 `64 * 2**i`（i<4），第 5 个 stage 恒为 512 | i=0..4 → 64,128,256,512,512 |
| 4 | 每个 stage 末尾一个 `MaxPool2d(2, 2, ceil_mode=ceil_mode)` | 共 5 次下采样 → 224/32 = 7 |
| 5 | classifier 输入维度 `512*7*7 = 25088` | **硬编码**：只接受 224×224（5 次 pool → 7×7）。换分辨率或改增强尺寸必须同步改这行，否则报 `mat1 and mat2 shapes cannot be multiplied` |
| 6 | `num_classes=-1` 表示“只做特征提取” | 此时不应创建 classifier |

- [ ] 4.2.1 每条约束都能说出“不这么做会怎样”

### 4.3 难点：`range_sub_modules` 是什么

因为 `features` 是一个扁平的 `nn.Sequential`（所有 conv/pool 平铺），但 `forward`、`_freeze_stages` 都希望**按 stage 操作**。
所以建层时顺便记录每个 stage 在 Sequential 中的下标区间 `[start, end)`：

```
stage 0: [0, 3)   conv, conv, pool
stage 1: [3, 6)   conv, conv, pool
stage 2: [6, 10)  conv, conv, conv, pool
...
```

- [ ] 4.3.1 想清楚 `num_modules = num_blocks + 1`（+1 是 pool）
- [ ] 4.3.2 想清楚 `with_last_pool=False` 时为什么要同时 `pop` 层、并把最后一个区间 `end -= 1`

### 4.4 校验脚本（写完立刻验证，别等训练）

> ⚠️ **先记住这一条，它能帮你省掉几小时**：权重对齐**必须**在 `eval` 模式下做。
> `Dropout` 和 `BatchNorm` 在 `train` 模式下会引入随机性（dropout 随机置零、BN 用 batch 统计量并更新 running stats），
> 这时候 `diff` 不可能为 0——**这是正常现象，不代表你结构写错了**。
> 所以脚本里的 `a.eval(); b.eval()` 两行绝对不能省；看到 diff 是 0.1 量级且每次运行都不同，先查这里，再查网络代码。

`tools/check_parity.py`：

```python
import torch
from mmpretrain.models.backbones import VGG as OfficialVGG

from vggnet.backbones.vgg import VGG as MyVGG

# ★ 必须 eval：train 模式下 Dropout / BN 会引入随机，diff 永远不为 0
a = OfficialVGG(16, num_classes=1000); a.eval()
b = MyVGG(16, num_classes=1000);       b.eval()

sa, sb = a.state_dict(), b.state_dict()
assert set(sa) == set(sb), f"键不一致: 多={set(sb)-set(sa)} 少={set(sa)-set(sb)}"
assert all(sa[k].shape == sb[k].shape for k in sa), "有层形状不对"

b.load_state_dict(sa, strict=True)          # 官方权重直接灌进你的实现
x = torch.randn(1, 3, 224, 224)             # ★ 同一个输入，别调两次 randn
with torch.no_grad():
    oa = a(x)[0]
    ob = b(x)[0]
print("max |diff| =", (oa - ob).abs().max().item())
print("params(M)  = %.2f" % (sum(p.numel() for p in b.parameters()) / 1e6))
```

期望：

```
max |diff| = 0.0        ← 必须是 0.0（同设备、同输入、同权重，浮点顺序一致）
params(M)  = 138.36
```

> 这条 `0.0` 的前提是**同一设备 + 同一 dtype**。如果你把脚本挪到 GPU 上跑，cuDNN 的非确定性可能带来
> `1e-6` 级差异，那时判定阈值放宽到 `1e-5` 即可，不必强求 `0.0`。

> 注意：**不要写 `a = OfficialVGG(...).eval()`**。官方 `VGG.train()` 没有 `return self`，
> 链式调用 `.eval()` 会返回 `None`，然后你会在下一行收到 `AttributeError: 'NoneType' object has no attribute 'state_dict'`。
> 分两行写；你自己的实现里请把 `return self` 补上。

- [ ] 4.4.1 `max |diff| = 0.0`
- [ ] 4.4.2 `params(M) = 138.36`
- [ ] 4.4.3 顺手验证 `out_indices=(4,)`、`num_classes=-1` 时输出 `(1, 512, 7, 7)`

### 4.5 把结构验证做满：8 个变体全过

只对齐 VGG16 说明不了什么——`arch_settings` 抄错一个数字、`norm_cfg` 传错位置，单测 VGG16 都可能照样 `diff=0`。
用同一份脚本把下面 8 个组合全跑一遍，才算结构复现完成：

| 变体 | 构造参数 | 期望参数量 |
| :--- | :--- | :---: |
| VGG11 | `depth=11, num_classes=1000` | 132.86 M |
| VGG13 | `depth=13` | 133.05 M |
| VGG16 | `depth=16` | 138.36 M |
| VGG19 | `depth=19` | 143.67 M |
| VGG11-BN | `depth=11, norm_cfg=dict(type="BN")` | 132.87 M |
| VGG13-BN | `depth=13, norm_cfg=dict(type="BN")` | 133.05 M |
| VGG16-BN | `depth=16, norm_cfg=dict(type="BN")` | 138.37 M |
| VGG19-BN | `depth=19, norm_cfg=dict(type="BN")` | 143.68 M |

```python
# tools/check_all_variants.py
import torch
from mmpretrain.models.backbones import VGG as OfficialVGG

from vggnet.backbones.vgg import VGG as MyVGG

CASES = [(11, None), (13, None), (16, None), (19, None),
         (11, {"type": "BN"}), (13, {"type": "BN"}),
         (16, {"type": "BN"}), (19, {"type": "BN"})]

for depth, norm_cfg in CASES:
    kw = dict(depth=depth, num_classes=1000, norm_cfg=norm_cfg)
    a, b = OfficialVGG(**kw), MyVGG(**kw)
    a.eval(); b.eval()          # ★ 必须 eval，否则 diff≠0（dropout / BN 随机性）
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), f"depth={depth} 键不一致"
    assert all(sa[k].shape == sb[k].shape for k in sa), f"depth={depth} 形状不一致"
    b.load_state_dict(sa, strict=True)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        diff = (a(x)[0] - b(x)[0]).abs().max().item()
    print(f"depth={depth:2d} type={'bn ' if norm_cfg else 'plain'} "
          f"params={sum(p.numel() for p in b.parameters()) / 1e6:7.2f}M diff={diff}")
```

期望输出（8 行，`diff` 全为 0.0，参数量与上表逐项吻合）：

```
depth=11 type=plain params= 132.86M diff=0.0
depth=13 type=plain params= 133.05M diff=0.0
depth=16 type=plain params= 138.36M diff=0.0
depth=19 type=plain params= 143.67M diff=0.0
depth=11 type=bn  params= 132.87M diff=0.0
depth=13 type=bn  params= 133.05M diff=0.0
depth=16 type=bn  params= 138.37M diff=0.0
depth=19 type=bn  params= 143.68M diff=0.0
```

> 如果 8 行的 `diff` 都是 `1e-1` 量级、而且每次运行数字都不一样，**先看是不是漏了 `.eval()`**，
> 再去怀疑层数、通道、padding——这是最常见的误判路径。

> 跑 BN 变体时还会多一条 `Unnecessary conv bias before batch/instance norm` 警告。
> 官方实现同样保留 `bias=True`（为了和已发布权重对齐），**这是预期的，别为了消警告把 bias 改掉**。

- [ ] 4.5.1 8 行全部 `diff=0.0`
- [ ] 4.5.2 8 个参数量全部对上（哪怕差 0.01M 也说明某层通道数错了）

### 4.6 四个必须知道的坑

1. **`init_cfg` 不能省**。不写 Kaiming/Constant(1)/Normal(0.01)，无 BN 的 VGG16 从零训练极易梯度爆炸。
   而且 `init_cfg` 只有在继承 `BaseBackbone`（即 `BaseModule`）并调用 `super().__init__(init_cfg)` 时才生效。
2. **`train()` 记得 `return self`**。官方也漏了，属于历史遗留；补上不影响权重对齐。
3. **`num_classes` 存在 backbone 里**，意味着 `data_preprocessor` 的 `num_classes` 必须和它一致，否则 one-hot 长度对不上，报形状错误。
4. **`frozen_stages` 在构造时不生效**。冻结动作写在 `train()` 里，只有调用过 `model.train()` 之后参数才是 `requires_grad=False`。
   Runner 训练前会自动 `model.train()`，所以训练没问题；但如果你在脚本里手动 `VGG(..., frozen_stages=2)` 后立刻检查梯度，会以为它没生效——**这是最容易误判的一条**。

### 4.7 结构复现验收清单（逐项打勾才算完成）

| # | 检查项 | 期望结果 |
| :--- | :--- | :--- |
| 1 | 8 个变体的 `state_dict` 键集合 | 与官方完全相同（VGG16 是 32 个键） |
| 2 | 8 个变体的参数量 | 与 4.5 表逐项吻合 |
| 3 | 加载官方权重后前向 | 最大误差 = 0.0 |
| 4 | `num_classes=-1, out_indices=(4,)` | 输出 `(B, 512, 7, 7)` |
| 5 | `num_classes=-1, out_indices=(2, 3, 4)` | 输出 `28×28 / 14×14 / 7×7` 三个尺度 |
| 6 | `with_last_pool=False` | 与官方同参数下 shape 一致（`14×14`） |
| 7 | `ceil_mode=True` + 奇数输入（如 225） | 与官方输出一致，不报错 |
| 8 | `dilations=(1,1,1,1,2)` | 能构建且输出 shape 与官方一致 |
| 9 | `norm_eval=True` 且 `model.train()` | 所有 BN 处于 eval 状态 |
| 10 | `frozen_stages=2` 且调用过 `model.train()` | 前 2 个 stage 冻结，第 3 个 stage 仍可训练 |
| 11 | 对齐前模型和官方实现**都**调用过 `.eval()` | `diff=0.0`；漏写时 diff 每次运行都不同 |

- [ ] 4.7.1 清单 11 项全部通过
- [ ] 4.7.2 能口头回答：误差为 0 证明了什么、**没有**证明什么（见 0.1 的提醒）
- [ ] 4.7.3 能解释：漏写 `.eval()` 时 diff 为什么不为 0（dropout / BN 的训练模式行为）

**验收**：4.4、4.5、4.7 全绿 → Phase 4 完成，进入 Phase 5。

---

## Phase 5：写 Config（0.5 天，只为“接线”）

结构复现不需要训练 ImageNet，但需要证明你的 backbone 能被 MMPretrain 的 Runner 正常构建、前向、反向。
这一 Phase 的唯一目标：**用 16 张假图把整条链路跑通**，证明它不是一个“只能在 notebook 里 forward 的玩具”。

### 5.1 四个 `_base_` 各管什么

| 文件 | 内容 |
| :--- | :--- |
| `_base_/models/vgg16.py` | `ImageClassifier` + `VGG(depth=16, num_classes=1000)` + `ClsHead` |
| `_base_/datasets/imagenet_bs32_pil_resize.py` | 数据管线（RandomResizedCrop 224 / ResizeEdge 256+CenterCrop）、batch 32、`Accuracy(topk=(1,5))` |
| `_base_/schedules/imagenet_bs256.py` | SGD lr0.1、MultiStepLR [30,60,90]、100 epoch、`auto_scale_lr.base_batch_size=256` |
| `_base_/default_runtime.py` | `default_scope='mmpretrain'`、hooks、日志、`load_from` |

引用安装包内的配置要用 **`mmpretrain::` 前缀，且路径从 `configs/` 下一层开始**：

```python
_base_ = [
    "mmpretrain::_base_/models/vgg16.py",
    "mmpretrain::_base_/datasets/imagenet_bs32_pil_resize.py",
    "mmpretrain::_base_/schedules/imagenet_bs256.py",
    "mmpretrain::_base_/default_runtime.py",
]
```

> 写成 `mmpretrain::configs/_base_/...` 会报 `FileNotFoundError`（前缀已经指向 `configs/` 了）。

- [ ] 5.1.1 四个文件都点开读过一遍，知道每个键从哪来

### 5.2 完整 config（`configs/vgg16net_8xb32_in1k.py`）

> 这份是**将来真训练**用的版本（ImageNet、100 epoch）。本次结构复现不用它，直接用 5.3 的覆盖项把它变成 mini 版即可。

```python
_base_ = [
    "mmpretrain::_base_/models/vgg16.py",
    "mmpretrain::_base_/datasets/imagenet_bs32_pil_resize.py",
    "mmpretrain::_base_/schedules/imagenet_bs256.py",
    "mmpretrain::_base_/default_runtime.py",
]

# 1) 把你的实现注册进来（模块被 import 时装饰器才生效）
#    多卡 / DDP 场景建议同时在入口脚本手动 import 一次，见 5.4
custom_imports = dict(imports=["vggnet.backbones.vgg"], allow_failed_imports=False)

# 2) 换成你自己的 backbone
model = dict(
    backbone=dict(type="VGGNet", depth=16, num_classes=1000),
    # 加 BN 的版本：backbone=dict(type="VGGNet", depth=16, num_classes=1000,
    #                            norm_cfg=dict(type="BN")),
)

# 3) lr：官方 recipe 是 8 卡 × bs32（总 256）配 lr 0.01
optim_wrapper = dict(optimizer=dict(type="SGD", lr=0.01, momentum=0.9,
                                    weight_decay=1e-4))

# 4) 数据
data_root = "data/imagenet"
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

# 5) 训练时长
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
```

> **关于 `custom_imports`**：它由 `Runner.from_cfg()` 负责执行，单进程训练没问题。
> 但 Windows 的 `spawn` 会在子进程里**重新执行入口脚本**，DDP 下每个 rank 还要各自构建模型；
> 因此在 `tools/train.py` 顶部再手动 `import vggnet.backbones.vgg` 做双保险（见 5.4）。
> `allow_failed_imports=False` 的作用是让导入错误当场暴露，别改成 `True` 把它吞掉——否则你只会看到一个
> 莫名其妙的 `KeyError`（提示 `VGGNet` 找不到），然后怀疑注册名写错了。

- [ ] 5.2.1 每一行都能解释为什么这么写
- [ ] 5.2.2 想清楚：单卡 bs32 时 lr 该怎么定？（本次不追求精度，能用即可；真要复现指标看附录 F）

### 5.3 先用假数据打通流程（强烈建议）

真 ImageNet 目录结构：`data/imagenet/train/<类别名>/*.JPEG`、`data/imagenet/val/<类别名>/*.JPEG`。

先造假数据（`tools/make_fake_imagenet.py`，只用于冒烟，**不代表能训出东西**）：

```python
import os
import numpy as np
from PIL import Image

root = "data/mini"
rng = np.random.default_rng(0)
for split in ("train", "val"):
    for cls in ("cat", "dog"):
        os.makedirs(os.path.join(root, split, cls), exist_ok=True)
        for i in range(8):
            arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(root, split, cls, f"{i}.JPEG"))
```

> ⚠️ **224×224 是硬约束**：`classifier[0]` 是 `Linear(512*7*7, 4096)`，其中的 7×7 来自 224 经过 5 次 pool。
> 所以数据管线**最终输出必须是 224**：`RandomResizedCrop(scale=224)`、`Resize(224)`、`CenterCrop(224)` 都行，
> 但你不能把它改成 256、也不能用随机尺寸——那会直接崩在 FC 层（附录 A 那条 `mat1 and mat2` 就是它）。
> 假图造成 256×256 是安全的，因为 `RandomResizedCrop` 会再裁成 224。

冒烟配置：把 5.2 的配置**复制**成 `configs/vgg16net_mini.py`，在末尾追加下面这些覆盖项（**少覆盖一处就会报错**）：

```python
data_root = "data/mini"
classes = ["cat", "dog"]                    # 不加会断言“子文件夹数 2 != 1000”
data_preprocessor = dict(num_classes=2)      # 必须和 backbone 的 num_classes 一致
model = dict(backbone=dict(type="VGGNet", depth=16, num_classes=2))
train_dataloader = dict(batch_size=2, num_workers=0,   # Windows 冒烟先用 0
                        dataset=dict(data_root=data_root, classes=classes))
val_dataloader = dict(batch_size=2, num_workers=0,
                      dataset=dict(data_root=data_root, classes=classes))
test_dataloader = val_dataloader
val_evaluator = dict(type="Accuracy", topk=(1,))       # 2 类算 top5 无意义
test_evaluator = val_evaluator
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
```

### 5.4 启动脚本 `tools/train.py`

```python
import argparse

import vggnet.backbones.vgg  # noqa: F401  ★ 手动注册，别只依赖 config 里的 custom_imports
from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--cfg-options", nargs="+", default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # --cfg-options optimizer.lr=0.001 这种点号路径可直接覆盖
    for opt in args.cfg_options or []:
        key, value = opt.split("=", 1)
        cfg.merge_from_dict({key: eval(value)})   # 仅本地调试用，别 eval 外部输入
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
```

> **为什么还要手动 `import` 一次**：`custom_imports` 在 `Runner.from_cfg()` 里执行，只覆盖“主进程构建模型”这条路径；
> Windows 的 `spawn` 会重新执行入口脚本、DDP 的每个 rank 也要各自构建模型。显式 `import` + `pip install -e .`
> 是最稳的组合。**单卡冒烟通过，不代表多卡 DDP 一定正常**，多卡前先看 6.5。

```powershell
$env:PYTHONUTF8 = "1"
python tools/train.py configs/vgg16net_mini.py
```

- [ ] 5.4.1 能跑完 1 个 epoch，最后打印 `accuracy/top1`
- [ ] 5.4.2 在 `work_dirs/` 下生成了 `.pth` 和 `vis_data/` 日志

**验收**：mini 配置 1 epoch 跑通，`val accuracy: 50.0000`（二分类瞎猜水平，正常）。

> **坑（实测）**：如果忘记把 lr 降到 0.01（base 里是 0.1），无 BN 的 VGG16 会在 **2–3 个 iter 内 loss 变 NaN**。
> 这不是代码错，是 lr 太大 + 无 BN。看到 `loss: nan` 先查 lr。

---

## Phase 6：训练/评测设施（接线验证，0.5 天）

结构复现的最后一公里：证明你的模型能被 checkpoint、评测器、推理接口正常使用，
而不是“forward 一次就完事”。本 Phase 全部在 mini 数据上做，不看精度。

### 6.1 日志怎么读

```
Epoch(train) [1][100/5000]  lr: 1.0000e-02  eta: 5:12:33  time: 0.412  data_time: 0.031  memory: 8123  loss: 2.301
```

| 字段 | 含义 | 异常信号 |
| :--- | :--- | :--- |
| `lr` | 当前学习率 | 一直是 0 → 优化器没参数 |
| `time` / `data_time` | 单 iter 总耗时 / 取数耗时 | `data_time` 接近 `time` → IO 瓶颈，加 workers |
| `memory` | 显存 MB | 接近上限会 OOM |
| `loss` | 训练损失 | `nan` → lr 太大 / 初始化错 |

- [ ] 6.1.1 能判断一次训练是“算力瓶颈”还是“IO 瓶颈”

### 6.2 checkpoint 三件套（别混）

| 需求 | 写法 | 说明 |
| :--- | :--- | :--- |
| 只加载权重（微调/评测） | `cfg.load_from = "xxx.pth"` | 不恢复 optimizer / epoch |
| 断点续训 | `cfg.resume = True`（或 `--resume`） | 恢复 optimizer、epoch、dataloader |
| 从预训练 backbone 初始化 | `model.backbone.init_cfg = dict(type="Pretrained", checkpoint=..., prefix="backbone.")` | 只灌 backbone |

- [ ] 6.2.1 用 `load_from` 加载官方 VGG16 权重，在 mini 验证集上跑完一次 `test`：无 shape 报错、能打印 accuracy 即通过（mini 数据上的数值没有意义）

### 6.3 评测与推理

```python
# 评测（自己的 checkpoint）
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile("configs/vgg16net_8xb32_in1k.py")
cfg.load_from = "work_dirs/vgg16net/epoch_1.pth"   # 换成你自己跑出来的 checkpoint
Runner.from_cfg(cfg).test()
```

```python
# 官方权重直接推理（不写代码验证结构，最快的一条路）
from mmpretrain import inference_model, get_model

model = get_model("vgg16_8xb32_in1k", pretrained=True)
print(inference_model(model, "图片路径.jpg")["pred_class"])
```

- [ ] 6.3.1 官方权重推理出合理的类别
- [ ] 6.3.2 自己的实现加载官方权重后，**推理结果与官方逐位一致**（这是结构复现的最终证明）

### 6.4 显存与吞吐（VGG 很吃显存）

- [ ] 6.4.1 记录：bs32 / 224 输入 / fp32 大约需要多少显存（VGG16 权重本身 0.55 GB，加上激活值通常 10 GB 以上，务必实测一遍）
- [ ] 6.4.2 OOM 时的处理顺序：降 batch → 开 AMP（`optim_wrapper=dict(type="AmpOptimWrapper")`）→ 梯度累积

### 6.5 多卡 / DDP（本次可跳过，但别踩坑）

结构复现不需要多卡。但如果你顺手想试 `torchrun`，注意 Windows 下这几件事：

- [ ] 6.5.1 入口脚本必须保留 `if __name__ == "__main__":`（spawn 会重新执行入口，没有 guard 会无限递归）
- [ ] 6.5.2 自定义模块在入口处**显式 import**（见 5.4），不要只靠 `custom_imports`
- [ ] 6.5.3 启动方式：`torchrun --nproc_per_node=2 tools/train.py configs/vgg16net_mini.py`
- [ ] 6.5.4 多卡时 lr 按附录 F.2 换算；`num_workers` 减半，Windows 上别开太高
- [ ] 6.5.5 记住：**单卡冒烟通过 ≠ 多卡一定正常**，结构复现本身也不需要多卡来证明

**验收**：能在 mini 数据上完成「加载权重 → 评测 → 出 accuracy」和「单张图推理」。

---

## Phase 7：加分项（选做，但很能体现深度）

结构复现完成后，下面这些都只是“换个参数再验一次”，成本很低，但能让简历上多两行：

- [ ] 7.1 **多尺度特征**：`num_classes=-1, out_indices=(3, 4)`，接一个假检测头，理解 FPN 为什么需要多尺度
- [ ] 7.2 **dilation**：`dilations=(1,1,1,1,2)`，感受野变大、分辨率不变（DeepLab 的 VGG16 backbone 就是这么改的）
- [ ] 7.3 **`with_last_pool=False`**：输出 14×14，理解检测/分割为什么不要最后那次下采样
- [ ] 7.4 **迁移学习接口**：`frozen_stages` + `norm_eval`，在小数据集上微调，说清两者的区别
- [ ] 7.5 **ONNX 导出**：`torch.onnx.export` + onnxruntime，对比输出是否一致
- [ ] 7.6 **写 README**：结构表、8 变体对齐结果（diff 全 0）、命令、踩坑记录

---

## 附录 A 报错速查

| 报错 | 原因 | 解决 |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'mmengine'` | `pip install mmpretrain` 不带 mmengine | 手动 `pip install mmengine==0.10.7` |
| `UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 ... collect_env` | 中文 Windows 下 mmengine 按 GBK 解码 `cl.exe` 输出 | `$env:PYTHONUTF8="1"` |
| `Fail to import MultiScaleDeformableAttention ... install mmcv rather than mmcv-lite` | 用了 mmcv-lite | **警告，忽略**；分类不需要该算子 |
| `Unnecessary conv bias before batch/instance norm` | BN 版仍保留 `bias=True` | **警告，忽略**；与官方权重对齐所需 |
| `UnboundLocalError: cannot access local variable 'config'` | `get_model(dict(...))` 传了普通 dict（1.2.0 bug） | 传 `Config(dict(model=...))` 或模型名字符串 |
| `KeyError: 'model'` | `get_model(Config(dict(model=...)))` 里漏了 `model` 键 | 包一层 `model=` |
| `KeyError: 'VGG' is already registered` | 注册名与官方冲突 | `@MODELS.register_module("VGGNet")` |
| `KeyError: 'VGGNet' is not in the registry`（大意：找不到注册名） | `custom_imports` 没执行，或 `allow_failed_imports=True` 把导入错误吞了 | 改回 `False` 看真实报错；入口脚本手动 `import vggnet.backbones.vgg` |
| 多卡 / DDP 子进程报找不到自定义模块 | spawn 子进程的 `sys.path` 与主进程不同 | `pip install -e .`，并在入口文件顶部显式 import |
| `FileNotFoundError: ...\.mim\configs\configs\_base_...` | `_base_` 前缀写成了 `mmpretrain::configs/...` | 改成 `mmpretrain::_base_/...` |
| `AssertionError: The number of subfolders (2) doesn't match ... (1000)` | 小数据集没传 `classes` | `dataset=dict(classes=["cat","dog"])` |
| `loss: nan` | lr 太大（无 BN 的 VGG 尤其敏感） | lr 降到 0.01 或更低 |
| `AttributeError: 'NoneType' object has no attribute 'state_dict'` | 官方 `train()` 没 `return self`，`.eval()` 返回 None | 分两行写；自己的实现补 `return self` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | 输入不是 224（改了 `RandomResizedCrop` 尺寸 / `Resize`），或 `with_last_pool=False` 让特征图不是 7×7 | 数据管线最终输出必须是 224×224；检查 pool 次数 |
| 对齐脚本 `diff` 每次都不一样、量级 0.1 | 模型没切 `eval()`，dropout / BN 在训练模式 | `a.eval(); b.eval()` |
| Windows 下 DataLoader 卡住 / 报 spawn 相关错误 | 多进程启动方式 | 冒烟用 `num_workers=0`；正式训练 4–8，入口包 `if __name__ == "__main__":` |

## 附录 B 源码地图

| 想看什么 | 路径（安装包内） |
| :--- | :--- |
| 官方 VGG 实现 | `mmpretrain/models/backbones/vgg.py` |
| Backbone 基类（`init_cfg`、`init_weights`） | `mmpretrain/models/backbones/base_backbone.py` |
| 模型组装 | `mmpretrain/models/classifiers/image.py` |
| 分类头 / loss 入口 | `mmpretrain/models/heads/cls_head.py` |
| 分类模型配置 | `mmpretrain/.mim/configs/_base_/models/vgg16.py` |
| 数据管线 | `mmpretrain/.mim/configs/_base_/datasets/imagenet_bs32_pil_resize.py` |
| 训练 recipe | `mmpretrain/.mim/configs/_base_/schedules/imagenet_bs256.py` |
| 注册表 | `mmpretrain/registry.py` |
| ImageNet 数据集 | `mmpretrain/datasets/imagenet.py` |

> 路径说明：`pip install` 安装时，配置在 `site-packages/mmpretrain/.mim/configs/`；从源码 clone 安装时，配置在仓库根目录的 `configs/`。
> 定位方法：`python -c "import mmpretrain, os; print(os.path.dirname(mmpretrain.__file__))"`

## 附录 C 参考资料

- 论文：<https://arxiv.org/abs/1409.1556>
- VGG 模型库（含预训练权重与训练 log，复现时拿来对曲线）：<https://github.com/open-mmlab/mmpretrain/tree/main/configs/vgg>
- 自定义模型官方文档：<https://mmpretrain.readthedocs.io/en/latest/advanced_guides/modules.html>
- 安装与 FAQ：<https://mmpretrain.readthedocs.io/en/latest/get_started.html>、<https://mmpretrain.readthedocs.io/en/latest/notes/faq.html>

## 附录 D 自测题（做完再写 README）

1. 两个 3×3 卷积的感受野等于一个几×几？参数量比是多少？
2. 为什么 VGG 的 `features` 用扁平 `Sequential` 而不是嵌套 `Sequential`？各有什么代价？
3. `range_sub_modules` 如果记录错了，会在哪一步暴露？
4. `num_classes=-1` 和 `out_indices=(4,)` 同时设置时，`forward` 返回什么？
5. 为什么官方把 FC 放进 backbone？如果拆到 head 里，还能加载官方权重吗？
6. `init_cfg` 里 `Constant(val=1., layer=["_BatchNorm"])` 初始化的是哪两个参数？
7. `norm_eval=True` 和 `frozen_stages=5` 的区别是什么？
8. 无 BN 的 VGG16 为什么对 lr 这么敏感？加了 BN 之后为什么可以用更大 lr？
9. 你的实现和官方 `max|diff| = 0`，能说明什么？**不能**说明什么？
10. 为什么 VGG 的 classifier 不能直接吃 320×320 的输入？（提示：5 次 pool 之后是几×几，`Linear(25088, ...)` 又意味着什么）
11. 如果要做成“任意分辨率都能跑”，该怎么改？改了以后**还能加载官方权重吗**？
12. 对齐脚本忘了写 `.eval()` 会发生什么？为什么 diff 每次运行都不一样？

<details>
<summary>第 10–12 题的参考思路（先自己想）</summary>

- **10**：320 经 5 次 pool 得到 10×10，`10*10*512 = 51200` ≠ 25088，FC 层形状对不上，直接报 `mat1 and mat2`。
- **11**：把 `classifier` 换成 `nn.AdaptiveAvgPool2d(1)` + `Linear(512, num_classes)`（或 1×1 卷积 + GAP），
  就能吃任意 ≥32 的输入。但代价是：① 官方 `classifier.*` 权重形状对不上，**不能直接 load**；
  ② 全局池化丢了空间信息，同数据下精度通常低于三层 FC（这正是后来网络普遍改用 GAP 的原因之一）。
- **12**：`Dropout` 随机置零、`BatchNorm` 用当前 batch 的均值方差并更新 running stats，两者都带随机性/状态变化，
  所以 `train()` 模式下两次前向结果都不一致。对齐权重必须 `eval()`（同时 `no_grad()`）。

</details>

## 附录 E 参考实现（卡住再看）

<details>
<summary>点击展开（先自己写完再对照）</summary>

`vggnet/backbones/vgg.py`：

```python
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS


def make_vgg_layer(in_channels,
                   out_channels,
                   num_blocks,
                   conv_cfg=None,
                   norm_cfg=None,
                   act_cfg=dict(type="ReLU"),
                   dilation=1,
                   ceil_mode=False):
    """一个 VGG stage：num_blocks 个 3x3 卷积 + 一个 2x2 MaxPool。"""
    layers = []
    for _ in range(num_blocks):
        layers.append(
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                bias=True,           # 原版 VGG 卷积带 bias
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,   # 传了 BN 就是 VGG16-BN
                act_cfg=act_cfg))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers


@MODELS.register_module("VGGNet")
class VGG(BaseBackbone):
    """从零复现的 VGG backbone。"""

    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4),
    }

    def __init__(self,
                 depth,
                 num_classes=-1,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=None,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type="ReLU"),
                 norm_eval=False,
                 ceil_mode=False,
                 with_last_pool=True,
                 init_cfg=[
                     dict(type="Kaiming", layer=["Conv2d"]),
                     dict(type="Constant", val=1., layer=["_BatchNorm"]),
                     dict(type="Normal", std=0.01, layer=["Linear"])
                 ]):
        super().__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for vgg")
        assert 1 <= num_stages <= 5
        assert len(dilations) == num_stages

        self.stage_blocks = self.arch_settings[depth][:num_stages]
        self.num_classes = num_classes
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        if out_indices is None:
            out_indices = (5,) if num_classes > 0 else (4,)
        assert max(out_indices) <= num_stages
        self.out_indices = out_indices

        self.in_channels = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks + 1           # +1 是 pool
            end_idx = start_idx + num_modules
            out_channels = 64 * 2**i if i < 4 else 512
            vgg_layers.extend(
                make_vgg_layer(
                    self.in_channels,
                    out_channels,
                    num_blocks,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dilation=dilations[i],
                    ceil_mode=ceil_mode))
            self.in_channels = out_channels
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx

        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = "features"
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                x = vgg_layers[j](x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        return tuple(outs)

    def _freeze_stages(self):
        vgg_layers = getattr(self, self.module_name)
        for i in range(self.frozen_stages):
            for j in range(*self.range_sub_modules[i]):
                m = vgg_layers[j]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self   # 官方实现漏了这行，补上后 .eval() 才能链式调用
```

</details>

---

## 附录 F 指标复现（以后想做再看）

本次不做。但 Phase 5/6 已经把“接线”做好了，将来想跑指标时只需要补数据和对齐 recipe。

**F.1 对齐这些设置**

| 项目 | 官方设置 |
| :--- | :--- |
| 数据 | ImageNet-1k，`data/imagenet/{train,val}/<wnid>/*.JPEG` |
| batch | 8 卡 × 32 = 256 |
| 优化器 | SGD, lr 0.01, momentum 0.9, wd 1e-4 |
| schedule | MultiStep [30,60,90] ×0.1，100 epoch |
| 增强 | RandomResizedCrop(224) + 水平翻转 |
| 测试 | ResizeEdge 256 + CenterCrop 224 |

**F.2 单卡 lr 换算**

- 线性缩放：`lr' = 0.01 × 总 batch / 256`（单卡 bs32 → 0.00125，稳但收敛慢）
- 或开自动缩放：`auto_scale_lr = dict(base_batch_size=256, enable=True)`

**F.3 目标与解释**

| 模型 | 目标 top-1 | 说明 |
| :--- | :---: | :--- |
| VGG16 | 71.62 | 论文 71.5（single-crop） |
| VGG16-BN | 73.74 | BN 让深网更好训 |

- 论文 74.5 是 dense + multi-crop 的结果，**不能拿来和单模型指标对标**
- 差 2 个点以上，按顺序查：lr → 数据管线是否真 224 → BN 的 `eps/momentum` → 初始化 → `ceil_mode`

**F.4 算力量级**：单张 4090 跑满 100 epoch 大约数天到一周，多卡按线性预估。建议先用 1/10 子集把 recipe 跑顺再上全量。

---

### 最后的建议

1. **先做 Phase 4，别急着训 ImageNet**。结构对齐只要半小时，能省掉后面一周的自我怀疑。
2. **每个 Phase 的验收标准都写进 README**，这是“复现”和“跑了个开源项目”的区别。
3. 卡住时优先看官方 `vgg.py` 与官方 log（`configs/vgg/README.md` 里的 `log` 链接），比任何博客都准。
