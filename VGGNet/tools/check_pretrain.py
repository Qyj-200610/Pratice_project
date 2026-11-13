"""Phase 5 验收：官方 1000 类权重 → 你的 VGG16，以及「1000 类 → 2 类」的替换。

用法（在 VGGNet/ 下）：
    python tools/check_pretrain.py
"""
import torch
from mmengine import Config
from mmengine.runner import load_checkpoint
from mmpretrain import get_model

from vggnet.backbones.vgg import VGG as MyVGG

URL = ("https://download.openmmlab.com/mmclassification/v0/vgg/"
       "vgg16_batch256_imagenet_20210208-db26f1a5.pth")

# ---------- 1) 1000 类：backbone 逐键加载 + 输出逐位一致 ----------
official = get_model("vgg16_8xb32_in1k", pretrained=True)
mine = MyVGG(16, num_classes=1000)
mine.eval()

mapping = {k[len("backbone."):]: v
           for k, v in official.state_dict().items()
           if k.startswith("backbone.")}
missing, unexpected = mine.load_state_dict(mapping, strict=False)
print("[1000 类] missing:", missing, "unexpected:", unexpected)
assert not missing and not unexpected, "有键没对上，检查 prefix 或层名"
print("[1000 类] keys:", len(mine.state_dict()))

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    diff = (official(x) - mine(x)[0]).abs().max().item()
print("[1000 类] max |diff| =", diff)
assert diff == 0.0, "输出不一致，结构还没对齐"

# ---------- 2) 2 类：只有 classifier.6 尺寸不匹配，其余全部复用 ----------
cfg = Config(dict(
    model=dict(
        type="ImageClassifier",
        backbone=dict(type="VGGNet", depth=16, num_classes=2),
        neck=None,
        head=dict(type="ClsHead",
                  loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
                  topk=(1,)),
    ),
    data_preprocessor=dict(num_classes=2),
))
model = get_model(cfg)

# ★ 必须用 mmengine 的 load_checkpoint：strict=False 时它只告警并跳过尺寸不匹配的键。
#   裸 torch 的 model.load_state_dict(ckpt, strict=False) 遇到尺寸不匹配会直接 RuntimeError。
load_checkpoint(model, URL, map_location="cpu", strict=False)

conv_ok = torch.equal(model.backbone.features[0].conv.weight,
                      official.backbone.features[0].conv.weight)
fc1_ok = torch.equal(model.backbone.classifier[0].weight,
                     official.backbone.classifier[0].weight)
print("[2 类] conv 权重一致:", conv_ok)
print("[2 类] FC1 权重一致:", fc1_ok)
print("[2 类] classifier.6 形状:", tuple(model.backbone.classifier[6].weight.shape))

with torch.no_grad():
    out = model(torch.randn(1, 3, 224, 224))

assert conv_ok and fc1_ok, "卷积 / 前两层 FC 没有复用官方权重"
assert tuple(model.backbone.classifier[6].weight.shape) == (2, 4096)
assert tuple(out.shape) == (1, 2)

print("[2 类] 输出:", tuple(out.shape))
print("\n全部通过 ✅")
