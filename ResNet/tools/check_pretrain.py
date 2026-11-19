
import torch
from mmengine.runner import load_checkpoint
from mmpretrain import get_model
from mmpretrain.registry import MODELS

from resnet.backbones.resnet import ResNet as MyResNet
from resnet.backbones.senet import SEResNet as MySEResNet

RESNET50 = ("https://download.openmmlab.com/mmclassification/v0/resnet/"
            "resnet50_8xb32_in1k_20210831-ea4938fc.pth")
SERESNET50 = ("https://download.openmmlab.com/mmclassification/v0/se-resnet/"
              "se-resnet50_batch256_imagenet_20200804-ae206104.pth")


def build(backbone, num_classes, in_channels):
    return MODELS.build(dict(
        type="ImageClassifier",
        backbone=backbone,
        neck=dict(type="GlobalAveragePooling"),
        head=dict(
            type="LinearClsHead",
            num_classes=num_classes,
            in_channels=in_channels,
            loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
            topk=(1,))))


def load_backbone(mine, official):
    mapping = {k[len("backbone."):]: v
               for k, v in official.state_dict().items()
               if k.startswith("backbone.")}
    missing, unexpected = mine.load_state_dict(mapping, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return len(mapping)


x = torch.randn(1, 3, 224, 224)

official = get_model("resnet50_8xb32_in1k", pretrained=True)
official.eval()
mine = MyResNet(depth=50)
mine.eval()
n = load_backbone(mine, official)
with torch.no_grad():
    diff = (official.backbone(x)[0] - mine(x)[0]).abs().max().item()
print("[ResNet-50 1000] backbone keys:", n, "max |diff| =", diff)
assert diff == 0.0

model = build(dict(type="MyResNet", depth=50), 2, 2048)
load_checkpoint(model, RESNET50, map_location="cpu", strict=False)
model.eval()
conv_ok = torch.equal(model.backbone.conv1.weight,
                      official.backbone.conv1.weight)
print("[ResNet-50 2cls] conv 权重一致:", conv_ok,
      "| head.fc:", tuple(model.head.fc.weight.shape))
assert conv_ok
with torch.no_grad():
    out = model(x)
assert tuple(out.shape) == (1, 2)
print("[ResNet-50 2cls] 输出:", tuple(out.shape))

official_se = get_model("seresnet50_8xb32_in1k", pretrained=True)
official_se.eval()
mine_se = MySEResNet(depth=50)
mine_se.eval()
n = load_backbone(mine_se, official_se)
with torch.no_grad():
    diff = (official_se.backbone(x)[0] - mine_se(x)[0]).abs().max().item()
print("[SE-ResNet-50 1000] backbone keys:", n, "max |diff| =", diff)
assert diff == 0.0

model_se = build(dict(type="MySEResNet", depth=50), 2, 2048)
load_checkpoint(model_se, SERESNET50, map_location="cpu", strict=False)
model_se.eval()
print("[SE-ResNet-50 2cls] head.fc:",
      tuple(model_se.head.fc.weight.shape))
with torch.no_grad():
    assert tuple(model_se(x).shape) == (1, 2)

model_se2 = build(dict(type="MySEResNet", depth=50), 2, 2048)
own, sd = model_se2.state_dict(), official.state_dict()
missing = [k for k in own if k not in sd]
bad = [k for k in own if k in sd and own[k].shape != sd[k].shape]
print("[ResNet-50 -> SE 2cls] missing:", len(missing),
      "| 全部是 se_layer:", all("se_layer" in k for k in missing),
      "| size mismatch:", bad)
assert len(missing) == 64
assert bad == ["head.fc.weight", "head.fc.bias"]

print("\n全部通过 ✅")