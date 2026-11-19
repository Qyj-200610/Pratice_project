
import mmengine
import mmpretrain
import torch
from dotenv import load_dotenv
from mmengine import Config
from mmpretrain import get_model

load_dotenv()

print("torch      :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("mmpretrain :", mmpretrain.__version__, "| mmengine:", mmengine.__version__)

cfg = Config(
    dict(
        model=dict(
            type="ImageClassifier",
            backbone=dict(type="ResNet", depth=50, out_indices=(3,)),
            neck=dict(type="GlobalAveragePooling"),
            head=dict(
                type="LinearClsHead",
                num_classes=1000,
                in_channels=2048,
                loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
                topk=(1, 5),
            ),
        )
    )
)
model = get_model(cfg)
model.eval()
with torch.no_grad():
    out = model(torch.randn(2, 3, 224, 224))
    feat = model.backbone(torch.randn(2, 3, 224, 224))[0]

print("logits     :", tuple(out.shape))
print("params(M)  : %.2f" % (sum(p.numel() for p in model.parameters()) / 1e6))
print("feat       :", tuple(feat.shape))
print("bb keys    :", len(model.backbone.state_dict()))
