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
model.eval()

with torch.no_grad():
    out = model(torch.randn(2, 3, 224, 224))

print("logits     :", tuple(out.shape))
print("params(M)  : %.2f" % (sum(p.numel() for p in model.parameters()) / 1e6))
