
import torch

from mmpretrain.models.backbones import SEResNet as OfficialSEResNet
from resnet.backbones.senet import SEResNet as MySEResNet

depth = 50
x = torch.randn(1, 3, 224, 224)

a = OfficialSEResNet(depth=depth)
b = MySEResNet(depth=depth)
a.eval()
b.eval()

sa, sb = a.state_dict(), b.state_dict()
assert set(sa) == set(sb), f"键不一致: {set(sa) ^ set(sb)}"
assert all(sa[k].shape == sb[k].shape for k in sa), "形状不一致"

b.load_state_dict(sa, strict=True)
with torch.no_grad():
    diff = (a(x)[0] - b(x)[0]).abs().max().item()

params = sum(p.numel() for p in b.parameters()) / 1e6
print(f"depth={depth} keys={len(sa)} params={params:.2f}M diff={diff}")
