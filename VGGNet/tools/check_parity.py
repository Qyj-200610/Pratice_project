
import torch
from mmpretrain.models.backbones import VGG as OfficialVGG

from vggnet.backbones.vgg import VGG as MyVGG

a = OfficialVGG(16, num_classes=1000)
a.eval()

b = MyVGG(16, num_classes=1000)
b.eval()

sa, sb = a.state_dict(), b.state_dict()
assert set(sa) == set(sb), f"键不一致: 多={set(sb)-set(sa)} 少={set(sa)-set(sb)}"
assert all(sa[k].shape == sb[k].shape for k in sa), "有层形状不对"

b.load_state_dict(sa, strict=True)         
x = torch.randn(1, 3, 224, 224)             
with torch.no_grad():
    oa = a(x)[0]
    ob = b(x)[0]
print("max |diff| =", (oa - ob).abs().max().item())
print("params(M)  = %.2f" % (sum(p.numel() for p in b.parameters()) / 1e6))

print(f"out_indices={b.out_indices}")
