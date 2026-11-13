

import torch
from mmpretrain.models.backbones import VGG as OfficialVGG

from vggnet.backbones.vgg import VGG as MyVGG

CASES = [
    (11, None), 
    (13, None), 
    (16, None), 
    (19, None),
    (11, {"type": "BN"}), 
    (13, {"type": "BN"}),
    (16, {"type": "BN"}), 
    (19, {"type": "BN"})
  ]

for depth, norm_cfg in CASES:
    kw = dict(depth=depth, num_classes=1000, norm_cfg=norm_cfg)
    a, b = OfficialVGG(**kw), MyVGG(**kw)
    a.eval()
    b.eval()          
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), f"depth={depth} 键不一致"
    assert all(sa[k].shape == sb[k].shape for k in sa), f"depth={depth} 形状不一致"
    b.load_state_dict(sa, strict=True)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        diff = (a(x)[0] - b(x)[0]).abs().max().item()
    print(f"depth={depth:2d} type={'bn ' if norm_cfg else 'plain'} "
          f"params={sum(p.numel() for p in b.parameters()) / 1e6:7.2f}M diff={diff}")
