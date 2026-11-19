
import argparse

import compat  
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_state_dict
from mmpretrain import get_model
from mmpretrain.registry import TRANSFORMS

import resnet  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("image")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model = get_model(Config(dict(model=cfg.model,
                                  data_preprocessor=cfg.data_preprocessor)))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_state_dict(model, ckpt["state_dict"], strict=False)
    model.eval()

    gates = {}
    hooks = []
    for name, module in model.backbone.named_modules():
        if name.endswith("se_layer.conv2"):
            def hook(m, i, o, name=name):
                gates[name] = o.detach().flatten(1)
            hooks.append(module.register_forward_hook(hook))

    pipeline = Compose([TRANSFORMS.build(t) for t in cfg.test_pipeline])
    sample = pipeline(dict(img_path=args.image))
    batch = dict(inputs=sample["inputs"].unsqueeze(0),
                 data_samples=[sample["data_samples"]])

    with torch.no_grad():
        pred = model.test_step(batch)[0]

    print("预测:", cfg.classes[pred.pred_label.item()],
          "| 概率: %.4f" % pred.pred_score.max().item())
    for stage in ("layer1", "layer2", "layer3", "layer4"):
        vals = torch.cat([v for k, v in gates.items()
                          if k.startswith(stage + ".")])
        print(f"{stage}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"min={vals.min():.4f} max={vals.max():.4f}")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
