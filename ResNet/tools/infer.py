
import argparse

import compat  
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_state_dict
from mmpretrain import get_model
from mmpretrain.registry import TRANSFORMS

import resnet  


def build_model(cfg, checkpoint):
    model = get_model(Config(dict(model=cfg.model,
                                  data_preprocessor=cfg.data_preprocessor)))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_state_dict(model, ckpt["state_dict"], strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model = build_model(cfg, args.checkpoint)
    pipeline = Compose([TRANSFORMS.build(t) for t in cfg.test_pipeline])
    classes = cfg.classes

    for path in args.images:
        sample = pipeline(dict(img_path=path))
        batch = dict(inputs=sample["inputs"].unsqueeze(0),
                     data_samples=[sample["data_samples"]])
        with torch.no_grad():
            pred = model.test_step(batch)[0]

        probs = pred.pred_score
        k = min(args.topk, len(probs))
        top = torch.topk(probs, k)
        print(f"\n{path}")
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {classes[idx]:>4s}  {score:.4f}")


if __name__ == "__main__":
    main()
