
import argparse
import glob
import os
import re
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))     
sys.path.insert(0, TOOLS_DIR)

from infer import main as infer_main  

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
WORK_DIR = os.path.join(PROJECT_ROOT, "work_dirs", "resnet50_cats_dogs")
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "ResNet", "config",
                              "resnet50_cats_dogs.py")


def latest_checkpoint(work_dir):
    """优先取 epoch 最大的 best_*.pth；没有就退到 epoch_*.pth。"""
    for pattern in ("best_accuracy_top1_epoch_*.pth", "epoch_*.pth"):
        paths = glob.glob(os.path.join(work_dir, pattern))
        if paths:
            return max(paths, key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default=os.path.join(PROJECT_ROOT, "validation_photos"))
    parser.add_argument("--checkpoint", default=None,
                        help="默认自动取 epoch 最大的 best_accuracy_top1_epoch_*.pth")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", default=WORK_DIR)
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()

    img_dir = os.path.abspath(args.img_dir)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"图片目录不存在：{img_dir}")

    images = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(IMG_EXT)
    )
    if not images:
        raise SystemExit(f"{img_dir} 里没有图片（支持 {', '.join(IMG_EXT)}）")

    work_dir = os.path.abspath(args.work_dir)
    ckpt = args.checkpoint or latest_checkpoint(work_dir)
    if not ckpt:
        raise SystemExit(f"{work_dir} 下没有 checkpoint，先训练。")
    ckpt = os.path.abspath(ckpt)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"checkpoint 不存在：{ckpt}")

    print(f"config    : {args.config}")
    print(f"checkpoint: {ckpt}")
    print(f"图片      : {len(images)} 张（{img_dir}）")

    sys.argv = ["infer.py", os.path.abspath(args.config), ckpt, *images,
                "--topk", str(args.topk)]
    infer_main()


if __name__ == "__main__":
    main()
