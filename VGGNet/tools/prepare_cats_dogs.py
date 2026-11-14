
import argparse
import os
import random
import shutil
import tempfile
import zipfile

IMG_EXT = (".jpg", ".jpeg", ".png")
PARENT_NAMES = {"cat": "cat", "cats": "cat", "dog": "dog", "dogs": "dog"}


def collect_images(src):
    """递归收集图片，同时兼容「文件名前缀」和「父目录名」两种类别标注方式。"""
    by_name = {"cat": [], "dog": []}
    by_dir = {"cat": [], "dog": []}
    skipped_empty = 0
    for root, _, files in os.walk(src):
        parent = os.path.basename(root).lower()
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue
            path = os.path.join(root, f)
            if os.path.getsize(path) == 0:      
                skipped_empty += 1
                continue
            stem = f.split(".")[0].lower()
            if stem in by_name:
                by_name[stem].append(path)
            elif parent in PARENT_NAMES:
                by_dir[PARENT_NAMES[parent]].append(path)
    if skipped_empty:
        print(f"跳过 {skipped_empty} 张 0 字节的损坏图片")
    if by_name["cat"] or by_name["dog"]:
        return by_name
    return by_dir


def resolve_source(src):
    """返回 (可用的图片根目录, 分组结果, 临时目录或 None)。"""
    groups = collect_images(src)
    if groups["cat"] or groups["dog"]:
        return src, groups, None

    zip_path = os.path.join(src, "train.zip")
    if not os.path.isfile(zip_path):
        return src, groups, None

    tmp = tempfile.mkdtemp(prefix="kaggle_dvc_")
    print(f"没有现成图片，但找到 {zip_path}\n解压到临时目录 {tmp} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    return tmp, collect_images(tmp), tmp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="PetImages 目录（Microsoft）或含 train.zip 的目录（Kaggle）")
    parser.add_argument("--dst", default="data/cats_dogs")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--force", action="store_true", help="目标已存在时覆盖")
    args = parser.parse_args()

    dst_abs = os.path.abspath(args.dst)
    if os.path.exists(dst_abs) and os.listdir(dst_abs) and not args.force:
        raise SystemExit(f"{dst_abs} 已存在且非空；确认要重建就加 --force")

    root, groups, tmp = None, None, None
    try:
        root, groups, tmp = resolve_source(args.src)
        print(f"来源: {os.path.abspath(root)}")
        print(f"找到: cat={len(groups['cat'])}, dog={len(groups['dog'])}")
        if not groups["cat"] or not groups["dog"]:
            raise SystemExit(
                "没找到猫狗图片。请确认 --src 指向 PetImages 目录"
                "（里面应有 Cat/、Dog/ 两个子目录），或该目录下至少有 train.zip"
            )

        rng = random.Random(args.seed)
        for split in ("train", "val"):
            for cls in groups:
                os.makedirs(os.path.join(dst_abs, split, cls), exist_ok=True)

        for cls, items in groups.items():
            items = sorted(items)
            rng.shuffle(items)
            n_val = int(len(items) * args.val_ratio)
            parts = {"val": items[:n_val], "train": items[n_val:]}
            for split, names in parts.items():
                for path in names:
                    target = os.path.join(dst_abs, split, cls,
                                          os.path.basename(path))
                    if os.path.exists(target):
                        raise SystemExit(f"文件名冲突: {target}")
                    if args.mode == "hardlink":
                        try:
                            os.link(path, target)
                            continue
                        except OSError:
                            pass  
                    shutil.copy2(path, target)
            print(f"{cls}: total={len(items)} "
                  f"train={len(parts['train'])} val={len(parts['val'])}")

        print(f"完成 → {dst_abs}")
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
