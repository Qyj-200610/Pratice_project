
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import mmcv

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".ppm")


def check_one(path):
    """返回 (path, 错误原因或 None)。"""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            return path, "0 字节文件"
        img = mmcv.imfrombytes(data, channel_order="bgr")
        if img is None:
            return path, "cv2 解码失败"
        if img.ndim != 3 or img.shape[2] != 3:
            return path, f"通道数异常 {img.shape}"
        if img.shape[0] < 2 or img.shape[1] < 2:
            return path, f"尺寸异常 {img.shape}"
        return path, None
    except Exception as e:  
        return path, f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/cats_dogs", help="数据集根目录")
    parser.add_argument("--delete", action="store_true", help="真的删除坏图（默认只报告）")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = os.path.abspath(args.src)
    if not os.path.isdir(root):
        raise SystemExit(f"目录不存在：{root}")

    files = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(IMG_EXT):
                files.append(os.path.join(dirpath, n))
    files.sort()
    print(f"扫描 {root}，共 {len(files)} 张图，线程数 {args.workers} ...")

    bad = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(check_one, p) for p in files]
        for fut in as_completed(futs):
            path, err = fut.result()
            done += 1
            if err:
                bad.append((path, err))
                print(f"  [坏图] {os.path.relpath(path, root)}  ← {err}", flush=True)
            if done % 2000 == 0:
                print(f"  ... {done}/{len(files)}", flush=True)

    print(f"\n扫描完成：{len(files)} 张，坏图 {len(bad)} 张")
    if not bad:
        print("没有发现损坏图片 ✅")
        return

    if not args.delete:
        print("当前是「只检查」模式；确认后加 --delete 删除这些文件")
        return

    for path, _ in bad:
        os.remove(path)
    print(f"已删除 {len(bad)} 张坏图")

    for split in sorted(os.listdir(root)):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        parts = []
        for cls in sorted(os.listdir(d)):
            p = os.path.join(d, cls)
            if os.path.isdir(p):
                parts.append(f"{cls}={len(os.listdir(p))}")
        if parts:
            print(f"  {split}: " + ", ".join(parts))


if __name__ == "__main__":
    sys.exit(main())
