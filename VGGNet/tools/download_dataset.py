
import argparse
import os
import sys
import urllib.error
import urllib.request
import zipfile

URL = ("https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/"
       "kagglecatsanddogs_5340.zip")
EXPECTED_SIZE = 824887076         
CHUNK = 1024 * 1024


def human(n):
    return f"{n / 1024 ** 2:.1f} MB"


def download(url, dst):
    """支持断点续传：已下载的部分会被保留，重跑接着下。"""
    pos = os.path.getsize(dst) if os.path.exists(dst) else 0
    if pos == EXPECTED_SIZE:
        print(f"已下载完成，跳过：{dst}（{human(pos)}）")
        return dst
    if pos > EXPECTED_SIZE:
        print(f"文件比预期大，删掉重下：{dst}")
        os.remove(dst)
        pos = 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if pos:
        headers["Range"] = f"bytes={pos}-"
        print(f"发现已下载 {human(pos)}，尝试续传 ...")
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if pos else "wb"
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(dst, mode) as f:
            total = pos + int(r.headers.get("Content-Length", 0) or 0)
            done = pos
            last_pct = -1
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct // 5 != last_pct // 5:
                        last_pct = pct
                        print(f"  {pct:3d}%  {human(done)} / {human(total)}", flush=True)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"下载失败：HTTP {e.code}（{url}）") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise SystemExit(
            f"下载中断：{e}\n已下载的部分保留在 {dst}，重跑本脚本会断点续传。"
        ) from e

    size = os.path.getsize(dst)
    if size != EXPECTED_SIZE:
        raise SystemExit(
            f"文件大小不对：{size} != {EXPECTED_SIZE}。删掉 {dst} 后重跑。"
        )
    print(f"下载完成：{dst}（{human(size)}）")
    return dst


def extract(zip_path, out_dir):
    """解压出 PetImages/，已存在则跳过（加 --force 可重解压）。"""
    pet = os.path.join(out_dir, "PetImages")
    if os.path.isdir(pet) and os.listdir(pet):
        print(f"已解压，跳过：{pet}")
        return pet
    os.makedirs(out_dir, exist_ok=True)
    print(f"解压到 {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"解压完成：{pet}")
    return pet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", default="data/raw", help="下载/解压目录（默认 data/raw）")
    parser.add_argument("--no-extract", action="store_true", help="只下载，不解压")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args()

    dst = os.path.abspath(args.dst)
    os.makedirs(dst, exist_ok=True)
    zip_path = os.path.join(dst, os.path.basename(URL))
    if args.force and os.path.exists(zip_path):
        os.remove(zip_path)

    download(URL, zip_path)
    if not args.no_extract:
        pet = extract(zip_path, dst)
        for cls in ("Cat", "Dog"):
            d = os.path.join(pet, cls)
            n = len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"  {cls}: {n} 张")
        print(f"\n下一步：python VGGNet/tools/prepare_cats_dogs.py --src {pet}")


if __name__ == "__main__":
    sys.exit(main())
