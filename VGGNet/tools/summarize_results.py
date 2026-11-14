"""汇总一次（或多次）训练的结果，生成可存档的 summary.md / summary.json。

数据来源（都是 mmengine 自动写的，不需要改训练脚本）：
- `vis_data/scalars.json`：train 的 loss / lr / 耗时
- `<run>.log`：每个 epoch 的实际迭代数、验证集 `accuracy/top1`

用法（在 Pratice_project/ 下）：
    python VGGNet/tools/summarize_results.py
    python VGGNet/tools/summarize_results.py --work-dir work_dirs/vgg16net_cats_dogs
    python VGGNet/tools/summarize_results.py --out results/summary.md
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

TRAIN_RE = re.compile(r"Epoch\(train\) \[\s*(\d+)\]\[\s*\d+/(\d+)\]")
VAL_RE = re.compile(r"Epoch\(val\) \[\s*(\d+)\]\[\s*\d+/\d+\].*accuracy/top1:\s*([\d.]+)")


def find_scalars(work_dir):
    return sorted(glob.glob(os.path.join(work_dir, "**", "vis_data", "scalars.json"),
                            recursive=True))


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_log(run_dir):
    """从 mmengine 的 .log 里取每个 epoch 的总迭代数和验证准确率。"""
    logs = glob.glob(os.path.join(run_dir, "*.log"))
    iters, val = {}, {}
    if not logs:
        return iters, val
    with open(logs[0], encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                iters[int(m.group(1))] = int(m.group(2))
            m = VAL_RE.search(line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
    return iters, val


def summarize(rows, iters, val):
    """返回 per-epoch 列表。"""
    train = {}
    for r in rows:
        if "loss" in r and "epoch" in r:
            train.setdefault(int(r["epoch"]), []).append(r)

    table = []
    for epoch in sorted(set(train) | set(val)):
        items = train.get(epoch, [])
        losses = [float(i["loss"]) for i in items if "loss" in i]
        times = [float(i["time"]) for i in items if "time" in i]
        lrs = [float(i["lr"]) for i in items if "lr" in i]
        table.append({
            "epoch": epoch,
            "iters": iters.get(epoch, len(items)),
            "loss_avg": round(statistics.fmean(losses), 4) if losses else None,
            "sec_per_iter": round(statistics.fmean(times), 4) if times else None,
            "lr": lrs[-1] if lrs else None,
            "val_top1": round(val[epoch], 2) if epoch in val else None,
        })
    return table


def render(title, table):
    def fmt(v, nd=4):
        return "—" if v is None else f"{v:.{nd}f}"

    lines = [f"### {title}", "",
             "| epoch | iters | loss(avg) | s/iter | lr | val top-1 |",
             "| :-: | :-: | :-: | :-: | :-: | :-: |"]
    for r in table:
        lines.append(
            f"| {r['epoch']} | {r['iters']} | {fmt(r['loss_avg'])} | "
            f"{fmt(r['sec_per_iter'], 3)} | {fmt(r['lr'], 6)} | {fmt(r['val_top1'], 2)} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="work_dirs/vgg16net_cats_dogs")
    parser.add_argument("--out", default=None, help="summary.md 输出路径（默认放到 --work-dir 下）")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if not os.path.isdir(work_dir):
        raise SystemExit(f"目录不存在：{work_dir}")

    scalars = find_scalars(work_dir)
    if not scalars:
        raise SystemExit(f"{work_dir} 下没找到 vis_data/scalars.json，训练还没跑过？")

    results = {}
    blocks = ["# 训练结果汇总", "", f"来源：`{work_dir}`", ""]
    best = None
    for path in scalars:
        run_dir = os.path.dirname(os.path.dirname(path))
        rel = os.path.relpath(run_dir, work_dir)
        iters, val = parse_log(run_dir)
        # 兼容：没有 log 时，验证指标在 scalars 里以 step 表示 epoch
        if not val:
            for r in load_rows(path):
                if "accuracy/top1" in r and "step" in r:
                    val[int(r["step"])] = float(r["accuracy/top1"])
        table = summarize(load_rows(path), iters, val)
        results[rel] = table
        blocks += [render(rel, table), ""]
        print(render(rel, table) + "\n")
        for r in table:
            if r["val_top1"] is not None and (best is None or r["val_top1"] > best["val_top1"]):
                best = {"run": rel, **r}

    if best:
        line = f"最佳验证准确率：{best['val_top1']:.2f}%（{best['run']}，epoch {best['epoch']}）"
        print(line)
        blocks += ["---", "", f"**{line}**", ""]
    else:
        print("还没有验证记录（训练可能没跑完，或 log 被删了）")
        blocks += ["---", "", "还没有验证记录。", ""]

    out = args.out or os.path.join(work_dir, "summary.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"work_dir": work_dir, "runs": results, "best": best},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入：{out}")
    print(f"已写入：{os.path.splitext(out)[0] + '.json'}")


if __name__ == "__main__":
    sys.exit(main())
