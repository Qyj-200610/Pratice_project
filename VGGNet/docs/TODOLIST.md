# VGGNet 复现 TODO List（总览）

> 文档已拆成两份，**按顺序做**：先代码、后训练。
>
> | 文件 | 内容 | 目标 |
> | :--- | :--- | :--- |
> | **[01_代码部分.md](01_代码部分.md)** | Phase 1–5：环境冒烟 → 读懂结构 → 项目骨架 → 手写 VGG16 + 8 变体对齐 → 加载官方权重 | 1、2 |
> | **[02_训练部分.md](02_训练部分.md)** | Phase 6–11：猫狗数据 → 迁移学习（>95%）→ TensorBoard → 实拍推理 → 恒源云 DDP → 交付 | 3、4 + DDP |
>
> **四个目标（外加 DDP 支持）**：
>
> 1. **代码实现**：PyTorch 手写 VGG16，接进 MMPretrain，`state_dict` 与官方逐键一致、`max|diff| = 0.0`。
> 2. **加载预训练权重**：官方 ImageNet-1k（1000 类）权重灌进你的实现，2 类时正确替换 `classifier.6`。
> 3. **迁移学习**：真实猫狗 24,997 张（微软版），验证集 **top-1 > 95%**。
> 4. **可视化 + 推理**：TensorBoard 记录 loss / acc / lr；手机实拍猫狗照片推理验证。
>
> **外加 DDP 多卡**：代码按多卡规范写，训练/评测在**恒源云 2 卡**上验收。
>
> **环境**：`mmcv-lite`；本机 Windows + RTX 5060 Laptop 8GB 只做单卡，多卡上恒源云。
> **版本基线**：python 3.12.10 / torch 2.11.0+cu128 / mmpretrain 1.2.0 / mmengine 0.10.7 / mmcv-lite 2.2.0。

---

## 一、进度追踪

| Phase | 内容 | 所在文件 | 验收 |
| :-: | :--- | :--- | :--- |
| 1 | 环境冒烟 | [01](01_代码部分.md) | `params(M) 138.36` / `(2,1000)` |
| 2 | 读懂结构 | [01](01_代码部分.md) | 能默写 VGG16 stage 表 |
| 3 | 项目骨架 | [01](01_代码部分.md) | `pip install -e .` 成功 |
| 4 | 手写 backbone + 8 变体对齐 | [01](01_代码部分.md) | 8 行 `diff = 0.0`，参数量全对 |
| 5 | 加载官方权重 | [01](01_代码部分.md) | `max \|diff\| = 0.0`；2 类替换正确 |
| 6 | 猫狗数据 | [02](02_训练部分.md) | 22.5k / 2.5k 张，`cat`/`dog` 目录 |
| 7 | 迁移学习 | [02](02_训练部分.md) | val top-1 **> 95%** |
| 8 | TensorBoard | [02](02_训练部分.md) | loss / acc / lr 曲线 |
| 9 | 实拍推理 | [02](02_训练部分.md) | 3 张以上预测正确 |
| 10 | DDP（恒源云 2 卡） | [02](02_训练部分.md) | 2 卡 8 epoch + 评测 + 结果带回 |
| 11 | README / 交付 | [02](02_训练部分.md) | 证据齐全 |

---

## 二、三条关键结论（先记住，能省掉几小时）

1. **本机只做单卡**：无 NCCL、gloo+CUDA 崩溃、`torchrun` 报 libuv，多卡一律上恒源云。
2. **torch ≥2.6 读 mmengine 的 ckpt 会报 `UnpicklingError`**：必须加 `tools/compat.py` 补丁
   （01 的 5.5），否则续训 / `load_from` / `Runner.test()` 全部失败。
3. **`get_model` 只认 `Config(dict(model=...))`**：传普通 dict 会抛 `UnboundLocalError`（mmpretrain 1.2.0 的 bug）。

---

## 附录 A 自测题（做完再写 README）

1. 两个 3×3 卷积的感受野 = 一个几×几？参数量比是多少？
2. 为什么 VGG 的 `features` 用扁平 `Sequential`？`range_sub_modules` 记错会在哪一步暴露？
3. 为什么 FC 放在 backbone 里？拆到 head 里还能加载官方权重吗？
4. `num_classes=-1` 与 `out_indices=(4,)` 同时设置时返回什么？
5. 为什么 `init_cfg` 里要有 `Normal(std=0.01, layer=["Linear"])`？
6. `state_dict` 对齐时 `max|diff| = 0` 证明了什么？**没有**证明什么？
7. 猫狗 2 分类时，为什么 `classifier.6` 的 `size mismatch` 是预期行为？
8. 迁移学习为什么必须用 ImageNet 的 mean/std？用错会怎样？
9. `load_from` 和 `init_cfg=Pretrained` 在 Runner 里的执行顺序？谁覆盖谁？
10. 为什么本机 8GB 能跑 bs32，却跑不了 GPU DDP？
11. `torch.load(weights_only=True)` 这个默认值变化影响了哪三个流程？
12. 2 卡 DDP 时 lr 该怎么变？`auto_scale_lr` 的 `base_batch_size` 该填多少？

<details>
<summary>第 3、7、9、10、11 题的参考思路（先自己想）</summary>

- **3**：官方权重的键是 `backbone.classifier.0/3/6.*`，FC 在 backbone 内才能对上键；
  拆到 head 需要改键名映射，且 head 的 `Linear` 形状和 checkpoint 不一致，加载不了。
- **7**：1000 类 → 2 类，最后一层必须重建；`load_from` 的 `strict=False` 会跳过尺寸不匹配的键，
  前 30 个键全部复用，这正是迁移学习的标准做法。
- **9**：`Runner.train()` 先 `_init_model_weights()`（执行 `init_cfg`），再 `load_or_resume()`（执行 `load_from`），
  所以 `load_from` 覆盖 `init_cfg`。
- **10**：显存够但**后端不行**：Windows 无 NCCL，gloo 不支持 CUDA all_reduce（实测崩溃），torchrun 还有 libuv 问题。
- **11**：`--resume`、`load_from` 指向 mmengine ckpt、`Runner.test()`；都会走 `load_checkpoint → torch.load`。

</details>


---

## 附录 B 本次修订记录（对照旧版 TODOLIST）

| # | 旧版内容 | 问题 | 本次处理 |
| :-: | :--- | :--- | :--- |
| 1 | 定位「只做结构复现，不做指标复现」 | 与最新目标（迁移学习 + TB + 推理 + DDP）不符 | 全文按 4 条目标重写 |
| 2 | 6.5「Windows 下也能 `torchrun --nproc_per_node=2`」 | 本机实测不可行：无 NCCL、gloo+CUDA 崩溃、torchrun 报 libuv | 改为「本机只做单卡，多卡在恒源云 2 卡跑」，见 Phase 10 |
| 3 | 6.2/6.3 用 `load_from` 读自己的 ckpt / resume | torch≥2.6 默认 `weights_only=True`，mmengine ckpt 读不了（实测） | 新增 5.5 补丁与 `tools/compat.py` |
| 4 | 6.2 把 `init_cfg=Pretrained` 当通用方式 | 只在 `Runner.train()` 生效，`get_model()`/`test()` 不调 `init_weights`（实测） | 5.1 三种方式对比表 |
| 5 | `train.py` 里 `merge_from_dict({key: eval(value)})` | 点号键不会嵌套，`eval` 不安全 | 改用 `mmengine.config.DictAction` |
| 6 | 5.3/6.2 用 `classes=["cat","dog"]` | 写法本身可用（已验证），但与 `num_classes` 的关系没讲清 | 7.2 三处一致性表 |
| 7 | 缺少 TensorBoard 配置 | 目标 4 无法满足，且本机没装 tensorboard | 新增 Phase 8 + `vis_backends` |
| 8 | 缺少猫狗数据/迁移学习内容 | 目标 3 无法落地 | 新增 Phase 6、7 + 数据划分脚本 |
| 9 | 1.4 期望输出 4 行 | 与仓库里 `check_env.py` 实际 6 行不一致 | 期望输出改为实际值 |
| 10 | 01 文件附录 C 参考实现 `make_vgg_layer` 无 `with_norm` | 官方 1.2.0 签名里有该参数（函数体未用） | 补上参数，并说明历史原因 |
| 11 | 4.5 参数量表 | 经本机官方实现复算，**数值全部正确**，保留 | 原样保留，补「实测 9.5 秒」 |
| 12 | 4.4「官方 `train()` 没有 `return self`」 | 经源码核对，**属实**，保留 | 保留并在参考实现里补上 |
