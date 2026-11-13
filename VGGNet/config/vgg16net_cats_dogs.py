# 真实猫狗训练配置（自包含，不继承 _base_，避免 1000 类设置漏覆盖）
# 数据：Pratice_project/data/cats_dogs（tools/prepare_cats_dogs.py 生成）
# 启动：在 Pratice_project/ 下执行
#   python VGGNet/tools/train.py VGGNet/config/vgg16net_cats_dogs.py
data_root = "data/cats_dogs"      # 相对当前工作目录
classes = ("cat", "dog")

model = dict(
    type="ImageClassifier",
    backbone=dict(type="VGGNet", depth=16, num_classes=2),
    neck=None,
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1,),
    ),
)

data_preprocessor = dict(
    num_classes=2,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomResizedCrop", scale=224, backend="pillow"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="PackInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="ResizeEdge", scale=256, edge="short", backend="pillow"),
    dict(type="CenterCrop", crop_size=224),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type="ImageNet",
        data_root=data_root,
        split="train",
        classes=classes,
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type="ImageNet",
        data_root=data_root,
        split="val",
        classes=classes,
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="Accuracy", topk=(1,))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# 本机实测：bs32/AMP 峰值约 3GB、~118 img/s
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="SGD", lr=1e-4, momentum=0.9, weight_decay=1e-4),
)
param_scheduler = [
    dict(type="CosineAnnealingLR", T_max=8, eta_min=1e-6, by_epoch=True,
         begin=0, end=8),
]

train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# 单卡 bs32 为基准；2 卡 DDP 时总 batch 64，lr 自动 ×2
auto_scale_lr = dict(base_batch_size=32, enable=True)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1,
                    save_best="accuracy/top1", max_keep_ckpts=2),
)

# TensorBoard：装了 tensorboard 才加 backend，没装时训练照常跑。
# 注意：config 里不能写 `import tensorboard` / `from x import y` 再调用——
# mmengine 会把它们变成 LazyObject（调用报 RuntimeError），
# 且模块名会作为全局变量存进 Config，Runner 深拷贝时报 "cannot pickle 'module' object"。
vis_backends = [dict(type="LocalVisBackend")]
try:
    __import__("tensorboard")
    vis_backends.append(dict(type="TensorboardVisBackend"))
except ImportError:
    pass
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)

# Windows 本机不跑 DDP；恒源云 2 卡用 nccl
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# 官方 1000 类权重：卷积 + 前两层 FC 复用，classifier.6 重建（会打 2 条 size mismatch 警告，正常）
load_from = ("https://download.openmmlab.com/mmclassification/v0/vgg/"
             "vgg16_batch256_imagenet_20210208-db26f1a5.pth")

default_scope = "mmpretrain"
log_level = "INFO"
work_dir = "work_dirs/vgg16net_cats_dogs"
