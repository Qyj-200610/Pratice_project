
data_root = "data/cats_dogs"
classes = ("cat", "dog")

model = dict(
    type="ImageClassifier",
    backbone=dict(type="MyResNet", depth=50, out_indices=(3,)),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=2,
        in_channels=2048,
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

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=1e-4),
)
param_scheduler = [
    dict(type="CosineAnnealingLR", T_max=10, eta_min=1e-6, by_epoch=True,
         begin=0, end=10),
]

train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=32, enable=True)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1,
                    save_best="accuracy/top1", max_keep_ckpts=2),
)

vis_backends = [dict(type="LocalVisBackend")]
try:
    __import__("tensorboard")
    vis_backends.append(dict(type="TensorboardVisBackend"))
except ImportError:
    pass
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

load_from = ("https://download.openmmlab.com/mmclassification/v0/resnet/"
             "resnet50_8xb32_in1k_20210831-ea4938fc.pth")

default_scope = "mmpretrain"
log_level = "INFO"
work_dir = "work_dirs/resnet50_cats_dogs"
