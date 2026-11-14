auto_scale_lr = dict(base_batch_size=32, enable=True)
classes = (
    'cat',
    'dog',
)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=2,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
data_root = 'data/cats_dogs'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        save_best='accuracy/top1',
        type='CheckpointHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(depth=16, num_classes=2, type='VGGNet'),
    head=dict(
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        topk=(1, ),
        type='ClsHead'),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        T_max=8,
        begin=0,
        by_epoch=True,
        end=8,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        classes=(
            'cat',
            'dog',
        ),
        data_root='data/cats_dogs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(topk=(1, ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        classes=(
            'cat',
            'dog',
        ),
        data_root='data/cats_dogs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='ImageNet'),
    num_workers=4,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        classes=(
            'cat',
            'dog',
        ),
        data_root='data/cats_dogs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(topk=(1, ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'work_dirs/vgg16net_cats_dogs'
