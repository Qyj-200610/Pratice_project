
_base_ = [
    "mmpretrain::_base_/models/vgg16.py",
    "mmpretrain::_base_/datasets/imagenet_bs32_pil_resize.py",
    "mmpretrain::_base_/schedules/imagenet_bs256.py",
    "mmpretrain::_base_/default_runtime.py",
]

custom_imports = dict(imports=["vggnet.backbones.vgg"], allow_failed_imports=False)

model = dict(
    backbone=dict(type="VGGNet", depth=16, num_classes=1000),
)

optim_wrapper = dict(optimizer=dict(type="SGD", lr=0.00125, momentum=0.9,
                                    weight_decay=1e-4))

data_root = "data/imagenet"
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
