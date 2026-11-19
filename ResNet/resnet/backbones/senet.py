import torch.utils.checkpoint as cp
import torch

from mmpretrain.registry import MODELS
from .resnet import Bottleneck, ResLayer, ResNet
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from mmpretrain.models.utils import make_divisible


class SELayer(BaseModule):
    def __init__(self, channels, ratio=16, divisor=8, conv_cfg=None,
                 act_cfg=(dict(type="ReLU"), dict(type="Sigmoid")),
                 init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)
        squeeze_channels = make_divisible(channels // ratio, divisor)
        self.conv1 = ConvModule(
            channels, squeeze_channels, 1, bias="auto",
            conv_cfg=conv_cfg, act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            squeeze_channels, channels, 1, bias="auto",
            conv_cfg=conv_cfg, act_cfg=act_cfg[1])

    def forward(self, x):
        return x * self.conv2(self.conv1(self.global_avgpool(x)))


class SEBottleneck(Bottleneck):

    def __init__(self, in_channels, out_channels, se_ratio=16, **kwargs):
        super(SEBottleneck, self).__init__(in_channels, out_channels, **kwargs)
        self.se_layer = SELayer(out_channels, ratio=se_ratio)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.se_layer(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


@MODELS.register_module("MySEResNet")
class SEResNet(ResNet):

    arch_settings = {
        50: (SEBottleneck, (3, 4, 6, 3)),
        101: (SEBottleneck, (3, 4, 23, 3)),
        152: (SEBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, se_ratio=16, **kwargs):
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for SEResNet')
        self.se_ratio = se_ratio
        super(SEResNet, self).__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(se_ratio=self.se_ratio, **kwargs)