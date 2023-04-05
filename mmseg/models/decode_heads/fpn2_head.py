# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer, build_conv_layer

from mmseg.ops import Upsample, resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import BaseModule, ModuleList, Sequential

# class RB(BaseModule):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.in_layers = Sequential(
#             nn.GroupNorm(32, in_channels),
#             nn.SiLU(),
#             Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         )

#         self.out_layers = Sequential(
#             nn.GroupNorm(32, out_channels),
#             nn.SiLU(),
#             Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         )

#         if out_channels == in_channels:
#             self.skip = nn.Identity()
#         else:
#             self.skip = Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         h = self.in_layers(x)
#         h = self.out_layers(h)
#         return h + self.skip(x)

class RB(BaseModule):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(RB, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

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
    
@HEADS.register_module()
class FPN2Head(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super(FPN2Head, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(Sequential(*scale_head))

        self.Predict = Sequential(
            RB(inplanes = 64 + 32, planes = 64), 
            RB(inplanes = 64, planes = 64),
            Conv2d(64, 2, kernel_size=1))    

        self.TBHead = Sequential(
            RB(inplanes = 128, planes = 64), 
            RB(inplanes = 64, planes = 64))    

        self.UpLast = Upsample(
                    scale_factor=4,
                    mode='bilinear',
                    align_corners=self.align_corners)

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
        output =self.UpLast(output)
        output = self.TBHead(output)
        output = torch.cat((output, inputs[-1]), dim=1)

        output = self.Predict(output)
        # output = self.cls_seg(output)
        return output
