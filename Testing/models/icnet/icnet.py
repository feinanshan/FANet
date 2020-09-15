import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .utils import (
    get_interp_size,
    cascadeFeatureFusion,
    conv2DBatchNormRelu,
    residualBlockPSP,
    pyramidPooling,
)

icnet_specs = {
    "cityscapes": {"n_classes": 19, "input_size": (1025, 2049), "block_config": [3, 4, 6, 3]}
}


class ICNet(nn.Module):

    """
    Image Cascade Network
    URL: https://arxiv.org/abs/1704.08545
    References:
    1) Original Author's code: https://github.com/hszhao/ICNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/ICNet-tensorflow
    """

    def __init__(
        self,
        n_classes=19,
        block_config=[3, 4, 6, 3],
        input_size=(1025, 2049),
        version=None,
        is_batchnorm=False,
    ):

        super(ICNet, self).__init__()

        bias = not is_batchnorm

        self.block_config = (
            icnet_specs[version]["block_config"] if version is not None else block_config
        )
        self.n_classes = icnet_specs[version]["n_classes"] if version is not None else n_classes
        self.input_size = icnet_specs[version]["input_size"] if version is not None else input_size

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=64,
            padding=1,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(
            self.block_config[0], 64, 32, 128, 1, 1, is_batchnorm=is_batchnorm
        )
        self.res_block3_conv = residualBlockPSP(
            self.block_config[1],
            128,
            64,
            256,
            2,
            1,
            include_range="conv",
            is_batchnorm=is_batchnorm,
        )
        self.res_block3_identity = residualBlockPSP(
            self.block_config[1],
            128,
            64,
            256,
            2,
            1,
            include_range="identity",
            is_batchnorm=is_batchnorm,
        )

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(
            self.block_config[2], 256, 128, 512, 1, 2, is_batchnorm=is_batchnorm
        )
        self.res_block5 = residualBlockPSP(
            self.block_config[3], 512, 256, 1024, 1, 4, is_batchnorm=is_batchnorm
        )

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(
            1024, [6, 3, 2, 1], model_name="icnet", fusion_mode="sum", is_batchnorm=is_batchnorm
        )

        # Final conv layer with kernel 1 in sub4 branch
        self.conv5_4_k1 = conv2DBatchNormRelu(
            in_channels=1024,
            k_size=1,
            n_filters=256,
            padding=0,
            stride=1,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )

        # High-resolution (sub1) branch
        self.convbnrelu1_sub1 = conv2DBatchNormRelu(
            in_channels=3,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu2_sub1 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=32,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.convbnrelu3_sub1 = conv2DBatchNormRelu(
            in_channels=32,
            k_size=3,
            n_filters=64,
            padding=1,
            stride=2,
            bias=bias,
            is_batchnorm=is_batchnorm,
        )
        self.classification = nn.Conv2d(128, self.n_classes, 1, 1, 0)

        # Cascade Feature Fusion Units
        self.cff_sub24 = cascadeFeatureFusion(
            self.n_classes, 256, 256, 128, is_batchnorm=is_batchnorm
        )
        self.cff_sub12 = cascadeFeatureFusion(
            self.n_classes, 128, 64, 128, is_batchnorm=is_batchnorm
        )


    def forward(self, x):
        h, w = x.shape[2:]

        # H, W -> H/2, W/2
        x_sub2 = F.interpolate(
            x, size=get_interp_size(x, s_factor=2), mode="bilinear", align_corners=True
        )

        # H/2, W/2 -> H/4, W/4
        x_sub2 = self.convbnrelu1_1(x_sub2)
        x_sub2 = self.convbnrelu1_2(x_sub2)
        x_sub2 = self.convbnrelu1_3(x_sub2)

        # H/4, W/4 -> H/8, W/8
        x_sub2 = F.max_pool2d(x_sub2, 3, 2, 1)

        # H/8, W/8 -> H/16, W/16
        x_sub2 = self.res_block2(x_sub2)
        x_sub2 = self.res_block3_conv(x_sub2)
        # H/16, W/16 -> H/32, W/32
        x_sub4 = F.interpolate(
            x_sub2, size=get_interp_size(x_sub2, s_factor=2), mode="bilinear", align_corners=True
        )
        x_sub4 = self.res_block3_identity(x_sub4)

        x_sub4 = self.res_block4(x_sub4)
        x_sub4 = self.res_block5(x_sub4)

        x_sub4 = self.pyramid_pooling(x_sub4)
        x_sub4 = self.conv5_4_k1(x_sub4)

        x_sub1 = self.convbnrelu1_sub1(x)
        x_sub1 = self.convbnrelu2_sub1(x_sub1)
        x_sub1 = self.convbnrelu3_sub1(x_sub1)

        x_sub24, sub4_cls = self.cff_sub24(x_sub4, x_sub2)
        x_sub12, sub24_cls = self.cff_sub12(x_sub24, x_sub1)

        x_sub12 = F.interpolate(
            x_sub12, size=get_interp_size(x_sub12, z_factor=2), mode="bilinear", align_corners=True
        )
        x_sub4 = self.res_block3_identity(x_sub4)
        sub124_cls = self.classification(x_sub12)


        sub124_cls = F.interpolate(
                sub124_cls,
                size=get_interp_size(sub124_cls, z_factor=4),
                mode="bilinear",
                align_corners=True,
            )
        return sub124_cls
