

import torch
import torch.nn as nn

from models.base.base_blocks import Base3DStem
from models.base.base_blocks import STEM_REGISTRY

@STEM_REGISTRY.register()
class X3DStem(Base3DStem):
    def __init__(self, cfg):
        super(X3DStem, self).__init__(cfg)

    def _construct_block(
        self, 
        cfg,
        dim_in, 
        num_filters, 
        kernel_sz,
        stride,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        self.a = nn.Conv3d(
            dim_in,
            num_filters,
            kernel_size = [1, kernel_sz[1], kernel_sz[2]],
            stride      = [1, stride[1], stride[2]],
            padding     = [0, kernel_sz[1]//2, kernel_sz[2]//2],
            bias        = False
        )
        self.a2 = nn.Conv3d(
            num_filters,
            num_filters,
            kernel_size = [kernel_sz[0], 1, 1],
            stride      = [stride[0], 1, 1], 
            padding     = [kernel_sz[0]//2, 0, 0],
            bias        = False,
            groups      = num_filters,
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x)
        x = self.a2(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x

