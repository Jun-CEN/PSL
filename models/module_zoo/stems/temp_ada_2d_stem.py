#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Temporal Adaptive 2D Stem. """

import torch
import torch.nn as nn

from models.base.base_blocks import Base2DStem
from models.base.base_blocks import STEM_REGISTRY
from models.module_zoo.branches.temporal_adaptive_spatialconv import (
    route_func_mlp,
    route_func_mlp_with_global_info,
    TemporalAdaptiveSpatialConv, 
    TemporalAdaptiveSpatialConvCinAdaptive
)

class route_func_mlp_stem(nn.Module):

    def __init__(self, c_in, c_out, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_stem, self).__init__()
        self.c_in = c_in
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.avgpool_2 = nn.AdaptiveAvgPool3d((None,2,2))
        self.avgpool_4 = nn.AdaptiveAvgPool3d((None,4,4))
        self.a = nn.Conv3d(
            in_channels=c_in * (1+4+16),
            out_channels=round((c_in * (1+4+16))/ratio),
            kernel_size=[3,1,1],
            padding=[1,0,0],
        )
        self.bn = nn.BatchNorm3d(round((c_in * (1+4+16))/ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=round((c_in * (1+4+16))/ratio),
            out_channels=c_out,
            kernel_size=[3,1,1],
            padding=[1,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()

    def forward(self, x):
        x = torch.cat(
            (self.avgpool(x),
            self.avgpool_2(x).permute(0,1,3,4,2).reshape(
                x.shape[0], x.shape[1]*4, x.shape[2], 1, 1), 
            self.avgpool_4(x).permute(0,1,3,4,2).reshape(
                x.shape[0], x.shape[1]*16, x.shape[2], 1, 1)
            ), dim=1 )
        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

@STEM_REGISTRY.register()
class TemporalAdaptive2DStem(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStem, self).__init__(cfg)

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
        self.a = TemporalAdaptiveSpatialConv(
            dim_in,
            num_filters,
            kernel_size = [1, kernel_sz[1], kernel_sz[2]],
            stride      = [1, stride[1], stride[2]],
            padding     = [0, kernel_sz[1]//2, kernel_sz[2]//2],
            bias        = False
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.rf = nn.Parameter(
            torch.ones(1, num_filters, cfg.DATA.NUM_INPUT_FRAMES,1,1)
        )
    
    def forward(self, x):
        b = x.shape[0]
        x = self.a(x, self.rf.repeat(b,1,1,1,1))
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x


@STEM_REGISTRY.register()
class TemporalAdaptive2DStemV2(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemV2, self).__init__(cfg)

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
        self.a = TemporalAdaptiveSpatialConv(
            dim_in,
            num_filters,
            kernel_size = [1, kernel_sz[1], kernel_sz[2]],
            stride      = [1, stride[1], stride[2]],
            padding     = [0, kernel_sz[1]//2, kernel_sz[2]//2],
            bias        = False
        )
        self.a_rf = route_func_mlp_stem(
            c_in=dim_in,
            c_out=num_filters,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x, self.a_rf(x))
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x

@STEM_REGISTRY.register()
class TemporalAdaptive2DStemV3(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemV3, self).__init__(cfg)

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
        self.a = TemporalAdaptiveSpatialConvCinAdaptive(
            dim_in,
            num_filters,
            kernel_size = [1, kernel_sz[1], kernel_sz[2]],
            stride      = [1, stride[1], stride[2]],
            padding     = [0, kernel_sz[1]//2, kernel_sz[2]//2],
            bias        = False
        )
        self.a_rf = route_func_mlp(
            c_in=dim_in,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=0.5,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x, self.a_rf(x))
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x


@STEM_REGISTRY.register()
class TemporalAdaptive2DStemV4(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemV4, self).__init__(cfg)

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
        self.a_rf = route_func_mlp(
            c_in=num_filters,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x)
        x = self.a_rf(x) * x
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x

@STEM_REGISTRY.register()
class TemporalAdaptive2DStemV5(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemV5, self).__init__(cfg)

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
        self.a_rf = route_func_mlp(
            c_in=num_filters,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.avg_pool = nn.AvgPool3d(
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x)
        x = self.a_rf(x) * x
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.avg_pool(x) + x
        return x

@STEM_REGISTRY.register()
class TemporalAdaptive2DStemV6(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemV6, self).__init__(cfg)

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
        self.a_rf = route_func_mlp_with_global_info(
            c_in=num_filters,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x)
        x = self.a_rf(x) * x
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x

@STEM_REGISTRY.register()
class TemporalAdaptive2DStemV7(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemV7, self).__init__(cfg)

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
        self.a_rf = route_func_mlp(
            c_in=num_filters,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.avg_pool = nn.AvgPool3d(
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x)
        x = self.a_rf(x) * x
        x = self.avg_pool(x) + x
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x

@STEM_REGISTRY.register()
class TemporalAdaptive2DStemV8(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemV8, self).__init__(cfg)

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
        self.a_rf = route_func_mlp(
            c_in=num_filters,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.avg_pool = nn.AvgPool3d(
            kernel_size=[3,7,7],
            stride=1,
            padding=[1,3,3],
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x)
        x = self.a_rf(x) * x
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.avg_pool(x) + x
        return x
    

@STEM_REGISTRY.register()
class TemporalAdaptive2DStemDownSample(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemDownSample, self).__init__(cfg)

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
        self.a_rf = route_func_mlp(
            c_in=num_filters,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.maxpool = nn.MaxPool3d(
            kernel_size = (1, 3, 3),
            stride      = (1, 2, 2),
            padding     = (0, 1, 1)
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x)
        x = self.a_rf(x) * x
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.maxpool(x)
        return x

@STEM_REGISTRY.register()
class TemporalAdaptive2DStemMixPool(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemMixPool, self).__init__(cfg)

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
        self.a = TemporalAdaptiveSpatialConv(
            dim_in,
            num_filters,
            kernel_size = [1, kernel_sz[1], kernel_sz[2]],
            stride      = [1, stride[1], stride[2]],
            padding     = [0, kernel_sz[1]//2, kernel_sz[2]//2],
            bias        = False
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.rf = nn.Parameter(
            torch.ones(1, num_filters, cfg.DATA.NUM_INPUT_FRAMES,1,1)
        )

        self.maxpool = nn.MaxPool3d(
            kernel_size=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]
            ],
            stride=1,
            padding=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]//2
            ],
        )
        self.avgpool = nn.AvgPool3d(
            kernel_size=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]
            ],
            stride=1,
            padding=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]//2
            ],
        )
        if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
            self.maxpool_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
            self.maxpool_bn.no_init=True
            self.maxpool_bn.weight.data.zero_()
            self.maxpool_bn.bias.data.zero_()
            self.avgpool_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
            self.avgpool_bn.no_init=True
            self.avgpool_bn.weight.data.zero_()
            self.avgpool_bn.bias.data.zero_()
    
    def forward(self, x):
        b = x.shape[0]
        x = self.a(x, self.rf.repeat(b,1,1,1,1))
        x = self.a_bn(x) + self.avgpool_bn(self.avgpool(x)) + self.maxpool_bn(self.maxpool(x))
        x = self.a_relu(x)
        return x

@STEM_REGISTRY.register()
class TemporalAdaptive2DStemV4MixPool(Base2DStem):
    """
    Inherits base 3D stem and adds a maxpool as downsampling.
    """
    def __init__(self, cfg):
        super(TemporalAdaptive2DStemV4MixPool, self).__init__(cfg)

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
        self.a_rf = route_func_mlp(
            c_in=num_filters,
            num_frames=cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)

        self.maxpool = nn.MaxPool3d(
            kernel_size=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]
            ],
            stride=1,
            padding=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]//2
            ],
        )
        self.avgpool = nn.AvgPool3d(
            kernel_size=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]
            ],
            stride=1,
            padding=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]//2
            ],
        )
        if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
            self.maxpool_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
            self.maxpool_bn.no_init=True
            self.maxpool_bn.weight.data.zero_()
            self.maxpool_bn.bias.data.zero_()
            self.avgpool_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
            self.avgpool_bn.no_init=True
            self.avgpool_bn.weight.data.zero_()
            self.avgpool_bn.bias.data.zero_()

        self.a_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.a(x)
        x = self.a_rf(x) * x
        x = self.a_bn(x) + self.avgpool_bn(self.avgpool(x)) + self.maxpool_bn(self.maxpool(x))
        x = self.a_relu(x)
        return x