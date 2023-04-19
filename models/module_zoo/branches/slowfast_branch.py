#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" SlowFast architectures. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseBranch
from models.base.base_blocks import BRANCH_REGISTRY
from models.utils.init_helper import _init_convnet_weights

from models.module_zoo.branches.temporal_adaptive_spatialconv import (
    TemporalAdaptiveSpatialConvCinAdaptive, 
    route_func_mlp_with_global_info
)

@BRANCH_REGISTRY.register()
class SlowfastBranch(BaseBranch):
    """
    Constructs SlowFast conv branch.

    See Christoph Feichtenhofer et al.
    SlowFast Networks for Video Recognition.
    """
    def __init__(self, cfg, block_idx):
        super(SlowfastBranch, self).__init__(cfg, block_idx)

    def _construct_simple_block(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters,
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels     = self.num_filters,
            out_channels    = self.num_filters,
            kernel_size     = self.kernel_size,
            stride          = 1,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_bn.transform_final_bn = True
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [3, 1, 1] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 1,
            stride          = 1,
            padding         = [1, 0, 0] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.c_bn.transform_final_bn = True
    
    def forward(self, x):
        if self.transformation == 'simple_block':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            return x
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SlowfastBranchCinAdaptiveWithGlobalInfo(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SlowfastBranchCinAdaptiveWithGlobalInfo, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [3, 1, 1] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 1,
            stride          = 1,
            padding         = [1, 0, 0] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SlowfastBranchCinAdaptiveWithGlobalInfoMixPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SlowfastBranchCinAdaptiveWithGlobalInfoMixPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [3, 1, 1] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 1,
            stride          = 1,
            padding         = [1, 0, 0] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_maxpool = nn.MaxPool3d(
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
        self.b_avgpool = nn.AvgPool3d(
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
            self.b_maxpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_maxpool_bn.no_init=True
            self.b_maxpool_bn.weight.data.zero_()
            self.b_maxpool_bn.bias.data.zero_()
            self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_avgpool_bn.no_init=True
            self.b_avgpool_bn.weight.data.zero_()
            self.b_avgpool_bn.bias.data.zero_()

        self.b_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x)) + self.b_maxpool_bn(self.b_maxpool(x))
            else:
                x = self.b_bn(x) + self.b_avgpool(x) + self.b_maxpool(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SlowfastBranchDilated(BaseBranch):
    """
    Constructs SlowFast conv branch with control on the dilation rate.
    """
    def __init__(self, cfg, block_idx):
        super(SlowfastBranchDilated, self).__init__(cfg, block_idx)

    def _construct_simple_block(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters,
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels     = self.num_filters,
            out_channels    = self.num_filters,
            kernel_size     = self.kernel_size,
            stride          = 1,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_bn.transform_final_bn = True
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [3, 1, 1] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 1,
            stride          = 1,
            padding         = [1, 0, 0] if self.cfg.VIDEO.BACKBONE.TEMPORAL_CONV_BOTTLENECK[self.stage_id] else 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2 + 1, self.kernel_size[2]//2 + 1] if self.stage_id == 4 else [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            dilation        = 2 if self.stage_id == 4 else 1
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.c_bn.transform_final_bn = True
    
    def forward(self, x):
        if self.transformation == 'simple_block':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            return x
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x