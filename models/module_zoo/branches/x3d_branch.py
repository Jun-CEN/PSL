#!/usr/bin/env python3

""" X3D branch. """ 

import torch
import torch.nn as nn

from models.base.base_blocks import BaseBranch, BaseHead
from models.base.base_blocks import BRANCH_REGISTRY

from models.module_zoo.modules.se import SE
from models.module_zoo.modules.swish import Swish
from models.module_zoo.branches.temporal_adaptive_spatialconv import (
    TemporalAdaptive3DConv, 
    route_func_mlp_with_global_info
)

@BRANCH_REGISTRY.register()
class X3DBranch(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(X3DBranch, self).__init__(cfg, block_idx)
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = int(self.num_filters*self.expansion_ratio),
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(int(self.num_filters*self.expansion_ratio), eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels     = int(self.num_filters*self.expansion_ratio),
            out_channels    = int(self.num_filters*self.expansion_ratio),
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            groups          = int(self.num_filters*self.expansion_ratio)
        )
        self.b_bn = nn.BatchNorm3d(int(self.num_filters*self.expansion_ratio), eps=self.bn_eps, momentum=self.bn_mmt)
        if self.cfg.VIDEO.BACKBONE.BRANCH.SWISH:
            self.b_relu = Swish()
        else:
            self.b_relu = nn.ReLU(inplace=True)

        if ((self.block_id+1)%2) and self.cfg.VIDEO.BACKBONE.BRANCH.SE_RATIO > 0.0:
            self.se = SE(int(self.num_filters*self.expansion_ratio), self.cfg.VIDEO.BACKBONE.BRANCH.SE_RATIO)

        self.c = nn.Conv3d(
            in_channels     = int(self.num_filters*self.expansion_ratio),
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
            raise NotImplementedError
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            if hasattr(self, "se"):
                x = self.se(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class X3DBranchTemporalAdaptive(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(X3DBranchTemporalAdaptive, self).__init__(cfg, block_idx)
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = int(self.num_filters*self.expansion_ratio),
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(int(self.num_filters*self.expansion_ratio), eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = TemporalAdaptive3DConv(
            in_channels     = int(self.num_filters*self.expansion_ratio),
            out_channels    = int(self.num_filters*self.expansion_ratio),
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            groups          = int(self.num_filters*self.expansion_ratio),
            adaptive_dim    = "cout"
        )
        self.b_rf = route_func_mlp_with_global_info(
            c_in=int(self.num_filters*self.expansion_ratio),
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
        )
        self.b_bn = nn.BatchNorm3d(int(self.num_filters*self.expansion_ratio), eps=self.bn_eps, momentum=self.bn_mmt)
        if self.cfg.VIDEO.BACKBONE.BRANCH.SWISH:
            self.b_relu = Swish()
        else:
            self.b_relu = nn.ReLU(inplace=True)

        if ((self.block_id+1)%2) and self.cfg.VIDEO.BACKBONE.BRANCH.SE_RATIO > 0.0:
            self.se = SE(int(self.num_filters*self.expansion_ratio), self.cfg.VIDEO.BACKBONE.BRANCH.SE_RATIO)

        self.c = nn.Conv3d(
            in_channels     = int(self.num_filters*self.expansion_ratio),
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
            raise NotImplementedError
        elif self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            if hasattr(self, "se"):
                x = self.se(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x