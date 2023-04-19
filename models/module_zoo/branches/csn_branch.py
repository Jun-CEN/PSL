#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" CSN Branch. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseBranch, Base3DStem, BaseHead
from models.base.base_blocks import BRANCH_REGISTRY, STEM_REGISTRY

from models.module_zoo.branches.temporal_adaptive_spatialconv import (
    TemporalAdaptive3DConvCoutAdaptive, 
    route_func_mlp_with_global_info
)

@BRANCH_REGISTRY.register()
class CSNBranch(BaseBranch):
    """
    The ir-CSN branch.
    
    See Du Tran et al.
    Video Classification with Channel-Separated Convolutional Networks.
    """
    def __init__(self, cfg, block_idx):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(CSNBranch, self).__init__(cfg, block_idx)
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
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
            bias            = False,
            groups          = self.num_filters//self.expansion_ratio,
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
        if self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

class route_func_mlp_with_global_info_with_stride(nn.Module):

    def __init__(self, c_in, num_frames, ratio, stride, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_with_stride, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[3,1,1],
            padding=[1,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[3,1,1],
            padding=[1,0,0],
            stride=stride,
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

@BRANCH_REGISTRY.register()
class CSNBranchCoutAdaptiveWithGlobalInfo(BaseBranch):
    """
    The ir-CSN branch.
    
    See Du Tran et al.
    Video Classification with Channel-Separated Convolutional Networks.
    """
    def __init__(self, cfg, block_idx):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(CSNBranchCoutAdaptiveWithGlobalInfo, self).__init__(cfg, block_idx)
    
    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = TemporalAdaptive3DConvCoutAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = self.kernel_size,
            stride          = self.stride,
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            groups          = self.num_filters//self.expansion_ratio,
        )
        self.b_rf = route_func_mlp_with_global_info_with_stride(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            stride=self.stride,
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
        if self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x
