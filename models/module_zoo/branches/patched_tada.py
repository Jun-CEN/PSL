
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.modules.utils import _triple

from models.base.base_blocks import BaseBranch
from models.base.base_blocks import BRANCH_REGISTRY, STEM_REGISTRY

class route_func_mlp_with_global_info_not_patched(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_not_patched, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
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
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        
    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x) + self.g(global_x) # modification here
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class route_func_mlp_with_global_info_not_patched_v2(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_not_patched_v2, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g1 = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=1,
            padding=0,
        )
        self.bn_g = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.g2 = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.g2.no_init=True
        self.g2.weight.data.zero_()

        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[3,1,1],
            padding=[1,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[3,1,1],
            padding=[1,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        
    def forward(self, x):
        global_x = self.globalpool(x)
        g = self.g1(global_x)
        g = self.bn_g(g)
        g = self.relu(g)
        g = self.g2(g)

        x = self.avgpool(x)
        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + g + 1
        return x