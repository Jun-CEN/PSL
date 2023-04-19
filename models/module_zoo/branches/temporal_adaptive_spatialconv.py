
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.modules.utils import _triple

from models.base.base_blocks import BaseBranch
from models.base.base_blocks import BRANCH_REGISTRY, STEM_REGISTRY

from models.module_zoo.branches.patched_tada import (
    route_func_mlp_with_global_info_not_patched,
    route_func_mlp_with_global_info_not_patched_v2
)

from models.module_zoo.modules.se import SE

class route_func_mlp_local_with_global_info(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_local_with_global_info, self).__init__()
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
            kernel_size=1,
            padding=0,
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=1,
            padding=0,
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

class route_func_mlp_with_global_info_asym(nn.Module):

    def __init__(self, c_in, c_out, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_asym, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
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
            out_channels=c_out,
            kernel_size=[3,1,1],
            padding=[1,0,0],
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

class route_func_mlp_with_global_info(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info, self).__init__()
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

class route_func_mlp_with_global_info_randinit(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_randinit, self).__init__()
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
            bias=False
        )
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x)
        return x

class route_func_mlp_with_global_info_norm(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_norm, self).__init__()
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
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        self.weight_bn = nn.BatchNorm3d(int(c_in), eps=bn_eps, momentum=bn_mmt)
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x)
        x = self.weight_bn(x) + 1
        return x

class route_func_mlp_with_global_info_dropout(nn.Module):

    def __init__(self, c_in, num_frames, ratio, dropout_ratio=0.2, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_dropout, self).__init__()
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
        self.dropout = nn.Dropout(p=dropout_ratio)
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
        x = self.a(self.dropout(x + self.g(global_x)))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class route_func_mlp_with_global_info_spatial_adaptive(nn.Module):

    def __init__(self, c_in, num_frames, ratio, num_spatial_weight, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_spatial_adaptive, self).__init__()
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
            out_channels=c_in*num_spatial_weight,
            kernel_size=[3,1,1],
            padding=[1,0,0],
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

class route_func_mlp_with_global_info_pure_spatial_adaptive(nn.Module):

    def __init__(self, c_in, num_frames, ratio, num_spatial_weight, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_pure_spatial_adaptive, self).__init__()
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
            out_channels=num_spatial_weight,
            kernel_size=[3,1,1],
            padding=[1,0,0],
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

class route_func_mlp_with_global_info_kernel_size(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
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

class route_func_mlp_with_global_info_kernel_size_add(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_add, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
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
        x = self.b(x)
        return x

class route_func_mlp_with_global_info_kernel_size_gelu(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_gelu, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.gelu = nn.GELU()
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.gelu(x)
        x = self.b(x) + 1
        return x

class route_func_mlp_with_global_info_kernel_size_gelu_instancenorm(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_gelu_instancenorm, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.gelu = nn.GELU()
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        self.b_in = nn.InstanceNorm3d(c_in, eps=bn_eps, momentum=bn_mmt)
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.gelu(x)
        x = self.b_in(self.b(x)) + 1
        return x

class route_func_mlp_with_global_info_kernel_size_gelu_instancenorm_newinit(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_gelu_instancenorm_newinit, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.gelu = nn.GELU()
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b_in = nn.InstanceNorm3d(c_in, eps=bn_eps, momentum=bn_mmt, affine=True)
        self.b_in.no_init = True
        self.b_in.weight.data.zero_()
        self.b_in.bias.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.gelu(x)
        x = self.b_in(self.b(x)) + 1
        return x

class route_func_mlp_with_global_info_kernel_size_gelu_batchnorm_newinit(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_gelu_batchnorm_newinit, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.gelu = nn.GELU()
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b_bn = nn.BatchNorm3d(c_in, eps=bn_eps, momentum=bn_mmt, affine=True)
        self.b_bn.no_init = True
        self.b_bn.weight.data.zero_()
        self.b_bn.bias.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.gelu(x)
        x = self.b_bn(self.b(x)) + 1
        return x

class route_func_mlp_with_global_info_kernel_size_groupnorm(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_groupnorm, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.gn = nn.GroupNorm(1, int(c_in//ratio), eps=bn_eps)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.gn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class route_func_temporal_invariant(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_temporal_invariant, self).__init__()
        self.c_in = c_in
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.g.no_init=True
        self.g.weight.data.zero_()
        self.g.bias.data.zero_()
        

    def forward(self, x):
        x = self.globalpool(x)
        x = self.g(x) + 1
        return x

class route_func_temporal_invariant_randinit(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_temporal_invariant_randinit, self).__init__()
        self.c_in = c_in
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        

    def forward(self, x):
        x = self.globalpool(x)
        x = self.g(x)
        return x

class route_func_temporal_invariant_local_generation_randinit(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_temporal_invariant_local_generation_randinit, self).__init__()
        self.c_in = c_in
        self.spatialpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.t1 = nn.Conv3d(
            in_channels=num_frames,
            out_channels=num_frames//ratio,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.t2 = nn.Conv3d(
            in_channels=num_frames//ratio,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=False
        )
        

    def forward(self, x):
        x = self.spatialpool(x)
        x = x.permute(0,2,1,3,4) # B, C, T, 1, 1 -> B, T, C, 1, 1
        x = self.t1(x)
        x = self.relu(x)
        x = self.t2(x)
        return x.permute(0,2,1,3,4)

class route_func_temporal_invariant_local_generation(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_temporal_invariant_local_generation, self).__init__()
        self.c_in = c_in
        self.spatialpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.t1 = nn.Conv3d(
            in_channels=num_frames,
            out_channels=num_frames//ratio,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.t2 = nn.Conv3d(
            in_channels=num_frames//ratio,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.t2.no_init=True
        self.t2.weight.data.zero_()
        

    def forward(self, x):
        x = self.spatialpool(x)
        x = x.permute(0,2,1,3,4) # B, C, T, 1, 1 -> B, T, C, 1, 1
        x = self.t1(x)
        x = self.relu(x)
        x = self.t2(x) + 1
        return x.permute(0,2,1,3,4)

class route_func_temporal_invariant_localglobal_generation_randinit(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_temporal_invariant_localglobal_generation_randinit, self).__init__()
        self.c_in = c_in
        self.spatialpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.t1 = nn.Conv3d(
            in_channels=num_frames,
            out_channels=num_frames//ratio,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.t2 = nn.Conv3d(
            in_channels=num_frames//ratio,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=False
        )

        self.t3 = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in//ratio,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.t4 = nn.Conv3d(
            in_channels=c_in//ratio,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.spatialpool(x)
        x1 = x.permute(0,2,1,3,4) # B, C, T, 1, 1 -> B, T, C, 1, 1
        x1 = self.t1(x1)
        x1 = self.relu(x1)
        x1 = self.t2(x1)

        x2 = self.t3(x)
        x2 = self.relu(x2)
        x2 = self.t4(x2)
        x2 = self.sigmoid(x2)
        return x1.permute(0,2,1,3,4) * x2

class route_func_temporal_invariant_localglobal_generation(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_temporal_invariant_localglobal_generation, self).__init__()
        self.c_in = c_in
        self.spatialpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.t1 = nn.Conv3d(
            in_channels=num_frames,
            out_channels=num_frames//ratio,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.t2 = nn.Conv3d(
            in_channels=num_frames//ratio,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.t2.no_init=True
        self.t2.weight.data.zero_()

        self.t3 = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in//ratio,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.t4 = nn.Conv3d(
            in_channels=c_in//ratio,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.spatialpool(x)
        x1 = x.permute(0,2,1,3,4) # B, C, T, 1, 1 -> B, T, C, 1, 1
        x1 = self.t1(x1)
        x1 = self.relu(x1)
        x1 = self.t2(x1)

        x2 = self.t3(x)
        x2 = self.relu(x2)
        x2 = self.t4(x2)
        x2 = self.sigmoid(x2)
        return x1.permute(0,2,1,3,4) * x2 + 1

class route_func_temporal_variant(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_temporal_variant, self).__init__()
        self.c_in = c_in
        self.num_frames = num_frames
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in*num_frames,
            kernel_size=1,
            padding=0,
        )
        self.g.no_init=True
        self.g.weight.data.zero_()
        self.g.bias.data.zero_()
        

    def forward(self, x):
        x = self.globalpool(x)
        x = self.g(x) + 1
        x = x.reshape(x.shape[0], x.shape[1]//self.num_frames, self.num_frames, 1, 1)
        return x

class route_func_temporal_variant_randinit(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_temporal_variant_randinit, self).__init__()
        self.c_in = c_in
        self.num_frames = num_frames
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in*num_frames,
            kernel_size=1,
            padding=0,
        )
        

    def forward(self, x):
        x = self.globalpool(x)
        x = self.g(x)
        x = x.reshape(x.shape[0], x.shape[1]//self.num_frames, self.num_frames, 1, 1)
        return x

class route_func_mlp_with_global_info_kernel_size_bnpool(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_bnpool, self).__init__()
        self.c_in = c_in
        self.pool_bn = nn.BatchNorm3d(c_in, eps=bn_eps, momentum=bn_mmt)
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        x = self.pool_bn(x)
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class route_func_mlp_with_global_info_kernel_size_poolbn(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_poolbn, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.avgpool_bn = nn.BatchNorm3d(c_in, eps=bn_eps, momentum=bn_mmt)
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.globalpool_bn = nn.BatchNorm3d(c_in, eps=bn_eps, momentum=bn_mmt)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        global_x = self.globalpool_bn(global_x)
        x = self.avgpool(x)
        x = self.avgpool_bn(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class route_func_mlp_with_global_info_kernel_size_poolbnrelu(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_poolbnrelu, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.avgpool_bn = nn.BatchNorm3d(c_in, eps=bn_eps, momentum=bn_mmt)
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.globalpool_bn = nn.BatchNorm3d(c_in, eps=bn_eps, momentum=bn_mmt)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        global_x = self.globalpool_bn(global_x)
        global_x = self.relu(global_x)
        x = self.avgpool(x)
        x = self.avgpool_bn(x)
        x = self.relu(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class route_func_mlp_with_global_info_kernel_size_with_bias(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_with_bias, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()

        self.c = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.c.no_init=True
        self.c.weight.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x_conv = self.b(x) + 1
        x_bias = self.c(x)
        return (x_conv, x_bias)

class route_func_mlp_with_global_info_kernel_size_with_bias_decoupled(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_with_bias_decoupled, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()

        self.c = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn2 = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.d = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.d.no_init=True
        self.d.weight.data.zero_()
        

    def forward(self, x):
        x = self.avgpool(x)
        g = self.g(self.globalpool(x))

        x_conv = self.a(x + g)
        x_conv = self.bn(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = self.b(x_conv) + 1

        x_bias = self.c(x + g)
        x_bias = self.bn2(x_bias)
        x_bias = self.relu(x_bias)
        x_bias = self.d(x_bias)
        return (x_conv, x_bias)

class route_func_mlp_with_global_info_kernel_size_constrained(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_constrained, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x)
        x = self.sigmoid(x)
        return x

class route_func_mlp_with_global_info_kernel_size_constrained_v2(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_kernel_size_constrained_v2, self).__init__()
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
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.relu(x)
        x = self.b(x)
        x = self.sigmoid(x)
        return x

class route_func_transformer(nn.Module):

    def __init__(self, c_in, num_frames, ratio, num_heads, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_transformer, self).__init__()
        self.c_in = c_in
        self.c_mid = int(c_in//ratio)
        self.num_heads = num_heads
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.to_qkv = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio)*3,
            kernel_size=1,
            padding=0,
        )
        self.scale = self.c_mid ** -0.5

        self.proj = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.to_qkv.no_init=True
        self.proj.no_init=True

        from models.utils.init_helper import trunc_normal_
        trunc_normal_(self.to_qkv.weight, std=.02)
        self.to_qkv.bias.data.zero_()
        self.proj.weight.data.zero_()
        self.proj.bias.data.fill_(1.)
        

    def forward(self, x):
        B, _, T, _, _ = x.shape
        x = self.avgpool(x)
        # B, C, T, 1, 1 -> B, 3C, T, 1, 1
        qkv = self.to_qkv(x).reshape(B, 3, self.num_heads, self.c_mid//self.num_heads, T)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]

        attn = (q.transpose(-2,-1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v.transpose(-1,-2)).reshape(B, self.c_mid, T, 1, 1)
        x = self.proj(x)
        return x

class route_func_mlp_with_global_info_partial(nn.Module):

    def __init__(self, c_in, num_frames, ratio, proportion=1.0, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_partial, self).__init__()
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
            out_channels=int(c_in*proportion),
            kernel_size=[3,1,1],
            padding=[1,0,0],
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

class route_func_mlp_two_channel(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_two_channel, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[3,1,1],
            padding=[1,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b1 = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[3,1,1],
            padding=[1,0,0],
            bias=False
        )
        self.b2 = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[3,1,1],
            padding=[1,0,0],
            bias=False
        )
        self.b1.no_init=True
        self.b2.no_init=True
        self.b1.weight.data.zero_()
        self.b2.weight.data.zero_()
        

    def forward(self, x):
        x = self.avgpool(x)
        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x1 = self.b1(x) + 1
        x2 = self.b2(x) + 1
        return (x1, x2)

class route_func_mlp_with_global_info_two_channel(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_two_channel, self).__init__()
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
        self.b1 = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[3,1,1],
            padding=[1,0,0],
            bias=False
        )
        self.b2 = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[3,1,1],
            padding=[1,0,0],
            bias=False
        )
        self.b1.no_init=True
        self.b2.no_init=True
        self.b1.weight.data.zero_()
        self.b2.weight.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x1 = self.b1(x) + 1
        x2 = self.b2(x) + 1
        return (x1, x2)

class route_func_mlp_with_global_info_two_channel_v2(nn.Module):

    def __init__(self, c_in, num_frames, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_two_channel_v2, self).__init__()
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
        self.b1 = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.b2 = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.b1.no_init=True
        self.b2.no_init=True
        self.b1.weight.data.zero_()
        self.b2.weight.data.zero_()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x1 = self.b1(x) + 1
        x2 = self.b2(x) + 1
        return (x1, x2)

class route_func_mlp(nn.Module):

    def __init__(self, c_in, num_frames, ratio, meta_arch="ResNet3D", bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp, self).__init__()
        self.c_in = c_in
        self.meta_arch = meta_arch
        self.num_frames = num_frames
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
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
        if self.meta_arch == "ResNet2D": 
            b_t, c, h, w = x.size()
            x = x.reshape(-1, self.num_frames, c, h, w).permute(0,2,1,3,4)
        x = self.avgpool(x)
        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class route_func_mlp_kernel_size(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_kernel_size, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        x = self.avgpool(x)
        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class route_func_mlp_kernel_size_randinit(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_kernel_size_randinit, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        

    def forward(self, x):
        x = self.avgpool(x)
        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x)
        return x

class route_func_mlp_linear(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_linear, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.a.no_init=True
        self.a.weight.data.zero_()
        self.a.bias.data.fill_(1.)
        

    def forward(self, x):
        x = self.avgpool(x)
        x = self.a(x)
        return x

class route_func_mlp_linear_global_info(nn.Module):

    def __init__(self, c_in, num_frames, ratio, kernel_size, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_linear_global_info, self).__init__()
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
            out_channels=c_in,
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.a.no_init=True
        self.a.weight.data.zero_()
        self.a.bias.data.fill_(1.)
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.a(x)
        return x

class TemporalAdaptiveSpatialConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(TemporalAdaptiveSpatialConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        _, _, c_out, c_in, kh, kw = self.weight.size()
        x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(-1) * self.weight).reshape(-1, c_in, kh, kw)
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=None, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        return output

class TemporalAdaptiveSpatialConvCinAdaptive(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1, meta_arch="ResNet3D", num_frames=8):
        super(TemporalAdaptiveSpatialConvCinAdaptive, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts
        self.meta_arch = meta_arch
        self.num_frames = num_frames

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight, with_bias=False):
        _, _, c_out, c_in, kh, kw = self.weight.size()
        if self.meta_arch == "ResNet3D":
            b, c_in, t, h, w = x.size()
            x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        else:
            b_t, c_in, h, w = x.size()
            b = b_t // self.num_frames
            t = self.num_frames
            x = x.reshape(1, -1, h, w)
        if with_bias:
            weight = (routing_weight[0].permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
            bias = routing_weight[1].permute(0,2,1,3,4).reshape(-1)
        else:
            weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
            bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        if self.meta_arch == "ResNet3D":
            output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        elif self.meta_arch == "ResNet2D":
            output = output.view(b_t, c_out, output.size(-2), output.size(-1))
        return output

class TemporalAdaptiveSpatialConvCinAdaptiveAdd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1, meta_arch="ResNet3D", num_frames=8):
        super(TemporalAdaptiveSpatialConvCinAdaptiveAdd, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts
        self.meta_arch = meta_arch
        self.num_frames = num_frames

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight, with_bias=False):
        _, _, c_out, c_in, kh, kw = self.weight.size()
        if self.meta_arch == "ResNet3D":
            b, c_in, t, h, w = x.size()
            x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        else:
            b_t, c_in, h, w = x.size()
            b = b_t // self.num_frames
            t = self.num_frames
            x = x.reshape(1, -1, h, w)
        if with_bias:
            weight = (routing_weight[0].permute(0,2,1,3,4).unsqueeze(2) + self.weight).reshape(-1, c_in, kh, kw)
            bias = routing_weight[1].permute(0,2,1,3,4).reshape(-1)
        else:
            weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(2) + self.weight).reshape(-1, c_in, kh, kw)
            bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        if self.meta_arch == "ResNet3D":
            output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        elif self.meta_arch == "ResNet2D":
            output = output.view(b_t, c_out, output.size(-2), output.size(-1))
        return output

class TemporalAdaptiveSpatialConvCinAdaptiveWithKernelBias(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1, meta_arch="ResNet3D", num_frames=8):
        super(TemporalAdaptiveSpatialConvCinAdaptiveWithKernelBias, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts
        self.meta_arch = meta_arch
        self.num_frames = num_frames

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight, with_kernel_bias=False):
        _, _, c_out, c_in, kh, kw = self.weight.size()
        if self.meta_arch == "ResNet3D":
            b, c_in, t, h, w = x.size()
            x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        else:
            b_t, c_in, h, w = x.size()
            b = b_t // self.num_frames
            t = self.num_frames
            x = x.reshape(1, -1, h, w)
        bias = None
        if with_kernel_bias:
            weight = (routing_weight[0].permute(0,2,1,3,4).unsqueeze(2) * self.weight + routing_weight[1].permute(0,2,1,3,4).unsqueeze(2)).reshape(-1, c_in, kh, kw)
        else:
            weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        if self.meta_arch == "ResNet3D":
            output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        elif self.meta_arch == "ResNet2D":
            output = output.view(b_t, c_out, output.size(-2), output.size(-1))
        return output

class TemporalAdaptive3DConvCinAdaptive(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(TemporalAdaptive3DConvCinAdaptive, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        _, _, c_out, c_in, kt, kh, kw = self.weight.size()
        x = torch.nn.functional.pad(x, (0,0,0,0,kt//2,kt//2), "constant", 0).unfold(
                dimension=2, size=kt, step=1
            ).permute(0,2,1,5,3,4).reshape(1, -1, kt, h, w)
        weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(2).unsqueeze(-1) * self.weight).reshape(-1, c_in, kt, kh, kw)
        bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv3d(
                    x, weight, bias=bias, stride=[1] + list(self.stride[1:]), padding=[0]+ list(self.padding[1:]), 
                    dilation=[1] + list(self.dilation[1:]), groups=self.groups * b * t
                )

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        return output

class TemporalAdaptive3DConvCoutAdaptive(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(TemporalAdaptive3DConvCoutAdaptive, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        _, _, c_out, c_in, kt, kh, kw = self.weight.size()
        x = torch.nn.functional.pad(x, (0,0,0,0,kt//2,kt//2), "constant", 0).unfold(
                dimension=2, size=kt, step=self.stride[0]
            ).permute(0,2,1,5,3,4).reshape(1, -1, kt, h, w)
        weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(3).unsqueeze(-1) * self.weight).reshape(-1, c_in, kt, kh, kw)
        bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv3d(
                    x, weight, bias=bias, stride=[1] + list(self.stride[1:]), padding=[0]+ list(self.padding[1:]), 
                    dilation=[1] + list(self.dilation[1:]), groups=self.groups * b * (t//self.stride[0])
                )

        output = output.view(b, (t//self.stride[0]), c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        return output

class TemporalAdaptive3DConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1, adaptive_dim="cin"):
        super(TemporalAdaptive3DConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts
        self.adaptive_dim = adaptive_dim

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        _, _, c_out, c_in, kt, kh, kw = self.weight.size()
        x = torch.nn.functional.pad(x, (0,0,0,0,kt//2,kt//2), "constant", 0).unfold(
                dimension=2, size=kt, step=1
            ).permute(0,2,1,5,3,4).reshape(1, -1, kt, h, w)
        if self.adaptive_dim == "cin":
            weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(2).unsqueeze(-1) * self.weight).reshape(-1, c_in, kt, kh, kw)
        elif self.adaptive_dim == "cout":
            weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(3).unsqueeze(-1) * self.weight).reshape(-1, c_in, kt, kh, kw)
        bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv3d(
                    x, weight, bias=bias, stride=[1] + list(self.stride[1:]), padding=[0]+ list(self.padding[1:]), 
                    dilation=[1] + list(self.dilation[1:]), groups=self.groups * b * t
                )

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        return output

class TemporalAdaptiveSpatialConvSpatialAdaptive(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(TemporalAdaptiveSpatialConvSpatialAdaptive, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight, with_bias=False):
        b, c_in, t, h, w = x.size()
        _, _, c_out, c_in, kh, kw = self.weight.size()
        x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        if with_bias:
            weight = (routing_weight[0].reshape(b, kh, kw, 1, t).permute(0,4,3,1,2).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
            bias = routing_weight[1].permute(0,2,1,3,4).reshape(-1)
        else:
            weight = (routing_weight.reshape(b, kh, kw, 1, t).permute(0,4,3,1,2).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
            bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        return output

class TemporalAdaptiveSpatialConvCinAdaptiveSpatialAdaptive(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(TemporalAdaptiveSpatialConvCinAdaptiveSpatialAdaptive, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight, with_bias=False):
        b, c_in, t, h, w = x.size()
        _, _, c_out, c_in, kh, kw = self.weight.size()
        x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        if with_bias:
            weight = (routing_weight[0].reshape(b, kh, kw, c_in,t).permute(0,4,3,1,2).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
            bias = routing_weight[1].permute(0,2,1,3,4).reshape(-1)
        else:
            weight = (routing_weight.reshape(b, kh, kw, c_in,t).permute(0,4,3,1,2).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
            bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        return output

class TemporalAdaptiveSpatialConvCinAdaptivePartial(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 proportion=1.0,
                 num_experts=1):
        super(TemporalAdaptiveSpatialConvCinAdaptivePartial, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.dynamic_channels = int(out_channels*proportion)
        self.shared_channels = out_channels - self.dynamic_channels

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        _, _, c_out, c_in, kh, kw = self.weight.size()
        x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        weight = torch.cat(
            (self.weight[:,:,:,:self.shared_channels,:,:].repeat(b,t,1,1,1,1),
            routing_weight.permute(0,2,1,3,4).unsqueeze(2) * self.weight[:,:,:,self.shared_channels:,:,:]), 
            dim=3
        ).reshape(-1, c_in, kh, kw)
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=None, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        return output

class TemporalAdaptiveSpatialConvBothAdaptive(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(TemporalAdaptiveSpatialConvBothAdaptive, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, t, h, w = x.size()
        _, _, c_out, c_in, kh, kw = self.weight.size()
        x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        weight = (
                routing_weight[0].permute(0,2,1,3,4).unsqueeze(2) * self.weight * routing_weight[1].permute(0,2,1,3,4).unsqueeze(3) 
            ).reshape(-1, c_in, kh, kw)
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=None, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        return output

###########################################################################################
###################################     Blocks      #######################################
###########################################################################################


@BRANCH_REGISTRY.register()
class BaselineSpatialConvBlock(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(BaselineSpatialConvBlock, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
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
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
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
class Baseline3DConvBlock(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(Baseline3DConvBlock, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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
            kernel_size     = [self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]],
            stride          = [self.stride[0], self.stride[1], self.stride[2]],
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
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
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
class Baseline2Plus1DBlock(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(Baseline2Plus1DBlock, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

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

            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class Baseline2Plus1DBlockReversed(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(Baseline2Plus1DBlockReversed, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
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
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlock(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlock, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConv(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp(
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
class TemporalAdaptiveSpatialConvBlockWithGlobalInfoAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockWithGlobalInfoAvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConv(
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptive3DConvBlock(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptive3DConvBlock, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptive3DConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]],
            stride          = [self.stride[0], self.stride[1], self.stride[2]],
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
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
class TemporalAdaptiveSpatialConvBlockCinAdaptive(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptive, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        if self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet3D":
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

            self.b = TemporalAdaptiveSpatialConvCinAdaptive(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
                bias            = False,
                meta_arch       = self.cfg.VIDEO.BACKBONE.META_ARCH, 
                num_frames      = self.cfg.DATA.NUM_INPUT_FRAMES,
            )
            self.b_rf = route_func_mlp(
                c_in=self.num_filters//self.expansion_ratio,
                num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
                ratio=4,
                meta_arch = self.cfg.VIDEO.BACKBONE.META_ARCH, 
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
        elif self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet2D":
            self.a = nn.Conv2d(
                in_channels     = self.dim_in,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.a_relu = nn.ReLU(inplace=True)

            self.b = TemporalAdaptiveSpatialConvCinAdaptive(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
                bias            = False,
                meta_arch = self.cfg.VIDEO.BACKBONE.META_ARCH,
            )
            self.b_rf = route_func_mlp(
                c_in=self.num_filters//self.expansion_ratio,
                num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
                ratio=4,
                meta_arch = self.cfg.VIDEO.BACKBONE.META_ARCH, 
            )
            self.b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_relu = nn.ReLU(inplace=True)

            self.c = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
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
class TemporalAdaptive2Plus1DBlockCinAdaptive(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptive2Plus1DBlockCinAdaptive, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        assert self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet3D"

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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            meta_arch       = self.cfg.VIDEO.BACKBONE.META_ARCH, 
            num_frames      = self.cfg.DATA.NUM_INPUT_FRAMES,
        )
        self.b_rf = route_func_mlp(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            meta_arch = self.cfg.VIDEO.BACKBONE.META_ARCH, 
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

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

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptive2Plus1DBlockCinAdaptiveTempConvAsWell(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptive2Plus1DBlockCinAdaptiveTempConvAsWell, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        assert self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet3D"

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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            meta_arch       = self.cfg.VIDEO.BACKBONE.META_ARCH, 
            num_frames      = self.cfg.DATA.NUM_INPUT_FRAMES,
        )
        self.b_rf = route_func_mlp(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            meta_arch = self.cfg.VIDEO.BACKBONE.META_ARCH, 
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = TemporalAdaptive3DConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_rf = route_func_mlp(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            meta_arch = self.cfg.VIDEO.BACKBONE.META_ARCH, 
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

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

            x = self.b2(x, self.b2_rf(x))
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptive2Plus1DBlockCinAdaptiveReversed(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptive2Plus1DBlockCinAdaptiveReversed, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        assert self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet3D"

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

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            meta_arch       = self.cfg.VIDEO.BACKBONE.META_ARCH, 
            num_frames      = self.cfg.DATA.NUM_INPUT_FRAMES,
        )
        self.b_rf = route_func_mlp(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            meta_arch = self.cfg.VIDEO.BACKBONE.META_ARCH, 
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

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x


@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_kernel_size(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveV3WithGlobalInfo(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveV3WithGlobalInfo, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveLinear(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveLinear, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_linear(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveLinearWithGlobalInfo(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveLinearWithGlobalInfo, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_linear_global_info(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalFull(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalFull, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_rf = route_func_mlp_with_global_info_asym(
            c_in=self.dim_in,
            c_out=self.dim_in,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
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
        self.b_rf = route_func_mlp(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.c = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_rf = route_func_mlp_with_global_info_asym(
            c_in=self.num_filters//self.expansion_ratio,
            c_out=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=2,
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x, self.a_rf(x))
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x, self.c_rf(x))
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockWithGlobalInfo(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockWithGlobalInfo, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConv(
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfo(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfo, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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
class TemporalAdaptive3DConvBlockCinAdaptiveWithGlobalInfo(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptive3DConvBlockCinAdaptiveWithGlobalInfo, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptive3DConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]],
            stride          = [self.stride[0], self.stride[1], self.stride[2]],
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoPartial(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoPartial, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptivePartial(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            proportion      = self.cfg.VIDEO.BACKBONE.BRANCH.PROPORTION,
        )
        self.b_rf = route_func_mlp_with_global_info_partial(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            proportion=self.cfg.VIDEO.BACKBONE.BRANCH.PROPORTION,
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoPartialStage(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoPartialStage, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = self.stage_id in self.cfg.VIDEO.BACKBONE.BRANCH.ENABLE_STAGES

        self._construct_branch()
    
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

        if self.enable:
            self.b = TemporalAdaptiveSpatialConvCinAdaptivePartial(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
                bias            = False,
                proportion      = self.cfg.VIDEO.BACKBONE.BRANCH.PROPORTION,
            )
            self.b_rf = route_func_mlp_with_global_info_partial(
                c_in=self.num_filters//self.expansion_ratio,
                num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
                ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
                proportion=self.cfg.VIDEO.BACKBONE.BRANCH.PROPORTION,
            )
        else:
            self.b = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
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
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            if self.enable:
                x = self.b(x, self.b_rf(x))
            else:
                x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoPartialAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoPartialAvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptivePartial(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False,
            proportion      = self.cfg.VIDEO.BACKBONE.BRANCH.PROPORTION,
        )
        self.b_rf = route_func_mlp_with_global_info_partial(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            proportion=self.cfg.VIDEO.BACKBONE.BRANCH.PROPORTION,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoPartialStageAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoPartialStageAvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = self.stage_id in self.cfg.VIDEO.BACKBONE.BRANCH.ENABLE_STAGES

        self._construct_branch()
    
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

        if self.enable:
            self.b = TemporalAdaptiveSpatialConvCinAdaptivePartial(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
                bias            = False,
                proportion      = self.cfg.VIDEO.BACKBONE.BRANCH.PROPORTION,
            )
            self.b_rf = route_func_mlp_with_global_info_partial(
                c_in=self.num_filters//self.expansion_ratio,
                num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
                ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
                proportion=self.cfg.VIDEO.BACKBONE.BRANCH.PROPORTION,
            )
        else:
            self.b = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
                bias            = False
            )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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

            if self.enable:
                x = self.b(x, self.b_rf(x))
            else:
                x = self.b(x)

            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.avg_pool = nn.AvgPool3d(
            kernel_size=3,
            stride=1,
            padding=1,
        )

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

            x = self.avg_pool(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveLocalWithGlobalInfo(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveLocalWithGlobalInfo, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_local_with_global_info(
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
class TemporalAdaptiveSpatialConvBlockBothAdaptive(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockBothAdaptive, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvBothAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_two_channel(
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
class TemporalAdaptiveSpatialConvBlockBothAdaptiveWithGlobalInfo(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockBothAdaptiveWithGlobalInfo, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvBothAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_two_channel(
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
class TemporalAdaptiveSpatialConvBlockBothAdaptiveWithGlobalInfoV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockBothAdaptiveWithGlobalInfoV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvBothAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_two_channel_v2(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
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
class TemporalAdaptiveSpatialConvBlockBothAdaptiveWithGlobalInfoAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockBothAdaptiveWithGlobalInfoAvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvBothAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_two_channel(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveTransformer(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveTransformer, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_transformer(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            num_heads=self.cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS,
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
class TemporalConvBaseline(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalConvBaseline, self).__init__(cfg, block_idx)
    
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
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

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

            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoConstrained(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoConstrained, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_constrained(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoConstrainedV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoConstrainedV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_constrained_v2(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoTemporalConv(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoTemporalConv, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

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

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoWithBias(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoWithBias, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_with_bias(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
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

            x = self.b(x, self.b_rf(x), with_bias=True)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoWithKernelBiasAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoWithKernelBiasAvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptiveWithKernelBias(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_with_bias(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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

            x = self.b(x, self.b_rf(x), with_kernel_bias=True)
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoWithBiasDecoupled(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoWithBiasDecoupled, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_with_bias_decoupled(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
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

            x = self.b(x, self.b_rf(x), with_bias=True)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMaxPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMaxPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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
        if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
            self.b_maxpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_maxpool_bn.no_init=True
            self.b_maxpool_bn.weight.data.zero_()
            self.b_maxpool_bn.bias.data.zero_()

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
                x = self.b_bn(x) + self.b_maxpool_bn(self.b_maxpool(x))
            else:
                x = self.b_bn(x + self.b_maxpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()

    def _construct_simple_block(self):
        self.a = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.a_rf = route_func_mlp_with_global_info(
            c_in=self.dim_in,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        if self.stride[0] > 1:
            self.striding = nn.AvgPool3d(
                kernel_size=1,
                stride=[self.stride[0], 1, 1],
                padding=0
            )
        self.a_avgpool = nn.AvgPool3d(
            kernel_size=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1],
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]
            ],
            stride=[self.stride[0], 1, 1],
            padding=[
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1]//2,
                self.cfg.VIDEO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[2]//2
            ],
        )
        self.a_avgpool_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.a_avgpool_bn.no_init=True
        self.a_avgpool_bn.weight.data.zero_()
        self.a_avgpool_bn.bias.data.zero_()
        self.a_relu = nn.ReLU(inplace=True)

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters,
            out_channels    = self.num_filters,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = 1,
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info(
            c_in=self.num_filters,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
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
        self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_avgpool_bn.no_init=True
        self.b_avgpool_bn.weight.data.zero_()
        self.b_avgpool_bn.bias.data.zero_()
    
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
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x
        elif self.transformation == "simple_block":
            x = self.a(x, self.a_rf(x))
            if hasattr(self, "striding"):
                x = self.a_bn(self.striding(x)) + self.a_avgpool_bn(self.a_avgpool(x))
            else:
                x = self.a_bn(x) + self.a_avgpool_bn(self.a_avgpool(x))
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolFull(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolFull, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.a = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.a_rf = route_func_mlp_with_global_info(
            c_in=self.dim_in,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
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
            self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_avgpool_bn.no_init=True
            self.b_avgpool_bn.weight.data.zero_()
            self.b_avgpool_bn.bias.data.zero_()
        
        self.b_relu = nn.ReLU(inplace=True)

        self.c = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.c_rf = route_func_mlp_with_global_info(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x, self.a_rf(x))
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x, self.c_rf(x))
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolNotPatched(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolNotPatched, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_not_patched(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolNotPatchedV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolNotPatchedV2, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_not_patched_v2(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolNorm(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolNorm, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_norm(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveAvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_kernel_size(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoOnlyAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoOnlyAvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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
            x = self.b_bn(self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMaxPoolDropout(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMaxPoolDropout, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_dropout(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            dropout_ratio=self.cfg.VIDEO.BACKBONE.BRANCH.DROPOUT_RATIO,
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
        if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
            self.b_maxpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_maxpool_bn.no_init=True
            self.b_maxpool_bn.weight.data.zero_()
            self.b_maxpool_bn.bias.data.zero_()

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
                x = self.b_bn(x) + self.b_maxpool_bn(self.b_maxpool(x))
            else:
                x = self.b_bn(x + self.b_maxpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMixPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMixPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x) + self.b_maxpool_bn(self.b_maxpool(x)))
            else:
                x = self.b_bn(x + self.b_avgpool(x) + self.b_maxpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMixPoolCorrected(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMixPoolCorrected, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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
class TemporalAdaptive3DConvBlockCinAdaptiveWithGlobalInfoMixPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptive3DConvBlockCinAdaptiveWithGlobalInfoMixPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptive3DConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]],
            stride          = [self.stride[0], self.stride[1], self.stride[2]],
            padding         = [self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
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
                x = self.b_bn(x + self.b_avgpool(x) + self.b_maxpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoSpatialAdaptive(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoSpatialAdaptive, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptiveSpatialAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_spatial_adaptive(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            num_spatial_weight=self.kernel_size[1]*self.kernel_size[2]
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
class TemporalAdaptiveSpatialConvBlockSpatialAdaptiveWithGlobalInfoAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockSpatialAdaptiveWithGlobalInfoAvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvSpatialAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_pure_spatial_adaptive(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            num_spatial_weight=self.kernel_size[1]*self.kernel_size[2]
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x) + self.b_avgpool(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoSpatialAdaptiveMixPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoSpatialAdaptiveMixPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptiveSpatialAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_spatial_adaptive(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            num_spatial_weight=self.kernel_size[1]*self.kernel_size[2]
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x) + self.b_maxpool_bn(self.b_maxpool(x)))
            else:
                x = self.b_bn(x + self.b_avgpool(x) + self.b_maxpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoSpatialAdaptiveAvgPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoSpatialAdaptiveAvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptiveSpatialAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_spatial_adaptive(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            num_spatial_weight=self.kernel_size[1]*self.kernel_size[2]
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x) + self.b_avgpool(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x


@BRANCH_REGISTRY.register()
class AvgPoolBlock(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(AvgPoolBlock, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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

            x = self.b(x)
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class OnlyAvgPoolBlock(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(OnlyAvgPoolBlock, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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

            x = self.b(x)
            x = self.b_bn(self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMixPoolSE(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMixPoolSE, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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
        if ((self.block_id+1)%2) and self.cfg.VIDEO.BACKBONE.BRANCH.SE_RATIO > 0.0:
            self.se = SE(self.num_filters, self.cfg.VIDEO.BACKBONE.BRANCH.SE_RATIO)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x) + self.b_maxpool_bn(self.b_maxpool(x)))
            else:
                x = self.b_bn(x + self.b_avgpool(x) + self.b_maxpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            if hasattr(self, "se"):
                x = self.se(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMixPoolCorrectedSE(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoMixPoolCorrectedSE, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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
        if ((self.block_id+1)%2) and self.cfg.VIDEO.BACKBONE.BRANCH.SE_RATIO > 0.0:
            self.se = SE(self.num_filters, self.cfg.VIDEO.BACKBONE.BRANCH.SE_RATIO)
    
    def forward(self, x):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x)) + self.b_maxpool_bn(self.b_maxpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x) + self.b_maxpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            if hasattr(self, "se"):
                x = self.se(x)
            return x


# -------- tadav2 -------
class route_func_mlp_with_global_info_sigmoid(nn.Module):

    def __init__(self, c_in, ratio, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_sigmoid, self).__init__()
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
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()

        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        x = self.sigmoid(self.b(x)) + 0.5
        return x

class TAda2D_CinAda(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1, meta_arch="ResNet3D", num_frames=8):
        super(TAda2D_CinAda, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts
        self.meta_arch = meta_arch
        self.num_frames = num_frames

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight, with_bias=False):
        _, _, c_out, c_in, kh, kw = self.weight.size()
        if self.meta_arch == "ResNet3D":
            b, c_in, t, h, w = x.size()
            x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        else:
            b_t, c_in, h, w = x.size()
            b = b_t // self.num_frames
            t = self.num_frames
            x = x.reshape(1, -1, h, w)
        if with_bias:
            weight = (routing_weight[0].permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
            bias = routing_weight[1].permute(0,2,1,3,4).reshape(-1)
        else:
            weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
            bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
                dilation=self.dilation[1:], groups=self.groups * b * t)

        if self.meta_arch == "ResNet3D":
            output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        elif self.meta_arch == "ResNet2D":
            output = output.view(b_t, c_out, output.size(-2), output.size(-1))
        return output

@BRANCH_REGISTRY.register()
class TAda2DBlock_CinAda_AvgPool_Sigmoid(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TAda2DBlock_CinAda_AvgPool_Sigmoid, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TAda2D_CinAda(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_sigmoid(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=4,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSize(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSize, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizePoolBN(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizePoolBN, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_poolbn(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeBNPool(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeBNPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_bnpool(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizePoolBNReLU(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizePoolBNReLU, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_poolbnrelu(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveLearnable(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveLearnable, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = nn.Parameter(torch.ones(1,self.num_filters//self.expansion_ratio, self.cfg.DATA.NUM_INPUT_FRAMES, 1, 1))
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

            x = self.b(x, self.b_rf.repeat(x.shape[0],1,1,1,1))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x


@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveLearnableShared(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveLearnableShared, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = nn.Parameter(torch.ones(1,self.num_filters//self.expansion_ratio, 1, 1, 1))
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

            x = self.b(x, self.b_rf.repeat(x.shape[0],1,x.shape[2],1,1))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariant(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariant, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_temporal_invariant(
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

            x = self.b(x, self.b_rf(x).repeat(1,1,x.shape[2],1,1))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveVariant(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveVariant, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_temporal_variant(
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantLocalGeneration(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantLocalGeneration, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_temporal_invariant_local_generation(
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

            x = self.b(x, self.b_rf(x).repeat(1,1,x.shape[2],1,1))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantLocalGenerationRandInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantLocalGenerationRandInit, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_temporal_invariant_local_generation_randinit(
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

            x = self.b(x, self.b_rf(x).repeat(1,1,x.shape[2],1,1))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantLocalGlobalGenerationRandInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantLocalGlobalGenerationRandInit, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_temporal_invariant_localglobal_generation_randinit(
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantLocalGlobalGeneration(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantLocalGlobalGeneration, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_temporal_invariant_localglobal_generation(
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveV2RandInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveV2RandInit, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_kernel_size_randinit(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_KERNEL_SIZE,
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantRandInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveInvariantRandInit, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_temporal_invariant_randinit(
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

            x = self.b(x, self.b_rf(x).repeat(1,1,x.shape[2],1,1))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveVariantRandInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveVariantRandInit, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_temporal_variant_randinit(
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
class TemporalAdaptiveSpatialConvBlockCinAdaptiveLearnableRandInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveLearnableRandInit, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = nn.Parameter(torch.randn(1,self.num_filters//self.expansion_ratio, self.cfg.DATA.NUM_INPUT_FRAMES, 1, 1))
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

            x = self.b(x, self.b_rf.repeat(x.shape[0],1,1,1,1))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x


@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveLearnableSharedRandInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveLearnableSharedRandInit, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = nn.Parameter(torch.randn(1,self.num_filters//self.expansion_ratio, 1, 1, 1))
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

            x = self.b(x, self.b_rf.repeat(x.shape[0],1,x.shape[2],1,1))
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoRandInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoRandInit, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_randinit(
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

class route_func_mlp_with_global_info_evolving(nn.Module):

    def __init__(self, c_in, num_frames, ratio, prev_linear=False, bn_eps=1e-5, bn_mmt=0.1):
        super(route_func_mlp_with_global_info_evolving, self).__init__()
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
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()

        if prev_linear:
            self.prev_linear = nn.Conv3d(
                in_channels=c_in//2,
                out_channels=c_in,
                kernel_size=1,
                padding=0,
            )
            self.prev_linear.no_init = True
            self.prev_linear.weight.data = self.prev_linear.weight.data.softmax(1)
            self.prev_linear.bias.data.zero_()
        

    def forward(self, x, cal_prev=None):
        global_x = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(global_x))
        x = self.bn(x)
        x = self.relu(x)
        if cal_prev == None:
            x = self.b(x) + 1
        else:
            if hasattr(self, "prev_linear"):
                x = self.b(x) + self.prev_linear(cal_prev)
            else:
                x = self.b(x) + cal_prev
        return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolEvolving(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolEvolving, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_evolving(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            prev_linear=self.block_id==0 and self.stage_id > 1,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
    
    def forward(self, x, cal_prev=None):
        if  self.transformation == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            cal = self.b_rf(x, cal_prev)
            x = self.b(x, cal)
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
        return x, cal

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeAct(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeAct, self).__init__(cfg, block_idx, construct_branch=False)
        self.act = cfg.VIDEO.BACKBONE.ACTIVATION
        self._construct_branch()
    
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
        if self.act == "ReLU":
            self.a_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.a_gelu = nn.GELU()

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
            self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_avgpool_bn.no_init=True
            self.b_avgpool_bn.weight.data.zero_()
            self.b_avgpool_bn.bias.data.zero_()
        
        if self.act == "ReLU":
            self.b_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.b_gelu = nn.GELU()

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
            if self.act == "ReLU":
                x = self.a_relu(x)
            elif self.act == "GELU":
                x = self.a_gelu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            if self.act == "ReLU":
                x = self.b_relu(x)
            elif self.act == "GELU":
                x = self.b_gelu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGELURF(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGELURF, self).__init__(cfg, block_idx, construct_branch=False)
        self.act = cfg.VIDEO.BACKBONE.ACTIVATION
        self._construct_branch()
    
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
        if self.act == "ReLU":
            self.a_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.a_gelu = nn.GELU()

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_gelu(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
            self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_avgpool_bn.no_init=True
            self.b_avgpool_bn.weight.data.zero_()
            self.b_avgpool_bn.bias.data.zero_()
        
        if self.act == "ReLU":
            self.b_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.b_gelu = nn.GELU()

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
            if self.act == "ReLU":
                x = self.a_relu(x)
            elif self.act == "GELU":
                x = self.a_gelu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            if self.act == "ReLU":
                x = self.b_relu(x)
            elif self.act == "GELU":
                x = self.b_gelu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGN(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGN, self).__init__(cfg, block_idx, construct_branch=False)
        self.act = cfg.VIDEO.BACKBONE.ACTIVATION
        self._construct_branch()
    
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
        if self.act == "ReLU":
            self.a_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.a_gelu = nn.GELU()

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_groupnorm(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
            self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_avgpool_bn.no_init=True
            self.b_avgpool_bn.weight.data.zero_()
            self.b_avgpool_bn.bias.data.zero_()
        
        if self.act == "ReLU":
            self.b_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.b_gelu = nn.GELU()

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
            if self.act == "ReLU":
                x = self.a_relu(x)
            elif self.act == "GELU":
                x = self.a_gelu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            if self.act == "ReLU":
                x = self.b_relu(x)
            elif self.act == "GELU":
                x = self.b_gelu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGELURFInstanceNorm(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGELURFInstanceNorm, self).__init__(cfg, block_idx, construct_branch=False)
        self.act = cfg.VIDEO.BACKBONE.ACTIVATION
        self._construct_branch()
    
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
        if self.act == "ReLU":
            self.a_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.a_gelu = nn.GELU()

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_gelu_instancenorm(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
            self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_avgpool_bn.no_init=True
            self.b_avgpool_bn.weight.data.zero_()
            self.b_avgpool_bn.bias.data.zero_()
        
        if self.act == "ReLU":
            self.b_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.b_gelu = nn.GELU()

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
            if self.act == "ReLU":
                x = self.a_relu(x)
            elif self.act == "GELU":
                x = self.a_gelu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            if self.act == "ReLU":
                x = self.b_relu(x)
            elif self.act == "GELU":
                x = self.b_gelu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGELURFInstanceNormNewInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGELURFInstanceNormNewInit, self).__init__(cfg, block_idx, construct_branch=False)
        self.act = cfg.VIDEO.BACKBONE.ACTIVATION
        self._construct_branch()
    
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
        if self.act == "ReLU":
            self.a_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.a_gelu = nn.GELU()

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_gelu_instancenorm_newinit(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
            self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_avgpool_bn.no_init=True
            self.b_avgpool_bn.weight.data.zero_()
            self.b_avgpool_bn.bias.data.zero_()
        
        if self.act == "ReLU":
            self.b_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.b_gelu = nn.GELU()

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
            if self.act == "ReLU":
                x = self.a_relu(x)
            elif self.act == "GELU":
                x = self.a_gelu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            if self.act == "ReLU":
                x = self.b_relu(x)
            elif self.act == "GELU":
                x = self.b_gelu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGELURFBatchNormNewInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeActGELURFBatchNormNewInit, self).__init__(cfg, block_idx, construct_branch=False)
        self.act = cfg.VIDEO.BACKBONE.ACTIVATION
        self._construct_branch()
    
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
        if self.act == "ReLU":
            self.a_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.a_gelu = nn.GELU()

        self.b = TemporalAdaptiveSpatialConvCinAdaptive(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_gelu_batchnorm_newinit(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
            self.b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_avgpool_bn.no_init=True
            self.b_avgpool_bn.weight.data.zero_()
            self.b_avgpool_bn.bias.data.zero_()
        
        if self.act == "ReLU":
            self.b_relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.b_gelu = nn.GELU()

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
            if self.act == "ReLU":
                x = self.a_relu(x)
            elif self.act == "GELU":
                x = self.a_gelu(x)

            x = self.b(x, self.b_rf(x))
            if self.cfg.VIDEO.BACKBONE.BRANCH.SEPARATE_BN:
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            if self.act == "ReLU":
                x = self.b_relu(x)
            elif self.act == "GELU":
                x = self.b_gelu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeAddCalibration(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(TemporalAdaptiveSpatialConvBlockCinAdaptiveWithGlobalInfoAvgPoolWithKernelSizeAddCalibration, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
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

        self.b = TemporalAdaptiveSpatialConvCinAdaptiveAdd(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_rf = route_func_mlp_with_global_info_kernel_size_add(
            c_in=self.num_filters//self.expansion_ratio,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            ratio=4,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.TADA_KERNEL_SIZE,
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
                if hasattr(self, "striding"):
                    x = self.b_bn(self.striding(x)) + self.b_avgpool_bn(self.b_avgpool(x))
                else:
                    x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
            else:
                x = self.b_bn(x + self.b_avgpool(x))
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x