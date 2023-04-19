import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from models.base.base_blocks import BRANCH_REGISTRY
from models.base.base_blocks_av import BaseAVBranchV2
from models.module_zoo.branches.tada_av import (
    TAdaConv2D_CinAda_AV,
    visual2visual_route_func_mlp_with_g
)

class a2v_routefunc_audioconvdown(nn.Module):

    def __init__(
        self, visual_c_in, audio_c_in, ratio, 
        video_length, audio_length, 
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(a2v_routefunc_audioconvdown, self).__init__()
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=audio_c_in,
            out_channels=audio_c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.audio_a = nn.Conv3d(
            in_channels=audio_c_in,
            out_channels=int(audio_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a_bn = nn.BatchNorm3d(int(audio_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(audio_c_in//ratio),
            out_channels=visual_c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xa):
        xa = F.pad(xa, (0,0,0,self.audio_pad_length), "constant", 0)
        xa = self.audio_proj(xa)
        xa = self.audio_pool(xa).unsqueeze(-1)

        xa = self.audio_a(xa)
        xa = self.audio_a_bn(xa)
        xa = self.relu(xa)
        xa = self.b(xa) + 1

        return xa

class av2v_routefunc_audioconvdown_jointbn(nn.Module):

    def __init__(
        self, visual_c_in, audio_c_in, ratio, 
        video_length, audio_length, 
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(av2v_routefunc_audioconvdown_jointbn, self).__init__()
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=audio_c_in,
            out_channels=audio_c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=visual_c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=int(visual_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=audio_c_in,
            out_channels=int(audio_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.a_bn = nn.BatchNorm3d(int(audio_c_in//ratio)+int(visual_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(visual_c_in//ratio) + int(audio_c_in//ratio),
            out_channels=visual_c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xv, xa):
        xa = F.pad(xa, (0,0,0,self.audio_pad_length), "constant", 0)
        xa = self.audio_proj(xa)
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        xv = self.visual_a(xv)
        xa = self.audio_a(xa)

        x = torch.cat((xv, xa), dim=1)
        x = self.a_bn(x)
        x = self.relu(x)
        x = self.b(x) + 1

        return x

class av2v_routefunc_audioconvdown(nn.Module):

    def __init__(
        self, visual_c_in, audio_c_in, ratio, 
        video_length, audio_length, 
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(av2v_routefunc_audioconvdown, self).__init__()
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=audio_c_in,
            out_channels=audio_c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=visual_c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=int(visual_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=audio_c_in,
            out_channels=int(audio_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a_bn = nn.BatchNorm3d(int(audio_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.visual_a_bn = nn.BatchNorm3d(int(visual_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(visual_c_in//ratio) + int(audio_c_in//ratio),
            out_channels=visual_c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xv, xa):
        xa = F.pad(xa, (0,0,0,self.audio_pad_length), "constant", 0)
        xa = self.audio_proj(xa)
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        xv = self.visual_a(xv)
        xv = self.visual_a_bn(xv)
        xa = self.audio_a(xa)
        xa = self.audio_a_bn(xa)

        x = torch.cat((xv, xa), dim=1)

        x = self.relu(x)
        x = self.b(x) + 1

        return x

class av2v_routefunc_audioconvdown_dropaudio(nn.Module):

    def __init__(
        self, visual_c_in, audio_c_in, ratio, 
        video_length, audio_length, 
        drop_audio_rate=0.0,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(av2v_routefunc_audioconvdown_dropaudio, self).__init__()
        self.drop_audio_rate = drop_audio_rate
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=audio_c_in,
            out_channels=audio_c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=visual_c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=int(visual_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=audio_c_in,
            out_channels=int(audio_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a_bn = nn.BatchNorm3d(int(audio_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.visual_a_bn = nn.BatchNorm3d(int(visual_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(visual_c_in//ratio) + int(audio_c_in//ratio),
            out_channels=visual_c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xv, xa):
        xa = F.pad(xa, (0,0,0,self.audio_pad_length), "constant", 0)
        xa = self.audio_proj(xa)
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        xv = self.visual_a(xv)
        xv = self.visual_a_bn(xv)
        xa = self.audio_a(xa)
        xa = self.audio_a_bn(xa)

        if self.drop_audio_rate > 0.0:
            drop_audio = (torch.rand(xa.shape[0],1,1,1,1,device=xa.device) > self.drop_audio_rate)*1
            xa = xa * drop_audio

        x = torch.cat((xv, xa), dim=1)

        x = self.relu(x)
        x = self.b(x) + 1

        return x

class av2v_routefunc_audioconvdown_dropaudio_midadd(nn.Module):

    def __init__(
        self, visual_c_in, audio_c_in, ratio, 
        video_length, audio_length, 
        drop_audio_rate=0.0,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(av2v_routefunc_audioconvdown_dropaudio_midadd, self).__init__()
        self.drop_audio_rate = drop_audio_rate
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=audio_c_in,
            out_channels=audio_c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=visual_c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=int(visual_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=audio_c_in,
            out_channels=int(visual_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a_bn = nn.BatchNorm3d(int(visual_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.visual_a_bn = nn.BatchNorm3d(int(visual_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(visual_c_in//ratio),
            out_channels=visual_c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xv, xa):
        xa = F.pad(xa, (0,0,0,self.audio_pad_length), "constant", 0)
        xa = self.audio_proj(xa)
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        xv = self.visual_a(xv)
        xv = self.visual_a_bn(xv)
        xa = self.audio_a(xa)
        xa = self.audio_a_bn(xa)

        if self.drop_audio_rate > 0.0:
            drop_audio = (torch.rand(xa.shape[0],1,1,1,1,device=xa.device) > self.drop_audio_rate)*1
            xa = xa * drop_audio

        x = xv+xa

        x = self.relu(x)
        x = self.b(x) + 1

        return x

class av2a_routefunc_audioconvdown(nn.Module):

    def __init__(
        self, visual_c_in, audio_c_in, ratio, 
        video_length, audio_length, 
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(av2a_routefunc_audioconvdown, self).__init__()
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=audio_c_in,
            out_channels=audio_c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=visual_c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=visual_c_in,
            out_channels=int(visual_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=audio_c_in,
            out_channels=int(audio_c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a_bn = nn.BatchNorm3d(int(audio_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.visual_a_bn = nn.BatchNorm3d(int(visual_c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(visual_c_in//ratio) + int(audio_c_in//ratio),
            out_channels=audio_c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xv, xa):
        xa = F.pad(xa, (0,0,0,self.audio_pad_length), "constant", 0)
        xa = self.audio_proj(xa)
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        xv = self.visual_a(xv)
        xv = self.visual_a_bn(xv)
        xa = self.audio_a(xa)
        xa = self.audio_a_bn(xa)

        x = torch.cat((xv, xa), dim=1)

        x = self.relu(x)
        x = self.b(x) + 1

        return x

@BRANCH_REGISTRY.register()
class BaselineAVBlock_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(BaselineAVBlock_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self.video_branch_enable = cfg.VIDEO.ENABLE
        self.audio_branch_enable = cfg.AUDIO.ENABLE

        self._construct_branch()
    
    def _construct_bottleneck(self):
        if self.video_branch_enable:
            self.visual_a = nn.Conv3d(
                in_channels     = self.visual_dim_in,
                out_channels    = self.visual_num_filters//self.expansion_ratio,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.visual_a_relu = nn.ReLU(inplace=True)

            self.visual_b = nn.Conv3d(
                in_channels     = self.visual_num_filters//self.expansion_ratio,
                out_channels    = self.visual_num_filters//self.expansion_ratio,
                kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
                stride          = [1, self.visual_stride[1], self.visual_stride[2]],
                padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
                bias            = False
            )
            self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.visual_b_relu = nn.ReLU(inplace=True)

            self.visual_c = nn.Conv3d(
                in_channels     = self.visual_num_filters//self.expansion_ratio,
                out_channels    = self.visual_num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        if self.audio_branch_enable:
            self.audio_a = nn.Conv2d(
                in_channels     = self.audio_dim_in,
                out_channels    = self.audio_num_filters//self.expansion_ratio,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_a_relu = nn.ReLU(inplace=True)

            self.audio_b = nn.Conv2d(
                in_channels     = self.audio_num_filters//self.expansion_ratio,
                out_channels    = self.audio_num_filters//self.expansion_ratio,
                kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride          = [self.audio_stride[0], self.audio_stride[1]],
                padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias            = False
            )
            self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_b_relu = nn.ReLU(inplace=True)

            self.audio_c = nn.Conv2d(
                in_channels     = self.audio_num_filters//self.expansion_ratio,
                out_channels    = self.audio_num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

            self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            if self.video_branch_enable:
                xv = self.visual_a(xv)
                xv = self.visual_a_bn(xv)
                xv = self.visual_a_relu(xv)

                xv = self.visual_b(xv)
                xv = self.visual_b_bn(xv)
                xv = self.visual_b_relu(xv)

                xv = self.visual_c(xv)
                xv = self.visual_c_bn(xv)
            
            if self.audio_branch_enable:

                xa = self.audio_a(xa)
                xa = self.audio_a_bn(xa)
                xa = self.audio_a_relu(xa)

                xa = self.audio_b(xa)
                xa = self.audio_b_bn(xa)
                xa = self.audio_b_relu(xa)

                xa = self.audio_dropout(xa)

                xa = self.audio_c(xa)
                xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_SelfRoute_AvgPool_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_SelfRoute_AvgPool_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.visual2visual_rf = visual2visual_route_func_mlp_with_g(
            c_in=self.visual_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool = nn.AvgPool3d(
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xv = self.visual_b(xv, self.visual2visual_rf(xv))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            xa = self.audio_b(xa)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)

            xa = self.audio_dropout(xa)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_CrossRoute_AvgPool_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_CrossRoute_AvgPool_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.a2v_rf = a2v_routefunc_audioconvdown(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool = nn.AvgPool3d(
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)

    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            # ---- conv a
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            # ---- conv b
            xv = self.visual_b(xv, self.a2v_rf(xa))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            xa = self.audio_b(xa)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)
            xa = self.audio_dropout(xa)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRouteMidCat_AvgPool_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteMidCat_AvgPool_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.av2v_rf = av2v_routefunc_audioconvdown_jointbn(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool = nn.AvgPool3d(
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)

    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            # ---- conv a
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            # ---- conv b
            xv = self.visual_b(xv, self.av2v_rf(xv, xa))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            xa = self.audio_b(xa)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)
            xa = self.audio_dropout(xa)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa


@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRouteMidCatSepBN_AvgPool_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteMidCatSepBN_AvgPool_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.av2v_rf = av2v_routefunc_audioconvdown(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool = nn.AvgPool3d(
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)

    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            # ---- conv a
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            # ---- conv b
            xv = self.visual_b(xv, self.av2v_rf(xv, xa))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            xa = self.audio_b(xa)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)
            xa = self.audio_dropout(xa)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRouteMidCatSepBN_AvgPool_AudioDrop_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteMidCatSepBN_AvgPool_AudioDrop_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.av2v_rf = av2v_routefunc_audioconvdown_dropaudio(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            drop_audio_rate=self.cfg.VIDEO.BACKBONE.BRANCH.DROP_AUDIO_RATE,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool = nn.AvgPool3d(
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)

    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            # ---- conv a
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            # ---- conv b
            xv = self.visual_b(xv, self.av2v_rf(xv, xa))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            xa = self.audio_b(xa)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)
            xa = self.audio_dropout(xa)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRouteMidAddSepBN_AvgPool_AudioDrop_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteMidAddSepBN_AvgPool_AudioDrop_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.av2v_rf = av2v_routefunc_audioconvdown_dropaudio_midadd(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            drop_audio_rate=self.cfg.VIDEO.BACKBONE.BRANCH.DROP_AUDIO_RATE,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool = nn.AvgPool3d(
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)

    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            # ---- conv a
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            # ---- conv b
            xv = self.visual_b(xv, self.av2v_rf(xv, xa))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            xa = self.audio_b(xa)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)
            xa = self.audio_dropout(xa)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_ATAda_JointRouteMidCatSepBN_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_ATAda_JointRouteMidCatSepBN_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a" # visual tada conv
        )
        self.av2a_rf = av2a_routefunc_audioconvdown(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)

    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            # ---- conv a
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            # ---- conv b
            xa = self.audio_b(xa, self.av2a_rf(xv, xa))
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)
            xa = self.audio_dropout(xa)

            xv = self.visual_b(xv)
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_ATAda_JointRouteMidCatSepBN_AvgPool_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_ATAda_JointRouteMidCatSepBN_AvgPool_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_avgpool = nn.AvgPool2d(
            kernel_size=[
                self.cfg.AUDIO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0],
                self.cfg.AUDIO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1],
            ],
            stride=1,
            padding=[
                self.cfg.AUDIO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[0]//2,
                self.cfg.AUDIO.BACKBONE.BRANCH.POOL_KERNEL_SIZE[1]//2,
            ],
        )
        self.audio_b_avgpool_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_avgpool_bn.no_init=True
        self.audio_b_avgpool_bn.weight.data.zero_()
        self.audio_b_avgpool_bn.bias.data.zero_()
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a" # visual tada conv
        )
        self.av2a_rf = av2a_routefunc_audioconvdown(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)

    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            # ---- conv a
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            # ---- conv b
            xa = self.audio_b(xa, self.av2a_rf(xv, xa))
            xa = self.audio_b_bn(xa) + self.audio_b_avgpool_bn(self.audio_b_avgpool(xa))
            xa = self.audio_b_relu(xa)
            xa = self.audio_dropout(xa)

            xv = self.visual_b(xv)
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_AVTAda_JointRouteMidCatSepBN_AvgPool_AVSepChannel(BaseAVBranchV2):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_AVTAda_JointRouteMidCatSepBN_AvgPool_AVSepChannel, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.visual_dim_in,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.av2v_rf = av2v_routefunc_audioconvdown(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool = nn.AvgPool3d(
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.visual_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.visual_num_filters//self.expansion_ratio,
            out_channels    = self.visual_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.audio_dim_in,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a" # visual tada conv
        )
        self.av2a_rf = av2a_routefunc_audioconvdown(
            visual_c_in=self.visual_num_filters//self.expansion_ratio,
            audio_c_in=self.audio_num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.audio_num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.audio_num_filters//self.expansion_ratio,
            out_channels    = self.audio_num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)

    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            # ---- conv a
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            # ---- conv b
            av2v_rf = self.av2v_rf(xv, xa)
            av2a_rf = self.av2a_rf(xv, xa)
            xv = self.visual_b(xv, av2v_rf)
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            xa = self.audio_b(xa, av2a_rf)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)
            xa = self.audio_dropout(xa)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa