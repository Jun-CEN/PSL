import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.modules.utils import _triple, _pair

from models.base.base_blocks import BRANCH_REGISTRY
from models.base.base_blocks_av import BaseAVBranch

class visual2visual_route_func_mlp_with_g(nn.Module):

    def __init__(
        self, c_in, ratio, kernel_size=[3, 3],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(visual2visual_route_func_mlp_with_g, self).__init__()
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

class audio2visual_route_func_melpool_fpool_mlp_v1(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audio2visual_route_func_melpool_fpool_mlp_v1, self).__init__()
        self.c_in = c_in
        self.pool = nn.AdaptiveAvgPool2d((video_length,1))
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1],
            padding=[kernel_size[0]//2,0],
        )
        self.bn = nn.BatchNorm2d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1],
            padding=[kernel_size[1]//2,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        # x: B, C, F, M
        x = self.pool(x) # -> B, C, T, 1

        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x.unsqueeze(-1) # B, C, T, 1, 1

class audio2visual_route_func_melpool_fpool_mlp_v2_additional_conv(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audio2visual_route_func_melpool_fpool_mlp_v2_additional_conv, self).__init__()
        self.c_in = c_in
        self.proj = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0
        )
        self.pool = nn.AdaptiveAvgPool2d((video_length,1))
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1],
            padding=[kernel_size[0]//2,0],
        )
        self.bn = nn.BatchNorm2d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1],
            padding=[kernel_size[1]//2,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        # x: B, C, F, M
        x = self.proj(x)
        x = self.pool(x) # -> B, C, T, 1

        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x.unsqueeze(-1) # B, C, T, 1, 1

class audio2visual_route_func_melpool_fpool_mlp_v3_conv_downsample(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length, audio_length,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audio2visual_route_func_melpool_fpool_mlp_v3_conv_downsample, self).__init__()
        self.c_in = c_in
        self.video_length = video_length
        self.pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.proj = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.pool = nn.AdaptiveAvgPool2d((None,1))
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1],
            padding=[kernel_size[0]//2,0],
        )
        self.bn = nn.BatchNorm2d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1],
            padding=[kernel_size[1]//2,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        # x: B, C, F, M
        x = F.pad(x, (0,0,0,self.pad_length), "constant", 0)
        x = self.proj(x) # -> B, C, T, M
        x = self.pool(x) # -> B, C, T, 1

        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x.unsqueeze(-1) # B, C, T, 1, 1

class audio2visual_route_func_melpool_fpool_mlp_v4_convbn_downsample(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length, audio_length,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audio2visual_route_func_melpool_fpool_mlp_v4_convbn_downsample, self).__init__()
        self.c_in = c_in
        self.video_length = video_length
        self.pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.proj = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.proj_bn = nn.BatchNorm2d(c_in, eps=bn_eps, momentum=bn_mmt)
        self.proj_relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((None,1))
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1],
            padding=[kernel_size[0]//2,0],
        )
        self.bn = nn.BatchNorm2d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1],
            padding=[kernel_size[1]//2,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        # x: B, C, F, M
        x = F.pad(x, (0,0,0,self.pad_length), "constant", 0)
        x = self.proj(x) # -> B, C, T, M
        x = self.proj_bn(x)
        x = self.proj_relu(x)
        x = self.pool(x) # -> B, C, T, 1

        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x.unsqueeze(-1) # B, C, T, 1, 1

class audiovisual2visual_route_func_melpool_fpool_mlp_v1_early_concat(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audiovisual2visual_route_func_melpool_fpool_mlp_v1_early_concat, self).__init__()
        self.c_in = c_in
        self.audio_pool = nn.AdaptiveAvgPool2d((video_length,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in*2,
            out_channels=int(c_in*2//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in*2//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in*2//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xv, xa):
        # x: B, C, F, M
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        x = torch.cat((xv, xa), dim=1)

        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1

        return x

class audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat, self).__init__()
        self.c_in = c_in
        self.audio_pool = nn.AdaptiveAvgPool2d((video_length,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in*2//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in*2//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xv, xa):
        # x: B, C, F, M
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        xv = self.visual_a(xv)
        xa = self.audio_a(xa)

        x = torch.cat((xv, xa), dim=1)

        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1

        return x

class audiovisual2visual_route_func_melpool_fpool_mlp_v3_mid_concat_glu(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audiovisual2visual_route_func_melpool_fpool_mlp_v3_mid_concat_glu, self).__init__()
        self.c_in = c_in
        self.audio_pool = nn.AdaptiveAvgPool2d((video_length,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio)*2,
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in*2//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in*2//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1,1],
            padding=[kernel_size[1]//2,0,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, xv, xa):
        # x: B, C, F, M
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        xv = self.visual_a(xv)
        xa = self.audio_a(xa)
        xv, gate = xv.chunk(2, dim=1)

        x = torch.cat((xv, xa*gate), dim=1)

        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1

        return x

class audiovisual2visual_route_func_melpool_fpool_mlp_v4_mid_concat_convdownsample(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length, audio_length, 
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audiovisual2visual_route_func_melpool_fpool_mlp_v4_mid_concat_convdownsample, self).__init__()
        self.c_in = c_in
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.bn = nn.BatchNorm3d(int(c_in*2//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in*2//ratio),
            out_channels=c_in,
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

        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1

        return x

class audiovisual2visual_route_func_melpool_fpool_mlp_v5_mid_concat_convdownsample_sepbn(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length, audio_length, 
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audiovisual2visual_route_func_melpool_fpool_mlp_v5_mid_concat_convdownsample_sepbn, self).__init__()
        self.c_in = c_in
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.visual_bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.audio_bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in*2//ratio),
            out_channels=c_in,
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
        xv = self.visual_bn(xv)
        xa = self.audio_a(xa)
        xa = self.audio_bn(xa)

        x = torch.cat((xv, xa), dim=1)

        x = self.relu(x)
        x = self.b(x) + 1

        return x

class audiovisual2visual_route_func_melpool_fpool_mlp_v6_mid_add_convdownsample_sepbn(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length, audio_length, 
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audiovisual2visual_route_func_melpool_fpool_mlp_v6_mid_add_convdownsample_sepbn, self).__init__()
        self.c_in = c_in
        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio_proj = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio_pool = nn.AdaptiveAvgPool2d((None,1))
        self.visual_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.visual_globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.visual_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.audio_a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1,1],
            padding=[kernel_size[0]//2,0,0],
        )
        self.visual_bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.audio_bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
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
        

    def forward(self, xv, xa):
        xa = F.pad(xa, (0,0,0,self.audio_pad_length), "constant", 0)
        xa = self.audio_proj(xa)
        xa = self.audio_pool(xa).unsqueeze(-1)
        xv = self.visual_pool(xv)
        xv_global = self.visual_globalpool(xv)
        xv = xv + self.g(xv_global)

        xv = self.visual_a(xv)
        xv = self.visual_bn(xv)
        xa = self.audio_a(xa)
        xa = self.audio_bn(xa)

        x = xv + xa

        x = self.relu(x)
        x = self.b(x) + 1

        return x

class audio2audio_route_func_melpool_fpool_mlp_v1(nn.Module):

    def __init__(
        self, c_in, ratio, 
        video_length,
        kernel_size=[1, 1],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(audio2audio_route_func_melpool_fpool_mlp_v1, self).__init__()
        self.c_in = c_in
        self.pool = nn.AdaptiveAvgPool2d((video_length,1))
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernel_size[0],1],
            padding=[kernel_size[0]//2,0],
        )
        self.bn = nn.BatchNorm2d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernel_size[1],1],
            padding=[kernel_size[1]//2,0],
            bias=False
        )
        self.b.no_init=True
        self.b.weight.data.zero_()
        

    def forward(self, x):
        # x: B, C, F, M
        x = self.pool(x) # -> B, C, T, 1

        x = self.a(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x.unsqueeze(-1) # B, C, T, 1, 1

class visual2audio_route_func_mlp_with_g(nn.Module):

    def __init__(
        self, c_in, ratio, kernel_size=[3, 3],
        bn_eps=1e-5, bn_mmt=0.1
    ):
        super(visual2audio_route_func_mlp_with_g, self).__init__()
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

class TAdaConv2D_CinAda_AV(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 modality='v', num_frames=8):
        """
        modality (str): 'v' or 'a' representing the convolution is performed on
            video or audio features.
        """
        super(TAdaConv2D_CinAda_AV, self).__init__()

        if modality == "v":
            kernel_size = _triple(kernel_size)
            stride = _triple(stride)
            padding = _triple(padding)
            dilation = _triple(dilation)

            assert kernel_size[0] == 1
            assert stride[0] == 1
            assert padding[0] == 0
            assert dilation[0] == 1
        else:
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_frames = num_frames

        assert modality in ['v', 'a']
        self.modality = modality

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[-2], kernel_size[-1]))
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
        _, _, c_out, c_in, kh, kw = self.weight.size()
        if self.modality == "v":
            b, c_in, t, h, w = x.size()
            x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)
        elif self.modality == "a":
            b, c_in, f, m = x.size()
            t = routing_weight.shape[2]
            pad_length = math.ceil(f/t)*t-f
            x_pad = F.pad(x, (0,0,0,pad_length), "constant", 0)
            padded_f = x_pad.shape[2]
            x = x_pad.unfold(dimension=2, size=padded_f//t, step=padded_f//t).permute(0,2,1,4,3).reshape(1, -1, padded_f//t, m)

        weight = (routing_weight.permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in, kh, kw)
        bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(
                x, weight=weight, bias=bias, stride=self.stride[-2:], padding=self.padding[-2:],
                dilation=self.dilation[-2:], groups=self.groups * b * t)

        if self.modality == "v":
            output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)
        elif self.modality == "a":
            output_f = int((f-self.kernel_size[0]+2*self.padding[0])/self.stride[0] + 1)
            output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4).reshape(b,c_out,-1,m//self.stride[-1])[
                :,:,:output_f,:
            ]
        return output

@BRANCH_REGISTRY.register()
class BaselineAVBlock(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(BaselineAVBlock, self).__init__(cfg, block_idx, construct_branch=False)

        self.video_branch_enable = cfg.VIDEO.ENABLE
        self.audio_branch_enable = cfg.AUDIO.ENABLE

        self._construct_branch()
    
    def _construct_bottleneck(self):
        if self.video_branch_enable:
            self.visual_a = nn.Conv3d(
                in_channels     = self.dim_in,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.visual_a_relu = nn.ReLU(inplace=True)

            self.visual_b = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
                stride          = [1, self.visual_stride[1], self.visual_stride[2]],
                padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
                bias            = False
            )
            self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.visual_b_relu = nn.ReLU(inplace=True)

            self.visual_c = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        if self.audio_branch_enable:
            self.audio_a = nn.Conv2d(
                in_channels     = self.dim_in,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_a_relu = nn.ReLU(inplace=True)

            self.audio_b = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride          = [self.audio_stride[0], self.audio_stride[1]],
                padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias            = False
            )
            self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_b_relu = nn.ReLU(inplace=True)

            self.audio_c = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
class TAdaConvBlock_CinAda_VTAda_SelfRoute_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_SelfRoute_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.visual2visual_rf = visual2visual_route_func_mlp_with_g(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xv = self.visual_b(xv, self.visual2visual_rf(xv))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_SelfRoute_ANormal_AvgPool(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_SelfRoute_ANormal_AvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.visual2visual_rf = visual2visual_route_func_mlp_with_g(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
class TAdaConvBlock_CinAda_VTAda_CrossRoute_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_CrossRoute_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audio2visual_rf = audio2visual_route_func_melpool_fpool_mlp_v1(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audio2visual_rf(xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_CrossRouteV2_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_CrossRouteV2_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audio2visual_rf = audio2visual_route_func_melpool_fpool_mlp_v2_additional_conv(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audio2visual_rf(xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_CrossRouteV3_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_CrossRouteV3_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audio2visual_rf = audio2visual_route_func_melpool_fpool_mlp_v3_conv_downsample(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audio2visual_rf(xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_CrossRouteV3Prime_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_CrossRouteV3Prime_ANormal, self).__init__(cfg, block_idx, construct_branch=False)
        self.last = last
        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audio2visual_rf = audio2visual_route_func_melpool_fpool_mlp_v3_conv_downsample(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
            self.audio_b = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride          = [self.audio_stride[0], self.audio_stride[1]],
                padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias            = False
            )
            self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_b_relu = nn.ReLU(inplace=True)

            self.audio_c = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audio2visual_rf(xa))
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
                xa = self.audio_b(xa)
                xa = self.audio_b_bn(xa)
                xa = self.audio_b_relu(xa)
                xa = self.audio_dropout(xa)

                xa = self.audio_c(xa)
                xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_CrossRouteV4_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_CrossRouteV4_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audio2visual_rf = audio2visual_route_func_melpool_fpool_mlp_v4_convbn_downsample(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audio2visual_rf(xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_CrossRoute_ANormal_AvgPool(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_CrossRoute_ANormal_AvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audio2visual_rf = audio2visual_route_func_melpool_fpool_mlp_v1(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audio2visual_rf(xa))
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
class TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audio2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v1_early_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audio2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal_V2_MidConcat(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal_V2_MidConcat, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_JointRouteV2Prime_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteV2Prime_ANormal, self).__init__(cfg, block_idx, construct_branch=False)
        self.last = last
        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
            self.audio_b = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride          = [self.audio_stride[0], self.audio_stride[1]],
                padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias            = False
            )
            self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_b_relu = nn.ReLU(inplace=True)

            self.audio_c = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
                xa = self.audio_b(xa)
                xa = self.audio_b_bn(xa)
                xa = self.audio_b_relu(xa)
                xa = self.audio_dropout(xa)

                xa = self.audio_c(xa)
                xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRouteV2Prime_ANormal_AvgPool(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteV2Prime_ANormal_AvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.last = last
        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
            self.audio_b = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride          = [self.audio_stride[0], self.audio_stride[1]],
                padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias            = False
            )
            self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_b_relu = nn.ReLU(inplace=True)

            self.audio_c = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
                xa = self.audio_b(xa)
                xa = self.audio_b_bn(xa)
                xa = self.audio_b_relu(xa)
                xa = self.audio_dropout(xa)

                xa = self.audio_c(xa)
                xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal_V2_MidConcat_AvgPool(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal_V2_MidConcat_AvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
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
class TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal_V2_MidConcat_AVAvgPool(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal_V2_MidConcat_AVAvgPool, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        # audio to video fusion
        video_length = self.cfg.DATA.NUM_INPUT_FRAMES
        audio_length = self.audio_t

        self.audio_pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
        self.audio2video_proj = nn.Conv2d(
            in_channels=self.num_filters//self.expansion_ratio,
            out_channels=self.num_filters//self.expansion_ratio,
            kernel_size=[math.ceil(audio_length/video_length), 1],
            stride=[math.ceil(audio_length/video_length), 1],
            padding=0
        )
        self.audio2video_avgpool = nn.AdaptiveAvgPool2d((None,1))
        self.audio2video_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio2video_bn.no_init=True
        self.audio2video_bn.weight.data.zero_()
        self.audio2video_bn.bias.data.zero_()

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            x_a2v = F.pad(xa, (0,0,0,self.audio_pad_length), "constant", 0)
            x_a2v = self.audio2video_avgpool(self.audio2video_proj(x_a2v))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv)) + self.audio2video_bn(x_a2v.unsqueeze(-1))
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
class TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal_V3_MidConcat_GLU(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRoute_ANormal_V3_MidConcat_GLU, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v3_mid_concat_glu(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_JointRouteV4_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteV4_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v4_mid_concat_convdownsample(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_JointRouteV5_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteV5_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v5_mid_concat_convdownsample_sepbn(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_JointRouteV6_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteV6_ANormal, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v6_mid_add_convdownsample_sepbn(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv)
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
class TAdaConvBlock_CinAda_VTAda_JointRouteV6Prime_ANormal(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteV6Prime_ANormal, self).__init__(cfg, block_idx, construct_branch=False)
        self.last = last

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v6_mid_add_convdownsample_sepbn(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
            self.audio_b = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride          = [self.audio_stride[0], self.audio_stride[1]],
                padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias            = False
            )
            self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_b_relu = nn.ReLU(inplace=True)

            self.audio_c = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
                xa = self.audio_b(xa)
                xa = self.audio_b_bn(xa)
                xa = self.audio_b_relu(xa)
                xa = self.audio_dropout(xa)

                xa = self.audio_c(xa)
                xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRouteV6Prime_ANormal_AvgPool(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteV6Prime_ANormal_AvgPool, self).__init__(cfg, block_idx, construct_branch=False)
        self.last = last

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v6_mid_add_convdownsample_sepbn(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            audio_length=self.audio_t,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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
        self.visual_b_avgpool_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_avgpool_bn.no_init=True
        self.visual_b_avgpool_bn.weight.data.zero_()
        self.visual_b_avgpool_bn.bias.data.zero_()
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
            self.audio_b = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride          = [self.audio_stride[0], self.audio_stride[1]],
                padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias            = False
            )
            self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_b_relu = nn.ReLU(inplace=True)

            self.audio_c = nn.Conv2d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )
            self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

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
            xv = self.visual_b(xv, self.audiovisual2visual_rf(xv, xa))
            xv = self.visual_b_bn(xv) + self.visual_b_avgpool_bn(self.visual_b_avgpool(xv))
            xv = self.visual_b_relu(xv)

            # ---- conv c
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            if not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
                xa = self.audio_b(xa)
                xa = self.audio_b_bn(xa)
                xa = self.audio_b_relu(xa)
                xa = self.audio_dropout(xa)

                xa = self.audio_c(xa)
                xa = self.audio_c_bn(xa)

            return xv, xa



@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VNormal_ATAda_SelfRoute(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VNormal_ATAda_SelfRoute, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a"
        )
        self.audio2audio_rf = audio2audio_route_func_melpool_fpool_mlp_v1(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xv = self.visual_b(xv)
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            xa = self.audio_b(xa, self.audio2audio_rf(xa))
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)

            xa = self.audio_dropout(xa)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VNormal_ATAda_CrossRoute(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VNormal_ATAda_CrossRoute, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a"
        )
        self.visual2audio_rf = visual2audio_route_func_mlp_with_g(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_b(xa, self.visual2audio_rf(xv))
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)

            xv = self.visual_b(xv)
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)

            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)
            xa = self.audio_dropout(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VNormal_ATAda_JointRoute(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VNormal_ATAda_JointRoute, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a"
        )
        #  audiovisual 2 visual is identical to audiovisual 2 audio
        self.audiovisual2audio_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v1_early_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_b(xa, self.audiovisual2audio_rf(xv, xa))
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)

            xv = self.visual_b(xv)
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)
            
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)
            xa = self.audio_dropout(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VNormal_ATAda_JointRoute_V2_MidConcat(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VNormal_ATAda_JointRoute_V2_MidConcat, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a"
        )
        #  audiovisual 2 visual is identical to audiovisual 2 audio
        self.audiovisual2audio_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xa = self.audio_b(xa, self.audiovisual2audio_rf(xv, xa))
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)

            xv = self.visual_b(xv)
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)
            
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)
            xa = self.audio_dropout(xa)

            return xv, xa


# ----------------------------------- VTAda ATAda -----------------------------------
# ----------------------------------- VTAda ATAda -----------------------------------
# ----------------------------------- VTAda ATAda -----------------------------------
# ----------------------------------- VTAda ATAda -----------------------------------

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRoute_ATAda_JointRoute(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRoute_ATAda_JointRoute, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v1_early_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a"
        )
        #  audiovisual 2 visual is identical to audiovisual 2 audio
        self.audiovisual2audio_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v1_early_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xv_rf = self.audiovisual2visual_rf(xv, xa)
            xa_rf = self.audiovisual2audio_rf(xv, xa)

            xa = self.audio_b(xa, xa_rf)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)

            xv = self.visual_b(xv, xv_rf)
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)
            
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)
            xa = self.audio_dropout(xa)

            return xv, xa

@BRANCH_REGISTRY.register()
class TAdaConvBlock_CinAda_VTAda_JointRouteV2_ATAda_JointRouteV2(BaseAVBranch):
    def __init__(self, cfg, block_idx, last=False):
        super(TAdaConvBlock_CinAda_VTAda_JointRouteV2_ATAda_JointRouteV2, self).__init__(cfg, block_idx, construct_branch=False)

        self._construct_branch()
    
    def _construct_bottleneck(self):
        self.visual_a = nn.Conv3d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_a_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_a_relu = nn.ReLU(inplace=True)

        self.visual_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
            stride          = [1, self.visual_stride[1], self.visual_stride[2]],
            padding         = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
            bias            = False,
            modality        = "v" # visual tada conv
        )
        self.audiovisual2visual_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.visual_b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.visual_b_relu = nn.ReLU(inplace=True)

        self.visual_c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.visual_c_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_a = nn.Conv2d(
            in_channels     = self.dim_in,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_a_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_a_relu = nn.ReLU(inplace=True)

        self.audio_b = TAdaConv2D_CinAda_AV(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
            stride          = [self.audio_stride[0], self.audio_stride[1]],
            padding         = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
            bias            = False,
            modality        = "a"
        )
        #  audiovisual 2 visual is identical to audiovisual 2 audio
        self.audiovisual2audio_rf = audiovisual2visual_route_func_melpool_fpool_mlp_v2_mid_concat(
            c_in=self.num_filters//self.expansion_ratio,
            ratio=self.cfg.VIDEO.BACKBONE.BRANCH.RATIO,
            video_length=self.cfg.DATA.NUM_INPUT_FRAMES, 
            kernel_size=self.cfg.VIDEO.BACKBONE.BRANCH.RF_KERNEL_SIZE
        )
        self.audio_b_bn = nn.BatchNorm2d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.audio_b_relu = nn.ReLU(inplace=True)

        self.audio_c = nn.Conv2d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False
        )
        self.audio_c_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)

        self.audio_dropout = nn.Dropout(p=self.cfg.AUDIO.DROPOUT_RATE)
    
    def forward(self, xv, xa):
        if  self.transformation == 'bottleneck':
            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)

            xv_rf = self.audiovisual2visual_rf(xv, xa)
            xa_rf = self.audiovisual2audio_rf(xv, xa)

            xa = self.audio_b(xa, xa_rf)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)

            xv = self.visual_b(xv, xv_rf)
            xv = self.visual_b_bn(xv)
            xv = self.visual_b_relu(xv)

            xa = self.audio_c(xa)
            xa = self.audio_c_bn(xa)
            
            xv = self.visual_c(xv)
            xv = self.visual_c_bn(xv)
            xa = self.audio_dropout(xa)

            return xv, xa