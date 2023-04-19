#!/usr/bin/env python3

import math
import torch
import torch.nn as nn

from models.base.base_blocks import BaseBranch
from models.base.base_blocks import BRANCH_REGISTRY, STEM_REGISTRY

@BRANCH_REGISTRY.register()
class MSMBranch(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MSMBranch, self).__init__(cfg, block_idx, construct_branch=False)
        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.patchwise = cfg.VIDEO.BACKBONE.BRANCH.PATCHWISE
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

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

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

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                if self.num_heads > 1:
                    b,c,t,h,w = x.shape
                    x_3_attn = x*x_3
                    x_mix_attn = x*x_mix
                    x_3_attn = x_3_attn.reshape(b, self.num_heads, c//self.num_heads, t, h, w).sum(2,keepdim=True)
                    x_mix_attn = x_mix_attn.reshape(b, self.num_heads, c//self.num_heads, t, h, w).sum(2,keepdim=True)
                    attn = torch.cat((x_3_attn, x_mix_attn), dim=2).softmax(2)
                    x_3 = x_3.reshape(b, self.num_heads, c//self.num_heads, t, h, w) * attn[:,:,0:1,:]
                    x_mix = x_mix.reshape(b, self.num_heads, c//self.num_heads, t, h, w) * attn[:,:,1:2,:]
                    x_3 = x_3.reshape(b, -1, t, h, w)
                    x_mix = x_mix.reshape(b, -1, t, h, w)

                else:
                    x_3_attn = (x*x_3).sum(1, keepdim=True)
                    x_mix_attn = (x*x_mix).sum(1, keepdim=True)
                    attn = torch.cat((x_3_attn, x_mix_attn), dim=1).softmax(1)

                    x_3 = x_3 * attn[:,0:1,:]
                    x_mix = x_mix * attn[:,1:2,:]
                if self.output == "cat":
                    x = torch.cat((x, x_3+x_mix), dim=1)
                elif self.output == "add":
                    x = x + x_3 + x_mix
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class MSMBranchV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MSMBranchV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.patchwise = cfg.VIDEO.BACKBONE.BRANCH.PATCHWISE
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

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
        if self.stride[0] > 1:
            self.avg_pool = nn.AvgPool3d(
                kernel_size = [3, 1, 1],
                stride = [self.stride[0], 1, 1],
                padding = [1, 0, 0]
            )
        
        if self.output == "cat":
            self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio*2, eps=self.bn_eps, momentum=self.bn_mmt)
        elif self.output == "add":
            self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            q = self.b(x)
            x = self.b_bn(q)
            x = self.b_relu(x)

            x_3 = self.b2(x)

            if self.enable: 
                x_mix = self.b2_mix(x)
                if hasattr(self, "avg_pool"):
                    q = self.avg_pool(q)

                if self.num_heads > 1:
                    b,c,t,h,w = q.shape
                    x_3_attn = q*x_3
                    x_mix_attn = q*x_mix
                    x_3_attn = x_3_attn.reshape(b, self.num_heads, c//self.num_heads, t, h, w).sum(2,keepdim=True) * ((c//self.num_heads)**-0.5)
                    x_mix_attn = x_mix_attn.reshape(b, self.num_heads, c//self.num_heads, t, h, w).sum(2,keepdim=True) * ((c//self.num_heads)**-0.5)
                    attn = torch.cat((x_3_attn, x_mix_attn), dim=2).softmax(2)
                    x_3 = x_3.reshape(b, self.num_heads, c//self.num_heads, t, h, w) * attn[:,:,0:1,:]
                    x_mix = x_mix.reshape(b, self.num_heads, c//self.num_heads, t, h, w) * attn[:,:,1:2,:]
                    x_3 = x_3.reshape(b, -1, t, h, w)
                    x_mix = x_mix.reshape(b, -1, t, h, w)

                else:
                    x_3_attn = (q*x_3).sum(1, keepdim=True) * (c**-0.5)
                    x_mix_attn = (q*x_mix).sum(1, keepdim=True) * (c**-0.5)
                    attn = torch.cat((x_3_attn, x_mix_attn), dim=1).softmax(1)

                    x_3 = x_3 * attn[:,0:1,:]
                    x_mix = x_mix * attn[:,1:2,:]
                if self.output == "cat":
                    x = torch.cat((q, x_3+x_mix), dim=1)
                elif self.output == "add":
                    x = q + x_3 + x_mix
                x = self.b2_bn(x)
                x = self.b2_relu(x)
            else:
                raise NotImplementedError

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class MSMBranchV3Add(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MSMBranchV3Add, self).__init__(cfg, block_idx, construct_branch=False)
        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.patchwise = cfg.VIDEO.BACKBONE.BRANCH.PATCHWISE
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

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.num_filters//self.expansion_ratio,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
        if self.stride[0] > 1:
            self.avg_pool = nn.AvgPool3d(
                kernel_size = [3, 1, 1],
                stride = [self.stride[0], 1, 1],
                padding = [1, 0, 0]
            )

        self.proj = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
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

            q = self.b(x)
            x = self.b_bn(q)
            x = self.b_relu(x)

            x_3 = self.b2(x)

            if self.enable: 
                x_mix = self.b2_mix(x)
                if hasattr(self, "avg_pool"):
                    q = self.avg_pool(q)

                if self.num_heads > 1:
                    b,c,t,h,w = q.shape
                    x_3_attn = q*x_3
                    x_mix_attn = q*x_mix
                    x_3_attn = x_3_attn.reshape(b, self.num_heads, c//self.num_heads, t, h, w).sum(2,keepdim=True) * ((c//self.num_heads)**-0.5)
                    x_mix_attn = x_mix_attn.reshape(b, self.num_heads, c//self.num_heads, t, h, w).sum(2,keepdim=True) * ((c//self.num_heads)**-0.5)
                    attn = torch.cat((x_3_attn, x_mix_attn), dim=2).softmax(2)
                    x_3 = x_3.reshape(b, self.num_heads, c//self.num_heads, t, h, w) * attn[:,:,0:1,:]
                    x_mix = x_mix.reshape(b, self.num_heads, c//self.num_heads, t, h, w) * attn[:,:,1:2,:]
                    x_3 = x_3.reshape(b, -1, t, h, w)
                    x_mix = x_mix.reshape(b, -1, t, h, w)

                else:
                    x_3_attn = (q*x_3).sum(1, keepdim=True) * (c**-0.5)
                    x_mix_attn = (q*x_mix).sum(1, keepdim=True) * (c**-0.5)
                    attn = torch.cat((x_3_attn, x_mix_attn), dim=1).softmax(1)

                    x_3 = x_3 * attn[:,0:1,:]
                    x_mix = x_mix * attn[:,1:2,:]
                x_add = self.proj(x_3+x_mix)
                x = q + x_add
                x = self.b2_bn(x)
                x = self.b2_relu(x)
            else:
                raise NotImplementedError

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class MSMBranchOriginal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MSMBranchOriginal, self).__init__(cfg, block_idx, construct_branch=False)
        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL
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

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )
        self.b2_mix_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_mix_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.z1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
            kernel_size     = [self.mid_conv_kernel, 1, 1],
            stride          = 1, 
            padding         = [self.mid_conv_kernel//2, 0, 0],
            bias            = False
        )
        self.z1_bn = nn.BatchNorm3d(max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel), eps=self.bn_eps, momentum=self.bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
            out_channels    = self.num_filters//self.expansion_ratio, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z2_3_sig = nn.Sigmoid()

        self.z2_mix = nn.Conv3d(
            in_channels     = max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
            out_channels    = self.num_filters//self.expansion_ratio, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z2_mix_sig = nn.Sigmoid()

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            x_mix = self.b2_mix(x)
            x_mix = self.b2_mix_bn(x_mix)
            x_mix = self.b2_mix_relu(x_mix)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            x_joint = self.z1(x_joint)
            x_joint = self.z1_bn(x_joint)
            x_joint = self.z1_relu(x_joint)

            x_3_attn = self.z2_3(x_joint)
            x_mix_attn = self.z2_mix(x_joint)

            x_attn = torch.stack(
                (self.temporal_pool(x_3_attn), self.temporal_pool(x_mix_attn)), dim=1
            ).softmax(dim=1)

            x_3_attn = self.z2_3_sig(x_3_attn) * x_attn[:,0,:]
            x_mix_attn = self.z2_mix_sig(x_mix_attn) * x_attn[:,1,:]

            x_3 = x_3 * x_3_attn
            x_mix = x_mix * x_mix_attn

            x = torch.cat((x, x_3+x_mix), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x


@BRANCH_REGISTRY.register()
class MSMBranchOriginalV2(BaseBranch):
    # temporal sigmoid is performed with C aggregated
    def __init__(self, cfg, block_idx):
        super(MSMBranchOriginalV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL
        # self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT
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

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )
        self.b2_mix_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_mix_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.z1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
            kernel_size     = [self.mid_conv_kernel, 1, 1],
            stride          = 1, 
            padding         = [self.mid_conv_kernel//2, 0, 0],
            bias            = False
        )
        self.z1_bn = nn.BatchNorm3d(max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel), eps=self.bn_eps, momentum=self.bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
            out_channels    = self.num_filters//self.expansion_ratio, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z2_3_sig = nn.Sigmoid()

        self.z2_mix = nn.Conv3d(
            in_channels     = max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
            out_channels    = self.num_filters//self.expansion_ratio, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z2_mix_sig = nn.Sigmoid()

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            x_mix = self.b2_mix(x)
            x_mix = self.b2_mix_bn(x_mix)
            x_mix = self.b2_mix_relu(x_mix)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            x_joint = self.z1(x_joint)
            x_joint = self.z1_bn(x_joint)
            x_joint = self.z1_relu(x_joint)

            x_3_attn = self.z2_3(x_joint)
            x_mix_attn = self.z2_mix(x_joint)

            x_attn = torch.stack(
                (self.temporal_pool(x_3_attn), self.temporal_pool(x_mix_attn)), dim=1
            ).softmax(dim=1)

            x_3_attn = self.z2_3_sig(x_3_attn.sum(1, keepdim=True)) * x_attn[:,0,:]
            x_mix_attn = self.z2_mix_sig(x_mix_attn.sum(1, keepdim=True)) * x_attn[:,1,:]

            x_3 = x_3 * x_3_attn
            x_mix = x_mix * x_mix_attn

            x = torch.cat((x, x_3+x_mix), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSK(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSK, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )
        self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_mix_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=self.mid_conv_kernel,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            x_mix = self.b2_mix(x)
            x_mix = self.b2_mix_bn(x_mix)
            x_mix = self.b2_mix_relu(x_mix)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

class TemporalSelectiveKernelUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z1.linear = True
        self.z2_3.linear = True
        self.z2_mix.linear = True

    def forward(self, x):
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        xout = torch.stack(
            (self.temporal_pool(x1), self.temporal_pool(x2)), dim=1
        ).softmax(1)

        return xout[:,0,:], xout[:,1,:]

class TemporalSelectiveKernelUnitFramewise(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x):
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        xout = torch.stack(
            (x1, x2), dim=1
        ).softmax(1)

        return xout[:,0,:], xout[:,1,:]

class MultiHeadTemporalSelectiveKernelUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.mid_channels = mid_channels
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels*num_heads,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_ln = nn.BatchNorm3d(mid_channels*num_heads, eps=1e-5)
        self.z1_relu = nn.ReLU(inplace=True)

        for i in range(num_heads):
            mod = nn.Conv3d(
                in_channels     = mid_channels,
                out_channels    = in_channels, 
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
            )
            setattr(self, f"z2_3_{i}", mod)
            mod = nn.Conv3d(
                in_channels     = mid_channels,
                out_channels    = in_channels, 
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
            )
            setattr(self, f"z2_mix_{i}", mod)

    def forward(self, x):
        x = self.z1(x)
        x = self.z1_ln(x)
        x = self.z1_relu(x)

        for i in range(self.num_heads):
            x1_ = getattr(self, f"z2_3_{i}")(x[:,i*self.mid_channels:(i+1)*self.mid_channels,:])
            x2_ = getattr(self, f"z2_mix_{i}")(x[:,i*self.mid_channels:(i+1)*self.mid_channels,:])
            if i == 0:
                x1, x2 = x1_, x2_
            else:
                x1 = torch.cat((x1, x1_), dim=1)
                x2 = torch.cat((x2, x2_), dim=1)

        xout = torch.stack(
            (self.temporal_pool(x1), self.temporal_pool(x2)), dim=1
        ).softmax(1)

        return xout[:,0,:], xout[:,1,:]

@BRANCH_REGISTRY.register()
class ExpandSKSpatial(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKSpatial, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
            dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
            bias            = False
        )
        self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_mix_relu = nn.ReLU(inplace=True)

        # ---- spatial selection ---- 
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=1,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            x_mix = self.b_mix(x)
            x_mix = self.b_mix_bn(x_mix)
            x_mix = self.b_mix_relu(x_mix)

            x_joint = self.global_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                if i == 0:
                    x = x_3 * x_3_attn + x_mix * x_mix_attn
                else:
                    x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

            # ---- temporal convolutions ----
            x_ = self.b2(x)
            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = torch.cat((x, x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x


@BRANCH_REGISTRY.register()
class ExpandSKSpatialTemporal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKSpatialTemporal, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size_spatial = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL_SPATIAL
        self.mix_kernel_size_temporal = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL_TEMPORAL

        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL_TEMPORAL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size_spatial//2, self.mix_kernel_size_spatial//2],
            dilation        = [1, self.mix_kernel_size_spatial//2, self.mix_kernel_size_spatial//2],
            bias            = False
        )
        self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_mix_relu = nn.ReLU(inplace=True)

        # ---- spatial selection ---- 
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=1,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit_spatial{i}", unit)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size_temporal//2, 0, 0],
            dilation        = [self.mix_kernel_size_temporal//2, 1, 1],
            bias            = False
        )
        self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_mix_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=self.mid_conv_kernel,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit_temporal{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            x_mix = self.b_mix(x)
            x_mix = self.b_mix_bn(x_mix)
            x_mix = self.b_mix_relu(x_mix)

            x_joint = self.global_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit_spatial{i}")(x_joint)
                if i == 0:
                    x = x_3 * x_3_attn + x_mix * x_mix_attn
                else:
                    x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            x_mix = self.b2_mix(x)
            x_mix = self.b2_mix_bn(x_mix)
            x_mix = self.b2_mix_relu(x_mix)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit_temporal{i}")(x_joint)
                x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKTemporalV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKTemporalV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )

        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=self.mid_conv_kernel,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit{i}", unit)

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

            # ---- spatial convolutions ----
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)
            x_mix = self.b2_mix(x)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                if i == 0:
                    x_ = x_3 * x_3_attn + x_mix * x_mix_attn
                else:
                    x_ = torch.cat((x_, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = x + x_

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKSpatialV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKSpatialV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
            dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
            bias            = False
        )

        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        # ---- spatial selection ---- 
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=1,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x_3 = self.b(x)

            x_mix = self.b_mix(x)

            x_joint = self.global_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                if i == 0:
                    x = x_3 * x_3_attn + x_mix * x_mix_attn
                else:
                    x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_ = self.b2(x)
            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = torch.cat((x, x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKSpatialNEWV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKSpatialNEWV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
            dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
            bias            = False
        )

        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        # ---- spatial selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnitFramewise(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=1,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x_3 = self.b(x)

            x_mix = self.b_mix(x)

            x_joint = self.spatial_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                if i == 0:
                    x = x_3 * x_3_attn + x_mix * x_mix_attn
                else:
                    x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_ = self.b2(x)
            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = torch.cat((x, x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKSpatialTemporalV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKSpatialTemporalV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size_spatial = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL_SPATIAL
        self.mix_kernel_size_temporal = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL_TEMPORAL

        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL_TEMPORAL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size_spatial//2, self.mix_kernel_size_spatial//2],
            dilation        = [1, self.mix_kernel_size_spatial//2, self.mix_kernel_size_spatial//2],
            bias            = False
        )

        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        # ---- spatial selection ---- 
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=1,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit_spatial{i}", unit)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size_temporal//2, 0, 0],
            dilation        = [self.mix_kernel_size_temporal//2, 1, 1],
            bias            = False
        )

        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        for i in range(self.num_heads):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=self.mid_conv_kernel,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit_temporal{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x_3 = self.b(x)

            x_mix = self.b_mix(x)

            x_joint = self.global_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit_spatial{i}")(x_joint)
                if i == 0:
                    x = x_3 * x_3_attn + x_mix * x_mix_attn
                else:
                    x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)

            x_mix = self.b2_mix(x)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            for i in range(self.num_heads):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit_temporal{i}")(x_joint)
                if i == 0:
                    x_ = x_3 * x_3_attn + x_mix * x_mix_attn
                else:
                    x_ = torch.cat((x_, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = torch.cat((x, x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKTemporalV3(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKTemporalV3, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        assert self.num_heads >= 2

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )

        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        for i in range(self.num_heads-2):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=self.mid_conv_kernel,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)
            x_mix = self.b2_mix(x)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            x_ = torch.cat((x_3, x_mix), dim=1)
            for i in range(self.num_heads-2):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                x_ = torch.cat((x_, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = torch.cat((x, x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKSpatialV3(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKSpatialV3, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        assert self.num_heads >= 2

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
            dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
            bias            = False
        )

        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        # ---- spatial selection ---- 
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for i in range(self.num_heads-2):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=1,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x_3 = self.b(x)

            x_mix = self.b_mix(x)

            x_joint = self.global_pool(x_3 + x_mix)
            x = torch.cat((x_3, x_mix), dim=1)
            for i in range(self.num_heads-2):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_ = self.b2(x)
            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = torch.cat((x, x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKSpatialTemporalV3(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKSpatialTemporalV3, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        assert self.num_heads >= 2

        self.mix_kernel_size_spatial = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL_SPATIAL
        self.mix_kernel_size_temporal = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL_TEMPORAL

        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL_TEMPORAL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size_spatial//2, self.mix_kernel_size_spatial//2],
            dilation        = [1, self.mix_kernel_size_spatial//2, self.mix_kernel_size_spatial//2],
            bias            = False
        )

        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        # ---- spatial selection ---- 
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for i in range(self.num_heads-2):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=1,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit_spatial{i}", unit)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size_temporal//2, 0, 0],
            dilation        = [self.mix_kernel_size_temporal//2, 1, 1],
            bias            = False
        )

        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        for i in range(self.num_heads-2):
            unit = TemporalSelectiveKernelUnit(
                in_channels=self.sk_channel,
                mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                kernel_size=self.mid_conv_kernel,
                bn_mmt=self.bn_mmt
            )
            setattr(self, f"sk_unit_temporal{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x_3 = self.b(x)

            x_mix = self.b_mix(x)

            x_joint = self.global_pool(x_3 + x_mix)
            x = torch.cat((x_3, x_mix), dim=1)
            for i in range(self.num_heads-2):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit_spatial{i}")(x_joint)
                x = torch.cat((x, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)

            x_mix = self.b2_mix(x)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            x_ = torch.cat((x_3, x_mix), dim=1)
            for i in range(self.num_heads-2):
                x_3_attn, x_mix_attn = getattr(self, f"sk_unit_temporal{i}")(x_joint)
                x_ = torch.cat((x_, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = torch.cat((x, x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKTemporalV4(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKTemporalV4, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )
        self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_mix_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.multi_head_sk = MultiHeadTemporalSelectiveKernelUnit(
            in_channels=self.sk_channel,
            mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
            kernel_size=self.mid_conv_kernel,
            num_heads=self.num_heads
        )

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            x_mix = self.b2_mix(x)
            x_mix = self.b2_mix_bn(x_mix)
            x_mix = self.b2_mix_relu(x_mix)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)
            x_3_attn, x_mix_attn = self.multi_head_sk(x_joint)
            x_3_attn = x_3_attn.reshape(x_3_attn.shape[0],self.num_heads,-1,1,1,1)
            x_mix_attn = x_mix_attn.reshape(x_mix_attn.shape[0],self.num_heads,-1,1,1,1)

            x_ = x_3.unsqueeze(1) * x_3_attn + x_mix.unsqueeze(1) * x_mix_attn
            x_ = x_.reshape(x_.shape[0],-1,x_.shape[3],x_.shape[4],x_.shape[5])
            x = torch.cat((x,x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class ExpandSKTemporalV5(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(ExpandSKTemporalV5, self).__init__(cfg, block_idx, construct_branch=False)
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = self.num_filters//self.expansion_ratio//self.num_heads

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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )

        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        # ---- temporal selection ---- 
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.multi_head_sk = MultiHeadTemporalSelectiveKernelUnit(
            in_channels=self.sk_channel,
            mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
            kernel_size=self.mid_conv_kernel,
            num_heads=self.num_heads
        )

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2,
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

            # ---- spatial convolutions ----
            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            # ---- temporal convolutions ----
            x_3 = self.b2(x)
            x_mix = self.b2_mix(x)

            # ---- temporal selection ----
            x_joint = self.spatial_pool(x_3 + x_mix)

            x_3_attn, x_mix_attn = self.multi_head_sk(x_joint)
            x_3_attn = x_3_attn.reshape(x_3_attn.shape[0],self.num_heads,-1,1,1,1)
            x_mix_attn = x_mix_attn.reshape(x_mix_attn.shape[0],self.num_heads,-1,1,1,1)

            x_ = x_3.unsqueeze(1) * x_3_attn + x_mix.unsqueeze(1) * x_mix_attn
            x_ = x_.reshape(x_.shape[0],-1,x_.shape[3],x_.shape[4],x_.shape[5])

            x_ = self.b2_bn(x_)
            x_ = self.b2_relu(x_)

            x = torch.cat((x, x_), dim=1)

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporal, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            # B,C,T,H,W = x.shape
            # similarity = torch.matmul(torch.nn.functional.normalize(x.reshape(B,C,T*H*W), dim=2, p=2) , torch.nn.functional.normalize(x.reshape(B,C,T*H*W),dim=2, p=2).transpose(-1,-2))
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # sim = similarity.mean(0).cpu().numpy()
            # im = ax.imshow(sim)
            # fig.tight_layout()
            # fig.savefig(f"visualizations/e004/x_input_{self.stage_id}_{self.block_id}.png")
            # print(f"Stage {self.stage_id}, block {self.block_id}")
            # with open("visualizations/e004/x_input_sim_max.txt", "a") as f:
            #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).max(-1)[0].mean()}\n")
            # with open("visualizations/e004/x_input_sim_mean.txt", "a") as f:
            #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).mean()}\n")
            # with open("visualizations/e004/x_input_sim_var.txt", "a") as f:
            #         f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).var()}\n")
            # print(f"x Average Max Similarity: {(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).max(-1)[0].mean()}")
            # print(f"x Average mean similarity: {(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).mean()}")

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            # B,C,T,H,W = x.shape
            # similarity = torch.matmul(torch.nn.functional.normalize(x.reshape(B,C,T*H*W), dim=2, p=2) , torch.nn.functional.normalize(x.reshape(B,C,T*H*W),dim=2, p=2).transpose(-1,-2))
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # sim = similarity.mean(0).cpu().numpy()
            # im = ax.imshow(sim)
            # fig.tight_layout()
            # fig.savefig(f"visualizations/e004/x_{self.stage_id}_{self.block_id}.png")
            # print(f"Stage {self.stage_id}, block {self.block_id}")
            # with open("visualizations/e004/x_sim_max.txt", "a") as f:
            #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).max(-1)[0].mean()}\n")
            # with open("visualizations/e004/x_sim_mean.txt", "a") as f:
            #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).mean()}\n")
            # with open("visualizations/e004/x_sim_var.txt", "a") as f:
            #         f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).var()}\n")
            # print(f"x Average Max Similarity: {(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).max(-1)[0].mean()}")
            # print(f"x Average mean similarity: {(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).mean()}")

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

                # B,C,T,H,W = x_temp.shape
                # similarity = torch.matmul(torch.nn.functional.normalize(x_temp.reshape(B,C,T*H*W), dim=2, p=2) , torch.nn.functional.normalize(x_temp.reshape(B,C,T*H*W),dim=2, p=2).transpose(-1,-2))
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots()
                # sim = similarity.mean(0).cpu().numpy()
                # im = ax.imshow(sim)
                # fig.tight_layout()
                # fig.savefig(f"visualizations/e004/x_temp_{self.stage_id}_{self.block_id}.png")
                # with open("visualizations/e004/x_temp_sim_max.txt", "a") as f:
                #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).max(-1)[0].mean()}\n")
                # with open("visualizations/e004/x_temp_sim_mean.txt", "a") as f:
                #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).mean()}\n")
                # with open("visualizations/e004/x_temp_sim_var.txt", "a") as f:
                #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).var()}\n")
                # print(f"x_temp Average Max Similarity: {(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).max(-1)[0].mean()}")
                # print(f"x_temp Average mean similarity: {(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).mean()}")
                # print("------------")

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                # B,C,T,H,W = x_3.shape
                # similarity = torch.matmul(torch.nn.functional.normalize(x_3.reshape(B,C,T*H*W), dim=2, p=2) , torch.nn.functional.normalize(x_3.reshape(B,C,T*H*W),dim=2, p=2).transpose(-1,-2))
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots()
                # sim = similarity.mean(0).cpu().numpy()
                # im = ax.imshow(sim)
                # fig.tight_layout()
                # fig.savefig(f"visualizations/e004/x_temp_{self.stage_id}_{self.block_id}.png")
                # with open("visualizations/e004/x_temp_sim_max.txt", "a") as f:
                #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).max(-1)[0].mean()}\n")
                # with open("visualizations/e004/x_temp_sim_mean.txt", "a") as f:
                #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).mean()}\n")
                # with open("visualizations/e004/x_temp_sim_var.txt", "a") as f:
                #     f.writelines(f"{(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).var()}\n")
                # print(f"x_temp Average Max Similarity: {(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).max(-1)[0].mean()}")
                # print(f"x_temp Average mean similarity: {(similarity.mean(0) * (1-torch.eye(similarity.shape[-1]).cuda())).mean()}")
                # print("------------")
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatial(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatial, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporal, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"spat_sk_unit{i}", unit)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"temp_sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"spat_sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                x_joint = self.spatial_pool(x_3_temp + x_mix_temp)
                for i in range(self.num_heads):
                    x_3_temp_attn, x_mix_temp_attn = getattr(self, f"temp_sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3_temp * x_3_temp_attn + x_mix_temp * x_mix_temp_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3_temp * x_3_temp_attn + x_mix_temp * x_mix_temp_attn), dim=1)

                if self.output == "cat":
                    x = torch.cat((x_spat, x_temp), dim=1)
                elif self.output == "add":
                    x = x_spat + x_temp
                elif self.output == "out":
                    x = x_temp
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class GroupSelectStackBranchTemporal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(GroupSelectStackBranchTemporal, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.conv_channel = self.num_filters//self.expansion_ratio

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.mlp_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.MLP_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

            self.alpha = cfg.VIDEO.BACKBONE.BRANCH.ALPHA
            self.group_size = cfg.VIDEO.BACKBONE.BRANCH.GROUP_SIZE

            self.conv_channel = self.num_filters//self.expansion_ratio//self.alpha
            self.head_channel = self.num_filters//self.expansion_ratio//self.num_heads

            self.num_groups = self.head_channel // self.group_size

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

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.conv_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.conv_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.conv_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.conv_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = GroupGen(
                    in_channels=self.conv_channel,
                    mid_channels=max(self.conv_channel//self.mlp_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    num_groups=self.num_groups,
                    group_size=self.group_size,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"groupgen_{i}", unit)
                unit = GSKV2(
                    in_channels=self.conv_channel,
                    mid_channels=max(self.conv_channel//self.mlp_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    num_groups=self.num_groups,
                    group_size=self.group_size,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_{i}", unit)


        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                if torch.isnan(x_joint).sum() > 0:
                        print("x_joint")
                for i in range(self.num_heads):
                    g_idx = getattr(self, f"groupgen_{i}")(x_joint)
                    x_3_f, x_mix_f = self.filter_feature_group(x_3, x_mix, g_idx)
                    x_3_attn, x_mix_attn = getattr(self, f"sk_{i}")(x_3_f, x_mix_f)
                    if i == 0:
                        x_temp = x_3_f * x_3_attn + x_mix_f * x_mix_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3_f * x_3_attn + x_mix_f * x_mix_attn), dim=1)
                x_temp = x_temp.reshape(x_temp.shape[0], -1, x_temp.shape[3], x_temp.shape[4], x_temp.shape[5])

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

    def filter_feature_group(self, x1, x2, g_idx):
        # B,N_G,_,_,_ = g_idx.shape
        # _,C,T,H,W = x1.shape
        # for group_idx in range(N_G):
        #     grid_a = torch.linspace(-1, 1, H).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        #     grid_b = torch.linspace(-1, 1, W).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cuda()
        #     grid_c_offset = torch.tensor([i*2/(C-1) for i in range(self.group_size)]).view(1,1,1,-1).cuda()
        #     # grid_c = (2*torch.tensor([0., 0.1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()-1) + grid_c_offset
        #     grid_c = ((2*g_idx[:,group_idx].cuda()-1) + grid_c_offset)%(-1)
        #     grid = torch.stack(
        #         (grid_b.repeat(B,H,1,self.group_size), # x
        #         grid_a.repeat(B,1,W,self.group_size),  # y
        #         grid_c.repeat(1,H,W,1)),          # z
        #         dim=-1
        #     )
        #     sampled_g1 = torch.nn.functional.grid_sample(
        #         x1.permute(0,2,1,3,4), # N, C, T, H, W -> N, T, C, H, W
        #         grid, 
        #         align_corners=True,
        #         mode="bilinear"
        #     ).permute(0,4,1,2,3) # after sampling, the shape becomes N, T, H, W, C
        #                         # N, T, H, W, C -> N, C, T, H, W
            
        #     sampled_g2 = torch.nn.functional.grid_sample(
        #         x2.permute(0,2,1,3,4), # N, C, T, H, W -> N, T, C, H, W
        #         grid, 
        #         align_corners=True,
        #         mode="bilinear"
        #     ).permute(0,4,1,2,3) # after sampling, the shape becomes N, T, H, W, C
        #                         # N, T, H, W, C -> N, C, T, H, W
        #     if group_idx == 0:
        #         g1 = sampled_g1.unsqueeze(1)
        #         g2 = sampled_g2.unsqueeze(1)
        #     else:
        #         g1 = torch.cat((g1, sampled_g1.unsqueeze(1)), dim=1)
        #         g2 = torch.cat((g2, sampled_g2.unsqueeze(1)), dim=1)


        B,C,T,H,W = x1.shape
        g_idx = (g_idx * 2 - 1)
        grid_c_offset = g_idx.permute(0,2,1).reshape(B,-1).unsqueeze(1).unsqueeze(1)
        # for group_idx in range(self.num_groups):
        #     grid_c_offset = g_idx[:,:,group_idx].unsqueeze(1).unsqueeze(1)
        #     grid_c = grid_c_base[:,:,:,group_idx:group_idx+1] + torch.tensor([i*2/(C-1) for i in range(self.group_size)]).view(1,1,1,-1).cuda().repeat(B,1,1,1) + grid_c_offset
        #     grid_c = grid_c % (-1)
        #     grid = torch.stack(
        #         (grid_x, grid_y, grid_c),
        #         dim=-1
        #     )
        #     sampled_g1 = torch.nn.functional.grid_sample(
        #         x1.permute(0,2,1,3,4), # N, C, T, H, W -> N, T, C, H, W
        #         grid, 
        #         align_corners=True,
        #         mode="bilinear"
        #     ).permute(0,4,1,2,3) # after sampling, the shape becomes N, T, H, W, C
        #                         # N, T, H, W, C -> N, C, T, H, W
            
        #     sampled_g2 = torch.nn.functional.grid_sample(
        #         x2.permute(0,2,1,3,4), # N, C, T, H, W -> N, T, C, H, W
        #         grid, 
        #         align_corners=True,
        #         mode="bilinear"
        #     ).permute(0,4,1,2,3) # after sampling, the shape becomes N, T, H, W, C
        #                         # N, T, H, W, C -> N, C, T, H, W
        #     if group_idx == 0:
        #         g1 = sampled_g1.unsqueeze(1)
        #         g2 = sampled_g2.unsqueeze(1)
        #     else:
        #         g1 = torch.cat((g1, sampled_g1.unsqueeze(1)), dim=1)
        #         g2 = torch.cat((g2, sampled_g2.unsqueeze(1)), dim=1)
        grid_x, grid_y, grid_c_base = generate_grid(x1, self.group_size, self.num_groups)
        grid_c = grid_c_base + grid_c_offset
        grid = torch.stack(
            (grid_x, grid_y, grid_c),
            dim=-1
        )
        g1 = torch.nn.functional.grid_sample(
            x1.permute(0,2,1,3,4), # N, C, T, H, W -> N, T, C, H, W
            grid, 
            align_corners=True,
            mode="bilinear"
        ).permute(0,4,1,2,3).reshape(
            B,self.num_groups,self.group_size,T,H,W
        )

        g2 = torch.nn.functional.grid_sample(
            x2.permute(0,2,1,3,4), # N, C, T, H, W -> N, T, C, H, W
            grid, 
            align_corners=True,
            mode="bilinear"
        ).permute(0,4,1,2,3).reshape(
            B,self.num_groups,self.group_size,T,H,W
        )




        # g_idx = g_idx * x1.shape[1]
        # select_idx_c = torch.stack(
        #     (g_idx.view(g_idx.shape[0], g_idx.shape[1]).long(), g_idx.view(g_idx.shape[0], g_idx.shape[1]).long()+1), 
        #     dim=1
        # )
        # select_idx_c = torch.stack(
        #     (select_idx_c, select_idx_c+1, select_idx_c+2, select_idx_c+3),
        #     dim=-1
        # ) % x1.shape[1]
        # if torch.isnan(select_idx_c).sum() > 0:
        #     print("select_idx_c")
        # select_idx_0 = torch.linspace(0,x1.shape[0]-1,x1.shape[0],dtype=torch.long).view(-1,1,1,1)
        # w = torch.stack((1-(g_idx - g_idx.long()), (g_idx - g_idx.long())), dim=1).unsqueeze(-1)
        # if torch.isnan(w).sum() > 0:
        #     print("w")
        # g1 = (x1[select_idx_0, select_idx_c] * w).sum(1)
        # g2 = (x2[select_idx_0, select_idx_c] * w).sum(1)
        # if torch.isnan(g1).sum() > 0:
        #     print("g1")
        return g1, g2

# grid_a = torch.linspace(-1, 1, H).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
# grid_b = torch.linspace(-1, 1, W).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cuda()
# grid_c_offset = torch.tensor([i*2/(C-1) for i in range(self.group_size)]).view(1,1,1,-1).cuda()
# grid_c = (2*g_idx[:,0,:,:,:]-1) + grid_c_offset
# grid = torch.stack(
#     (grid_a.repeat(B,1,W,self.group_size),
#     grid_b.repeat(B,H,1,self.group_size),
#     (grid_c + grid_c_offset).repeat(1,H,W,1)), dim=-1
# )
# sampled = torch.nn.functional.grid_sample(x1.permute(0,2,3,4,1), grid).permute(0,4,1,2,3)

def generate_grid(x, group_size, num_groups):
    B,C,T,H,W = x.shape
    grid_y = torch.linspace(-1, 1, H).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(
        B,1,W,group_size*num_groups
    ).cuda() # h: y
    grid_x = torch.linspace(-1, 1, W).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(
        B,H,1,group_size*num_groups
    ).cuda() # w: x
    grid_c_base = torch.linspace(-1, 1, num_groups).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,H,W,1).cuda()
    grid_c_base = grid_c_base.unsqueeze(-1).repeat(1,1,1,1,group_size).reshape(1,H,W,-1)
    return grid_x, grid_y, grid_c_base

class GSK(nn.Module):
    def __init__(
        self, 
        in_channels, 
        mid_channels, 
        kernel_size, 
        num_groups,
        group_size, 
        temporal_pool=False,
        bn_mmt=0.1
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = num_groups, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = num_groups, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_group_filter = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = num_groups*group_size, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        setattr(self.z2_group_filter, "no_init", True)
        self.z2_group_filter.weight.data.zero_()
        self.z2_group_filter.bias.data.zero_()

    def forward(self, x):
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        g_idx = self.z2_group_filter(
            self.temporal_pool(x)
        ).reshape(-1, self.group_size, self.num_groups).sigmoid()

        xout = torch.stack(
            (self.temporal_pool(x1), self.temporal_pool(x2)), dim=1
        ).softmax(1)

        return xout[:,0,:].unsqueeze(-1), xout[:,1,:].unsqueeze(-1), g_idx

class GroupGen(nn.Module):
    def __init__(
        self, 
        in_channels, 
        mid_channels, 
        kernel_size, 
        num_groups,
        group_size, 
        temporal_pool=False,
        bn_mmt=0.1
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_group_filter = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = num_groups*group_size, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        setattr(self.z2_group_filter, "no_init", True)
        self.z2_group_filter.weight.data.zero_()
        self.z2_group_filter.bias.data.zero_()

    def forward(self, x):
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        g_idx = self.z2_group_filter(
            self.temporal_pool(x)
        ).reshape(-1, self.group_size, self.num_groups).sigmoid()

        return g_idx

class GSKV2(nn.Module):
    def __init__(
        self, 
        in_channels, 
        mid_channels, 
        kernel_size, 
        num_groups,
        group_size, 
        temporal_pool=False,
        bn_mmt=0.1
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = num_groups,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = num_groups, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = num_groups, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x1, x2):
        x = self.spatial_pool((x1 + x2).mean(2))
        
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        xout = torch.stack(
            (self.temporal_pool(x1), self.temporal_pool(x2)), dim=1
        ).softmax(1)

        return xout[:,0,:].unsqueeze(-1), xout[:,1,:].unsqueeze(-1)


@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporalV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporalV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"spat_sk_unit{i}", unit)
            self.b_spat_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_spat_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"temp_sk_unit{i}", unit)
            self.b_temp_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_temp_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"spat_sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
                x_spat = self.b_spat_bn(x_spat)
                x_spat = self.b_spat_relu(x_spat)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                x_joint = self.spatial_pool(x_3_temp + x_mix_temp)
                for i in range(self.num_heads):
                    x_3_temp_attn, x_mix_temp_attn = getattr(self, f"temp_sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3_temp * x_3_temp_attn + x_mix_temp * x_mix_temp_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3_temp * x_3_temp_attn + x_mix_temp * x_mix_temp_attn), dim=1)
                x_temp = self.b_temp_bn(x_temp)
                x_temp = self.b_temp_relu(x_temp)

                if self.output == "cat":
                    x = torch.cat((x_spat, x_temp), dim=1)
                elif self.output == "add":
                    x = x_spat + x_temp
                elif self.output == "out":
                    x = x_temp
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporalV3(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporalV3, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"spat_sk_unit{i}", unit)
            self.b_spat_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_spat_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"temp_sk_unit{i}", unit)
            self.b_temp_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_temp_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)

            if self.enable:
                x_mix = self.b_mix(x)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"spat_sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
                x_spat = self.b_spat_bn(x_spat)
                x_spat = self.b_spat_relu(x_spat)
            
                x_3_temp = self.b2(x_spat)

                x_mix_temp = self.b2_mix(x_spat)

                x_joint = self.spatial_pool(x_3_temp + x_mix_temp)
                for i in range(self.num_heads):
                    x_3_temp_attn, x_mix_temp_attn = getattr(self, f"temp_sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3_temp * x_3_temp_attn + x_mix_temp * x_mix_temp_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3_temp * x_3_temp_attn + x_mix_temp * x_mix_temp_attn), dim=1)
                x_temp = self.b_temp_bn(x_temp)
                x_temp = self.b_temp_relu(x_temp)

                if self.output == "cat":
                    x = torch.cat((x_spat, x_temp), dim=1)
                elif self.output == "add":
                    x = x_spat + x_temp
                elif self.output == "out":
                    x = x_temp
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)


        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

            self.b_temp_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_temp_relu = nn.ReLU(inplace=True)
        else:
            self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

                x_temp = self.b_temp_bn(x_temp)
                x_temp = self.b_temp_relu(x_temp)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                x_3 = self.b2_bn(x_3)
                x_3 = self.b2_relu(x_3)
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)
            self.b_spat_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_spat_relu = nn.ReLU(inplace=True)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
                x_spat = self.b_spat_bn(x_spat)
                x_spat = self.b_spat_relu(x_spat)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalV3(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalV3, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )

        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

            self.b_temp_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_temp_relu = nn.ReLU(inplace=True)
        else:
            self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)

            if self.enable: 
                x_mix = self.b2_mix(x)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

                x_temp = self.b_temp_bn(x_temp)
                x_temp = self.b_temp_relu(x_temp)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                x_3 = self.b2_bn(x_3)
                x_3 = self.b2_relu(x_3)
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialV3(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialV3, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)
            self.b_spat_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_spat_relu = nn.ReLU(inplace=True)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)

            if self.enable:
                x_mix = self.b_mix(x)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
                x_spat = self.b_spat_bn(x_spat)
                x_spat = self.b_spat_relu(x_spat)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalV4(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalV4, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)


        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

            self.b_temp_relu = nn.ReLU(inplace=True)
        else:
            self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

                x_temp = self.b_temp_relu(x_temp)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                x_3 = self.b2_bn(x_3)
                x_3 = self.b2_relu(x_3)
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialV4(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialV4, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)
            self.b_spat_relu = nn.ReLU(inplace=True)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
                x_spat = self.b_spat_relu(x_spat)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalV5(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalV5, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)


        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

            self.b_temp_relu = nn.ReLU(inplace=True)
        else:
            self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

                x_temp = self.b_temp_relu(x_temp)

                x = x_temp + x # shortcut
            else:
                x_3 = self.b2_bn(x_3)
                x_3 = self.b2_relu(x_3)
                x = x_3 + x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialV5(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialV5, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)

        if self.stride[1] > 1:
            self.spatial_downsampling = nn.AvgPool3d(
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            )

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)
            self.b_spat_relu = nn.ReLU(inplace=True)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
                x_spat = self.b_spat_relu(x_spat)
                if hasattr(self, "spatial_downsampling"):
                    x_spat = x_spat + self.spatial_downsampling(x)
                else:
                    x_spat = x_spat + x
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalV6(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalV6, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)


        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        else:
            self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)

                x = x_temp + x # shortcut
            else:
                x_3 = self.b2_bn(x_3)
                x_3 = self.b2_relu(x_3)
                x = x_3 + x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialV6(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialV6, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.stride[1] > 1:
            self.spatial_downsampling = nn.AvgPool3d(
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            )

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)
            self.b_spat_relu = nn.ReLU(inplace=True)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads):
                    x_3_attn, x_mix_attn = getattr(self, f"sk_unit{i}")(x_joint)
                    if i == 0:
                        x_spat = x_3 * x_3_attn + x_mix * x_mix_attn
                    else:
                        x_spat = torch.cat((x_spat, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
                if hasattr(self, "spatial_downsampling"):
                    x_spat = x_spat + self.spatial_downsampling(x)
                else:
                    x_spat = x_spat + x
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalV7(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalV7, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            self.decompose = nn.Conv3d(
                in_channels     = self.sk_channel,
                out_channels    = self.num_heads,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitFuse(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                decompose_map = self.decompose(x_3+x_mix).softmax(1)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:,i:i+1])
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:,i:i+1])), dim=1)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialV7(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialV7, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.decompose = nn.Conv3d(
                in_channels     = self.sk_channel,
                out_channels    = self.num_heads,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitFuse(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                decompose_map = self.decompose(x_3+x_mix).softmax(1)

                for i in range(self.num_heads):
                    if i == 0:
                        x_spat = getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:, i:i+1])
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:, i:i+1])), dim=1)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalV7Shortcut(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalV7Shortcut, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            self.decompose = nn.Conv3d(
                in_channels     = self.sk_channel,
                out_channels    = self.num_heads,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitFuse(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                decompose_map = self.decompose(x_3+x_mix).softmax(1)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:,i:i+1])
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:,i:i+1])), dim=1)

                x = x + x_temp
            else:
                x = x + x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialV7Shortcut(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialV7Shortcut, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.stride[1] > 1:
            self.spatial_downsampling = nn.AvgPool3d(
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            )
        
        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.decompose = nn.Conv3d(
                in_channels     = self.sk_channel,
                out_channels    = self.num_heads,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitFuse(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                decompose_map = self.decompose(x_3+x_mix).softmax(1)

                for i in range(self.num_heads):
                    if i == 0:
                        x_spat = getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:, i:i+1])
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:, i:i+1])), dim=1)
                if hasattr(self, "spatial_downsampling"):
                    x_spat = x_spat + self.spatial_downsampling(x)
                else:
                    x_spat = x_spat + x
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalV8(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalV8, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            self.decompose = nn.Conv3d(
                in_channels     = self.sk_channel,
                out_channels    = self.num_heads,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitFuseV2(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                decompose_map = self.decompose(x_3+x_mix).softmax(1)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:,i:i+1])
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:,i:i+1])), dim=1)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialV8(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialV8, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.decompose = nn.Conv3d(
                in_channels     = self.sk_channel,
                out_channels    = self.num_heads,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitFuseV2(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                decompose_map = self.decompose(x_3+x_mix).softmax(1)

                for i in range(self.num_heads):
                    if i == 0:
                        x_spat = getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:, i:i+1])
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_unit{i}")(x_3, x_mix, decompose_map[:, i:i+1])), dim=1)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalShuffle(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalShuffle, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffle(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_unit{i}")(x_3, x_mix, i)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_unit{i}")(x_3, x_mix, i)), dim=1)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialShuffle(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialShuffle, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffle(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                for i in range(self.num_heads):
                    
                    if i == 0:
                        x_spat = getattr(self, f"sk_unit{i}")(x_3, x_mix, i)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_unit{i}")(x_3, x_mix, i)), dim=1)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporalShuffle(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporalShuffle, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffle(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_spat_unit{i}", unit)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffle(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_temp_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                for i in range(self.num_heads):
                    
                    if i == 0:
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x_3, x_mix, i)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x_3, x_mix, i)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, i)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, i)), dim=1)

                if self.output == "cat":
                    x = torch.cat((x_spat, x_temp), dim=1)
                elif self.output == "add":
                    x = x_spat + x_temp
                elif self.output == "out":
                    x = x_temp
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

class TemporalSelectiveKernelUnitShuffle(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x1, x2, unit_id):
        if unit_id == 0:
            # no suffle
            x1_ = x1
            x2_ = x2
        elif unit_id == 1:
            # shuffle
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//2, 2, T, H, W).flip(2).reshape(B,C,T,H,W)
        elif unit_id == 2:
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//4, 4, T, H, W).flip(2).reshape(B,C,T,H,W)
        elif unit_id == 3:
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//8, 8, T, H, W).flip(2).reshape(B,C,T,H,W)
        else:
            raise NotImplementedError

        
        x = self.spatial_pool(x1_ + x2_)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        xout = torch.stack(
            (self.temporal_pool(x1), self.temporal_pool(x2)), dim=1
        ).softmax(1)

        x1_ = x1_ * xout[:,0,:]
        x2_ = x2_ * xout[:,1,:]
        x_ = x1_ + x2_

        return x_, xout[:, 0, :], xout[:, 1, :]

class TemporalSelectiveKernelUnitFuse(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x1, x2, decompose_map):
        x = self.spatial_pool((x1 + x2)*decompose_map)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (self.temporal_pool(x1_), self.temporal_pool(x2_)), dim=1
        ).softmax(1)

        x1 = x1 * xout[:,0,:]
        x2 = x2 * xout[:,1,:]
        x_ = x1 + x2

        return x_

class TemporalSelectiveKernelUnitShuffleNaive(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [1, 1, 1],
            stride          = 1, 
            padding         = 0,
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x1, x2, unit_id):
        if unit_id == 0:
            # no suffle
            x1_ = x1
            x2_ = x2
        elif unit_id == 1:
            # shuffle
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//2, 2, T, H, W).flip(2).reshape(B,C,T,H,W)
        elif unit_id == 2:
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//4, 4, T, H, W).flip(2).reshape(B,C,T,H,W)
        elif unit_id == 3:
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//8, 8, T, H, W).flip(2).reshape(B,C,T,H,W)
        else:
            raise NotImplementedError

        
        x = self.global_pool(x1_ + x2_)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        xout = torch.stack(
            (x1, x2), dim=1
        ).softmax(1)

        x1_ = x1_ * xout[:,0,:]
        x2_ = x2_ * xout[:,1,:]
        x_ = x1_ + x2_

        return x_, xout[:,0,:], xout[:,1,:]
        # return x_

class TemporalSelectiveKernelUnitShuffleSigmoid(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x1, x2, unit_id):
        if unit_id == 0:
            # no suffle
            x1_ = x1
            x2_ = x2
        elif unit_id == 1:
            # shuffle
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//2, 2, T, H, W).flip(2).reshape(B,C,T,H,W)
        elif unit_id == 2:
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//4, 4, T, H, W).flip(2).reshape(B,C,T,H,W)
        elif unit_id == 3:
            B,C,T,H,W = x2.shape
            x1_ = x1
            x2_ = x2.reshape(B, C//8, 8, T, H, W).flip(2).reshape(B,C,T,H,W)
        else:
            raise NotImplementedError

        
        x = self.spatial_pool(x1_ + x2_)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        x1_ = x1_ * self.temporal_pool(x1).sigmoid()
        x2_ = x2_ * self.temporal_pool(x2).sigmoid()
        x_ = x1_ + x2_

        return x_

class TemporalSelectiveKernelUnitFuseV2(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x1, x2, decompose_map):
        x1_ = x1 * decompose_map
        x2_ = x2 * decompose_map
        x = self.spatial_pool(x1_ + x2_)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_attn = self.z2_3(x)
        x2_attn = self.z2_mix(x)

        xout = torch.stack(
            (self.temporal_pool(x1_attn), self.temporal_pool(x2_attn)), dim=1
        ).softmax(1)

        x1_ = x1_ * xout[:,0,:]
        x2_ = x2_ * xout[:,1,:]
        x_ = x1_ + x2_

        return x_

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalSE(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalSE, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            for i in range(self.num_heads):
                unit = SEUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=1,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"se_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"se_unit{i}")(x_3)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"se_unit{i}")(x_3)), dim=1)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialSE(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialSE, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            for i in range(self.num_heads):
                unit = SEUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=1,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"se_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                for i in range(self.num_heads):
                    if i == 0:
                        x_spat = getattr(self, f"se_unit{i}")(x_3)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"se_unit{i}")(x_3)), dim=1)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

class SEUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size=1, bn_mmt=0.1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = 1,
            stride          = 1, 
            padding         = 0,
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x):
        x_ = self.global_pool(x)
        x_ = self.z1(x_)
        x_ = self.z1_bn(x_)
        x_ = self.z1_relu(x_)

        x_ = self.z2_3(x_).sigmoid()

        x = x * x_

        return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalShuffleEnhanced(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalShuffleEnhanced, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitEnhanced(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_unit{i}")(x_3, x_mix, i)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_unit{i}")(x_3, x_mix, i)), dim=1)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

class TemporalSelectiveKernelUnitEnhanced(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [kernel_size, 1, 1],
            stride          = 1, 
            padding         = [kernel_size//2, 0, 0],
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z1.linear=True
        self.z2_3.linear=True
        self.z2_mix.linear=True

    def forward(self, x1, x2, unit_id):
        x1_ = x1
        x2_ = x2
        # if unit_id == 0:
        #     # no suffle
        #     x1_ = x1
        #     x2_ = x2
        # elif unit_id == 1:
        #     # shuffle
        #     B,C,T,H,W = x2.shape
        #     x1_ = x1
        #     x2_ = x2.reshape(B, C//2, 2, T, H, W).flip(2).reshape(B,C,T,H,W)
        # elif unit_id == 2:
        #     B,C,T,H,W = x2.shape
        #     x1_ = x1
        #     x2_ = x2.reshape(B, C//4, 4, T, H, W).flip(2).reshape(B,C,T,H,W)
        # elif unit_id == 3:
        #     B,C,T,H,W = x2.shape
        #     x1_ = x1
        #     x2_ = x2.reshape(B, C//8, 8, T, H, W).flip(2).reshape(B,C,T,H,W)
        # else:
        #     raise NotImplementedError

        
        x = self.spatial_pool(x1_ + x2_)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        xout = torch.stack(
            (self.temporal_pool(x1), self.temporal_pool(x2)), dim=1
        ).softmax(1)
        xout_2 = torch.stack(
            (x1, x2), dim=1
        ).sigmoid()

        xout = xout * xout_2

        x1_ = x1_ * xout[:,0,:]
        x2_ = x2_ * xout[:,1,:]
        x_ = x1_ + x2_

        return x_


# B,C,T,H,W = x_temp.shape
# similarity = torch.matmul(torch.nn.functional.normalize(x_temp.reshape(B,C,T*H*W), dim=2, p=2) , torch.nn.functional.normalize(x_temp.reshape(B,C,T*H*W),dim=2, p=2).transpose(-1,-2))
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# sim = similarity[0].cpu().numpy()
# im = ax.imshow(sim)
# fig.tight_layout()
# fig.savefig("test.png")
# print("finished")

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalBaselineStack(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalBaselineStack, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

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

        self.b1 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio//2,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio//2, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)
        
        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio//2,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )
        self.b2_mix_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio//2, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_mix_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            x_mix = self.b2_mix(x)
            x_mix = self.b2_mix_bn(x_mix)
            x_mix = self.b2_mix_relu(x_mix)

            x_temp = torch.cat((x_3, x_mix), dim=1)

            if self.output == "cat":
                x = torch.cat((x, x_temp), dim=1)
            elif self.output == "add":
                x = x + x_temp
            else:
                x = x_temp

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialBaselineStack(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialBaselineStack, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

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
            out_channels    = self.num_filters//self.expansion_ratio//2,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio//2, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio//2,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
            dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
            bias            = False
        )
        self.b_mix_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio//2, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_mix_relu = nn.ReLU(inplace=True)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            x_mix = self.b_mix(x)
            x_mix = self.b_mix_bn(x_mix)
            x_mix = self.b_mix_relu(x_mix)

            x_spat = torch.cat((x_3, x_mix), dim=1)
        
            x = self.b2(x_spat)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            if self.output == "cat":
                x = torch.cat((x_spat, x), dim=1)
            elif self.output == "add":
                x = x_spat + x
            elif self.output == "out":
                x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporalBaselineStack(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporalBaselineStack, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
        self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
        self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
        self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

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
            out_channels    = self.num_filters//self.expansion_ratio//2,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio//2, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        self.b_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio//2,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
            dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
            bias            = False
        )
        self.b_mix_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio//2, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_mix_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio//2,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio//2, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        self.b2_mix = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.num_filters//self.expansion_ratio//2,
            kernel_size     = [3, 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.mix_kernel_size//2, 0, 0],
            dilation        = [self.mix_kernel_size//2, 1, 1],
            bias            = False
        )
        self.b2_mix_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio//2, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_mix_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            x_mix = self.b_mix(x)
            x_mix = self.b_mix_bn(x_mix)
            x_mix = self.b_mix_relu(x_mix)

            x_spat = torch.cat((x_3, x_mix), dim=1)
        
            x_3_temp = self.b2(x_spat)
            x_3_temp = self.b2_bn(x_3_temp)
            x_3_temp = self.b2_relu(x_3_temp)

            x_mix_temp = self.b2_mix(x_spat)
            x_mix_temp = self.b2_mix_bn(x_mix_temp)
            x_mix_temp = self.b2_mix_relu(x_mix_temp)

            x_temp = torch.cat((x_3_temp, x_mix_temp), dim=1)

            if self.output == "cat":
                x = torch.cat((x_spat, x_temp), dim=1)
            elif self.output == "add":
                x = x_spat + x_temp
            elif self.output == "out":
                x = x_temp

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalNaive(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalNaive, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffleNaive(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp, x_3_weights, x_mix_weights = getattr(self, f"sk_unit{i}")(x_3, x_mix, 0)
                    else:
                        x_temp_, x_3_weights_, x_mix_weights_ = getattr(self, f"sk_unit{i}")(x_3, x_mix, 0)
                        x_temp = torch.cat((x_temp, x_temp_), dim=1)
                        x_3_weights = torch.cat((x_3_weights, x_3_weights_), dim=1)
                        x_mix_weights = torch.cat((x_mix_weights, x_mix_weights_), dim=1)
                exp = "e060"
                import numpy as np
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, figsize=(8,8))
                idx_x = np.linspace(0, x_3_weights.shape[1]-1, x_3_weights.shape[1])
                ax[0].plot(idx_x, x_3_weights.mean(0).squeeze().cpu().detach().numpy(), label="x_3_weights")
                ax[1].plot(idx_x, x_mix_weights.mean(0).squeeze().cpu().detach().numpy(), label="x_mix_weights")
                fig.tight_layout()
                fig.savefig(f"visualizations/weights/{exp}/{self.stage_id}_{self.block_id}.png")
                np.save(f"visualizations/weights/{exp}/x_3_weights_temp_{self.stage_id}_{self.block_id}.npy", x_mix_weights.mean(0).squeeze().cpu().detach().numpy())
                sparsity_spat = (x==0).view(x.shape[0], -1)
                sparsity_spat_mean = (sparsity_spat.sum(-1) / float(sparsity_spat.shape[-1])).mean()

                sparsity_temp_3 = (x_3==0).view(x_3.shape[0], -1)
                sparsity_temp_3_mean = (sparsity_temp_3.sum(-1) / float(sparsity_temp_3.shape[-1])).mean()

                sparsity_temp_mix = (x_mix==0).view(x_mix.shape[0], -1)
                sparsity_temp_mix_mean = (sparsity_temp_mix.sum(-1) / float(sparsity_temp_mix.shape[-1])).mean()

                sparsity_temp = (x_temp==0).view(x_temp.shape[0], -1)
                sparsity_temp_mean = (sparsity_temp.sum(-1) / float(sparsity_temp.shape[-1])).mean()
                with open(f"visualizations/weights/{exp}/sparsity.txt", "a") as f:
                    f.writelines(f"0.0, 0.0, {sparsity_spat_mean}, {sparsity_temp_3_mean}, {sparsity_temp_mix_mean}, {sparsity_temp_mean}\n")
                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialNaive(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialNaive, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffleNaive(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                for i in range(self.num_heads):
                    
                    if i == 0:
                        x_spat, x_3_weights, x_mix_weights = getattr(self, f"sk_unit{i}")(x_3, x_mix, 0)
                    else:
                        x_spat_, x_3_weights_, x_mix_weights_ = getattr(self, f"sk_unit{i}")(x_3, x_mix, 0)
                        x_spat = torch.cat((x_spat, x_spat_), dim=1)
                        x_3_weights = torch.cat((x_3_weights, x_3_weights_), dim=1)
                        x_mix_weights = torch.cat((x_mix_weights, x_mix_weights_), dim=1)

                exp = "e061"
                import numpy as np
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, figsize=(8,8))
                idx_x = np.linspace(0, x_3_weights.shape[1]-1, x_3_weights.shape[1])
                ax[0].plot(idx_x, x_3_weights.mean(0).squeeze().cpu().detach().numpy(), label="x_3_weights")
                ax[1].plot(idx_x, x_mix_weights.mean(0).squeeze().cpu().detach().numpy(), label="x_mix_weights")
                fig.tight_layout()
                fig.savefig(f"visualizations/weights/{exp}/{self.stage_id}_{self.block_id}.png")
                np.save(f"visualizations/weights/{exp}/x_3_weights_{self.stage_id}_{self.block_id}.npy", x_mix_weights.mean(0).squeeze().cpu().detach().numpy())

                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                sparsity_spat_3 = (x_3==0).view(x_3.shape[0], -1)
                sparsity_spat_3_mean = (sparsity_spat_3.sum(-1) / float(sparsity_spat_3.shape[-1])).mean()

                sparsity_spat_mix = (x_mix==0).view(x_mix.shape[0], -1)
                sparsity_spat_mix_mean = (sparsity_spat_mix.sum(-1) / float(sparsity_spat_mix.shape[-1])).mean()

                sparsity_spat = (x_spat==0).view(x_spat.shape[0], -1)
                sparsity_spat_mean = (sparsity_spat.sum(-1) / float(sparsity_spat.shape[-1])).mean()

                # sparsity_temp_3 = (x_3==0).view(x_3.shape[0], -1)
                # sparsity_temp_3_mean = (sparsity_temp_3.sum(-1) / float(sparsity_temp_3.shape[-1])).mean()

                # sparsity_temp_mix = (x_mix==0).view(x_mix.shape[0], -1)
                # sparsity_temp_mix_mean = (sparsity_temp_mix.sum(-1) / float(sparsity_temp_mix.shape[-1])).mean()

                sparsity_temp = (x==0).view(x.shape[0], -1)
                sparsity_temp_mean = (sparsity_temp.sum(-1) / float(sparsity_temp.shape[-1])).mean()

                with open(f"visualizations/weights/{exp}/sparsity.txt", "a") as f:
                    f.writelines(f"{sparsity_spat_3_mean}, {sparsity_spat_mix_mean}, {sparsity_spat_mean}, 0.0, 0.0, {sparsity_temp_mean}\n")

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporalNaive(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporalNaive, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffleNaive(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_spat_unit{i}", unit)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffle(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_temp_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                for i in range(self.num_heads):
                    
                    if i == 0:
                        x_spat, x_3_weights, x_mix_weights = getattr(self, f"sk_spat_unit{i}")(x_3, x_mix, 0)
                    else:
                        x_spat_, x_3_weights_, x_mix_weights_ = getattr(self, f"sk_spat_unit{i}")(x_3, x_mix, 0)
                        x_spat = torch.cat((x_spat, x_spat_), dim=1)
                        x_3_weights = torch.cat((x_3_weights, x_3_weights_), dim=1)
                        x_mix_weights = torch.cat((x_mix_weights, x_mix_weights_), dim=1)
                
                exp = "e062"
                import numpy as np
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, figsize=(8,8))
                idx_x = np.linspace(0, x_3_weights.shape[1]-1, x_3_weights.shape[1])
                ax[0].plot(idx_x, x_3_weights.mean(0).squeeze().cpu().detach().numpy(), label="x_3_weights")
                ax[1].plot(idx_x, x_mix_weights.mean(0).squeeze().cpu().detach().numpy(), label="x_mix_weights")
                fig.tight_layout()
                fig.savefig(f"visualizations/weights/{exp}/spat_{self.stage_id}_{self.block_id}.png")
                np.save(f"visualizations/weights/{exp}/x_3_weights_spat_{self.stage_id}_{self.block_id}.npy", x_mix_weights.mean(0).squeeze().cpu().detach().numpy())
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp, x_3_weights, x_mix_weights = getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, 0)
                    else:
                        x_temp_, x_3_weights_, x_mix_weights_ = getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, 0)
                        x_temp = torch.cat((x_temp, x_temp_), dim=1)
                        x_3_weights = torch.cat((x_3_weights, x_3_weights_), dim=1)
                        x_mix_weights = torch.cat((x_mix_weights, x_mix_weights_), dim=1)
                exp = "e062"
                import numpy as np
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, figsize=(8,8))
                idx_x = np.linspace(0, x_3_weights.shape[1]-1, x_3_weights.shape[1])
                ax[0].plot(idx_x, x_3_weights.mean(0).squeeze().cpu().detach().numpy(), label="x_3_weights")
                ax[1].plot(idx_x, x_mix_weights.mean(0).squeeze().cpu().detach().numpy(), label="x_mix_weights")
                fig.tight_layout()
                fig.savefig(f"visualizations/weights/{exp}/temp_{self.stage_id}_{self.block_id}.png")
                np.save(f"visualizations/weights/{exp}/x_3_weights_temp_{self.stage_id}_{self.block_id}.npy", x_mix_weights.mean(0).squeeze().cpu().detach().numpy())

                sparsity_spat_3 = (x_3==0).view(x_3.shape[0], -1)
                sparsity_spat_3_mean = (sparsity_spat_3.sum(-1) / float(sparsity_spat_3.shape[-1])).mean()

                sparsity_spat_mix = (x_mix==0).view(x_mix.shape[0], -1)
                sparsity_spat_mix_mean = (sparsity_spat_mix.sum(-1) / float(sparsity_spat_mix.shape[-1])).mean()

                sparsity_spat = (x_spat==0).view(x_spat.shape[0], -1)
                sparsity_spat_mean = (sparsity_spat.sum(-1) / float(sparsity_spat.shape[-1])).mean()

                sparsity_temp_3 = (x_3_temp==0).view(x_3_temp.shape[0], -1)
                sparsity_temp_3_mean = (sparsity_temp_3.sum(-1) / float(sparsity_temp_3.shape[-1])).mean()

                sparsity_temp_mix = (x_mix_temp==0).view(x_mix_temp.shape[0], -1)
                sparsity_temp_mix_mean = (sparsity_temp_mix.sum(-1) / float(sparsity_temp_mix.shape[-1])).mean()

                sparsity_temp = (x_temp==0).view(x_temp.shape[0], -1)
                sparsity_temp_mean = (sparsity_temp.sum(-1) / float(sparsity_temp.shape[-1])).mean()

                with open(f"visualizations/weights/{exp}/sparsity.txt", "a") as f:
                    f.writelines(f"{sparsity_spat_3_mean}, {sparsity_spat_mix_mean}, {sparsity_spat_mean}, {sparsity_temp_3_mean}, {sparsity_temp_mix_mean}, {sparsity_temp_mean}\n")

                if self.output == "cat":
                    x = torch.cat((x_spat, x_temp), dim=1)
                elif self.output == "add":
                    x = x_spat + x_temp
                elif self.output == "out":
                    x = x_temp
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchTemporalSigmoid(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchTemporalSigmoid, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffleSigmoid(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b2(x)
            x_3 = self.b2_bn(x_3)
            x_3 = self.b2_relu(x_3)

            if self.enable: 
                x_mix = self.b2_mix(x)
                x_mix = self.b2_mix_bn(x_mix)
                x_mix = self.b2_mix_relu(x_mix)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_unit{i}")(x_3, x_mix, 0)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_unit{i}")(x_3, x_mix, 0)), dim=1)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
                    x = x_temp
            else:
                if self.output == "cat":
                    x = torch.cat((x, x_3), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x_3

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialSigmoid(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialSigmoid, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffleSigmoid(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_unit{i}", unit)

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
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                for i in range(self.num_heads):
                    
                    if i == 0:
                        x_spat = getattr(self, f"sk_unit{i}")(x_3, x_mix, 0)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_unit{i}")(x_3, x_mix, 0)), dim=1)
            
                x = self.b2(x_spat)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_spat, x), dim=1)
                elif self.output == "add":
                    x = x_spat + x
                elif self.output == "out":
                    x = x
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporalSigmoid(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporalSigmoid, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffleSigmoid(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_spat_unit{i}", unit)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnitShuffle(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"sk_temp_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                for i in range(self.num_heads):
                    
                    if i == 0:
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x_3, x_mix, 0)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x_3, x_mix, 0)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, i)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, i)), dim=1)

                if self.output == "cat":
                    x = torch.cat((x_spat, x_temp), dim=1)
                elif self.output == "add":
                    x = x_spat + x_temp
                elif self.output == "out":
                    x = x_temp
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporalPlus(BaseBranch):
    def __init__(self, cfg, block_idx):
        # with different combined spatial features for different temporal convolutions
        super(SelectStackBranchSpatialTemporalPlus, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.output = cfg.VIDEO.BACKBONE.BRANCH.OUTPUT

        if self.enable: 
            self.mix_kernel_size = cfg.VIDEO.BACKBONE.BRANCH.MIX_KERNEL
            self.selection_reduction_ratio = cfg.VIDEO.BACKBONE.BRANCH.SEL_REDUCTION_RATIO
            self.min_reduced_channel = cfg.VIDEO.BACKBONE.BRANCH.MIN_REDUCED_CHANNEL // self.num_heads
            self.mid_conv_kernel = cfg.VIDEO.BACKBONE.BRANCH.MID_CONV_KERNEL

        self.sk_channel = (
            self.num_filters//self.expansion_ratio//self.num_heads 
            if self.enable else
            self.num_filters//self.expansion_ratio
        )
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
            out_channels    = self.sk_channel,
            kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
            stride          = [1, self.stride[1], self.stride[2]],
            padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
            bias            = False
        )
        self.b_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix_relu = nn.ReLU(inplace=True)

            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            for i in range(self.num_heads*2):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"spat_sk_unit{i//self.num_heads}_{i%self.num_heads}", unit)

        self.b2 = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio,
            out_channels    = self.sk_channel,
            kernel_size     = [self.kernel_size[0], 1, 1],
            stride          = [self.stride[0], 1, 1],
            padding         = [self.kernel_size[0]//2, 0, 0],
            bias            = False
        )
        self.b2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
        self.b2_relu = nn.ReLU(inplace=True)

        if self.enable:
            self.b2_mix = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = TemporalSelectiveKernelUnit(
                    in_channels=self.sk_channel,
                    mid_channels=max(self.sk_channel//self.selection_reduction_ratio, self.min_reduced_channel),
                    kernel_size=self.mid_conv_kernel,
                    bn_mmt=self.bn_mmt
                )
                setattr(self, f"temp_sk_unit{i}", unit)

        self.c = nn.Conv3d(
            in_channels     = self.num_filters//self.expansion_ratio*2 if self.output == "cat" else self.num_filters//self.expansion_ratio,
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

            x_3 = self.b(x)
            x_3 = self.b_bn(x_3)
            x_3 = self.b_relu(x_3)

            if self.enable:
                x_mix = self.b_mix(x)
                x_mix = self.b_mix_bn(x_mix)
                x_mix = self.b_mix_relu(x_mix)

                x_joint = self.spatial_pool(x_3 + x_mix)
                for i in range(self.num_heads*2):
                    x_3_attn, x_mix_attn = getattr(self, f"spat_sk_unit{i//self.num_heads}_{i%self.num_heads}")(x_joint)
                    if i == 0:
                        x_spat_3 = x_3 * x_3_attn + x_mix * x_mix_attn
                    elif i == self.num_heads:
                        x_spat_mix = x_3 * x_3_attn + x_mix * x_mix_attn
                    elif i < self.num_heads:
                        x_spat_3 = torch.cat((x_spat_3, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
                    else:
                        x_spat_mix = torch.cat((x_spat_mix, x_3 * x_3_attn + x_mix * x_mix_attn), dim=1)
            
                x_3_temp = self.b2(x_spat_3)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat_mix)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                x_joint = self.spatial_pool(x_3_temp + x_mix_temp)
                for i in range(self.num_heads):
                    x_3_temp_attn, x_mix_temp_attn = getattr(self, f"temp_sk_unit{i}")(x_joint)
                    if i == 0:
                        x_temp = x_3_temp * x_3_temp_attn + x_mix_temp * x_mix_temp_attn
                    else:
                        x_temp = torch.cat((x_temp, x_3_temp * x_3_temp_attn + x_mix_temp * x_mix_temp_attn), dim=1)

                if self.output == "cat":
                    x = torch.cat((x_spat, x_temp), dim=1)
                elif self.output == "add":
                    x = x_spat + x_temp
                elif self.output == "out":
                    x = x_temp
            else:
                x = self.b2(x_3)
                x = self.b2_bn(x)
                x = self.b2_relu(x)

                if self.output == "cat":
                    x = torch.cat((x_3, x), dim=1)
                elif self.output == "add":
                    x = x + x_3
                elif self.output == "out":
                    x = x

            x = self.c(x)
            x = self.c_bn(x)
            return x