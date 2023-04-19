
import math
import torch
import torch.nn as nn

from models.base.base_blocks import BaseBranch
from models.base.base_blocks import BRANCH_REGISTRY, STEM_REGISTRY
from models.module_zoo.branches.msm_branch import TemporalSelectiveKernelUnitShuffleNaive

@BRANCH_REGISTRY.register()
class MultiscaleAggregatorSpatialTemporal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MultiscaleAggregatorSpatialTemporal, self).__init__(cfg, block_idx, construct_branch=False)
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
                unit = SpatialMultiscaleAggregatorWithShuffleV1SeparatePool(
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
                unit = TemporalMultiscaleAggregatorWithShuffleV1SeparatePool(
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
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, 0)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, 0)), dim=1)

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
class MultiscaleAggregatorSpatialTemporalV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MultiscaleAggregatorSpatialTemporalV2, self).__init__(cfg, block_idx, construct_branch=False)
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
                unit = SpatialMultiscaleAggregatorWithShuffleV1SeparatePool(
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
                unit = TemporalMultiscaleAggregatorWithShuffleV2GAP(
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
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, 0)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, 0)), dim=1)

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
class MultiscaleAggregatorTemporal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MultiscaleAggregatorTemporal, self).__init__(cfg, block_idx, construct_branch=False)
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
                unit = TemporalMultiscaleAggregatorWithShuffleV1SeparatePool(
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
class MultiscaleAggregatorTemporalWithoutShuffle(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MultiscaleAggregatorTemporalWithoutShuffle, self).__init__(cfg, block_idx, construct_branch=False)
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
                unit = TemporalMultiscaleAggregatorWithShuffleV1SeparatePool(
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
class MultiscaleAggregatorTemporalSigmoidSelectionWithoutShuffle(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MultiscaleAggregatorTemporalSigmoidSelectionWithoutShuffle, self).__init__(cfg, block_idx, construct_branch=False)
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
                unit = TemporalMultiscaleAggregatorWithShuffleV3SeparatePoolSigmoidSelection(
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
class MultiscaleAggregatorTemporalDecomposeGAP(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MultiscaleAggregatorTemporalDecomposeGAP, self).__init__(cfg, block_idx, construct_branch=False)
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

            self.decompose = nn.Conv3d(
                in_channels     = self.sk_channel,
                out_channels    = self.num_heads,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                bias            = False
            )

            for i in range(self.num_heads):
                unit = TemporalMultiscaleAggregatorDecomposeGAP(
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

                decompose_map = self.decompose(x_3 + x_mix).softmax(1)

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


class SpatialMultiscaleAggregatorWithShuffleV1SeparatePool(nn.Module):
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

        return x_

class TemporalMultiscaleAggregatorWithShuffleV1SeparatePool(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [1, kernel_size, kernel_size],
            stride          = 1, 
            padding         = [0, kernel_size//2, kernel_size//2],
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

        
        x = self.temporal_pool(x1_ + x2_)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1 = self.z2_3(x)
        x2 = self.z2_mix(x)

        xout = torch.stack(
            (self.spatial_pool(x1), self.spatial_pool(x2)), dim=1
        ).softmax(1)

        x1_ = x1_ * xout[:,0,:]
        x2_ = x2_ * xout[:,1,:]
        x_ = x1_ + x2_

        return x_

class TemporalMultiscaleAggregatorWithShuffleV2GAP(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
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

        return x_

class TemporalMultiscaleAggregatorWithShuffleV3SeparatePoolSigmoidSelection(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = [1, kernel_size, kernel_size],
            stride          = 1, 
            padding         = [0, kernel_size//2, kernel_size//2],
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

        
        x = self.temporal_pool(x1_ + x2_)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        xout = self.spatial_pool(self.z2_3(x)).sigmoid()

        x1_ = x1_ * xout
        x2_ = x2_ * (1 - xout)
        x_ = x1_ + x2_

        return x_

class TemporalMultiscaleAggregatorDecomposeGAP(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, bn_mmt=0.1):
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

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = in_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x1, x2, decompose_map=None):
        if decompose_map is not None:
            x = self.global_pool((x1 + x2) * decompose_map)
        else:
            x = self.global_pool(x1 + x2)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        ).softmax(1)

        x_ = x1 * xout[:, 0, :] + x2 * xout[:, 1, :]

        return x_

@BRANCH_REGISTRY.register()
class MoETemporal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoETemporal, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFunc(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_temp = getattr(self, f"sk_unit{i}")(x, x_3, x_mix)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_unit{i}")(x, x_3, x_mix)), dim=1)

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
class MoESpatial(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatial, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFunc(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_unit{i}")(x, x_3, x_mix)), dim=1)
            
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
class MoESpatialTemporal(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporal, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFunc(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFunc(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_spat, x_3_temp, x_mix_temp)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_spat, x_3_temp, x_mix_temp)), dim=1)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalShortcut(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalShortcut, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFunc(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFunc(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_spat, x_3_temp, x_mix_temp)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_spat, x_3_temp, x_mix_temp)), dim=1)

                x = x_temp + x_spat
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
class MoESpatialTemporalGroupWise(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalGroupWise, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncGroupWise(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    group_size=4, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncGroupWise(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    group_size=4,
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_spat, x_3_temp, x_mix_temp)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_spat, x_3_temp, x_mix_temp)), dim=1)

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
class MoESpatialTemporalUniformInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInit, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_spat, x_3_temp, x_mix_temp)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_spat, x_3_temp, x_mix_temp)), dim=1)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1Conv(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1Conv, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp)), dim=1) # generate from x

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvNoBias(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvNoBias, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInitNoBias(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInitNoBias(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp)), dim=1) # generate from x

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvFullBias(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvFullBias, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInitFullBias(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInitFullBias(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp)), dim=1) # generate from x

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvNoBN(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvNoBN, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInitNoBN(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInitNoBN(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp)), dim=1) # generate from x

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvMultiHead(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvMultiHead, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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

            self.sk_spat_unit = RoutingFuncUniformInitMultiHead(
                in_channels=self.num_filters//self.expansion_ratio,
                mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                out_channels=self.sk_channel, 
                num_heads=self.num_heads,
                bn_eps=self.bn_eps,
                bn_mmt=self.bn_mmt
            )

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

            self.sk_temp_unit = RoutingFuncUniformInitMultiHead(
                in_channels=self.num_filters//self.expansion_ratio,
                mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                out_channels=self.sk_channel, 
                num_heads=self.num_heads,
                bn_eps=self.bn_eps,
                bn_mmt=self.bn_mmt
            )

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

                x_spat = self.sk_spat_unit(x, x_3, x_mix)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                x_temp = self.sk_temp_unit(x, x_3_temp, x_mix_temp)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvBNAfterSelective(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvBNAfterSelective, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
        self.b_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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

            for i in range(self.num_heads):
                unit = RoutingFuncUniformInit(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
        self.b2_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
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

            for i in range(self.num_heads):
                unit = RoutingFuncUniformInit(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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

            if self.enable:
                x_mix = self.b_mix(x)
                for i in range(self.num_heads):
                    if i == 0:
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)

                x_spat = self.b_bn(x_spat)
                x_spat = self.b_relu(x_spat)
            
                x_3_temp = self.b2(x_spat)
                x_mix_temp = self.b2_mix(x_spat)
                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp)), dim=1) # generate from x

                x_temp = self.b2_bn(x_temp)
                x_temp = self.b2_relu(x_temp)

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvWithTemperature(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvWithTemperature, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInitWithTemperature(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInitWithTemperature(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp)), dim=1) # generate from x

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvLinear(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvLinear, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInitLinear(
                    in_channels=self.num_filters//self.expansion_ratio,
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInitLinear(
                    in_channels=self.num_filters//self.expansion_ratio,
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp)), dim=1) # generate from x

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvTransformFinalBN(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvTransformFinalBN, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
        self.c_bn.transform_final_bn=True
    
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp)), dim=1) # generate from x

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightAfter1x1ConvWith4Experts(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightAfter1x1ConvWith4Experts, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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

            self.b_mix2 = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.kernel_size[1]//2, self.kernel_size[2]//2],
                bias            = False
            )
            self.b_mix2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix2_relu = nn.ReLU(inplace=True)

            self.b_mix3 = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [1, self.kernel_size[1], self.kernel_size[2]],
                stride          = [1, self.stride[1], self.stride[2]],
                padding         = [0, self.mix_kernel_size//2, self.mix_kernel_size//2],
                dilation        = [1, self.mix_kernel_size//2, self.mix_kernel_size//2],
                bias            = False
            )
            self.b_mix3_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b_mix3_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = RoutingFuncUniformInitWith4Experts(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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

            self.b2_mix2 = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [self.kernel_size[0], 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.kernel_size[0]//2, 0, 0],
                bias            = False
            )
            self.b2_mix2_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix2_relu = nn.ReLU(inplace=True)

            self.b2_mix3 = nn.Conv3d(
                in_channels     = self.num_filters//self.expansion_ratio,
                out_channels    = self.sk_channel,
                kernel_size     = [3, 1, 1],
                stride          = [self.stride[0], 1, 1],
                padding         = [self.mix_kernel_size//2, 0, 0],
                dilation        = [self.mix_kernel_size//2, 1, 1],
                bias            = False
            )
            self.b2_mix3_bn = nn.BatchNorm3d(self.sk_channel, eps=self.bn_eps, momentum=self.bn_mmt)
            self.b2_mix3_relu = nn.ReLU(inplace=True)

            for i in range(self.num_heads):
                unit = RoutingFuncUniformInitWith4Experts(
                    in_channels=self.num_filters//self.expansion_ratio,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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

                x_mix2 = self.b_mix2(x)
                x_mix2 = self.b_mix2_bn(x_mix2)
                x_mix2 = self.b_mix2_relu(x_mix2)

                x_mix3 = self.b_mix3(x)
                x_mix3 = self.b_mix3_bn(x_mix3)
                x_mix3 = self.b_mix3_relu(x_mix3)
                

                for i in range(self.num_heads):
                    
                    if i == 0:
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix, x_mix2, x_mix3)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x, x_3, x_mix, x_mix2, x_mix3)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                x_mix2_temp = self.b2_mix2(x_spat)
                x_mix2_temp = self.b2_mix2_bn(x_mix2_temp)
                x_mix2_temp = self.b2_mix2_relu(x_mix2_temp)

                x_mix3_temp = self.b2_mix3(x_spat)
                x_mix3_temp = self.b2_mix3_bn(x_mix3_temp)
                x_mix3_temp = self.b2_mix3_relu(x_mix3_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp, x_mix2_temp, x_mix3_temp) # generate from x
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x, x_3_temp, x_mix_temp, x_mix2_temp, x_mix3_temp)), dim=1) # generate from x

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightBefore1x1Conv(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightBefore1x1Conv, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.dim_in,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.dim_in,
                    mid_channels=max(self.num_filters//self.expansion_ratio//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
            x_in = x
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x_in, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x_in, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_in, x_3_temp, x_mix_temp) # generate from x_in
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_in, x_3_temp, x_mix_temp)), dim=1) # generate from x_in

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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
class MoESpatialTemporalUniformInitJointWeightBefore1x1ConvV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(MoESpatialTemporalUniformInitJointWeightBefore1x1ConvV2, self).__init__(cfg, block_idx, construct_branch=False)
        self.enable = cfg.VIDEO.BACKBONE.BRANCH.ENABLE_MSM
        self.num_heads = cfg.VIDEO.BACKBONE.BRANCH.NUM_HEADS
        self.num_heads = self.num_heads // max(cfg.VIDEO.BACKBONE.BRANCH.MIN_EXPERT_CHANNEL//(self.num_filters//self.expansion_ratio//self.num_heads), 1)
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.dim_in,
                    mid_channels=max(self.dim_in//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
                unit = RoutingFuncUniformInit(
                    in_channels=self.dim_in,
                    mid_channels=max(self.dim_in//self.selection_reduction_ratio, self.min_reduced_channel),
                    out_channels=self.sk_channel, 
                    bn_eps=self.bn_eps,
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
            x_in = x
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x_in, x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x_in, x_3, x_mix)), dim=1)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_in, x_3_temp, x_mix_temp) # generate from x_in
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_in, x_3_temp, x_mix_temp)), dim=1) # generate from x_in

                if self.output == "cat":
                    x = torch.cat((x, x_temp), dim=1)
                elif self.output == "add":
                    x = x + x_temp
                else:
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

class RoutingFunc(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bn_eps=1e-3, bn_mmt=0.1):
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
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        ).softmax(1)

        return x1 * xout[:, 0, :] + x2 * xout[:, 1, :]

class RoutingFuncGroupWise(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, group_size, bn_eps=1e-3, bn_mmt=0.1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.group_size = group_size
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = 1,
            stride          = 1, 
            padding         = 0,
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels//group_size, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels//group_size, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        ).softmax(1)

        B,C,T,H,W = x1.shape
        x1 = x1.reshape(B, self.group_size, C//self.group_size, T, H, W)
        x2 = x2.reshape(B, self.group_size, C//self.group_size, T, H, W)

        x_ = x1 * xout[:, 0:1, :] + x2 * xout[:, 1:2, :]

        x_ = x_.reshape(B,C,T,H,W)

        return x_

class RoutingFuncUniformInit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bn_eps=1e-3, bn_mmt=0.1):
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
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z1.linear=True
        self.z2_3.linear=True
        self.z2_mix.linear=True

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        ).softmax(1)

        return x1 * xout[:, 0, :] + x2 * xout[:, 1, :]

class RoutingFuncUniformInitNoBias(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bn_eps=1e-3, bn_mmt=0.1):
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
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = False,
        )
        self.z1.linear=True
        self.z2_3.linear=True
        self.z2_mix.linear=True

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        ).softmax(1)

        return x1 * xout[:, 0, :] + x2 * xout[:, 1, :]

class RoutingFuncUniformInitFullBias(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bn_eps=1e-3, bn_mmt=0.1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels,
            kernel_size     = 1,
            stride          = 1, 
            padding         = 0,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z1.linear=True
        self.z2_3.linear=True
        self.z2_mix.linear=True

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        ).softmax(1)

        return x1 * xout[:, 0, :] + x2 * xout[:, 1, :]

class RoutingFuncUniformInitNoBN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bn_eps=1e-3, bn_mmt=0.1):
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
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z1.linear=True
        self.z2_3.linear=True
        self.z2_mix.linear=True

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        ).softmax(1)

        return x1 * xout[:, 0, :] + x2 * xout[:, 1, :]

class RoutingFuncUniformInitMultiHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_heads, bn_eps=1e-3, bn_mmt=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.mid_channels = mid_channels
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = mid_channels*num_heads,
            kernel_size     = 1,
            stride          = 1, 
            padding         = 0,
            bias            = False,
        )
        self.z1_bn = nn.BatchNorm3d(mid_channels*num_heads, eps=bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        for head in range(num_heads):
            z2_3 = nn.Conv3d(
                in_channels     = mid_channels,
                out_channels    = out_channels, 
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
            )
            z2_3.linear=True

            z2_mix = nn.Conv3d(
                in_channels     = mid_channels,
                out_channels    = out_channels, 
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
            )
            z2_mix.linear=True
            setattr(self, f"z2_3_{head}", z2_3)
            setattr(self, f"z2_mix_{head}", z2_mix)
        self.z1.linear=True

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        for head_id in range(self.num_heads):
            x1_ = getattr(self, f"z2_3_{head_id}")(x[:,head_id*self.mid_channels:(head_id+1)*self.mid_channels])
            x2_ = getattr(self, f"z2_mix_{head_id}")(x[:,head_id*self.mid_channels:(head_id+1)*self.mid_channels])
            xout = torch.stack((x1_, x2_), dim=1).softmax(1)
            if head_id == 0:
                x_ = x1 * xout[:, 0, :] + x2 * xout[:, 1, :]
            else:
                x_ = torch.cat(
                    (x_, x1 * xout[:, 0, :] + x2 * xout[:, 1, :]), dim=1
                )
        return x_

class RoutingFuncUniformInitWithTemperature(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bn_eps=1e-3, bn_mmt=0.1):
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
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_3 = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_mix = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z1.linear=True
        self.z2_3.linear=True
        self.z2_mix.linear=True

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        )
        xout = (xout/30).softmax(1)

        return x1 * xout[:, 0, :] + x2 * xout[:, 1, :]

class RoutingFuncUniformInitLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bn_eps=1e-3, bn_mmt=0.1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.z1 = nn.Conv3d(
            in_channels     = in_channels,
            out_channels    = 2*out_channels,
            kernel_size     = 1,
            stride          = 1, 
            padding         = 0,
            bias            = False,
        )

        self.z1.linear=True

    def forward(self, x, x1, x2):
        x = self.z1(self.global_pool(x))

        B,C,_,_,_ = x.shape
        x = x.reshape(B, 2, C//2, 1, 1, 1).softmax(1)

        return x1 * x[:, 0, :] + x2 * x[:, 1, :]

class RoutingFuncUniformInitWith4Experts(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bn_eps=1e-3, bn_mmt=0.1):
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
        self.z1_bn = nn.BatchNorm3d(mid_channels, eps=bn_eps, momentum=bn_mmt)
        self.z1_relu = nn.ReLU(inplace=True)

        self.z2_a = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )

        self.z2_b = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z2_c = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z2_d = nn.Conv3d(
            in_channels     = mid_channels,
            out_channels    = out_channels, 
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
        )
        self.z1.linear=True
        self.z2_a.linear=True
        self.z2_b.linear=True
        self.z2_c.linear=True
        self.z2_d.linear=True

    def forward(self, x, x1, x2, x3, x4):
        x = self.z1(self.global_pool(x))
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_a(x)
        x2_ = self.z2_b(x)
        x3_ = self.z2_c(x)
        x4_ = self.z2_d(x)

        xout = torch.stack(
            (x1_, x2_, x3_, x4_), dim=1
        ).softmax(1)

        return x1 * xout[:, 0, :] + x2 * xout[:, 1, :] + x3 * xout[:, 2, :] + x4 * xout[:, 3, :]

@BRANCH_REGISTRY.register()
class SelectStackBranchSpatialTemporalNaiveV2(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporalNaiveV2, self).__init__(cfg, block_idx, construct_branch=False)
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
            self.middle = nn.Conv3d(
                self.num_filters//self.expansion_ratio,
                self.num_filters//self.expansion_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.middle_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.middle_relu = nn.ReLU(inplace=True)

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

                x_spat = self.middle(x_spat)
                x_spat = self.middle_bn(x_spat)
                x_spat = self.middle_relu(x_spat)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, 0)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp, 0)), dim=1)

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
class SelectStackBranchSpatialTemporalNaiveUniformInit(BaseBranch):
    def __init__(self, cfg, block_idx):
        super(SelectStackBranchSpatialTemporalNaiveUniformInit, self).__init__(cfg, block_idx, construct_branch=False)
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
                unit = TemporalSelectiveKernelUnitNaiveUniformInit(
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
            self.middle = nn.Conv3d(
                self.num_filters//self.expansion_ratio,
                self.num_filters//self.expansion_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.middle_bn = nn.BatchNorm3d(self.num_filters//self.expansion_ratio, eps=self.bn_eps, momentum=self.bn_mmt)
            self.middle_relu = nn.ReLU(inplace=True)

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
                unit = TemporalSelectiveKernelUnitNaiveUniformInit(
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
                        x_spat = getattr(self, f"sk_spat_unit{i}")(x_3, x_mix)
                    else:
                        x_spat = torch.cat((x_spat, getattr(self, f"sk_spat_unit{i}")(x_3, x_mix)), dim=1)

                x_spat = self.middle(x_spat)
                x_spat = self.middle_bn(x_spat)
                x_spat = self.middle_relu(x_spat)
            
                x_3_temp = self.b2(x_spat)
                x_3_temp = self.b2_bn(x_3_temp)
                x_3_temp = self.b2_relu(x_3_temp)

                x_mix_temp = self.b2_mix(x_spat)
                x_mix_temp = self.b2_mix_bn(x_mix_temp)
                x_mix_temp = self.b2_mix_relu(x_mix_temp)

                for i in range(self.num_heads):
                    if i == 0:
                        x_temp = getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp)
                    else:
                        x_temp = torch.cat((x_temp, getattr(self, f"sk_temp_unit{i}")(x_3_temp, x_mix_temp)), dim=1)

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

class TemporalSelectiveKernelUnitNaiveUniformInit(nn.Module):
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
        self.z1.linear=True
        self.z2_3.linear=True
        self.z2_mix.linear=True

    def forward(self, x1, x2):
        
        x = self.global_pool(x1 + x2)
        x = self.z1(x)
        x = self.z1_bn(x)
        x = self.z1_relu(x)

        x1_ = self.z2_3(x)
        x2_ = self.z2_mix(x)

        xout = torch.stack(
            (x1_, x2_), dim=1
        ).softmax(1)

        x1 = x1 * xout[:,0,:]
        x2 = x2 * xout[:,1,:]
        x = x1 + x2

        return x