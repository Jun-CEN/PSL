#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Basic audio-visual blocks. """

import abc
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from models.utils.params import (
    update_av_conv_params, 
    update_av_conv_params_v2,
    calculate_audio_length, 
    calculate_video_length
)
from models.base.base_blocks import (
    BRANCH_REGISTRY, STEM_REGISTRY, HEAD_REGISTRY,
    BaseModule
)
from torch.nn.modules.utils import _triple, _pair

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

@STEM_REGISTRY.register()
class Base2DAVStem(BaseModule):
    """
    Constructs basic AudioVisual ResNet 2D Stem.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base2DAVStem, self).__init__(cfg)

        self.cfg = cfg

        self.video_branch_enable = cfg.VIDEO.ENABLE
        self.audio_branch_enable = cfg.AUDIO.ENABLE

        # loading the config for downsampling
        visual_downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        if visual_downsampling:
            self.visual_stride = [1, 2, 2]
        else:
            self.visual_stride = [1, 1, 1]
        self.audio_downsampling_temporal    = cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        self.audio_downsampling_mel         = cfg.AUDIO.BACKBONE.DOWNSAMPLING_MEL[0]
        self.audio_pool_size = [
            2 if self.audio_downsampling_temporal else 1,
            2 if self.audio_downsampling_mel else 1
        ]
        self.visual_kernel_size = cfg.VIDEO.BACKBONE.KERNEL_SIZE[0]
        self.num_filters = cfg.VIDEO.BACKBONE.NUM_FILTERS[0]
        self.visual_num_input_channels = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS

        self.bn_eps = cfg.BN.EPS
        self.bn_mmt = cfg.BN.MOMENTUM

        self.audio_kernel_size  = cfg.AUDIO.BACKBONE.KERNEL_SIZE[0]
        self.window_size        = cfg.AUDIO.BACKBONE.STEM.WINDOW_SIZE
        self.hop_length         = cfg.AUDIO.BACKBONE.STEM.HOP_LENGTH
        self.mel_bins           = cfg.AUDIO.BACKBONE.STEM.MEL_BINS
        self.frange             = cfg.AUDIO.BACKBONE.STEM.FRANGE
        self.sample_rate        = cfg.AUDIO.SAMPLE_RATE
        self.audio_dropout      = cfg.AUDIO.DROPOUT_RATE
        self.audio_num_input_channels = cfg.AUDIO.BACKBONE.NUM_INPUT_CHANNELS

        self.audio_aug_time_drop_width = cfg.AUDIO.AUGMENTATION.TIME_DROP_WIDTH
        self.audio_aug_time_stripes_num = cfg.AUDIO.AUGMENTATION.TIME_STRIPES_NUM
        self.audio_aug_freq_drop_width = cfg.AUDIO.AUGMENTATION.FREQ_DROP_WIDTH
        self.audio_aug_freq_stripes_num = cfg.AUDIO.AUGMENTATION.FREQ_STRIPES_NUM
        
        self._construct_block()

    def _construct_block(
        self, 
    ):
        if self.video_branch_enable:
            # visual
            self.visual_a = nn.Conv3d(
                self.visual_num_input_channels,
                self.num_filters,
                kernel_size = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
                stride      = [1, self.visual_stride[1], self.visual_stride[2]],
                padding     = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
                bias        = False
            )
            self.visual_a_bn = nn.BatchNorm3d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.visual_a_relu = nn.ReLU(inplace=True)

        if self.audio_branch_enable:
            # audio

            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=self.window_size, hop_length=self.hop_length, 
                win_length=self.window_size, window='hann', center=True, pad_mode='reflect', 
                freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=self.sample_rate, n_fft=self.window_size, 
                n_mels=self.mel_bins, fmin=self.frange[0], fmax=self.frange[1], 
                ref=1.0, amin=1e-10, top_db=None, 
                freeze_parameters=True)

            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=self.audio_aug_time_drop_width, 
                time_stripes_num=self.audio_aug_time_stripes_num, 
                freq_drop_width=self.audio_aug_freq_drop_width, 
                freq_stripes_num=self.audio_aug_freq_stripes_num)
            
            self.audio_initial_bn = nn.BatchNorm2d(self.mel_bins, eps=self.bn_eps, momentum=self.bn_mmt)

            self.audio_a = nn.Conv2d(
                self.audio_num_input_channels,
                self.num_filters,
                kernel_size = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride      = [1, 1],
                padding     = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias=False
            )
            self.audio_a_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_a_relu = nn.ReLU(inplace=True)
            self.audio_b = nn.Conv2d(
                self.num_filters,
                self.num_filters,
                kernel_size = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride      = [1, 1],
                padding     = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias=False
            )
            self.audio_b_bn = nn.BatchNorm2d(self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_b_relu = nn.ReLU(inplace=True)

            if self.audio_downsampling_mel or self.audio_downsampling_temporal:
                self.audio_avgpool = nn.AvgPool2d(
                    kernel_size=self.audio_pool_size,
                )
            self.audio_dropout = nn.Dropout(p=self.audio_dropout)

    def forward(self, xv, xa):
        if self.video_branch_enable:
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)
        
        if self.audio_branch_enable:

            xa = self.spectrogram_extractor(xa.squeeze(1))
            xa = self.logmel_extractor(xa)

            xa = xa.transpose(1, 3)
            xa = self.audio_initial_bn(xa)
            xa = xa.transpose(1, 3)

            if self.training:
                xa = self.spec_augmenter(xa)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            xa = self.audio_b(xa)
            xa = self.audio_b_bn(xa)
            xa = self.audio_b_relu(xa)

            if hasattr(self, "audio_avgpool"):
                xa = self.audio_avgpool(xa)

            xa = self.audio_dropout(xa)

        return xv, xa

class BaseAVBlock(nn.Module):
    """
    Constructs a base audio-visual block, composed of one shortcut and one conv branch for 
    each of the modalities.
    """
    def __init__(
        self,
        cfg,
        block_idx,
        last=False,
    ):
        """
        Args: 
            cfg         (Config): global config object. 
            block_idx   (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(BaseAVBlock, self).__init__()
        
        self.cfg = cfg
        self.video_branch_enable = cfg.VIDEO.ENABLE
        self.audio_branch_enable = cfg.AUDIO.ENABLE
        update_av_conv_params(cfg, self, block_idx)
        self.last = last

        self._construct_block(
            cfg             = cfg,
            block_idx       = block_idx,
        )
            
    def _construct_block(
        self,
        cfg,
        block_idx,
    ):
        if (self.dim_in != self.num_filters or self.visual_downsampling) and self.video_branch_enable:
            # if the spatial size or the channel dimension is inconsistent for the input and the output 
            # of the block, a 1x1x1 3D conv need to be performed. 
            self.visual_short_cut = nn.Conv3d(
                self.dim_in,
                self.num_filters,
                kernel_size=1,
                stride=self.visual_stride,
                padding=0,
                bias=False
            )
            self.visual_short_cut_bn = nn.BatchNorm3d(
                self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt
            )
        if (self.dim_in != self.num_filters or self.audio_downsampling) and self.audio_branch_enable:
            self.audio_short_cut = nn.Conv2d(
                self.dim_in,
                self.num_filters,
                kernel_size=1,
                stride=self.audio_stride,
                padding=0,
                bias=False
            )
            self.audio_short_cut_bn = nn.BatchNorm2d(
                self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt
            )
        self.conv_branch = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, block_idx, self.last)
        self.relu = nn.ReLU(inplace=True)
            
    def forward(self, xv, xa):
        short_cut_v = xv
        short_cut_a = xa
        if hasattr(self, "visual_short_cut"):
            short_cut_v = self.visual_short_cut_bn(self.visual_short_cut(short_cut_v))
        if hasattr(self, "audio_short_cut"):
            short_cut_a = self.audio_short_cut_bn(self.audio_short_cut(short_cut_a))
        
        xv, xa = self.conv_branch(xv, xa)
        if self.video_branch_enable:
            xv = self.relu(short_cut_v + xv)
        if self.audio_branch_enable and not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
            xa = self.relu(short_cut_a + xa)

        return xv, xa

class BaseAudioVisualStage(nn.Module):
    """
    ResNet Stage containing several blocks.
    """
    def __init__(
        self,
        cfg,
        num_blocks,
        stage_idx,
    ):
        """
        Args:
            num_blocks (int): number of blocks contained in this res-stage.
            stage_idx  (int): the stage index of this res-stage.
        """
        super(BaseAudioVisualStage, self).__init__()
        self.cfg = cfg
        self.num_blocks = num_blocks       
        if stage_idx == 4 and cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE:
            self.only_video_feature = True
        else:
            self.only_video_feature = False
        self._construct_stage(
            cfg                     = cfg,
            stage_idx               = stage_idx,
        )
        
    def _construct_stage(
        self,
        cfg,
        stage_idx,
    ):
        for i in range(self.num_blocks):
            res_block = BaseAVBlock(
                cfg                 = cfg,
                block_idx           = [stage_idx, i],
                last                = self.only_video_feature and (i == self.num_blocks-1)
            )
            self.add_module("avres_{}".format(i+1), res_block)
        if (
            self.cfg.VIDEO.BACKBONE.FUSION_FROM_AUDIO.ENABLE[stage_idx-1] or
            self.cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.ENABLE[stage_idx-1]
        ):
            audio_visual_fusion = BaseAVFusion(
                cfg         = cfg,
                stage_idx   = stage_idx
            )
            self.add_module("avfusion", audio_visual_fusion)

    def forward(self, xv, xa):
        # performs computation on the convolutions
        for i in range(self.num_blocks):
            res_block = getattr(self, "avres_{}".format(i+1))
            xv, xa = res_block(xv, xa)
        
        if hasattr(self, "avfusion"):
            xv, xa = self.avfusion(xv, xa)

        return xv, xa

class BaseAVBranch(BaseModule):
    """
    Constructs the base convolution branch for ResNet based approaches.
    """
    def __init__(self, cfg, block_idx, construct_branch=True):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
            construct_branch (bool):   whether or not to automatically construct the branch.
                In the cases that the branch is not automatically contructed, e.g., some extra
                parameters need to be specified before branch construction, the branch could be
                constructed by "self._construct_branch" function.
        """
        super(BaseAVBranch, self).__init__(cfg)
        self.cfg = cfg
        update_av_conv_params(cfg, self, block_idx)
        if construct_branch:
            self._construct_branch()
    
    def _construct_branch(self):
        if self.transformation == 'simple_block':
            # for resnet with the number of layers lower than 34, simple block is constructed.
            self._construct_simple_block()
        elif self.transformation == 'bottleneck':
            # for resnet with the number of layers higher than 34, bottleneck is constructed.
            self._construct_bottleneck()
    
    @abc.abstractmethod
    def _construct_simple_block(self):
        return
    
    @abc.abstractmethod
    def _construct_bottleneck(self):
        return
    
    @abc.abstractmethod
    def forward(self, x):
        return

@HEAD_REGISTRY.register()
class BaseAVHead(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseAVHead, self).__init__()
        self.cfg = cfg
        self.dim_video       = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        self.dim_audio       = cfg.AUDIO.BACKBONE.NUM_OUT_FEATURES
        self.num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        self.dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        self.activation_func = cfg.VIDEO.HEAD.ACTIVATION

        self.audio_downsampling_mel = cfg.AUDIO.HEAD.DOWNSAMPLING_MEL
        self.audio_downsampling_temporal = cfg.AUDIO.HEAD.DOWNSAMPLING_TEMPORAL
        self.audio_dropout = cfg.AUDIO.DROPOUT_RATE
        
        self._construct_head()

    def _construct_head(self):
        if self.audio_downsampling_mel or self.audio_downsampling_temporal:
            self.audio_avgpool = nn.AvgPool2d(
                kernel_size=[
                    2 if self.audio_downsampling_temporal else 1,
                    2 if self.audio_downsampling_mel else 1
                ]
            )

        self.video_gap = nn.AdaptiveAvgPool3d(1)
        self.audio_gap = nn.AdaptiveAvgPool2d(1)
        
        if self.dropout_rate > 0.0: 
            self.dropout = nn.Dropout(self.dropout_rate)

        self.out = nn.Linear(self.dim_video+self.dim_audio, self.num_classes, bias=True)

        if self.activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif self.activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(self.activation_func)
            )

    def forward(self, x):
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xv = self.video_gap(xv)
        xv = xv.view(xv.shape[0], -1)
        xa = self.audio_gap(xa)
        xa = xa.view(xa.shape[0], -1)

        out = torch.cat((xv, xa), dim=1)

        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out = self.out(out)

        if not self.training:
            out = self.activation(out)
        
        out = out.view(out.shape[0], -1)
        return out, torch.cat((xv, xa), dim=1)

@HEAD_REGISTRY.register()
class BaseAVHeadx2(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseAVHeadx2, self).__init__()
        self.cfg = cfg
        self.dim_video       = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        self.dim_audio       = cfg.AUDIO.BACKBONE.NUM_OUT_FEATURES
        self.num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        self.dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        self.activation_func = cfg.VIDEO.HEAD.ACTIVATION

        self.audio_downsampling_mel = cfg.AUDIO.HEAD.DOWNSAMPLING_MEL
        self.audio_downsampling_temporal = cfg.AUDIO.HEAD.DOWNSAMPLING_TEMPORAL
        self.audio_dropout = cfg.AUDIO.DROPOUT_RATE
        
        self._construct_head()

    def _construct_head(self):
        if self.audio_downsampling_mel or self.audio_downsampling_temporal:
            self.audio_avgpool = nn.AvgPool2d(
                kernel_size=[
                    2 if self.audio_downsampling_temporal else 1,
                    2 if self.audio_downsampling_mel else 1
                ]
            )

        self.video_gap = nn.AdaptiveAvgPool3d(1)
        self.audio_gap = nn.AdaptiveAvgPool2d(1)
        
        if self.dropout_rate > 0.0: 
            self.dropout = nn.Dropout(self.dropout_rate)

        self.out1 = nn.Linear(self.dim_video+self.dim_audio, self.num_classes[0], bias=True)
        self.out2 = nn.Linear(self.dim_video+self.dim_audio, self.num_classes[1], bias=True)

        if self.activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif self.activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(self.activation_func)
            )

    def forward(self, x):
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xv = self.video_gap(xv)
        xv = xv.view(xv.shape[0], -1)
        xa = self.audio_gap(xa)
        xa = xa.view(xa.shape[0], -1)

        out = torch.cat((xv, xa), dim=1)

        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out1 = self.out1(out)
        out2 = self.out2(out)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        
        out1 = out1.view(out.shape[0], -1)
        out2 = out2.view(out.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, torch.cat((xv, xa), dim=1)

@HEAD_REGISTRY.register()
class BaseAVHeadAudioOnly(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseAVHeadAudioOnly, self).__init__()
        self.cfg = cfg
        self.dim_video       = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        self.dim_audio       = cfg.AUDIO.BACKBONE.NUM_OUT_FEATURES
        self.num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        self.dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        self.activation_func = cfg.VIDEO.HEAD.ACTIVATION

        self.audio_downsampling_mel = cfg.AUDIO.HEAD.DOWNSAMPLING_MEL
        self.audio_downsampling_temporal = cfg.AUDIO.HEAD.DOWNSAMPLING_TEMPORAL
        self.audio_dropout = cfg.AUDIO.DROPOUT_RATE
        
        self._construct_head()

    def _construct_head(self):
        if self.audio_downsampling_mel or self.audio_downsampling_temporal:
            self.audio_avgpool = nn.AvgPool2d(
                kernel_size=[
                    2 if self.audio_downsampling_temporal else 1,
                    2 if self.audio_downsampling_mel else 1
                ]
            )

        self.video_gap = nn.AdaptiveAvgPool3d(1)
        self.audio_gap = nn.AdaptiveAvgPool2d(1)
        
        if self.dropout_rate > 0.0: 
            self.dropout = nn.Dropout(self.dropout_rate)

        self.out = nn.Linear(self.dim_audio, self.num_classes, bias=True)

        if self.activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif self.activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(self.activation_func)
            )

    def forward(self, x):
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xa = self.audio_gap(xa)
        xa = xa.view(xa.shape[0], -1)

        if hasattr(self, "dropout"):
            out = self.dropout(xa)
        else:
            out = xa
        out = self.out(out)

        if not self.training:
            out = self.activation(out)
        
        out = out.view(out.shape[0], -1)
        return out, xa

@HEAD_REGISTRY.register()
class BaseAVHeadx2AudioOnly(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseAVHeadx2AudioOnly, self).__init__()
        self.cfg = cfg
        self.dim_video       = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        self.dim_audio       = cfg.AUDIO.BACKBONE.NUM_OUT_FEATURES
        self.num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        self.dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        self.activation_func = cfg.VIDEO.HEAD.ACTIVATION

        self.audio_downsampling_mel = cfg.AUDIO.HEAD.DOWNSAMPLING_MEL
        self.audio_downsampling_temporal = cfg.AUDIO.HEAD.DOWNSAMPLING_TEMPORAL
        self.audio_dropout = cfg.AUDIO.DROPOUT_RATE
        
        self._construct_head()

    def _construct_head(self):
        if self.audio_downsampling_mel or self.audio_downsampling_temporal:
            self.audio_avgpool = nn.AvgPool2d(
                kernel_size=[
                    2 if self.audio_downsampling_temporal else 1,
                    2 if self.audio_downsampling_mel else 1
                ]
            )

        self.audio_gap = nn.AdaptiveAvgPool2d(1)
        
        if self.dropout_rate > 0.0: 
            self.dropout = nn.Dropout(self.dropout_rate)

        self.out1 = nn.Linear(self.dim_audio, self.num_classes[0], bias=True)
        self.out2 = nn.Linear(self.dim_audio, self.num_classes[1], bias=True)

        if self.activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif self.activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(self.activation_func)
            )

    def forward(self, x):
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xa = self.audio_gap(xa)
        xa = xa.view(xa.shape[0], -1)

        if hasattr(self, "dropout"):
            out = self.dropout(xa)
        else:
            out = xa
        out1 = self.out1(out)
        out2 = self.out2(out)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        
        out1 = out1.view(out.shape[0], -1)
        out2 = out2.view(out.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, xa

@HEAD_REGISTRY.register()
class BaseAVHeadx2VideoOnly(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseAVHeadx2VideoOnly, self).__init__()
        self.cfg = cfg
        self.dim_video       = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        self.dim_audio       = cfg.AUDIO.BACKBONE.NUM_OUT_FEATURES
        self.num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        self.dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        self.activation_func = cfg.VIDEO.HEAD.ACTIVATION
        
        self._construct_head()

    def _construct_head(self):

        self.video_gap = nn.AdaptiveAvgPool3d(1)
        
        if self.dropout_rate > 0.0: 
            self.dropout = nn.Dropout(self.dropout_rate)

        self.out1 = nn.Linear(self.dim_video, self.num_classes[0], bias=True)
        self.out2 = nn.Linear(self.dim_video, self.num_classes[1], bias=True)

        if self.activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif self.activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(self.activation_func)
            )

    def forward(self, x):
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xv = self.video_gap(xv)
        xv = xv.view(xv.shape[0], -1)

        if hasattr(self, "dropout"):
            out = self.dropout(xv)
        else:
            out = xv
        out1 = self.out1(out)
        out2 = self.out2(out)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        
        out1 = out1.view(out.shape[0], -1)
        out2 = out2.view(out.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, xv

@HEAD_REGISTRY.register()
class BaseAVHeadx2AudioDownsize(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseAVHeadx2AudioDownsize, self).__init__()
        self.cfg = cfg
        self.dim_video       = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES

        self.dim_audio_original = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES

        self.dim_audio       = cfg.AUDIO.BACKBONE.NUM_OUT_FEATURES
        self.num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        self.dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        self.activation_func = cfg.VIDEO.HEAD.ACTIVATION

        self.audio_downsampling_mel = cfg.AUDIO.HEAD.DOWNSAMPLING_MEL
        self.audio_downsampling_temporal = cfg.AUDIO.HEAD.DOWNSAMPLING_TEMPORAL
        self.audio_dropout = cfg.AUDIO.DROPOUT_RATE
        
        self._construct_head()

    def _construct_head(self):
        if self.audio_downsampling_mel or self.audio_downsampling_temporal:
            self.audio_avgpool = nn.AvgPool2d(
                kernel_size=[
                    2 if self.audio_downsampling_temporal else 1,
                    2 if self.audio_downsampling_mel else 1
                ]
            )

        self.video_gap = nn.AdaptiveAvgPool3d(1)
        self.audio_gap = nn.AdaptiveAvgPool2d(1)

        self.audio_premap = nn.Linear(self.dim_audio_original, self.dim_audio, bias=True)
        self.audio_premap_relu = nn.ReLU(inplace=True)
        
        if self.dropout_rate > 0.0: 
            self.dropout = nn.Dropout(self.dropout_rate)

        self.out1 = nn.Linear(self.dim_video+self.dim_audio, self.num_classes[0], bias=True)
        self.out2 = nn.Linear(self.dim_video+self.dim_audio, self.num_classes[1], bias=True)

        if self.activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif self.activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(self.activation_func)
            )

    def forward(self, x):
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xv = self.video_gap(xv)
        xv = xv.view(xv.shape[0], -1)
        xa = self.audio_gap(xa)
        xa = xa.view(xa.shape[0], -1)

        xa = self.audio_premap(xa)
        xa = self.audio_premap_relu(xa)

        out = torch.cat((xv, xa), dim=1)

        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out1 = self.out1(out)
        out2 = self.out2(out)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        
        out1 = out1.view(out.shape[0], -1)
        out2 = out2.view(out.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, torch.cat((xv, xa), dim=1)

@HEAD_REGISTRY.register()
class BaseAVHeadx2MLP(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseAVHeadx2MLP, self).__init__()
        self.cfg = cfg
        self.dim_video       = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        self.dim_audio       = cfg.AUDIO.BACKBONE.NUM_OUT_FEATURES
        self.num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        self.dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        self.activation_func = cfg.VIDEO.HEAD.ACTIVATION

        self.audio_downsampling_mel = cfg.AUDIO.HEAD.DOWNSAMPLING_MEL
        self.audio_downsampling_temporal = cfg.AUDIO.HEAD.DOWNSAMPLING_TEMPORAL
        self.audio_dropout = cfg.AUDIO.DROPOUT_RATE
        
        self._construct_head()

    def _construct_head(self):
        if self.audio_downsampling_mel or self.audio_downsampling_temporal:
            self.audio_avgpool = nn.AvgPool2d(
                kernel_size=[
                    2 if self.audio_downsampling_temporal else 1,
                    2 if self.audio_downsampling_mel else 1
                ]
            )

        self.video_gap = nn.AdaptiveAvgPool3d(1)
        self.audio_gap = nn.AdaptiveAvgPool2d(1)
        
        if self.dropout_rate > 0.0: 
            self.dropout = nn.Dropout(self.dropout_rate)

        self.representation = nn.Linear(self.dim_video+self.dim_audio, (self.dim_video+self.dim_audio)//2)
        self.relu = nn.ReLU(inplace=True)

        self.out1 = nn.Linear((self.dim_video+self.dim_audio)//2, self.num_classes[0], bias=True)
        self.out2 = nn.Linear((self.dim_video+self.dim_audio)//2, self.num_classes[1], bias=True)

        if self.activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif self.activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(self.activation_func)
            )

    def forward(self, x):
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xv = self.video_gap(xv)
        xv = xv.view(xv.shape[0], -1)
        xa = self.audio_gap(xa)
        xa = xa.view(xa.shape[0], -1)

        out = torch.cat((xv, xa), dim=1)

        out = self.representation(out)
        out = self.relu(out)
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out1 = self.out1(out)
        out2 = self.out2(out)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        
        out1 = out1.view(out.shape[0], -1)
        out2 = out2.view(out.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, torch.cat((xv, xa), dim=1)

@HEAD_REGISTRY.register()
class BaseAVHeadx2SepDropout(nn.Module):
    """
    Constructs base head.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseAVHeadx2SepDropout, self).__init__()
        self.cfg = cfg
        self.dim_video       = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        self.dim_audio       = cfg.AUDIO.BACKBONE.NUM_OUT_FEATURES
        self.num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        self.dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        self.activation_func = cfg.VIDEO.HEAD.ACTIVATION

        self.audio_downsampling_mel = cfg.AUDIO.HEAD.DOWNSAMPLING_MEL
        self.audio_downsampling_temporal = cfg.AUDIO.HEAD.DOWNSAMPLING_TEMPORAL
        self.audio_dropout = cfg.AUDIO.DROPOUT_RATE
        
        self._construct_head()

    def _construct_head(self):
        if self.audio_downsampling_mel or self.audio_downsampling_temporal:
            self.audio_avgpool = nn.AvgPool2d(
                kernel_size=[
                    2 if self.audio_downsampling_temporal else 1,
                    2 if self.audio_downsampling_mel else 1
                ]
            )

        self.video_gap = nn.AdaptiveAvgPool3d(1)
        self.audio_gap = nn.AdaptiveAvgPool2d(1)
        
        if self.dropout_rate > 0.0: 
            self.dropout = nn.Dropout(self.dropout_rate)

        self.out1 = nn.Linear(self.dim_video+self.dim_audio, self.num_classes[0], bias=True)
        self.out2 = nn.Linear(self.dim_video+self.dim_audio, self.num_classes[1], bias=True)

        if self.activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif self.activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(self.activation_func)
            )

    def forward(self, x):
        if isinstance(x, dict):
            xv = x["video"]
            xa = x["audio"]

        xv = self.video_gap(xv)
        xv = xv.view(xv.shape[0], -1)
        xa = self.audio_gap(xa)
        xa = xa.view(xa.shape[0], -1)


        if hasattr(self, "dropout"):
            out = torch.cat((self.dropout(xv), self.dropout(xa)), dim=1)
        else:
            out = torch.cat((xv, xa), dim=1)
        out1 = self.out1(out)
        out2 = self.out2(out)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        
        out1 = out1.view(out.shape[0], -1)
        out2 = out2.view(out.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, torch.cat((xv, xa), dim=1)

class BaseAVFusion(nn.Module):
    def __init__(self, cfg, stage_idx):
        super().__init__()

        self.a2v_enable = cfg.VIDEO.BACKBONE.FUSION_FROM_AUDIO.ENABLE[stage_idx-1]

        self.v2a_enable = cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.ENABLE[stage_idx-1]
        self.v2a_kernel_size = cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.KERNEL_SIZE

        dim_in = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_idx]
        if self.v2a_enable:
            self.vpool = nn.AdaptiveAvgPool3d((1,1,1))
            self.v2a = nn.Conv3d(
                in_channels=dim_in,
                out_channels=dim_in,
                kernel_size=[self.v2a_kernel_size, 1, 1],
                stride=1,
                padding=[self.v2a_kernel_size//2, 0, 0]
            )
            self.v2a.no_init = True
            self.v2a.weight.data.zero_()
            self.v2a.bias.data.zero_()
        if self.a2v_enable:
            # block idx set to 1 as all the blocks are before fusion
            audio_length = calculate_audio_length(cfg, stage_idx, 1) 
            video_length = calculate_video_length(cfg, stage_idx, 1)
            self.pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
            self.apool = nn.AdaptiveAvgPool2d((None, 1))
            self.a2v = nn.Conv2d(
                in_channels = dim_in,
                out_channels = dim_in,
                kernel_size = [math.ceil(audio_length/video_length), 1],
                stride=[math.ceil(audio_length/video_length), 1],
                padding=0
            )
            self.a2v.no_init = True
            self.a2v.weight.data.zero_()
            self.a2v.bias.data.zero_()

    def forward(self, xv, xa):
        if self.v2a_enable:
            x_v2a = self.vpool(xv)
            x_v2a = self.v2a(x_v2a).squeeze(-1)
        if self.a2v_enable:
            x_a2v = F.pad(xa, (0,0,0,self.pad_length), "constant", 0)
            x_a2v = self.a2v(x_a2v)
            x_a2v = self.apool(x_a2v).unsqueeze(-1)
        xv = xv + x_a2v
        xa = xa + x_v2a
        return xv, xa


class BaseAudioVisualStageV2(nn.Module):
    """
    ResNet Stage containing several blocks.
    """
    def __init__(
        self,
        cfg,
        num_blocks,
        stage_idx,
    ):
        """
        Args:
            num_blocks (int): number of blocks contained in this res-stage.
            stage_idx  (int): the stage index of this res-stage.
        """
        super(BaseAudioVisualStageV2, self).__init__()
        self.cfg = cfg
        self.num_blocks = num_blocks       
        if stage_idx == 4 and cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE:
            self.only_video_feature = True
        else:
            self.only_video_feature = False
        self._construct_stage(
            cfg                     = cfg,
            stage_idx               = stage_idx,
        )
        
    def _construct_stage(
        self,
        cfg,
        stage_idx,
    ):
        if (
            self.cfg.VIDEO.BACKBONE.FUSION_FROM_AUDIO.ENABLE[stage_idx-1] or
            self.cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.ENABLE[stage_idx-1]
        ) and self.cfg.VIDEO.BACKBONE.FUSION_FROM_AUDIO.TADA:
            audio_visual_fusion = TAdaAVFusion(
                cfg         = cfg,
                stage_idx   = stage_idx
            )
            self.add_module("avfusion", audio_visual_fusion)
        elif (
            self.cfg.VIDEO.BACKBONE.FUSION_FROM_AUDIO.ENABLE[stage_idx-1] or
            self.cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.ENABLE[stage_idx-1]
        ):
            audio_visual_fusion = BaseAVFusionV2(
                cfg         = cfg,
                stage_idx   = stage_idx
            )
            self.add_module("avfusion", audio_visual_fusion)

        for i in range(self.num_blocks):
            res_block = BaseAVBlockV2(
                cfg                 = cfg,
                block_idx           = [stage_idx, i],
                last                = self.only_video_feature and (i == self.num_blocks-1)
            )
            self.add_module("avres_{}".format(i+1), res_block)

    def forward(self, xv, xa):

        if hasattr(self, "avfusion"):
            xv, xa = self.avfusion(xv, xa)

        # performs computation on the convolutions
        for i in range(self.num_blocks):
            res_block = getattr(self, "avres_{}".format(i+1))
            xv, xa = res_block(xv, xa)

        return xv, xa

class BaseAVBlockV2(nn.Module):
    """
    Constructs a base audio-visual block, composed of one shortcut and one conv branch for 
    each of the modalities.
    """
    def __init__(
        self,
        cfg,
        block_idx,
        last=False,
    ):
        """
        Args: 
            cfg         (Config): global config object. 
            block_idx   (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(BaseAVBlockV2, self).__init__()
        
        self.cfg = cfg
        self.video_branch_enable = cfg.VIDEO.ENABLE
        self.audio_branch_enable = cfg.AUDIO.ENABLE
        update_av_conv_params_v2(cfg, self, block_idx)
        self.last = last

        self._construct_block(
            cfg             = cfg,
            block_idx       = block_idx,
        )
            
    def _construct_block(
        self,
        cfg,
        block_idx,
    ):
        if (self.visual_dim_in != self.visual_num_filters or self.visual_downsampling) and self.video_branch_enable:
            # if the spatial size or the channel dimension is inconsistent for the input and the output 
            # of the block, a 1x1x1 3D conv need to be performed. 
            self.visual_short_cut = nn.Conv3d(
                self.visual_dim_in,
                self.visual_num_filters,
                kernel_size=1,
                stride=self.visual_stride,
                padding=0,
                bias=False
            )
            self.visual_short_cut_bn = nn.BatchNorm3d(
                self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt
            )
        if (self.audio_dim_in != self.audio_num_filters or self.audio_downsampling) and self.audio_branch_enable:
            self.audio_short_cut = nn.Conv2d(
                self.audio_dim_in,
                self.audio_num_filters,
                kernel_size=1,
                stride=self.audio_stride,
                padding=0,
                bias=False
            )
            self.audio_short_cut_bn = nn.BatchNorm2d(
                self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt
            )
        self.conv_branch = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, block_idx, self.last)
        self.relu = nn.ReLU(inplace=True)
            
    def forward(self, xv, xa):
        short_cut_v = xv
        short_cut_a = xa
        if hasattr(self, "visual_short_cut"):
            short_cut_v = self.visual_short_cut_bn(self.visual_short_cut(short_cut_v))
        if hasattr(self, "audio_short_cut"):
            short_cut_a = self.audio_short_cut_bn(self.audio_short_cut(short_cut_a))
        
        xv, xa = self.conv_branch(xv, xa)
        if self.video_branch_enable:
            xv = self.relu(short_cut_v + xv)
        if self.audio_branch_enable and not (self.last and self.cfg.VIDEO.BACKBONE.BRANCH.ONLY_VIDEO_FEATURE):
            xa = self.relu(short_cut_a + xa)

        return xv, xa

class BaseAVBranchV2(BaseModule):
    """
    Constructs the base convolution branch for ResNet based approaches.
    """
    def __init__(self, cfg, block_idx, construct_branch=True):
        """
        Args: 
            cfg              (Config): global config object. 
            block_idx        (list):   list of [stage_id, block_id], both starting from 0.
            construct_branch (bool):   whether or not to automatically construct the branch.
                In the cases that the branch is not automatically contructed, e.g., some extra
                parameters need to be specified before branch construction, the branch could be
                constructed by "self._construct_branch" function.
        """
        super(BaseAVBranchV2, self).__init__(cfg)
        self.cfg = cfg
        update_av_conv_params_v2(cfg, self, block_idx)
        if construct_branch:
            self._construct_branch()
    
    def _construct_branch(self):
        if self.transformation == 'simple_block':
            # for resnet with the number of layers lower than 34, simple block is constructed.
            self._construct_simple_block()
        elif self.transformation == 'bottleneck':
            # for resnet with the number of layers higher than 34, bottleneck is constructed.
            self._construct_bottleneck()
    
    @abc.abstractmethod
    def _construct_simple_block(self):
        return
    
    @abc.abstractmethod
    def _construct_bottleneck(self):
        return
    
    @abc.abstractmethod
    def forward(self, x):
        return

@STEM_REGISTRY.register()
class Base2DAVStemV2(BaseModule):
    """
    Constructs basic AudioVisual ResNet 2D Stem.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base2DAVStemV2, self).__init__(cfg)

        self.cfg = cfg

        self.video_branch_enable = cfg.VIDEO.ENABLE
        self.audio_branch_enable = cfg.AUDIO.ENABLE

        # loading the config for downsampling
        visual_downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        if visual_downsampling:
            self.visual_stride = [1, 2, 2]
        else:
            self.visual_stride = [1, 1, 1]
        self.audio_downsampling_temporal    = cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        self.audio_downsampling_mel         = cfg.AUDIO.BACKBONE.DOWNSAMPLING_MEL[0]
        self.audio_pool_size = [
            2 if self.audio_downsampling_temporal else 1,
            2 if self.audio_downsampling_mel else 1
        ]
        self.visual_kernel_size = cfg.VIDEO.BACKBONE.KERNEL_SIZE[0]
        self.visual_num_filters = cfg.VIDEO.BACKBONE.NUM_FILTERS[0]
        self.audio_num_filters = cfg.AUDIO.BACKBONE.NUM_FILTERS[0]
        self.visual_num_input_channels = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS

        self.bn_eps = cfg.BN.EPS
        self.bn_mmt = cfg.BN.MOMENTUM

        self.audio_kernel_size  = cfg.AUDIO.BACKBONE.KERNEL_SIZE[0]
        self.window_size        = cfg.AUDIO.BACKBONE.STEM.WINDOW_SIZE
        self.hop_length         = cfg.AUDIO.BACKBONE.STEM.HOP_LENGTH
        self.mel_bins           = cfg.AUDIO.BACKBONE.STEM.MEL_BINS
        self.frange             = cfg.AUDIO.BACKBONE.STEM.FRANGE
        self.sample_rate        = cfg.AUDIO.SAMPLE_RATE
        self.audio_dropout      = cfg.AUDIO.DROPOUT_RATE
        self.audio_num_input_channels = cfg.AUDIO.BACKBONE.NUM_INPUT_CHANNELS

        self.audio_aug_time_drop_width = cfg.AUDIO.AUGMENTATION.TIME_DROP_WIDTH
        self.audio_aug_time_stripes_num = cfg.AUDIO.AUGMENTATION.TIME_STRIPES_NUM
        self.audio_aug_freq_drop_width = cfg.AUDIO.AUGMENTATION.FREQ_DROP_WIDTH
        self.audio_aug_freq_stripes_num = cfg.AUDIO.AUGMENTATION.FREQ_STRIPES_NUM
        
        self._construct_block()

    def _construct_block(
        self, 
    ):
        if self.video_branch_enable:
            # visual
            self.visual_a = nn.Conv3d(
                self.visual_num_input_channels,
                self.visual_num_filters,
                kernel_size = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
                stride      = [1, self.visual_stride[1], self.visual_stride[2]],
                padding     = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
                bias        = False
            )
            self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.visual_a_relu = nn.ReLU(inplace=True)

        if self.audio_branch_enable:
            # audio

            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=self.window_size, hop_length=self.hop_length, 
                win_length=self.window_size, window='hann', center=True, pad_mode='reflect', 
                freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=self.sample_rate, n_fft=self.window_size, 
                n_mels=self.mel_bins, fmin=self.frange[0], fmax=self.frange[1], 
                ref=1.0, amin=1e-10, top_db=None, 
                freeze_parameters=True)

            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=self.audio_aug_time_drop_width, 
                time_stripes_num=self.audio_aug_time_stripes_num, 
                freq_drop_width=self.audio_aug_freq_drop_width, 
                freq_stripes_num=self.audio_aug_freq_stripes_num)
            
            self.audio_initial_bn = nn.BatchNorm2d(self.mel_bins, eps=self.bn_eps, momentum=self.bn_mmt)

            self.audio_a = nn.Conv2d(
                self.audio_num_input_channels,
                self.audio_num_filters,
                kernel_size = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride      = [1, 1],
                padding     = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias=False
            )
            self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_a_relu = nn.ReLU(inplace=True)

            if self.audio_downsampling_mel or self.audio_downsampling_temporal:
                self.audio_avgpool = nn.AvgPool2d(
                    kernel_size=self.audio_pool_size,
                )
            self.audio_dropout = nn.Dropout(p=self.audio_dropout)

    def forward(self, xv, xa):
        if self.video_branch_enable:
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)
        
        if self.audio_branch_enable:

            xa = self.spectrogram_extractor(xa.squeeze(1))
            xa = self.logmel_extractor(xa)

            xa = xa.transpose(1, 3)
            xa = self.audio_initial_bn(xa)
            xa = xa.transpose(1, 3)

            if self.training:
                xa = self.spec_augmenter(xa)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            if hasattr(self, "audio_avgpool"):
                xa = self.audio_avgpool(xa)

            xa = self.audio_dropout(xa)

        return xv, xa

class BaseAVFusionV2(nn.Module):
    def __init__(self, cfg, stage_idx):
        super().__init__()

        self.a2v_enable = cfg.VIDEO.BACKBONE.FUSION_FROM_AUDIO.ENABLE[stage_idx-1]

        self.v2a_enable = cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.ENABLE[stage_idx-1]
        self.v2a_kernel_size = cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.KERNEL_SIZE

        visual_dim_in = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_idx-1]
        audio_dim_in = cfg.AUDIO.BACKBONE.NUM_FILTERS[stage_idx-1]
        if self.v2a_enable:
            self.vpool = nn.AdaptiveAvgPool3d((1,1,1))
            self.visual_a = nn.Conv3d(
                in_channels=visual_dim_in,
                out_channels=audio_dim_in,
                kernel_size=[self.v2a_kernel_size, 1, 1],
                stride=1,
                padding=[self.v2a_kernel_size//2, 0, 0]
            )
            self.visual_a_bn = nn.BatchNorm3d(audio_dim_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
            self.visual_a_bn.no_init=True
            self.visual_a_bn.weight.data.zero_()
            self.visual_a_bn.bias.data.zero_()
            self.visual_a_relu = nn.ReLU(inplace=True)
        if self.a2v_enable:
            # block idx set to 1 as all the blocks are before fusion
            audio_length = calculate_audio_length(cfg, stage_idx, 0) 
            video_length = calculate_video_length(cfg, stage_idx, 0)
            self.pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
            self.apool = nn.AdaptiveAvgPool2d((None, 1))
            self.audio_a = nn.Conv2d(
                in_channels = audio_dim_in,
                out_channels = audio_dim_in,
                kernel_size = [math.ceil(audio_length/video_length), 1],
                stride=[math.ceil(audio_length/video_length), 1],
                padding=0
            )
            self.audio_b = nn.Conv2d(
                in_channels = audio_dim_in,
                out_channels = visual_dim_in,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.audio_b_bn = nn.BatchNorm2d(visual_dim_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
            self.audio_b_bn.no_init=True
            self.audio_b_bn.weight.data.zero_()
            self.audio_b_bn.bias.data.zero_()
            self.audio_b_relu = nn.ReLU(inplace=True)

    def forward(self, xv, xa):
        if self.v2a_enable:
            x_v2a = self.vpool(xv)
            x_v2a = self.visual_a(x_v2a)
            x_v2a = self.visual_a_bn(x_v2a).squeeze(-1)
            x_v2a = self.visual_a_relu(x_v2a)
        if self.a2v_enable:
            x_a2v = F.pad(xa, (0,0,0,self.pad_length), "constant", 0)
            x_a2v = self.audio_a(x_a2v)
            x_a2v = self.apool(x_a2v)
            x_a2v = self.audio_b(x_a2v)
            x_a2v = self.audio_b_bn(x_a2v)
            x_a2v = self.audio_b_relu(x_a2v).unsqueeze(-1)

        xv = xv + x_a2v
        xa = xa + x_v2a
        return xv, xa

@STEM_REGISTRY.register()
class Base2DAVStemV3(BaseModule):
    """
    Constructs basic AudioVisual ResNet 2D Stem.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base2DAVStemV3, self).__init__(cfg)

        self.cfg = cfg

        self.video_branch_enable = cfg.VIDEO.ENABLE
        self.audio_branch_enable = cfg.AUDIO.ENABLE

        # loading the config for downsampling
        visual_downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        if visual_downsampling:
            self.visual_stride = [1, 2, 2]
        else:
            self.visual_stride = [1, 1, 1]
        self.audio_downsampling_temporal    = cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        self.audio_downsampling_mel         = cfg.AUDIO.BACKBONE.DOWNSAMPLING_MEL[0]
        self.audio_pool_size = [
            2 if self.audio_downsampling_temporal else 1,
            2 if self.audio_downsampling_mel else 1
        ]
        self.visual_kernel_size = cfg.VIDEO.BACKBONE.KERNEL_SIZE[0]
        self.visual_num_filters = cfg.VIDEO.BACKBONE.NUM_FILTERS[0]
        self.audio_num_filters = cfg.AUDIO.BACKBONE.NUM_FILTERS[0]
        self.visual_num_input_channels = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS

        self.bn_eps = cfg.BN.EPS
        self.bn_mmt = cfg.BN.MOMENTUM

        self.n_fft = 2048
        self.audio_kernel_size  = cfg.AUDIO.BACKBONE.KERNEL_SIZE[0]
        self.window_size        = cfg.AUDIO.BACKBONE.STEM.WINDOW_SIZE
        self.hop_length         = cfg.AUDIO.BACKBONE.STEM.HOP_LENGTH
        self.mel_bins           = cfg.AUDIO.BACKBONE.STEM.MEL_BINS
        self.frange             = cfg.AUDIO.BACKBONE.STEM.FRANGE
        self.sample_rate        = cfg.AUDIO.SAMPLE_RATE
        self.audio_dropout      = cfg.AUDIO.DROPOUT_RATE
        self.audio_num_input_channels = cfg.AUDIO.BACKBONE.NUM_INPUT_CHANNELS

        self.audio_aug_time_drop_width = cfg.AUDIO.AUGMENTATION.TIME_DROP_WIDTH
        self.audio_aug_time_stripes_num = cfg.AUDIO.AUGMENTATION.TIME_STRIPES_NUM
        self.audio_aug_freq_drop_width = cfg.AUDIO.AUGMENTATION.FREQ_DROP_WIDTH
        self.audio_aug_freq_stripes_num = cfg.AUDIO.AUGMENTATION.FREQ_STRIPES_NUM
        
        self._construct_block()

    def _construct_block(
        self, 
    ):
        if self.video_branch_enable:
            # visual
            self.visual_a = nn.Conv3d(
                self.visual_num_input_channels,
                self.visual_num_filters,
                kernel_size = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
                stride      = [1, self.visual_stride[1], self.visual_stride[2]],
                padding     = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
                bias        = False
            )
            self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.visual_a_relu = nn.ReLU(inplace=True)

        if self.audio_branch_enable:
            # audio

            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, 
                win_length=self.window_size, window='hann', center=True, pad_mode='constant', 
                freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=self.sample_rate, n_fft=self.n_fft, 
                n_mels=self.mel_bins, fmin=self.frange[0], fmax=self.frange[1], 
                ref=1.0, amin=1e-10, top_db=None, 
                freeze_parameters=True)

            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=self.audio_aug_time_drop_width, 
                time_stripes_num=self.audio_aug_time_stripes_num, 
                freq_drop_width=self.audio_aug_freq_drop_width, 
                freq_stripes_num=self.audio_aug_freq_stripes_num)
            
            self.audio_initial_bn = nn.BatchNorm2d(self.mel_bins, eps=self.bn_eps, momentum=self.bn_mmt)

            self.audio_a = nn.Conv2d(
                self.audio_num_input_channels,
                self.audio_num_filters,
                kernel_size = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride      = [1, 1],
                padding     = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias=False
            )
            self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_a_relu = nn.ReLU(inplace=True)

            if self.audio_downsampling_mel or self.audio_downsampling_temporal:
                self.audio_avgpool = nn.AvgPool2d(
                    kernel_size=self.audio_pool_size,
                )
            self.audio_dropout = nn.Dropout(p=self.audio_dropout)

    def forward(self, xv, xa):
        if self.video_branch_enable:
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)
        
        if self.audio_branch_enable:

            xa = self.spectrogram_extractor(xa.squeeze(1))
            xa = self.logmel_extractor(xa)

            xa = xa.transpose(1, 3)
            xa = self.audio_initial_bn(xa)
            xa = xa.transpose(1, 3)

            if self.training:
                xa = self.spec_augmenter(xa)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            if hasattr(self, "audio_avgpool"):
                xa = self.audio_avgpool(xa)

            xa = self.audio_dropout(xa)

        return xv, xa

@STEM_REGISTRY.register()
class Base2DAVStemV4(BaseModule):
    """
    Constructs basic AudioVisual ResNet 2D Stem.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base2DAVStemV4, self).__init__(cfg)

        self.cfg = cfg

        self.video_branch_enable = cfg.VIDEO.ENABLE
        self.audio_branch_enable = cfg.AUDIO.ENABLE

        # loading the config for downsampling
        visual_downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        if visual_downsampling:
            self.visual_stride = [1, 2, 2]
        else:
            self.visual_stride = [1, 1, 1]
        self.audio_downsampling_temporal    = cfg.AUDIO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        self.audio_downsampling_mel         = cfg.AUDIO.BACKBONE.DOWNSAMPLING_MEL[0]
        self.audio_pool_size = [
            2 if self.audio_downsampling_temporal else 1,
            2 if self.audio_downsampling_mel else 1
        ]
        self.visual_kernel_size = cfg.VIDEO.BACKBONE.KERNEL_SIZE[0]
        self.visual_num_filters = cfg.VIDEO.BACKBONE.NUM_FILTERS[0]
        self.audio_num_filters = cfg.AUDIO.BACKBONE.NUM_FILTERS[0]
        self.visual_num_input_channels = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS

        self.bn_eps = cfg.BN.EPS
        self.bn_mmt = cfg.BN.MOMENTUM

        self.n_fft = 2048
        self.audio_kernel_size  = cfg.AUDIO.BACKBONE.KERNEL_SIZE[0]
        self.window_size        = cfg.AUDIO.BACKBONE.STEM.WINDOW_SIZE
        self.hop_length         = cfg.AUDIO.BACKBONE.STEM.HOP_LENGTH
        self.mel_bins           = cfg.AUDIO.BACKBONE.STEM.MEL_BINS
        self.frange             = cfg.AUDIO.BACKBONE.STEM.FRANGE
        self.sample_rate        = cfg.AUDIO.SAMPLE_RATE
        self.audio_dropout      = cfg.AUDIO.DROPOUT_RATE
        self.audio_num_input_channels = cfg.AUDIO.BACKBONE.NUM_INPUT_CHANNELS

        self.audio_aug_time_drop_width = cfg.AUDIO.AUGMENTATION.TIME_DROP_WIDTH
        self.audio_aug_time_stripes_num = cfg.AUDIO.AUGMENTATION.TIME_STRIPES_NUM
        self.audio_aug_freq_drop_width = cfg.AUDIO.AUGMENTATION.FREQ_DROP_WIDTH
        self.audio_aug_freq_stripes_num = cfg.AUDIO.AUGMENTATION.FREQ_STRIPES_NUM
        
        self._construct_block()

    def _construct_block(
        self, 
    ):
        if self.video_branch_enable:
            # visual
            self.visual_a = nn.Conv3d(
                self.visual_num_input_channels,
                self.visual_num_filters,
                kernel_size = [1, self.visual_kernel_size[1], self.visual_kernel_size[2]],
                stride      = [1, self.visual_stride[1], self.visual_stride[2]],
                padding     = [0, self.visual_kernel_size[1]//2, self.visual_kernel_size[2]//2],
                bias        = False
            )
            self.visual_a_bn = nn.BatchNorm3d(self.visual_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.visual_a_relu = nn.ReLU(inplace=True)

        if self.audio_branch_enable:
            # audio

            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, 
                win_length=self.window_size, window='hann', center=True, pad_mode='constant', 
                freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=self.sample_rate, n_fft=self.n_fft, 
                n_mels=self.mel_bins, fmin=self.frange[0], fmax=self.frange[1], 
                ref=1.0, amin=1e-10, top_db=None, 
                freeze_parameters=True)

            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=self.audio_aug_time_drop_width, 
                time_stripes_num=self.audio_aug_time_stripes_num, 
                freq_drop_width=self.audio_aug_freq_drop_width, 
                freq_stripes_num=self.audio_aug_freq_stripes_num)

            self.audio_a = nn.Conv2d(
                self.audio_num_input_channels,
                self.audio_num_filters,
                kernel_size = [self.audio_kernel_size[0], self.audio_kernel_size[1]],
                stride      = [1, 1],
                padding     = [self.audio_kernel_size[0]//2, self.audio_kernel_size[1]//2],
                bias=False
            )
            self.audio_a_bn = nn.BatchNorm2d(self.audio_num_filters, eps=self.bn_eps, momentum=self.bn_mmt)
            self.audio_a_relu = nn.ReLU(inplace=True)

            if self.audio_downsampling_mel or self.audio_downsampling_temporal:
                self.audio_avgpool = nn.AvgPool2d(
                    kernel_size=self.audio_pool_size,
                )
            self.audio_dropout = nn.Dropout(p=self.audio_dropout)

    def forward(self, xv, xa):
        if self.video_branch_enable:
            xv = self.visual_a(xv)
            xv = self.visual_a_bn(xv)
            xv = self.visual_a_relu(xv)
        
        if self.audio_branch_enable:

            xa = self.spectrogram_extractor(xa.squeeze(1))
            xa = self.logmel_extractor(xa)

            if self.training:
                xa = self.spec_augmenter(xa)

            xa = self.audio_a(xa)
            xa = self.audio_a_bn(xa)
            xa = self.audio_a_relu(xa)

            if hasattr(self, "audio_avgpool"):
                xa = self.audio_avgpool(xa)

            xa = self.audio_dropout(xa)

        return xv, xa

class TAdaAVFusion(nn.Module):
    def __init__(self, cfg, stage_idx):
        super().__init__()

        self.a2v_enable = cfg.VIDEO.BACKBONE.FUSION_FROM_AUDIO.ENABLE[stage_idx-1]

        self.v2a_enable = cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.ENABLE[stage_idx-1]
        self.v2a_kernel_size = cfg.AUDIO.BACKBONE.FUSION_FROM_VIDEO.KERNEL_SIZE

        visual_dim_in = cfg.VIDEO.BACKBONE.NUM_FILTERS[stage_idx-1]
        audio_dim_in = cfg.AUDIO.BACKBONE.NUM_FILTERS[stage_idx-1]
        audio_length = calculate_audio_length(cfg, stage_idx, 0) 
        video_length = calculate_video_length(cfg, stage_idx, 0)
        if self.v2a_enable:
            self.visual_a = TAdaConv2D_CinAda_AV(
                in_channels     = visual_dim_in,
                out_channels    = audio_dim_in,
                kernel_size     = [1, 3, 3],
                stride          = [1, 1, 1],
                padding         = [0, 1, 1],
                bias            = False,
                modality        = "v" # visual tada conv
            )
            self.visual_a_rf = av2v_routefunc_audioconvdown(
                visual_c_in=visual_dim_in,
                audio_c_in=audio_dim_in,
                ratio=4,
                video_length=video_length, 
                audio_length=audio_length,
                kernel_size=[1, 3]
            )
            self.visual_a_bn = nn.BatchNorm3d(audio_dim_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
            self.visual_a_relu = nn.ReLU(inplace=True)
            self.vpool = nn.AdaptiveAvgPool3d((1,1,1))
            self.visual_b = nn.Conv3d(
                in_channels=audio_dim_in,
                out_channels=audio_dim_in,
                kernel_size=1,
                stride=1,
                padding=0
            )
            self.visual_b_bn = nn.BatchNorm3d(audio_dim_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
            self.visual_b_bn.no_init=True
            self.visual_b_bn.weight.data.zero_()
            self.visual_b_bn.bias.data.zero_()
            self.visual_b_relu = nn.ReLU(inplace=True)
        if self.a2v_enable:
            self.audio_a = TAdaConv2D_CinAda_AV(
                in_channels     = audio_dim_in,
                out_channels    = visual_dim_in,
                kernel_size     = [3, 3],
                stride          = [1, 1],
                padding         = [1, 1],
                bias            = False,
                modality        = "a" # visual tada conv
            )
            self.audio_a_rf = av2a_routefunc_audioconvdown(
                visual_c_in=visual_dim_in,
                audio_c_in=audio_dim_in,
                ratio=4,
                video_length=video_length, 
                audio_length=audio_length,
                kernel_size=[1, 3]
            )
            self.audio_a_bn = nn.BatchNorm2d(visual_dim_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
            self.audio_a_relu = nn.ReLU(inplace=True)
            # block idx set to 1 as all the blocks are before fusion
            self.pad_length = math.ceil(audio_length/video_length)*video_length-audio_length
            self.apool = nn.AdaptiveAvgPool2d((None, 1))
            self.audio_b = nn.Conv2d(
                in_channels = visual_dim_in,
                out_channels = visual_dim_in,
                kernel_size = [math.ceil(audio_length/video_length), 1],
                stride=[math.ceil(audio_length/video_length), 1],
                padding=0
            )
            self.audio_c = nn.Conv2d(
                in_channels = visual_dim_in,
                out_channels = visual_dim_in,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.audio_c_bn = nn.BatchNorm2d(visual_dim_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
            self.audio_c_bn.no_init=True
            self.audio_c_bn.weight.data.zero_()
            self.audio_c_bn.bias.data.zero_()
            self.audio_c_relu = nn.ReLU(inplace=True)

    def forward(self, xv, xa):
        if self.v2a_enable:
            x_v2a = self.visual_a(xv, self.visual_a_rf(xv, xa))
            x_v2a = self.visual_a_bn(x_v2a)
            x_v2a = self.visual_a_relu(x_v2a)
            x_v2a = self.vpool(x_v2a)
            x_v2a = self.visual_b(x_v2a)
            x_v2a = self.visual_b_bn(x_v2a)
            x_v2a = self.visual_b_relu(x_v2a).squeeze(-1)
        if self.a2v_enable:
            x_a2v = self.audio_a(xa, self.audio_a_rf(xv, xa))
            x_a2v = self.audio_a_bn(x_a2v)
            x_a2v = self.audio_a_relu(x_a2v)
            x_a2v = F.pad(x_a2v, (0,0,0,self.pad_length), "constant", 0)
            x_a2v = self.audio_b(x_a2v)
            x_a2v = self.apool(x_a2v)
            x_a2v = self.audio_c(x_a2v)
            x_a2v = self.audio_c_bn(x_a2v)
            x_a2v = self.audio_c_relu(x_a2v).unsqueeze(-1)

        xv = xv + x_a2v
        xa = xa + x_v2a
        return xv, xa

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