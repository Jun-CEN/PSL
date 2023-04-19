#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" MLP head for contrastive learning. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY

class MLP(nn.Module):
    """
    Constructs a multi-layer perceptron.
    """
    def __init__(
        self, 
        dim_in,
        dim_mid,
        dim_out, 
        mid_bn,
        final_bn,
        bn_mmt,
        normalize=True,
        nonlinear=False,
    ):
        super(MLP, self).__init__()
        self.mid_bn     = mid_bn
        self.final_bn   = final_bn
        self.normalize  = normalize
        self.nonlinear  = nonlinear
    
        # first linear 
        self.linear_a = nn.Linear(dim_in, dim_mid)
        if self.mid_bn:
            self.linear_a_bn = nn.BatchNorm3d(dim_mid, eps=1e-5, momentum=bn_mmt)
        self.logits_out_relu = nn.ReLU(inplace=True)

        # second linear
        self.logits_out_b = nn.Linear(dim_mid, dim_out)
        if self.final_bn:
            self.final_bn = nn.BatchNorm3d(dim_out, eps=1e-5, momentum=bn_mmt)
        if self.nonlinear:
            self.final_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], 1, 1, 1, x.shape[1])

        # first linear
        x = self.linear_a(x)
        if self.mid_bn:
            x = self.linear_a_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x = self.logits_out_relu(x)

        # second linear
        x = self.logits_out_b(x)
        if self.final_bn:
            x = self.final_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        if self.nonlinear:
            x = self.final_relu(x)

        x = x.view(x.shape[0], -1)
        if self.normalize: 
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

@HEAD_REGISTRY.register()
class VideoMLPHead(BaseHead):
    """
    Constructs a MLP head for video contrastive learning.
    """
    def __init__(self, cfg):
        super(VideoMLPHead, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.mlp = MLP(
            dim_in=self.cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES,
            dim_mid=self.cfg.VIDEO.HEAD.MLP.MID_DIM,
            dim_out=self.cfg.VIDEO.HEAD.MLP.OUT_DIM, 
            mid_bn=self.cfg.VIDEO.HEAD.MLP.MID_BN,
            final_bn=self.cfg.VIDEO.HEAD.MLP.FINAL_BN,
            bn_mmt=self.cfg.BN.MOMENTUM,
            normalize=True,
            nonlinear=False,
        )

    
    def forward(self, x, deep_x=None):
        if x.ndim == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))

        x = self.mlp(x)
        
        return x

@HEAD_REGISTRY.register()
class VideoMLP_CE_Head(BaseHead):
    """
    Constructs a MLP head for video contrastive learning.
    """
    def __init__(self, cfg):
        super(VideoMLP_CE_Head, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.mlp = MLP(
            dim_in=self.cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES,
            dim_mid=self.cfg.VIDEO.HEAD.MLP.MID_DIM,
            dim_out=self.cfg.VIDEO.HEAD.MLP.OUT_DIM, 
            mid_bn=self.cfg.VIDEO.HEAD.MLP.MID_BN,
            final_bn=self.cfg.VIDEO.HEAD.MLP.FINAL_BN,
            bn_mmt=self.cfg.BN.MOMENTUM,
            normalize=True,
            nonlinear=False,
        )

        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        self.out = nn.Linear(dim, num_classes, bias=True)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )

    
    def forward(self, x, deep_x=None):
        if x.ndim == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))

        logits_ssl = self.mlp(x)

        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x
        out = self.out(out)

        if not self.training:
            out = self.activation(out)
        
        out = out.view(out.shape[0], -1)
        
        return out[::2], logits_ssl
    