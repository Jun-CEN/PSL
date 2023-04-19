
""" X3D heads. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY

@HEAD_REGISTRY.register()
class X3DHead(BaseHead):
    def __init__(self, cfg):
        super(X3DHead, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        self.out_a = nn.Conv3d(
            in_channels=dim, 
            out_channels=self.cfg.VIDEO.HEAD.MID_CHANNEL,
            kernel_size=(1,1,1),
            stride=(1,1,1),
            padding=0,
            bias=False
        )
        if self.cfg.VIDEO.HEAD.BN:
            self.out_a_bn = nn.BatchNorm3d(
                self.cfg.VIDEO.HEAD.MID_CHANNEL,
                eps=self.cfg.BN.EPS,
                momentum=self.cfg.BN.MOMENTUM
            )
        self.out_a_relu = nn.ReLU(inplace=True)
        self.out_b = nn.Linear(self.cfg.VIDEO.HEAD.MID_CHANNEL, num_classes, bias=True)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=4)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )
    
    def forward(self, x):
        out = {}
        x = self.global_avg_pool(x)
        x = self.out_a(x)
        if self.cfg.VIDEO.HEAD.BN:
            x = self.out_a_bn(x)
        x = self.out_a_relu(x)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        
        out = self.out_b(out)

        if not self.training:
            out = self.activation(out)
        out = out.view(out.shape[0], -1)
        return out, x.view(x.shape[0], -1)