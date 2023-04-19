
""" Head with Long-Term Feature Banks."""

import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY

@HEAD_REGISTRY.register()
class BaseHeadx2lfb(BaseHead):
    """
        Constructs base heads for EPIC-KITCHENS competitions with LFBs.

        See paper "Long-Term Feature Banks for Detailed Video Understanding",
        Wu et al., 2019 (https://arxiv.org/abs/1812.05038) 
        
        and paper "Towards Training Stronger Video Vision Transformers for EPIC-KITCHENS-100 Action Recognition",
        Huang et al., 2021 (https://arxiv.org/abs/2106.05058) 

        for details.
    """
    def __init__(self, cfg):
        super(BaseHeadx2lfb, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.scale = dim ** -0.5

        if self.cfg.DATA.LFB_FEAT_BACKBONE == "vivit":
            dim_lfb = 768 * self.cfg.DATA.NUM_LFB_VIEWS
        elif self.cfg.DATA.LFB_FEAT_BACKBONE == "csn":
            dim_lfb = 2048 * self.cfg.DATA.NUM_LFB_VIEWS

        self.to_q = nn.Linear(dim, dim_lfb)
        self.to_kv = nn.Linear(dim_lfb, dim_lfb*2)

        self.linear1 = nn.Linear(dim+dim_lfb, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim+dim_lfb, num_classes[1], bias=True)

        if dropout_rate > 0.0: 
            self.dropout = nn.Dropout(dropout_rate)

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_func == "identity":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )
    
    def forward(self, x, x_lfb):
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))
            x = x.reshape(x.shape[0], 1, x.shape[-1])

        B,N,C = x.shape
        B,N_LFB,C = x_lfb.shape

        q = self.to_q(x)
        kv = self.to_kv(x_lfb).reshape(B, N_LFB, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_lfb = (attn @ v)

        x = torch.cat((x, x_lfb), dim=-1)

        if hasattr(self, "dropout"):
            out1 = self.dropout(x)
            out2 = out1
        else:
            out1 = x
            out2 = x

        out1 = self.linear1(out1)
        out2 = self.linear2(out2)

        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, x