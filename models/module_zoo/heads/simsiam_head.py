
""" Heads for Video SimSiam. """

import torch
import torch.nn as nn

from models.base.base_blocks import BaseHead
from models.base.base_blocks import HEAD_REGISTRY

class SiamMLP(nn.Module):
    def __init__(
        self, 
        cfg, 
        dim_in_override=None, 
        dim_out_override=None, 
        normalize=True, 
        final=True
    ):
        super(SiamMLP, self).__init__()
        with_bn     = cfg.PRETRAIN.CONTRASTIVE.HEAD_BN
        final_bn    = cfg.PRETRAIN.CONTRASTIVE.FINAL_BN
        bn_mmt      = cfg.BN.MOMENTUM
        dim         = cfg.MODEL.NUM_OUT_FEATURES if dim_in_override is None else dim_in_override
        out_dim     = cfg.PRETRAIN.CONTRASTIVE.HEAD_OUT_DIM if dim_out_override is None else dim_out_override
        mid_dim     = out_dim // 4
        self.normalize = normalize and final

        self.num_layers = 2 if final else 3

        if self.num_layers == 2:
    
            self.logits_out_a = nn.Linear(dim, mid_dim)
            if with_bn:
                self.logits_out_a_bn = nn.BatchNorm3d(mid_dim, eps=1e-5, momentum=bn_mmt)

            self.logits_out_relu = nn.ReLU(inplace=True)

            self.logits_out_b = nn.Linear(mid_dim, out_dim)
        elif self.num_layers == 3:
            self.logits_out_a = nn.Linear(dim, mid_dim)
            if with_bn:
                self.logits_out_a_bn = nn.BatchNorm3d(mid_dim, eps=1e-5, momentum=bn_mmt)

            self.logits_out_relu = nn.ReLU(inplace=True)

            self.logits_out_b = nn.Linear(mid_dim, mid_dim)
            if with_bn:
                self.logits_out_b_bn = nn.BatchNorm3d(mid_dim, eps=1e-5, momentum=bn_mmt)

            self.logits_out_c = nn.Linear(mid_dim, out_dim)

        if final_bn:
            self.final_bn = nn.BatchNorm3d(out_dim, eps=1e-5, momentum=bn_mmt)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], 1, 1, 1, x.shape[1])

        if self.num_layers == 2:
            x = self.logits_out_a(x)
            if hasattr(self, 'logits_out_a_bn'):
                x = self.logits_out_a_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            x = self.logits_out_relu(x)
            x = self.logits_out_b(x)

        elif self.num_layers == 3:
            x = self.logits_out_a(x)
            if hasattr(self, 'logits_out_a_bn'):
                x = self.logits_out_a_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            x = self.logits_out_relu(x)
            x = self.logits_out_b(x)
            if hasattr(self, 'logits_out_b_bn'):
                x = self.logits_out_b_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
            x = self.logits_out_relu(x)
            x = self.logits_out_c(x)

        if hasattr(self, 'final_bn'):
            x = self.final_bn(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        x = x.view(x.shape[0], -1)
        if self.normalize: 
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

@HEAD_REGISTRY.register()
class SiamHead(BaseHead):
    def __init__(self, cfg, final=True):
        self.final = final
        super(SiamHead, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.mlp = SiamMLP(self.cfg, final=self.final)

    
    def forward(self, x, deep_x=None):
        logits = {}
        if x.ndim == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))

        logits = self.mlp(x)
        
        return logits