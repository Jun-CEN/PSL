#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Transformers. """

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from einops import rearrange, repeat
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import (
    STEM_REGISTRY, BRANCH_REGISTRY, HEAD_REGISTRY, DropPath, BaseHead
)

from models.module_zoo.branches.temporal_adaptive_spatialconv import (
    TemporalAdaptiveSpatialConvCinAdaptive,
    route_func_mlp_kernel_size
)
from models.utils.init_helper import lecun_normal_, trunc_normal_, _init_transformer_weights

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, ff_dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Self-attention module. 
    Currently supports both full self-attention on all the input tokens,
    or only-spatial/only-temporal self-attention. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer.
    and 
    Gedas Bertasius, Heng Wang, Lorenzo Torresani.
    Is Space-Time Attention All You Need for Video Understanding?

    Modified from 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(
        self,
        dim,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        einops_from=None,
        einops_to=None,
        **einops_dims,
    ):
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = nn.Linear(dim, dim * 3)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(
                x,
                self.einops_from,
                self.einops_to,
                **self.einops_dims,
            )
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # import os
        # for i in range(12):
        #     if not os.path.exists(f"./debug/transformer_visualization/layer_{i}.pyth"):
        #         break
        # torch.save(attn,f"./debug/transformer_visualization/layer_{i}.pyth")
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def forward_partial(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_attn = (cls_q @ k.transpose(1,2)).softmax(-1)
        cls_attn = self.attn_dropout(cls_attn)
        cls_out = cls_attn @ v

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        attn = (q_ @ k_.transpose(1, 2)).softmax(-1)
        attn = self.attn_dropout(attn)
        x = attn @ v_

        # merge back time or space
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        x = torch.cat((cls_out, x), dim = 1)

        # merge back the heads
        x = rearrange(x, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

@STEM_REGISTRY.register()
class PatchEmbedStem(nn.Module):
    """ 
    Video to Patch Embedding.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        channels        = cfg.DATA.NUM_INPUT_CHANNELS       if cfg is not None else 3   # default 3
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 16
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        if isinstance(dim, list):
            dim = dim[0]

        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(
            in_channels     =channels, 
            out_channels    =dim, 
            kernel_size     =[1, patch_size, patch_size], 
            stride          =[1, patch_size, patch_size], 
        )

    def forward(self, x):
        b, c, t, h, w, p = *x.shape, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x

@STEM_REGISTRY.register()
class TubeletEmbeddingStem(nn.Module):
    """ 
    Video to Tubelet Embedding.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        channels        = cfg.DATA.NUM_INPUT_CHANNELS       if cfg is not None else 3   # default 3
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 16
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE   if cfg is not None else 2

        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(
            in_channels     =channels, 
            out_channels    =dim, 
            kernel_size     =[tubelet_size, patch_size, patch_size], 
            stride          =[tubelet_size, patch_size, patch_size], 
        )

    def forward(self, x):
        b, c, t, h, w, p = *x.shape, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x

@BRANCH_REGISTRY.register()
class BaseTransformerLayer(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BRANCH_REGISTRY.register()
class TimesformerLayer(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        patch_size      = cfg.VIDEO.BACKBONE.PATCH_SIZE     if cfg is not None else 16  # default 16
        drop_path       = drop_path_rate
        
        num_patches = (image_size // patch_size) ** 2

        self.norm_temporal = nn.LayerNorm(dim, eps=1e-6)
        self.attn_temporal = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            einops_from='b (f n) d', einops_to='(b n) f d', n = num_patches
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            einops_from='b (f n) d', einops_to='(b f) n d', f = num_frames
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn_temporal(self.norm_temporal(x)))
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BACKBONE_REGISTRY.register()
class Transformer(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16
        if hasattr(backbone_cfg, "TUBELET_SIZE"):
            tubelet_size = backbone_cfg.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches_per_frame = (image_size // patch_size) ** 2
        num_patches = num_frames * num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        self.pos_embd = nn.Parameter(torch.zeros(1, num_patches + 1, num_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))

        # construct transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        x = self.stem(x)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x =  torch.cat((cls_token, x), dim = 1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)

        return x[:, 0]

@BACKBONE_REGISTRY.register()
class FactorizedTransformer(nn.Module):
    """
    The factorized transformer. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer. 
    """
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        depth_temp      = backbone_cfg.DEPTH_TEMP           if cfg is not None else 4   # default 4
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16
        if hasattr(backbone_cfg, "TUBELET_SIZE"):
            tubelet_size = backbone_cfg.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        # both spatial and temporal embeddings/cls_token needs to be constructed
        # for the factorized transformer video model 
        self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.temp_embd          = nn.Parameter(torch.zeros(1, num_frames // tubelet_size + 1, num_features))
        self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))
        self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth+depth_temp)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # construct temporal transformer layers
        self.layers_temporal = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i+depth])
            for i in range(depth_temp)])

        self.norm_out = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.temp_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_out, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training 
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x = torch.cat((cls_token, x), dim = 1)

        # to make the input video size changable
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            actual_num_pathces_per_side = int(math.sqrt(actual_num_patches_per_frame))
            if not hasattr(self, "new_pos_embd") or self.new_pos_embd.shape[1] != (actual_num_pathces_per_side**2+1):
                cls_pos_embd = self.pos_embd[:,0,:].unsqueeze(1)
                pos_embd = self.pos_embd[:, 1:, :]
                num_patches_per_side = int(math.sqrt(self.num_patches_per_frame))
                pos_embd = pos_embd.reshape(1, num_patches_per_side, num_patches_per_side, -1).permute(0,3,1,2)
                pos_embd = torch.nn.functional.interpolate(
                    pos_embd, size=(actual_num_pathces_per_side,actual_num_pathces_per_side), mode="bilinear"
                ).permute(0,2,3,1).reshape(1, actual_num_pathces_per_side**2, -1)
                self.new_pos_embd = torch.cat((cls_pos_embd, pos_embd), dim=1)
            x += self.new_pos_embd
        else:
            x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

        cls_token_out = self.cls_token_out.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token_out, x), dim=1)

        x += self.temp_embd

        x = self.layers_temporal(x)
        x = self.norm_out(x)

        return x[:, 0]

@BACKBONE_REGISTRY.register()
class FactorizedTransformerObj(nn.Module):
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        depth_temp      = backbone_cfg.DEPTH_TEMP           if cfg is not None else 4   # default 4
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16
        if hasattr(backbone_cfg, "TUBELET_SIZE"):
            tubelet_size = backbone_cfg.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.temp_embd          = nn.Parameter(torch.zeros(1, num_frames // tubelet_size + 1, num_features))
        self.obj_embd           = nn.Parameter(torch.zeros(1, backbone_cfg.NUM_OBJECTS, num_features))
        self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))
        self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth+depth_temp)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        self.obj_tokenize = nn.Linear(1024, num_features, bias=False)

        self.layers_temporal = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i+depth])
            for i in range(depth_temp)])

        self.norm_out = nn.LayerNorm(num_features, eps=1e-6)

        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.temp_embd, std=.02)
        trunc_normal_(self.obj_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_out, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x, x_obj):
        # x: video
        # x_obj: object
        x = self.stem(x)
        x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x =  torch.cat((cls_token, x), dim = 1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

        cls_token_out = self.cls_token_out.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token_out, x), dim=1)

        x += self.temp_embd

        x_obj = self.obj_tokenize(x_obj)
        x_obj += self.obj_embd

        x = torch.cat(
            (x, x_obj), dim=1
        )

        x = self.layers_temporal(x)
        x = self.norm_out(x)

        return x[:, 0]



@HEAD_REGISTRY.register()
class TransformerHeadx2Obj(BaseHead):
    def __init__(self, cfg):
        super(TransformerHeadx2Obj, self).__init__(cfg)
        self.apply(_init_transformer_weights)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):  
        self.scale = dim ** -0.5

        self.linear_obj = nn.Linear(1024, 512)
        self.relu = nn.ReLU(inplace=True)
        self.linear_obj_out = nn.Linear(512, dim)

        if self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_noun" or self.cfg.VIDEO.HEAD.OBJ_FEAT == "sa_cat_noun":
            self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
            self.linear2 = nn.Linear(dim*2, num_classes[1], bias=True)
        elif self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_x" or self.cfg.VIDEO.HEAD.OBJ_FEAT == "sa_cat_x":
            self.linear1 = nn.Linear(dim*2, num_classes[0], bias=True)
            self.linear2 = nn.Linear(dim*2, num_classes[1], bias=True)

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
    
    def forward(self, x, x_obj):
        x_obj = self.linear_obj(x_obj)
        x_obj = self.relu(x_obj)
        x_obj = self.linear_obj_out(x_obj) 

        x = x.reshape(x.shape[0], 1, x.shape[-1])
        if self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_x":
            x_obj = x_obj.mean(1, keepdim=True)
            x = torch.cat((x_obj, x), dim=-1)

            if hasattr(self, "dropout"):
                out1 = self.dropout(x)
                out2 = out1
            else:
                out1 = x
                out2 = x

            out1 = self.linear1(out1)
            out2 = self.linear2(out2)
        elif self.cfg.VIDEO.HEAD.OBJ_FEAT == "sa_cat_x":
            # q: x
            # k, v: x_obj
            attn = (x @ x_obj.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)         
            x = torch.cat(
                (x, (attn @ x_obj)), dim=-1
            )

            if hasattr(self, "dropout"):
                out1 = self.dropout(x)
                out2 = out1
            else:
                out1 = x
                out2 = x

            out1 = self.linear1(out1)
            out2 = self.linear2(out2)

        elif self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_noun":
            if hasattr(self, "dropout"):
                out1 = self.dropout(x)
                out2 = out1
            else:
                out1 = x
                out2 = x

            out1 = self.linear1(out1)
            x_obj = x_obj.mean(1, keepdim=True)
            out2 = self.linear2(torch.cat(
                (out2, x_obj), dim=-1
            ))
        
        elif self.cfg.VIDEO.HEAD.OBJ_FEAT == "sa_cat_noun":
            # q: x
            # k, v: x_obj
            attn = (x @ x_obj.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)         

            if hasattr(self, "dropout"):
                out1 = self.dropout(x)
                out2 = out1
            else:
                out1 = x
                out2 = x

            out1 = self.linear1(out1)
            out2 = torch.cat(
                (out2, (attn @ x_obj)), dim=-1
            )
            out2 = self.linear2(out2)
        
        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, x



@HEAD_REGISTRY.register()
class TransformerHeadx2lfb(BaseHead):
    def __init__(self, cfg):
        super(TransformerHeadx2lfb, self).__init__(cfg)
        self.apply(_init_transformer_weights)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.scale = dim ** -0.5

        if self.cfg.DATA.LFB_FEAT_BACKBONE == "vivit":
            dim_lfb = 768 * self.cfg.DATA.NUM_LFB_VIEWS
        elif self.cfg.DATA.LFB_FEAT_BACKBONE == "csn":
            dim_lfb = 2048 * self.cfg.DATA.NUM_LFB_VIEWS

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim_lfb, dim*2)

        self.linear1 = nn.Linear(dim*2, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim*2, num_classes[1], bias=True)

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
        x = x.reshape(x.shape[0], 1, x.shape[1])

        B,N,C = x.shape
        B,N_LFB,_ = x_lfb.shape

        q = self.to_q(x)
        kv = self.to_kv(x_lfb).reshape(B, N_LFB, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_lfb = (attn @ v)

        x = torch.cat((x, x_lfb), dim=-1)

        if hasattr(self, "dropout"):
            out1 = self.dropout(x)
            out2 = self.dropout(x)
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


@BACKBONE_REGISTRY.register()
class FactorizedTransformerWithIndex(nn.Module):
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        depth_temp      = backbone_cfg.DEPTH_TEMP           if cfg is not None else 4   # default 4
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16
        if hasattr(backbone_cfg, "TUBELET_SIZE"):
            tubelet_size = backbone_cfg.TUBELET_SIZE         if cfg is not None else 2
        else:
            tubelet_size = 1

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.temp_embd          = nn.Parameter(torch.zeros(1, num_frames // tubelet_size + 1, num_features))
        self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))
        self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth+depth_temp)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i], index=i)
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        self.layers_temporal = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i+depth], index=i+depth)
            for i in range(depth_temp)])

        self.norm_out = nn.LayerNorm(num_features, eps=1e-6)

        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.temp_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_out, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training 
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x = torch.cat((cls_token, x), dim = 1)

        if actual_num_patches_per_frame != self.num_patches_per_frame:
            actual_num_pathces_per_side = int(math.sqrt(actual_num_patches_per_frame))
            if not hasattr(self, "new_pos_embd") or self.new_pos_embd.shape[1] != (actual_num_pathces_per_side**2+1):
                cls_pos_embd = self.pos_embd[:,0,:].unsqueeze(1)
                pos_embd = self.pos_embd[:, 1:, :]
                num_patches_per_side = int(math.sqrt(self.num_patches_per_frame))
                pos_embd = pos_embd.reshape(1, num_patches_per_side, num_patches_per_side, -1).permute(0,3,1,2)
                pos_embd = torch.nn.functional.interpolate(
                    pos_embd, size=(actual_num_pathces_per_side,actual_num_pathces_per_side), mode="bilinear"
                ).permute(0,2,3,1).reshape(1, actual_num_pathces_per_side**2, -1)
                self.new_pos_embd = torch.cat((cls_pos_embd, pos_embd), dim=1)
            x += self.new_pos_embd
        else:
            x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

        cls_token_out = self.cls_token_out.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token_out, x), dim=1)

        x += self.temp_embd

        x = self.layers_temporal(x)
        x = self.norm_out(x)

        return x[:, 0]

class AttentionWithIndex(nn.Module):
    def __init__(
        self,
        dim,
        index=0,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        einops_from=None,
        einops_to=None,
        **einops_dims,
    ):
        super().__init__()
        self.index = index
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = nn.Linear(dim, dim * 3)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(
                x,
                self.einops_from,
                self.einops_to,
                **self.einops_dims,
            )
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        import matplotlib.pyplot as plt
        for head in range(attn.shape[1]):
            vis_attn = attn[0,head].detach()
            vis_attn = vis_attn[1:, 1:]
            vis_attn = (vis_attn - vis_attn.min(0,keepdim=True)[0]) / (vis_attn.max(0,keepdim=True)[0] - vis_attn.min(0, keepdim=True)[0])
            vis_attn = vis_attn.reshape(14, 14, 14, 14).permute(0,2,1,3).reshape(196,196)
            vis_attn = vis_attn.cpu().numpy()
            plt.imsave(f"./output/visualization/block{head}_{self.index}.jpg", vis_attn, cmap='gray')

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def forward_partial(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_attn = (cls_q @ k.transpose(1,2)).softmax(-1)
        cls_attn = self.attn_dropout(cls_attn)
        cls_out = cls_attn @ v

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        attn = (q_ @ k_.transpose(1, 2)).softmax(-1)
        attn = self.attn_dropout(attn)
        x = attn @ v_

        # merge back time or space
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        x = torch.cat((cls_out, x), dim = 1)

        # merge back the heads
        x = rearrange(x, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x


@BRANCH_REGISTRY.register()
class BaseTransformerLayerWithIndex(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0, index=0):
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = AttentionWithIndex(
            dim, index, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BACKBONE_REGISTRY.register()
class VisionTransformer(nn.Module):
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        tubelet_size    = backbone_cfg.TUBELET_SIZE
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        # both spatial and temporal embeddings/cls_token needs to be constructed
        # for the factorized transformer video model 
        # self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features))
        self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training 
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x = torch.cat((cls_token, x), dim = 1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

        x = x.mean(1)

        return x

class TemporallyCalibratedAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_temp_tokens,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        super().__init__()
        self.num_temp_tokens = num_temp_tokens
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = nn.Linear(dim, dim * 3)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

        self.conv_q = nn.Conv1d(
            dim, dim, kernel_size=3, stride=1, padding=1
        )
        self.conv_q.weight.data.zero_()
        self.conv_q.bias.data.zero_()
        self.conv_k = nn.Conv1d(
            dim, dim, kernel_size=3, stride=1, padding=1
        )
        self.conv_k.weight.data.zero_()
        self.conv_k.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape

        g = x[:,:1]
        # BxT, 1, C -> B, C, T
        g = g.reshape(g.shape[0]//self.num_temp_tokens, self.num_temp_tokens, -1).permute(0,2,1)
        temp_context_q = self.conv_q(g).permute(0,2,1).reshape(B, self.num_heads, 1, C//self.num_heads)
        temp_context_k = self.conv_k(g).permute(0,2,1).reshape(B, self.num_heads, 1, C//self.num_heads)

        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = ((q+temp_context_q) @ (k+temp_context_k).transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

@BRANCH_REGISTRY.register()
class TemporallyCalibratedTransformerLayer(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TemporallyCalibratedAttention(
            dim, num_temp_tokens=num_frames//tubelet_size, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

class route_func_linear_cls_token(nn.Module):

    def __init__(self, c_in, c_out, ratio, kernel_size):
        super(route_func_linear_cls_token, self).__init__()
        self.c_in = c_in
        self.a = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=True
        )
        self.a.no_init=True
        self.a.weight.data.zero_()
        self.a.bias.data.zero_()

    def forward(self, x):
        """
        Inputs:
            x (Tensor): shape B, C, T
        Outputs:
            x (Tensor): shape B, C, T
        """
        x = self.a(x) + 1
        return x

class TemporallyAdaptiveLinear(nn.Module):

    def __init__(self, in_channels, out_channels, num_temp_tokens,
                 adaptive_dimension="cin", mode="qkv", bias=True, bias_adaptive=True):
        super(TemporallyAdaptiveLinear, self).__init__()

        assert adaptive_dimension in ["cin", "cout"]
        assert mode in ["qkv", "normal"]

        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = num_temp_tokens
        self.adaptive_dimension = adaptive_dimension
        self.bias_adaptive = bias_adaptive

        if mode == "qkv":
            self.routing_func = route_func_linear_cls_token(
                c_in = in_channels,
                c_out = in_channels * 3,
                ratio = 4, 
                kernel_size= 3
            )
        elif mode == "normal":
            self.routing_func = route_func_linear_cls_token(
                c_in = in_channels,
                c_out = in_channels,
                ratio = 4, 
                kernel_size= 3
            )


        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels, 1)) # B, T, C_out, C_in, 1
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Inputs:
            x (Tensor): shape (B, N, C)
            routing_weight (Tensor): shape (B, T, C)
        """
        bt, n, c = x.shape
        b = bt // self.t
        # B, N, C -> B, C, N
        x = rearrange(x, "(b t) n c -> b c t n", t=self.t)
        routing_weight = self.routing_func(x[:,:,:,0]) # B, C, T
        _, _, c_out, c_in, _ = self.weight.size()
        x = rearrange(x, "b c t n -> (b t c) n")
        x = x.unsqueeze(0)
        if self.adaptive_dimension == "cin" and self.mode == "qkv":
            # routing weight -> B, T, 1, C, 1
            q_routing, k_routing, v_routing = torch.chunk(routing_weight.permute(0,2,1).unsqueeze(-2).unsqueeze(-1), 3, dim=-2)
            weight_q, weight_k, weight_v = torch.chunk(self.weight, 3, dim=2)
            weight_q = ( q_routing * weight_q).reshape(-1, c_in, 1)
            weight_k = ( k_routing * weight_k).reshape(-1, c_in, 1)
            weight_v = ( v_routing * weight_v).reshape(-1, c_in, 1)
            if self.bias is not None:
                bias_q, bias_k, bias_v = torch.chunk(self.bias, 3, dim=-1)
                bias_q = (bias_q * q_routing.squeeze(-3).squeeze(-1)).reshape(-1)
                bias_k = (bias_k * k_routing.squeeze(-3).squeeze(-1)).reshape(-1)
                bias_v = (bias_v * v_routing.squeeze(-3).squeeze(-1)).reshape(-1)
                output_q = F.conv1d(x, weight=weight_q, bias=bias_q, groups=b * self.t)
                output_k = F.conv1d(x, weight=weight_k, bias=bias_k, groups=b * self.t)
                output_v = F.conv1d(x, weight=weight_v, bias=bias_v, groups=b * self.t)
                output_q = rearrange(output_q.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output_k = rearrange(output_k.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output_v = rearrange(output_v.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output = torch.cat((output_q, output_k, output_v), dim=1)
            else:
                output_q = F.conv1d(x, weight=weight_q, bias=None, groups=b * self.t)
                output_k = F.conv1d(x, weight=weight_k, bias=None, groups=b * self.t)
                output_v = F.conv1d(x, weight=weight_v, bias=None, groups=b * self.t)
                output_q = rearrange(output_q.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output_k = rearrange(output_k.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output_v = rearrange(output_v.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output = torch.cat((output_q, output_k, output_v), dim=1)
        elif self.adaptive_dimension == "cin" and self.mode == "normal":
            weight = (routing_weight.permute(0,2,1).unsqueeze(-2).unsqueeze(-1) * self.weight).reshape(-1, c_in, 1)
            if self.bias is not None:
                if self.bias_adaptive:
                    bias = (routing_weight.permute(0,2,1) * self.bias).reshape(-1)
                else:
                    bias = self.bias.repeat(routing_weight.shape[0], routing_weight.shape[2], 1).reshape(-1)
                output = F.conv1d(
                    x, weight=weight, bias=bias, groups=b * self.t)
            else:
                output = F.conv1d(
                    x, weight=weight, bias=self.bias, groups=b * self.t)
            output = rearrange(output.squeeze(0), "(b t c) n -> (b t) n c", b=b, t=self.t, c=c_out)
        elif self.adaptive_dimension == "cout":
            # routing weight -> B, T, C, 1, 1
            weight = (routing_weight.permute(0,2,1).unsqueeze(-1).unsqueeze(-1) * self.weight).reshape(-1, c_in, 1)
            if self.bias is not None:
                bias = (routing_weight.permute(0,2,1) * self.bias).reshape(-1)
                output = F.conv1d(
                    x, weight=weight, bias=bias, groups=b * self.t)
            else:
                output = F.conv1d(
                    x, weight=weight, bias=None, groups=b * self.t)
            output = rearrange(output.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c_out)
        
        return output

class TemporallyAdaptiveProjectedAttentionCin(nn.Module):
    def __init__(
        self,
        dim,
        num_temp_tokens,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        super().__init__()
        self.num_temp_tokens = num_temp_tokens
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = TemporallyAdaptiveLinear(dim, dim * 3, num_temp_tokens=num_temp_tokens)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

    def forward(self, x):
        B, N, C = x.shape
        # B, N, 3, H, C//H -> 3, B, H, N, C//H
        qkv = self.to_qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

class TemporallyAdaptiveProjectedAttentionCout(nn.Module):
    def __init__(
        self,
        dim,
        num_temp_tokens,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        super().__init__()
        self.num_temp_tokens = num_temp_tokens
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = TemporallyAdaptiveLinear(dim, dim * 3, num_temp_tokens=num_temp_tokens, adaptive_dimension="cout")
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

    def forward(self, x):
        B, N, C = x.shape
        # B, N, 3, H, C//H -> 3, B, H, N, C//H
        qkv = self.to_qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

@BRANCH_REGISTRY.register()
class TemporallyAdaptiveTransformerLayerCin(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TemporallyAdaptiveProjectedAttentionCin(
            dim, num_temp_tokens=num_frames//tubelet_size, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BRANCH_REGISTRY.register()
class TemporallyAdaptiveTransformerLayerCout(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TemporallyAdaptiveProjectedAttentionCout(
            dim, num_temp_tokens=num_frames//tubelet_size, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

class route_func_linear_cls_token_moe(nn.Module):

    def __init__(self, c_in, c_out, ratio, kernel_size, num_experts):
        super(route_func_linear_cls_token_moe, self).__init__()
        self.c_in = c_in

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.kernel_size = kernel_size

        self.rf = nn.Conv1d(
            in_channels=c_in,
            out_channels=num_experts,
            kernel_size=1,
            padding=0,
            bias=True
        )
        self.weight = nn.Parameter(
            torch.zeros(num_experts, c_out, c_in, kernel_size)
        )
        self.bias = nn.Parameter(
            torch.zeros(num_experts, c_out)
        )

    def forward(self, x):
        """
        Inputs:
            x (Tensor): shape B, C, T
        Outputs:
            x (Tensor): shape B, C, T
        """
        b, c, t = x.shape
        num_e, c_out, c_in, k = self.weight.shape
        rf = self.rf(self.global_pool(x)).squeeze(-1) # B, N_Experts, 1
        weight = torch.mm(rf, self.weight.view(num_e, -1)).view(-1, c_in, k)
        bias = torch.mm(rf, self.bias).view(-1)
        x = x.reshape(1, -1, t)
        x = F.conv1d(
            x, weight=weight, bias=bias, padding=self.kernel_size//2, stride=1, groups=b
        )
        x = x.reshape(b, c_out, t) + 1
        return x

class TemporallyAdaptiveLinearMoE(nn.Module):

    def __init__(self, in_channels, out_channels, num_temp_tokens,
                 adaptive_dimension="cin", mode="qkv", bias=True):
        super(TemporallyAdaptiveLinearMoE, self).__init__()

        assert adaptive_dimension in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = num_temp_tokens
        self.adaptive_dimension = adaptive_dimension

        self.routing_func = route_func_linear_cls_token_moe(
            c_in = in_channels,
            c_out = in_channels * len(mode),
            ratio = 4, 
            kernel_size= 3,
            num_experts=4
        )

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels, 1)) # B, T, C_out, C_in, 1
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Inputs:
            x (Tensor): shape (B, N, C)
            routing_weight (Tensor): shape (B, T, C)
        """
        bt, n, c = x.shape
        b = bt // self.t
        # B, N, C -> B, C, N
        x = rearrange(x, "(b t) n c -> b c t n", t=self.t)
        routing_weight = self.routing_func(x[:,:,:,0]) # B, C, T
        _, _, c_out, c_in, _ = self.weight.size()
        x = rearrange(x, "b c t n -> (b t c) n")
        x = x.unsqueeze(0)
        if self.adaptive_dimension == "cin":
            # routing weight -> B, T, 1, C, 1
            q_routing, k_routing, v_routing = torch.chunk(routing_weight.permute(0,2,1).unsqueeze(-2).unsqueeze(-1), 3, dim=-2)
            weight_q, weight_k, weight_v = torch.chunk(self.weight, 3, dim=2)
            weight_q = ( q_routing * weight_q).reshape(-1, c_in, 1)
            weight_k = ( k_routing * weight_k).reshape(-1, c_in, 1)
            weight_v = ( v_routing * weight_v).reshape(-1, c_in, 1)
            if self.bias is not None:
                bias_q, bias_k, bias_v = torch.chunk(self.bias, 3, dim=-1)
                bias_q = (bias_q * q_routing.squeeze(-3).squeeze(-1)).reshape(-1)
                bias_k = (bias_k * k_routing.squeeze(-3).squeeze(-1)).reshape(-1)
                bias_v = (bias_v * v_routing.squeeze(-3).squeeze(-1)).reshape(-1)
                output_q = F.conv1d(x, weight=weight_q, bias=bias_q, groups=b * self.t)
                output_k = F.conv1d(x, weight=weight_k, bias=bias_k, groups=b * self.t)
                output_v = F.conv1d(x, weight=weight_v, bias=bias_v, groups=b * self.t)
                output_q = rearrange(output_q.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output_k = rearrange(output_k.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output_v = rearrange(output_v.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output = torch.cat((output_q, output_k, output_v), dim=1)
            else:
                output_q = F.conv1d(x, weight=weight_q, bias=None, groups=b * self.t)
                output_k = F.conv1d(x, weight=weight_k, bias=None, groups=b * self.t)
                output_v = F.conv1d(x, weight=weight_v, bias=None, groups=b * self.t)
                output_q = rearrange(output_q.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output_k = rearrange(output_k.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output_v = rearrange(output_v.squeeze(0), "(b t c) n -> (b t) c n", b=b, t=self.t, c=c)
                output = torch.cat((output_q, output_k, output_v), dim=1)
        elif self.adaptive_dimension == "cout":
            # routing weight -> B, T, C, 1, 1
            weight = (routing_weight.permute(0,2,1).unsqueeze(-1).unsqueeze(-1) * self.weight).reshape(-1, c_in, 1)
            if self.bias is not None:
                bias = (routing_weight.permute(0,2,1) * self.bias).reshape(-1)
                output = F.conv1d(
                    x, weight=weight, bias=bias, groups=b * self.t)
            else:
                output = F.conv1d(
                    x, weight=weight, bias=None, groups=b * self.t)
        
        return output

class TemporallyAdaptiveProjectedAttentionCinMoE(nn.Module):
    def __init__(
        self,
        dim,
        num_temp_tokens,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        super().__init__()
        self.num_temp_tokens = num_temp_tokens
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = TemporallyAdaptiveLinearMoE(dim, dim * 3, num_temp_tokens=num_temp_tokens)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

    def forward(self, x):
        B, N, C = x.shape
        # B, N, 3, H, C//H -> 3, B, H, N, C//H
        qkv = self.to_qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

@BRANCH_REGISTRY.register()
class TemporallyAdaptiveTransformerLayerCinMoE(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TemporallyAdaptiveProjectedAttentionCinMoE(
            dim, num_temp_tokens=num_frames//tubelet_size, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

class TemporallyAdaptiveFeedForward(nn.Module):
    def __init__(self, dim, num_temp_tokens, mult=4, ff_dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            TemporallyAdaptiveLinear(dim, dim * mult, num_temp_tokens, mode="normal", bias_adaptive=False),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            TemporallyAdaptiveLinear(dim * mult, dim, num_temp_tokens, mode="normal", bias_adaptive=False),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x):
        return self.net(x)

@BRANCH_REGISTRY.register()
class TemporallyAdaptiveFFNTransformerLayer(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = TemporallyAdaptiveFeedForward(
            dim=dim, 
            num_temp_tokens=num_frames//tubelet_size,
            mult=mlp_mult, 
            ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BACKBONE_REGISTRY.register()
class TemporallyAdaptiveVisionTransformer(nn.Module):
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        tubelet_size    = backbone_cfg.TUBELET_SIZE
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        # both spatial and temporal embeddings/cls_token needs to be constructed
        # for the factorized transformer video model 
        # self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features))
        self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 2, num_features))
        self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))
        self.temp_ada_token     = nn.Parameter(torch.randn(1, 1, num_features))

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.temp_ada_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training 
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        temp_adap_token = self.temp_ada_token.repeat((x.shape[0],1,1))
        x = torch.cat((temp_adap_token, cls_token, x), dim = 1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 1]

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

        x = x.mean(1)

        return x

class TemporallySharedAttention(nn.Module):
    """
    Self-attention module. 
    Currently supports both full self-attention on all the input tokens,
    or only-spatial/only-temporal self-attention. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer.
    and 
    Gedas Bertasius, Heng Wang, Lorenzo Torresani.
    Is Space-Time Attention All You Need for Video Understanding?

    Modified from 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(
        self,
        dim,
        num_temporal_tokens,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        einops_from=None,
        einops_to=None,
        **einops_dims,
    ):
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.num_temporal_tokens = num_temporal_tokens

        self.to_qk          = nn.Linear(dim, dim * 2)
        self.to_v           = nn.Linear(dim, dim)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(
                x,
                self.einops_from,
                self.einops_to,
                **self.einops_dims,
            )
        B, N, C = x.shape
        qk = self.to_qk(
            rearrange(x, "(b t) n c -> b t n c", t=self.num_temporal_tokens)[:,0:1] # obtain the first frame for each video
        ).reshape(B//self.num_temporal_tokens, 1, N, 2, self.num_heads, C // self.num_heads).permute(3,0,1,4,2,5)
        v = self.to_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k = qk[0], qk[1]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = rearrange(attn.repeat((1,self.num_temporal_tokens,1,1,1)), "b t h x y -> (b t) h x y")
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

@BRANCH_REGISTRY.register()
class TemporallySharedTransformerLayer(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TemporallySharedAttention(
            dim, 
            num_temporal_tokens=num_frames // tubelet_size, 
            num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

@BRANCH_REGISTRY.register()
class BaseTransformerLayerTemporalConvolution(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        self.temporal_conv = TemporalConv(cfg)

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.temporal_conv(x))
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x

class TemporalConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES
        num_frames      = cfg.DATA.NUM_INPUT_FRAMES
        tubelet_size    = cfg.VIDEO.BACKBONE.TUBELET_SIZE
        num_temp_tokens = num_frames // tubelet_size
        self.num_temp_tokens = num_temp_tokens

        self.norm_temp = nn.LayerNorm(dim, eps=1e-6)
        self.temp_conv = nn.Conv2d(dim, dim, kernel_size=[3, 1], stride=[1, 1], padding=[1, 0], bias=False)

    def forward(self, x):
        x = self.norm_temp(x)
        x = rearrange(x, "(b t) n c -> b c t n", t = self.num_temp_tokens)
        x = self.temp_conv(x)
        x = rearrange(x, "b c t n -> (b t) n c", t = self.num_temp_tokens)
        
        return x

@BRANCH_REGISTRY.register()
class BaseTransformerLayerWithDownsampling(nn.Module):
    def __init__(self, cfg, layer_idx, drop_path_rate=0.0):
        """
        Args: 
            cfg             (Config): global config object. 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = cfg.VIDEO.BACKBONE.NUM_FEATURES   if cfg is not None else 768 # default 768
        num_heads       = cfg.VIDEO.BACKBONE.NUM_HEADS      if cfg is not None else 1  # default 12
        attn_dropout    = cfg.VIDEO.BACKBONE.ATTN_DROPOUT   if cfg is not None else 0.1 # default 0.1
        ff_dropout      = cfg.VIDEO.BACKBONE.FF_DROPOUT     if cfg is not None else 0.1 # default 0.1
        mlp_mult        = cfg.VIDEO.BACKBONE.MLP_MULT       if cfg is not None else 4
        drop_path       = drop_path_rate

        self.layer_idx = layer_idx

        downsample_layer = cfg.VIDEO.BACKBONE.DOWNSAMPLE_LAYER

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_idx + 1 == cfg.VIDEO.BACKBONE.SPATIOTEMPORAL_LAYER_START:
            num_input_frames = cfg.DATA.NUM_INPUT_FRAMES
            tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE
            self.num_temporal_tokens = num_input_frames // tubelet_size
            self.st_rearrange = True
        else:
            self.st_rearrange = False
        if layer_idx + 1 in downsample_layer:
            self.downsample = PatchDownsample(cfg, 
            layer_idx + 1 > cfg.VIDEO.BACKBONE.SPATIOTEMPORAL_LAYER_START if cfg.VIDEO.BACKBONE.SPATIOTEMPORAL_LAYER_START is not None else False
            )

    def forward(self, x):
        if hasattr(self, "downsample"):
            x = self.downsample(x)
        if self.st_rearrange:
            cls_token = x[:,0:1,:]
            x_feat = x[:,1:,:]
            b, n, c = x_feat.shape
            cls_token = rearrange(cls_token, "(b t) n c -> b (t n) c", t=self.num_temporal_tokens).mean(1, keepdim=True)
            x_feat = rearrange(x_feat, "(b t) (h w) c -> b (t h w) c", t=self.num_temporal_tokens, h=int(n**0.5))
            x = torch.cat((cls_token, x_feat), dim=1)
        
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))

        return x

class PatchDownsample(nn.Module):
    def __init__(self, cfg, st_tokens):
        super().__init__()

        dim = cfg.VIDEO.BACKBONE.NUM_FEATURES

        self.st_tokens = st_tokens
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.conv = nn.Conv2d(
            dim,
            dim, 
            kernel_size=3 if cfg.VIDEO.BACKBONE.DOWNSAMPLE_OVERLAP else 2, 
            stride=2,
            padding=1 if cfg.VIDEO.BACKBONE.DOWNSAMPLE_OVERLAP else 0,
        )
        if st_tokens:
            num_input_frames = cfg.DATA.NUM_INPUT_FRAMES
            tubelet_size = cfg.VIDEO.BACKBONE.TUBELET_SIZE
            self.num_temporal_tokens = num_input_frames // tubelet_size
    
    def forward(self, x):
        if self.st_tokens:
            pass
        else:
            b, n, c = x.shape
            h = int((n-1)**0.5)
            x = self.norm(x)
            cls_token = x[:, 0:1, :]
            x_feat = rearrange(x[:, 1:], "b (h w) c -> b c h w", h=h)
            x_feat = self.conv(x_feat)
            x_feat = rearrange(x_feat, "b c h w -> b (h w) c", h=h//2)
            x = torch.cat((cls_token, x_feat), dim=1)
        return x

@BACKBONE_REGISTRY.register()
class VisionTransformerDownsample(nn.Module):
    def __init__(
        self,
        cfg,
        mode="video"
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super().__init__()

        mode = mode.upper()
        backbone_cfg = getattr(cfg, mode).BACKBONE

        num_frames      = cfg.DATA.NUM_INPUT_FRAMES         if cfg is not None else 8   # default 8
        image_size      = cfg.DATA.TRAIN_CROP_SIZE          if cfg is not None else 224 # default 224
        tubelet_size    = backbone_cfg.TUBELET_SIZE
        num_features    = backbone_cfg.NUM_FEATURES         if cfg is not None else 768 # default 768
        patch_size      = backbone_cfg.PATCH_SIZE           if cfg is not None else 16  # default 16
        depth           = backbone_cfg.DEPTH                if cfg is not None else 12  # default 12
        drop_path       = backbone_cfg.DROP_PATH            if cfg is not None else 16  # default 16

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.st_rearrange = False
        if backbone_cfg.SPATIOTEMPORAL_LAYER_START is not None and backbone_cfg.SPATIOTEMPORAL_LAYER_START <= depth:
            self.st_rearrange = True

        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        # constructs the tokenization module.
        self.stem = STEM_REGISTRY.get(backbone_cfg.STEM.NAME)(cfg) if cfg is not None else PatchEmbedStem(cfg)

        # both spatial and temporal embeddings/cls_token needs to be constructed
        # for the factorized transformer video model 
        # self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features))
        self.pos_embd           = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.cls_token          = nn.Parameter(torch.randn(1, 1, num_features))

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.layers = nn.Sequential(*[
            BRANCH_REGISTRY.get(backbone_cfg.BRANCH.NAME)(cfg, layer_idx=i, drop_path_rate=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        h, w = x.shape[-2:]

        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training 
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0],1,1))
        x = torch.cat((cls_token, x), dim = 1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        if not self.st_rearrange:
            x = rearrange(x, "(b t) c -> b t c", t=self.num_patches//self.num_patches_per_frame)

            x = x.mean(1)

        return x