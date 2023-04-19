#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Basic blocks. """

import os
import abc
import torch
import torch.nn as nn
from utils.registry import Registry
from models.utils.params import update_3d_conv_params
import matplotlib.pyplot as plt

from torchvision.utils import make_grid, save_image

from einops import rearrange, repeat

from models.utils.init_helper import lecun_normal_, trunc_normal_, _init_transformer_weights
from .bnn import BayesianPredictor, get_uncertainty

STEM_REGISTRY = Registry("Stem")
BRANCH_REGISTRY = Registry("Branch")
HEAD_REGISTRY = Registry("Head")

class BaseModule(nn.Module):
    """
    Constructs base module that contains basic visualization function and corresponding hooks.
    Note: The visualization function has only tested in the single GPU scenario.
        By default, the visualization is disabled.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseModule, self).__init__()
        self.cfg = cfg
        self.id = 0
        if self.cfg.VISUALIZATION.ENABLE and self.cfg.VISUALIZATION.FEATURE_MAPS.ENABLE:
            self.base_output_dir = self.cfg.VISUALIZATION.FEATURE_MAPS.BASE_OUTPUT_DIR
            self.register_forward_hook(self.visualize_features)
    
    def visualize_features(self, module, input, output_x):
        """
        Visualizes and saves the normalized output features for the module.
        """
        # feature normalization
        b,c,t,h,w = output_x.shape
        xmin, xmax = output_x.min(1).values.unsqueeze(1), output_x.max(1).values.unsqueeze(1)
        x_vis = ((output_x.detach() - xmin) / (xmax-xmin)).permute(0, 1, 3, 2, 4).reshape(b, c*h, t*w).detach().cpu().numpy()
        # x_vis = ((output_x.detach() - xmin) / (xmax-xmin)).permute(0, 1, 3, 2, 4).sum(1).reshape(b, h, t*w).detach().cpu().numpy()
        if hasattr(self, "stage_id"):
            stage_id = self.stage_id
            block_id = self.block_id
        else:
            stage_id = 0
            block_id = 0
        for i in range(b):
            if not os.path.exists(f'{self.base_output_dir}/{self.cfg.VISUALIZATION.NAME}/im_{self.id+i}/'):
                os.makedirs(f'{self.base_output_dir}/{self.cfg.VISUALIZATION.NAME}/im_{self.id+i}/')
            plt.imsave(
                f'{self.base_output_dir}/{self.cfg.VISUALIZATION.NAME}/im_{self.id+i}/layer_{stage_id}_{block_id}_feature.jpg', 
                x_vis[i]
            )
        self.id += b

class BaseBranch(BaseModule):
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
        super(BaseBranch, self).__init__(cfg)
        self.cfg = cfg
        update_3d_conv_params(cfg, self, block_idx)
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

class Base3DBlock(nn.Module):
    """
    Constructs a base 3D block, composed of a shortcut and a conv branch.
    """
    def __init__(
        self,
        cfg,
        block_idx,
    ):
        """
        Args: 
            cfg         (Config): global config object. 
            block_idx   (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(Base3DBlock, self).__init__()
        
        self.cfg = cfg
        self.act = cfg.VIDEO.BACKBONE.ACTIVATION
        update_3d_conv_params(cfg, self, block_idx)

        self._construct_block(
            cfg             = cfg,
            block_idx       = block_idx,
        )
            
    def _construct_block(
        self,
        cfg,
        block_idx,
    ):
        if self.dim_in != self.num_filters or self.downsampling:
            # if the spatial size or the channel dimension is inconsistent for the input and the output 
            # of the block, a 1x1x1 3D conv need to be performed. 
            self.short_cut = nn.Conv3d(
                self.dim_in,
                self.num_filters,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                bias=False
            )
            self.short_cut_bn = nn.BatchNorm3d(
                self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt
            )
        self.conv_branch = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, block_idx)
        if self.act == "ReLU":
            self.relu = nn.ReLU(inplace=True)
        elif self.act == "GELU":
            self.gelu = nn.GELU()
            
    def forward(self, x):
        short_cut = x
        if hasattr(self, "short_cut"):
            short_cut = self.short_cut_bn(self.short_cut(short_cut))
        
        if self.act == "ReLU":
            x = self.relu(short_cut + self.conv_branch(x))
        else:
            x = self.gelu(short_cut + self.conv_branch(x))
        return x

class Base3DResStage(nn.Module):
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
        super(Base3DResStage, self).__init__()
        self.cfg = cfg
        self.num_blocks = num_blocks       
        self._construct_stage(
            cfg                     = cfg,
            stage_idx               = stage_idx,
        )
        
    def _construct_stage(
        self,
        cfg,
        stage_idx,
    ):
        res_block = Base3DBlock(
            cfg                     = cfg,
            block_idx               = [stage_idx, 0],
        )
        self.add_module("res_{}".format(1), res_block)
        for i in range(self.num_blocks-1):
            res_block = Base3DBlock(
                cfg                 = cfg,
                block_idx           = [stage_idx, i+1],
            )
            self.add_module("res_{}".format(i+2), res_block)
        if cfg.VIDEO.BACKBONE.NONLOCAL.ENABLE and stage_idx+1 in cfg.VIDEO.BACKBONE.NONLOCAL.STAGES:
            non_local = BRANCH_REGISTRY.get('NonLocal')(
                cfg                 = cfg,
                block_idx           = [stage_idx, i+2]
            )
            self.add_module("nonlocal", non_local)

    def forward(self, x):

        # performs computation on the convolutions
        for i in range(self.num_blocks):
            res_block = getattr(self, "res_{}".format(i+1))
            x = res_block(x)

        # performs non-local operations if specified.
        if hasattr(self, "nonlocal"):
            non_local = getattr(self, "nonlocal")
            x = non_local(x)
        return x

class InceptionBaseConv3D(BaseModule):
    """
    Constructs basic inception 3D conv.
    Modified from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """
    def __init__(self, cfg, in_planes, out_planes, kernel_size, stride, padding=0):
        super(InceptionBaseConv3D, self).__init__(cfg)
        self.conv = nn.Conv3d(in_planes, out_planes, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

@STEM_REGISTRY.register()
class Base2DStem(BaseModule):
    """
    Constructs basic ResNet 2D Stem.
    A single 2D convolution is performed in the base 2D stem.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base2DStem, self).__init__(cfg)

        self.cfg = cfg
        self.act = cfg.VIDEO.BACKBONE.ACTIVATION

        # loading the config for downsampling
        _downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        _downsampling_temporal  = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        if _downsampling:
            _stride = [1, 2, 2]
        else:
            _stride = [1, 1, 1]
        
        self._construct_block(
            cfg             = cfg,
            dim_in          = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS,
            num_filters     = cfg.VIDEO.BACKBONE.NUM_FILTERS[0],
            kernel_sz       = cfg.VIDEO.BACKBONE.KERNEL_SIZE[0],
            stride          = _stride,
            bn_eps          = cfg.BN.EPS,
            bn_mmt          = cfg.BN.MOMENTUM
        )

    def _construct_block(
        self, 
        cfg,
        dim_in, 
        num_filters, 
        kernel_sz,
        stride,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        if cfg.VIDEO.BACKBONE.META_ARCH == "ResNet3D" or cfg.VIDEO.BACKBONE.META_ARCH == "TAda2DEvolving": 
            self.a = nn.Conv3d(
                dim_in,
                num_filters,
                kernel_size = [1, kernel_sz[1], kernel_sz[2]],
                stride      = [1, stride[1], stride[2]],
                padding     = [0, kernel_sz[1]//2, kernel_sz[2]//2],
                bias        = False
            )
            self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
            if self.act == "ReLU":
                self.a_relu = nn.ReLU(inplace=True)
            elif self.act == "GELU":
                self.a_gelu = nn.GELU()
        elif cfg.VIDEO.BACKBONE.META_ARCH == "ResNet2D":
            self.a = nn.Conv2d(
                dim_in,
                num_filters,
                kernel_size = [kernel_sz[1], kernel_sz[2]],
                stride      = [stride[1], stride[2]],
                padding     = [kernel_sz[1]//2, kernel_sz[2]//2],
                bias        = False
            )
            self.a_bn = nn.BatchNorm2d(num_filters, eps=bn_eps, momentum=bn_mmt)
            if self.act == "ReLU":
                self.a_relu = nn.ReLU(inplace=True)
            elif self.act == "GELU":
                self.a_gelu = nn.GELU()

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        if self.act == "ReLU":
            x = self.a_relu(x)
        elif self.act == "GELU":
            x = self.a_gelu(x)
        return x

@STEM_REGISTRY.register()
class Base3DStem(BaseModule):
    """
    Constructs basic ResNet 3D Stem.
    A single 3D convolution is performed in the base 3D stem.
    """
    def __init__(
        self, 
        cfg
    ):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(Base3DStem, self).__init__(cfg)

        self.cfg = cfg

        # loading the config for downsampling
        _downsampling           = cfg.VIDEO.BACKBONE.DOWNSAMPLING[0]
        _downsampling_temporal  = cfg.VIDEO.BACKBONE.DOWNSAMPLING_TEMPORAL[0]
        if _downsampling:
            if _downsampling_temporal:
                _stride = [2, 2, 2]
            else:
                _stride = [1, 2, 2]
        else:
            _stride = [1, 1, 1]
        
        self._construct_block(
            cfg             = cfg,
            dim_in          = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS,
            num_filters     = cfg.VIDEO.BACKBONE.NUM_FILTERS[0],
            kernel_sz       = cfg.VIDEO.BACKBONE.KERNEL_SIZE[0],
            stride          = _stride,
            bn_eps          = cfg.BN.EPS,
            bn_mmt          = cfg.BN.MOMENTUM
        )
        
    def _construct_block(
        self, 
        cfg,
        dim_in, 
        num_filters, 
        kernel_sz,
        stride,
        bn_eps=1e-5,
        bn_mmt=0.1
    ):
        self.a = nn.Conv3d(
            dim_in,
            num_filters,
            kernel_size = kernel_sz,
            stride      = stride,
            padding     = [kernel_sz[0]//2, kernel_sz[1]//2, kernel_sz[2]//2],
            bias        = False
        )
        self.a_bn = nn.BatchNorm3d(num_filters, eps=bn_eps, momentum=bn_mmt)
        self.a_relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        return x

@HEAD_REGISTRY.register()
class BaseHead(nn.Module):
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
        super(BaseHead, self).__init__()
        self.cfg = cfg
        dim             = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(
            dim,
            num_classes,
            dropout_rate,
            activation_func
        )

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

    def forward(self, x):
        
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))    
        elif len(x.shape) == 4 and self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet2D":
            b_t, c, h, w = x.size()
            b = b_t // self.cfg.DATA.NUM_INPUT_FRAMES
            t = self.cfg.DATA.NUM_INPUT_FRAMES
            x = self.global_avg_pool(x.reshape(b,t,c,h,w).permute(0,2,1,3,4))
            x = x.permute(0,2,3,4,1)
        if hasattr(self, "dropout"):
            logits = self.dropout(x)
        else:
            logits = x
        out = self.out(logits)
        out_logits = out

        if not self.training:
            out = self.activation(out)
        
        out = out.view(out.shape[0], -1)
        return out, logits
    
    def forward_dropout(self, x):
        
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))    
        elif len(x.shape) == 4 and self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet2D":
            b_t, c, h, w = x.size()
            b = b_t // self.cfg.DATA.NUM_INPUT_FRAMES
            t = self.cfg.DATA.NUM_INPUT_FRAMES
            x = self.global_avg_pool(x.reshape(b,t,c,h,w).permute(0,2,1,3,4))
            x = x.permute(0,2,3,4,1)
        
        out_list = []
        for i in range(5):
            if hasattr(self, "dropout"):
                logits = self.dropout(x)
            else:
                logits = x
            out = self.out(logits)
            out_logits = out

            if not self.training:
                out = self.activation(out)

            out = out.view(out.shape[0], -1)
            out_list.append(out)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        out_var = torch.var(out_list, dim=0)
        
        return out_mean, out_var

@HEAD_REGISTRY.register()
class BaseHeadBNN(nn.Module):
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
        super(BaseHeadBNN, self).__init__()
        self.cfg = cfg
        dim             = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(
            dim,
            num_classes,
            dropout_rate,
            activation_func,

        )

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

        # self.out = nn.Linear(dim, num_classes, bias=True)
        self.bnn_cls = BayesianPredictor(dim, num_classes)


        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )

    def forward(self, x):
        
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))    
        elif len(x.shape) == 4 and self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet2D":
            b_t, c, h, w = x.size()
            b = b_t // self.cfg.DATA.NUM_INPUT_FRAMES
            t = self.cfg.DATA.NUM_INPUT_FRAMES
            x = self.global_avg_pool(x.reshape(b,t,c,h,w).permute(0,2,1,3,4))
            x = x.permute(0,2,3,4,1)
        if hasattr(self, "dropout"):
            logits = self.dropout(x)
        else:
            logits = x
        # out = self.out(logits)
        outputs, log_priors, log_variational_posteriors = self.bnn_cls(x, npass=self.cfg.TRAIN.NPASS, testing=not self.training)

        # gather output dictionary
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        output_dict = {'log_prior': log_prior,
                       'log_posterior': log_variational_posterior}
        uncertain_alea, uncertain_epis = get_uncertainty(outputs)
        output_dict.update({'aleatoric': uncertain_alea,
                                'epistemic': uncertain_epis})

        cls_score = outputs.mean(0)
        # if not self.training:
        #     out = self.activation(out)
        
        output_dict.update({'pred_mean': cls_score.squeeze(1)})
        return output_dict, None
    
    def forward_dropout(self, x):
        
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))    
        elif len(x.shape) == 4 and self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet2D":
            b_t, c, h, w = x.size()
            b = b_t // self.cfg.DATA.NUM_INPUT_FRAMES
            t = self.cfg.DATA.NUM_INPUT_FRAMES
            x = self.global_avg_pool(x.reshape(b,t,c,h,w).permute(0,2,1,3,4))
            x = x.permute(0,2,3,4,1)
        
        out_list = []
        for i in range(5):
            if hasattr(self, "dropout"):
                logits = self.dropout(x)
            else:
                logits = x
            out = self.out(logits)
            out_logits = out

            if not self.training:
                out = self.activation(out)

            out = out.view(out.shape[0], -1)
            out_list.append(out)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        out_var = torch.var(out_list, dim=0)
        
        return out_mean, out_var

@HEAD_REGISTRY.register()
class BaseHeadRPL(nn.Module):
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
        super(BaseHeadRPL, self).__init__()
        self.cfg = cfg
        dim             = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        num_classes     = cfg.VIDEO.HEAD.NUM_CLASSES
        dropout_rate    = cfg.VIDEO.HEAD.DROPOUT_RATE
        activation_func = cfg.VIDEO.HEAD.ACTIVATION
        self._construct_head(
            dim,
            num_classes,
            dropout_rate,
            activation_func
        )

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

        self.fc_centers = nn.Linear(dim, num_classes, bias=False)
        self.num_classes = num_classes
        self.num_centers = 1
        self.radius = nn.Parameter(torch.zeros((self.num_classes, )))

        if activation_func == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )

    def forward(self, x):
        
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))    
        elif len(x.shape) == 4 and self.cfg.VIDEO.BACKBONE.META_ARCH == "ResNet2D":
            b_t, c, h, w = x.size()
            b = b_t // self.cfg.DATA.NUM_INPUT_FRAMES
            t = self.cfg.DATA.NUM_INPUT_FRAMES
            x = self.global_avg_pool(x.reshape(b,t,c,h,w).permute(0,2,1,3,4))
            x = x.permute(0,2,3,4,1)
        if hasattr(self, "dropout"):
            logits = self.dropout(x)
        else:
            logits = x
        dist = self.compute_dist(logits)
        if not self.training:
            dist = self.activation(dist)

        outputs = {'dist': dist, 'feature': logits, 'centers': self.fc_centers.weight, 'radius': self.radius}

        return outputs, logits
    
    def compute_dist(self, features, center=None, metric=None):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.fc_centers.weight, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * self.fc_centers(features) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                dist = self.fc_centers(features)
            else:
                dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)
        return dist

@HEAD_REGISTRY.register()
class BaseHeadx2(BaseHead):
    """
    Constructs two base heads in parallel.
    This is specifically for EPIC-KITCHENS dataset, where 'noun' and 'verb' class are predicted.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseHeadx2, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

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
    
    def forward(self, x):
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))    

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

@HEAD_REGISTRY.register()
class BaseHeadx2Obj(BaseHead):
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseHeadx2Obj, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.scale = dim ** -0.5

        if self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_noun" or self.cfg.VIDEO.HEAD.OBJ_FEAT == "sa_cat_noun":
            self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
            self.linear2 = nn.Linear(dim*2, num_classes[1], bias=True)
        elif self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_x" or self.cfg.VIDEO.HEAD.OBJ_FEAT == "sa_cat_x":
            self.linear1 = nn.Linear(dim*2, num_classes[0], bias=True)
            self.linear2 = nn.Linear(dim*2, num_classes[1], bias=True)

        if "sa_cat" in self.cfg.VIDEO.HEAD.OBJ_FEAT and self.cfg.VIDEO.HEAD.COMPLEX:
            self.linear_sa = nn.Linear(dim*2, dim*2)
            self.bn_train = nn.BatchNorm1d(dim*2, eps=1e-5, momentum=0.9)

        self.linear_obj = nn.Linear(1024, 512)
        self.bn = nn.BatchNorm1d(512, eps=1e-5, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.linear_obj_out = nn.Linear(512, dim)

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
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))   

        x_obj = self.linear_obj(x_obj)
        x_obj = self.bn(x_obj.permute(0,2,1)).permute(0,2,1)
        x_obj = self.relu(x_obj)
        x_obj = self.linear_obj_out(x_obj) 


        if self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_x":
            x_obj = x_obj.mean(1).reshape(x.shape[0], 1, 1, 1, x.shape[-1])
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
            x = x.reshape(x.shape[0], 1, x.shape[-1])
            attn = (x @ x_obj.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)         
            x = torch.cat(
                (x, (attn @ x_obj)), dim=-1
            )
            if self.cfg.VIDEO.HEAD.COMPLEX:
                x = self.linear_sa(x)
                x = self.bn_train(x.permute(0,2,1)).permute(0,2,1)
                x = self.relu(x)

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
            x_obj = x_obj.mean(1).reshape(x.shape[0], 1, 1, 1, x.shape[-1])
            out2 = self.linear2(torch.cat(
                (out2, x_obj), dim=-1
            ))
        
        elif self.cfg.VIDEO.HEAD.OBJ_FEAT == "sa_cat_noun":
            # q: x
            # k, v: x_obj
            x = x.reshape(x.shape[0], 1, x.shape[-1])
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
            if self.cfg.VIDEO.HEAD.COMPLEX:
                out2 = self.linear_sa(out2)
                out2 = self.bn_train(out2.permute(0,2,1)).permute(0,2,1)
                out2 = self.relu(out2)
            out2 = self.linear2(out2)
        
        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, x

@HEAD_REGISTRY.register()
class BaseHeadx2Objv2(BaseHead):
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseHeadx2Objv2, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.scale = dim ** -0.5

        if self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_noun":
            self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
            self.pre_classification_process = nn.Sequential(
                nn.BatchNorm1d(dim*2, eps=1e-5, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.Conv1d(dim*2, dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(dim, eps=1e-5, momentum=0.9),
                nn.ReLU(inplace=True)
            )
            self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

        self.linear_obj = nn.Linear(1024, 512)
        self.bn = nn.BatchNorm1d(512, eps=1e-5, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.linear_obj_out = nn.Linear(512, dim)

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
        if len(x.shape) == 5:
            x = self.global_avg_pool(x)
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))   

        x_obj = self.linear_obj(x_obj)
        x_obj = self.bn(x_obj.permute(0,2,1)).permute(0,2,1)
        x_obj = self.relu(x_obj)
        x_obj = self.linear_obj_out(x_obj) 

        if self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_noun":
            if hasattr(self, "dropout"):
                out1 = self.dropout(x)
                out2 = out1
            else:
                out1 = x
                out2 = x

            out1 = self.linear1(out1)
            x_obj = x_obj.mean(1).unsqueeze(-1)
            out2 = out2.squeeze().unsqueeze(-1)
            out2 = self.pre_classification_process(torch.cat((out2, x_obj), dim=1))
            out2 = self.linear2(out2.squeeze())
        
        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        return {"verb_class": out1, "noun_class": out2}, x

@HEAD_REGISTRY.register()
class SAPHead(BaseHead):
    def __init__(self, cfg):
        super(SAPHead, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

        self.conv_obj = nn.Conv1d(1024, 512, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(512, eps=1e-5, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv_obj_out = nn.Conv1d(512, dim, kernel_size=1, bias=False)

        if "verb" in self.cfg.VIDEO.HEAD.SAP:
            self.sap_verb = SAP(dim)
        if "noun" in self.cfg.VIDEO.HEAD.SAP:
            self.sap_noun = SAP(dim)

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
    
    def forward(self, x_verb, x_noun, x_obj):
        if len(x_verb.shape) == 5:
            x_verb = self.global_avg_pool(x_verb)
            x_noun = self.global_avg_pool(x_noun)

            x_verb = x_verb.view(x_verb.shape[0], -1)
            x_noun = x_noun.view(x_noun.shape[0], -1)

        x_obj = x_obj.permute(0, 2, 1) # B N C -> B C N
        x_obj = self.conv_obj(x_obj)
        x_obj = self.bn(x_obj)
        x_obj = self.relu(x_obj)
        x_obj = self.conv_obj_out(x_obj) 

        # x_verb (B,1,1,1,C)
        # x_noun (B,1,1,1,C)
        # x_obj (B,10,C)

        if hasattr(self, "sap_verb"):
            if self.cfg.VIDEO.HEAD.SAP_DETACH:
                x_verb = self.sap_verb(x_verb, x_obj, x_noun.detach())
            else:
                x_verb = self.sap_verb(x_verb, x_obj, x_noun)
        if hasattr(self, "sap_noun"):
            if self.cfg.VIDEO.HEAD.SAP_DETACH:
                x_noun = self.sap_noun(x_noun, x_obj, x_verb.detach())
            else:
                x_noun = self.sap_noun(x_noun, x_obj, x_verb)

        if hasattr(self, "dropout"):
            x_verb = self.dropout(x_verb)
            x_noun = self.dropout(x_noun)

        x_verb = self.linear1(x_verb)
        x_noun = self.linear2(x_noun)

        if not self.training:
            x_verb = self.activation(x_verb)
            x_noun = self.activation(x_noun)
        x_verb = x_verb.view(x_verb.shape[0], -1)
        x_noun = x_noun.view(x_noun.shape[0], -1)
        return {"verb_class": x_verb, "noun_class": x_noun}, x_verb

@HEAD_REGISTRY.register()
class SAPHeadAsym(BaseHead):
    def __init__(self, cfg):
        super(SAPHeadAsym, self).__init__(cfg)
    
    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func,
    ):
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        dim_verb = self.cfg.VIDEO.BACKBONE_VERB.NUM_OUT_FEATURES
        dim_noun = self.cfg.VIDEO.BACKBONE_NOUN.NUM_OUT_FEATURES

        dim = 1024

        self.pre_map_verb = nn.Linear(dim_verb, dim, bias=False)
        self.pre_map_noun = nn.Linear(dim_noun, dim, bias=False)

        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

        self.conv_obj = nn.Conv1d(1024, 512, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(512, eps=1e-5, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv_obj_out = nn.Conv1d(512, dim, kernel_size=1, bias=False)

        if "verb" in self.cfg.VIDEO.HEAD.SAP:
            self.sap_verb = SAP(dim)
        if "noun" in self.cfg.VIDEO.HEAD.SAP:
            self.sap_noun = SAP(dim)

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
    
    def forward(self, x_verb, x_noun, x_obj):
        if len(x_verb.shape) == 5:
            x_verb = self.global_avg_pool(x_verb)
            x_verb = x_verb.view(x_verb.shape[0], -1)
        
        if len(x_noun.shape) == 5:
            x_noun = self.global_avg_pool(x_noun)
            x_noun = x_noun.view(x_noun.shape[0], -1)
        
        x_verb = self.pre_map_verb(x_verb)
        x_noun = self.pre_map_noun(x_noun)


        x_obj = x_obj.permute(0, 2, 1) # B N C -> B C N
        x_obj = self.conv_obj(x_obj)
        x_obj = self.bn(x_obj)
        x_obj = self.relu(x_obj)
        x_obj = self.conv_obj_out(x_obj) 

        # x_verb (B,1,1,1,C)
        # x_noun (B,1,1,1,C)
        # x_obj (B,10,C)

        if hasattr(self, "sap_verb"):
            if self.cfg.VIDEO.HEAD.SAP_DETACH:
                x_verb = self.sap_verb(x_verb, x_obj, x_noun.detach())
            else:
                x_verb = self.sap_verb(x_verb, x_obj, x_noun)
        if hasattr(self, "sap_noun"):
            if self.cfg.VIDEO.HEAD.SAP_DETACH:
                x_noun = self.sap_noun(x_noun, x_obj, x_verb.detach())
            else:
                x_noun = self.sap_noun(x_noun, x_obj, x_verb)

        if hasattr(self, "dropout"):
            x_verb = self.dropout(x_verb)
            x_noun = self.dropout(x_noun)

        x_verb = self.linear1(x_verb)
        x_noun = self.linear2(x_noun)

        if not self.training:
            x_verb = self.activation(x_verb)
            x_noun = self.activation(x_noun)
        x_verb = x_verb.view(x_verb.shape[0], -1)
        x_noun = x_noun.view(x_noun.shape[0], -1)
        return {"verb_class": x_verb, "noun_class": x_noun}, x_verb

class SAP(nn.Module):
    def __init__(self, channel_size):
        super(SAP, self).__init__()

        self.csg = CSG(channel_size)
        self.arm = ARM(channel_size)

    def forward(self, x, det, q):
        q = q.unsqueeze(-1)
        out = self.csg(x, det, q)
        out = self.arm(q, out)

        return out 

class ARM(nn.Module):

    def __init__(self, channel_size):
        super(ARM, self).__init__()
        self.fc_q = nn.Conv1d(channel_size, channel_size//4, kernel_size=1)
        self.fc_k = nn.Conv1d(channel_size, channel_size//4, kernel_size=1)

    def forward(self, q, v):
        q = self.fc_q(q) 
        q = q.permute(0,2,1)
        v_ori = v.permute(0,2,1)
        v = self.fc_k(v)
        w = torch.matmul(q, v)
        w = torch.softmax(w, dim=2)
        o = torch.matmul(w, v_ori)
        o = o.squeeze(1)
    
        return o
        
class CSG(nn.Module):
    def __init__(self, channel_size):
        super(CSG, self).__init__()    
        self.fc_d = nn.Conv1d(channel_size*2, channel_size, kernel_size=1)      
        self.fc_g = nn.Conv1d(channel_size, channel_size, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, det, q):
        det = torch.cat((det, x.unsqueeze(2).expand(det.size())),dim=1)
        det = self.relu(self.fc_d(det))
        g = torch.sigmoid(self.fc_g(q))

        return g * det

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Base2DBlock(nn.Module):
    """
    Constructs a base 3D block, composed of a shortcut and a conv branch.
    """
    def __init__(
        self,
        cfg,
        block_idx,
    ):
        """
        Args: 
            cfg         (Config): global config object. 
            block_idx   (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(Base2DBlock, self).__init__()
        
        self.cfg = cfg
        update_3d_conv_params(cfg, self, block_idx)

        self._construct_block(
            cfg             = cfg,
            block_idx       = block_idx,
        )
            
    def _construct_block(
        self,
        cfg,
        block_idx,
    ):
        if self.dim_in != self.num_filters or self.downsampling:
            # if the spatial size or the channel dimension is inconsistent for the input and the output 
            # of the block, a 1x1x1 3D conv need to be performed. 
            self.short_cut = nn.Conv2d(
                self.dim_in,
                self.num_filters,
                kernel_size=1,
                stride=self.stride[1:] if len(self.stride) == 3 else self.stride,
                padding=0,
                bias=False
            )
            self.short_cut_bn = nn.BatchNorm2d(
                self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt
            )
        self.conv_branch = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, block_idx)
        self.relu = nn.ReLU(inplace=True)
            
    def forward(self, x):
        short_cut = x
        if hasattr(self, "short_cut"):
            short_cut = self.short_cut_bn(self.short_cut(short_cut))
        
        x = self.relu(short_cut + self.conv_branch(x))
        return x

class Base2DResStage(nn.Module):
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
        super(Base2DResStage, self).__init__()
        self.cfg = cfg
        self.num_blocks = num_blocks       
        self._construct_stage(
            cfg                     = cfg,
            stage_idx               = stage_idx,
        )
        
    def _construct_stage(
        self,
        cfg,
        stage_idx,
    ):
        res_block = Base2DBlock(
            cfg                     = cfg,
            block_idx               = [stage_idx, 0],
        )
        self.add_module("res_{}".format(1), res_block)
        for i in range(self.num_blocks-1):
            res_block = Base2DBlock(
                cfg                 = cfg,
                block_idx           = [stage_idx, i+1],
            )
            self.add_module("res_{}".format(i+2), res_block)

    def forward(self, x):

        # performs computation on the convolutions
        for i in range(self.num_blocks):
            res_block = getattr(self, "res_{}".format(i+1))
            x = res_block(x)

        return x

class EvolvingTAdaBlock(nn.Module):
    """
    Constructs a base 3D block, composed of a shortcut and a conv branch.
    """
    def __init__(
        self,
        cfg,
        block_idx,
    ):
        """
        Args: 
            cfg         (Config): global config object. 
            block_idx   (list):   list of [stage_id, block_id], both starting from 0.
        """
        super(EvolvingTAdaBlock, self).__init__()
        
        self.cfg = cfg
        update_3d_conv_params(cfg, self, block_idx)

        self._construct_block(
            cfg             = cfg,
            block_idx       = block_idx,
        )
            
    def _construct_block(
        self,
        cfg,
        block_idx,
    ):
        if self.dim_in != self.num_filters or self.downsampling:
            # if the spatial size or the channel dimension is inconsistent for the input and the output 
            # of the block, a 1x1x1 3D conv need to be performed. 
            self.short_cut = nn.Conv3d(
                self.dim_in,
                self.num_filters,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                bias=False
            )
            self.short_cut_bn = nn.BatchNorm3d(
                self.num_filters, eps=self.bn_eps, momentum=self.bn_mmt
            )
        self.conv_branch = BRANCH_REGISTRY.get(cfg.VIDEO.BACKBONE.BRANCH.NAME)(cfg, block_idx)
        self.relu = nn.ReLU(inplace=True)
            
    def forward(self, x, cal_prev=None):
        short_cut = x
        if hasattr(self, "short_cut"):
            short_cut = self.short_cut_bn(self.short_cut(short_cut))
        
        x, cal = self.conv_branch(x, cal_prev)
        x = self.relu(short_cut + x)
        return x, cal

class EvolvingTAdaResStage(nn.Module):
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
        super(EvolvingTAdaResStage, self).__init__()
        self.cfg = cfg
        self.num_blocks = num_blocks       
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
            res_block = EvolvingTAdaBlock(
                cfg                 = cfg,
                block_idx           = [stage_idx, i],
            )
            self.add_module("res_{}".format(i+1), res_block)

    def forward(self, x, prev_cal=None):

        # performs computation on the convolutions
        cal = prev_cal
        for i in range(self.num_blocks):
            res_block = getattr(self, "res_{}".format(i+1))
            x, cal = res_block(x, cal)

        return x, cal