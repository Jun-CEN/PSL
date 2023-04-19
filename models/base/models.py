#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

import torch
import torch.nn as nn
from utils.registry import Registry
import utils.logging as logging
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import HEAD_REGISTRY
from torch.autograd import Variable
import torch.nn.functional as F

MODEL_REGISTRY = Registry("Model")

logger = logging.get_logger(__name__)

class BaseVideoModel(nn.Module):
    """
    Standard video model.
    The model is divided into the backbone and the head, where the backbone
    extracts features and the head performs classification.

    The backbones can be defined in model/base/backbone.py or anywhere else
    as long as the backbone is registered by the BACKBONE_REGISTRY.
    The heads can be defined in model/module_zoo/heads/ or anywhere else
    as long as the head is registered by the HEAD_REGISTRY.

    The registries automatically finds the registered modules and construct 
    the base video model.
    """
    def __init__(self, cfg):
        """
        Args: 
            cfg (Config): global config object. 
        """
        super(BaseVideoModel, self).__init__()
        self.cfg = cfg
        
        # the backbone is created according to meta-architectures 
        # defined in models/base/backbone.py
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)

        # the head is created according to the heads 
        # defined in models/module_zoo/heads
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
        
    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self.cfg.TEST, "OPEN_METHOD") and self.cfg.TEST.OPEN_METHOD == "dropout":
            x = self.head.forward_dropout(x)
        else:
            x = self.head(x)
        return x
    
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(BaseVideoModel, self).train(mode)

        if "TSM" in self.cfg.VIDEO.BACKBONE.META_ARCH:
            if hasattr(self.cfg.OPTIMIZER, "FREEZE") and self.cfg.OPTIMIZER.FREEZE:
                cnt = 0
                for m in self.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                        cnt += 1
                        if cnt >= 2:
                            m.train(False)
                            # shutdown update in frozen mode
                            # logger.info(f"Freeze {m}")
            if hasattr(self.cfg.OPTIMIZER, "FREEZE_ALL") and self.cfg.OPTIMIZER.FREEZE_ALL:
                for m in self.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                        m.train(False)

        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self

class BaseVideoSiamModel(nn.Module):

    def __init__(self, cfg):
        super(BaseVideoSiamModel, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
        self.proj_head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg, final=False)
        self.pred_head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)

    def forward(self, x):
        x = self.backbone(x)
        z = self.proj_head(x)
        p = self.pred_head(z)

        return z, p

class BaseMultiModalModel(nn.Module):

    def __init__(self, cfg):
        super(BaseMultiModalModel, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)

        assert cfg.TEXT.ENABLE or cfg.AUDIO.ENABLE, "One of \" (Audio, Text) \" modality must be enabled for BaseMultiModalModel."

        if cfg.TEXT.ENABLE:
            self.text_head = HEAD_REGISTRY.get(cfg.TEXT.HEAD.NAME)(cfg=cfg)
        
        if cfg.AUDIO.ENABLE:
            self.audio_backbone = BACKBONE_REGISTRY.get(cfg.AUDIO.META_ARCH)(cfg=cfg)
            self.audio_head = HEAD_REGISTRY.get(cfg.AUDIO.HEAD.NAME)(cfg=cfg)
        
        if cfg.PRETRAIN.PROTOTYPE.ENABLE:
            if cfg.PRETRAIN.PROTOTYPE.DECOUPLE:
                self.prototypes_x = nn.Linear(cfg.PRETRAIN.CONTRASTIVE.HEAD_OUT_DIM, cfg.PRETRAIN.PROTOTYPE.NUM_PROTOTYPES, bias=False)
                self.prototypes_t = nn.Linear(cfg.PRETRAIN.CONTRASTIVE.HEAD_OUT_DIM, cfg.PRETRAIN.PROTOTYPE.NUM_PROTOTYPES, bias=False)
                with torch.no_grad():
                    self.prototypes_t.weight.copy_(self.prototypes_x.weight)
                if cfg.PRETRAIN.PROTOTYPE.MOMENTUM < 1 and cfg.PRETRAIN.PROTOTYPE.MOMENTUM > 0:
                    self.prototypes_t.weight.requires_grad = False
                    self.m = cfg.PRETRAIN.PROTOTYPE.MOMENTUM
            else:
                self.prototypes = nn.Linear(cfg.PRETRAIN.CONTRASTIVE.HEAD_OUT_DIM, cfg.PRETRAIN.PROTOTYPE.NUM_PROTOTYPES, bias=False)
                if cfg.PRETRAIN.PROTOTYPE.MOMENTUM < 1 and cfg.PRETRAIN.PROTOTYPE.MOMENTUM > 0:
                    self.prototypes.weight.requires_grad = False
                    self.m = cfg.PRETRAIN.PROTOTYPE.MOMENTUM

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm3d)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, xv, xt=None, xa=None):
        xv_mid = self.backbone(xv)
        xv = self.head(xv_mid)
        xv_mid = xv_mid.reshape(xv_mid.shape[0], xv_mid.shape[1], -1).mean(-1)
        xv_mid = torch.nn.functional.normalize(xv_mid, dim=-1, p=2)

        if isinstance(xv, tuple):
            if self.cfg.PRETRAIN.ASYM.HEAD_X_REVERSE:
                # if reverse: 
                # xv -> xv_proto, xv_nce
                xv_proto = xv[0]
                xv = xv[1]
            else:
                # proper sequence:
                # xv -> xv_nce, xv_proto
                xv_proto = xv[1]
                xv = xv[0]
        else:
            xv_proto = xv

        assert xt is not None or xa is not None, "One of \" (Audio, Text) \" modality must be fed to forward for BaseMultiModalModel."

        if self.cfg.TEXT.ENABLE and xt is not None:
            xt = self.text_head(xt)
        else:
            xt = None

        if self.cfg.AUDIO.ENABLE and xa is not None:
            xa = self.audio_backbone(xa)
            xa = self.audio_head(xa)
        else:
            xa = None

        # calculate prototypes 
        pv = None
        pt = None
        zt = None
        if self.cfg.PRETRAIN.PROTOTYPE.ENABLE and self.cfg.PRETRAIN.PROTOTYPE.DECOUPLE:
            with torch.no_grad():
                w = self.prototypes_x.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes_x.weight.copy_(w)
                w = self.prototypes_t.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes_t.weight.copy_(w)
            if self.cfg.PRETRAIN.PROTOTYPE.POOLING == "attentive-pool":
                weight = (xv.unsqueeze(1)*xt/self.cfg.PRETRAIN.PROTOTYPE.ATTENTIVE_POOLING_TEMP).sum(-1).softmax(-1).unsqueeze(-1).detach()
                if self.cfg.PRETRAIN.PROTOTYPE.POOLING_NORM:
                    zt = nn.functional.normalize((xt * weight).sum(1), dim=-1, p=2)
                else:
                    zt = (xt * weight).sum(1)
                pv = self.prototypes_x(xv)
                pt = self.prototypes_t(zt)
            else:
                pv = self.prototypes_x(xv)
                pt = self.prototypes_t(xt)
        elif self.cfg.PRETRAIN.PROTOTYPE.ENABLE and not self.cfg.PRETRAIN.PROTOTYPE.DECOUPLE:
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

            if self.cfg.PRETRAIN.PROTOTYPE.POOLING == "attentive-pool":
                weight = (xv.unsqueeze(1)*xt/self.cfg.PRETRAIN.PROTOTYPE.ATTENTIVE_POOLING_TEMP).sum(-1).softmax(-1).unsqueeze(-1).detach()

                if self.cfg.PRETRAIN.PROTOTYPE.POOLING_NORM:
                    zt = nn.functional.normalize((xt * weight).sum(1), dim=-1, p=2)
                else:
                    zt = (xt * weight).sum(1)
                
                pv = self.prototypes(xv_proto)
                pt = self.prototypes(zt)
            else:
                pv = self.prototypes(xv_proto)
                pt = self.prototypes(xt)

        return xv_mid, xv, pv, xt, pt, zt, xa
    
    def forward_video(self, xv):
        xv_mid = self.backbone(xv)
        xv = self.head(xv_mid)
        if isinstance(xv, tuple):
            if self.cfg.PRETRAIN.ASYM.HEAD_X_REVERSE:
                # if reverse: 
                # xv -> xv_proto, xv_nce
                xv_proto = xv[0]
                xv = xv[1]
            else:
                # proper sequence:
                # xv -> xv_nce, xv_proto
                xv_proto = xv[1]
                xv = xv[0]
        else:
            xv_proto = xv
        xv_mid = xv_mid.reshape(xv_mid.shape[0], xv_mid.shape[1], -1).mean(-1)
        return xv_mid, xv

    def forward_text(self, xt):
        xt = self.text_head(xt)
        return xt
    
    @torch.no_grad()
    def update_prototypes(self, incoming_weight):
        if self.cfg.PRETRAIN.PROTOTYPE.DECOUPLE:
            self.prototypes_t.weight.copy_(self.prototypes_t.weight * self.m + incoming_weight * (1 - self.m))
        else:
            self.prototypes.weight.copy_(self.prototypes.weight * self.m + incoming_weight * (1 - self.m))

    def regularization(self, q, zt):
        return ( (torch.matmul(q, self.prototypes.weight) - zt.detach())**2 / zt.shape[0]).sum()

@MODEL_REGISTRY.register()
class CMPNet(BaseVideoModel):
    def __init__(self, cfg):
        super(CMPNet, self).__init__(cfg)
    
    def forward(self, x):
        if isinstance(x, dict):
            x_data = x["video"]
        else:
            x_data = x
        b, n, c, t, h, w = x_data.shape
        x_data = x_data.reshape(b*n, c, t, h, w)
        res, logits = super(CMPNet, self).forward(x_data)
        pred = {}
        if isinstance(res, dict):
            for k, v in res.items():
                pred[k] = v
        else:
            pred["move_joint"] = res
        return pred, logits

@MODEL_REGISTRY.register()
class ContrastiveModel(BaseVideoModel):
    def __init__(self, cfg):
        super(ContrastiveModel, self).__init__(cfg)
    
    def forward(self, x):
        if isinstance(x, dict):
            x_data = x["video"]
        else:
            x_data = x
        b, n, c, t, h, w = x_data.shape
        x_data = x_data.reshape(b*n, c, t, h, w)
        logits = super(ContrastiveModel, self).forward(x_data)
        pred = {}
        return pred, logits

@MODEL_REGISTRY.register()
class Contrastive_CE_Model(BaseVideoModel):
    def __init__(self, cfg):
        super(Contrastive_CE_Model, self).__init__(cfg)
    
    def forward(self, x):
        if isinstance(x, dict):
            x_data = x["video"]
        else:
            x_data = x
        print(x_data.shape)
        b, n, c, t, h, w = x_data.shape
        x_data = x_data.reshape(b*n, c, t, h, w)
        pred, logits = super(Contrastive_CE_Model, self).forward(x_data)
        return pred, logits

@MODEL_REGISTRY.register()
class PrototypeModel(BaseVideoModel):
    def __init__(self, cfg):
        super(PrototypeModel, self).__init__(cfg)

        self.prototypes = torch.eye(cfg.MODEL.NUM_CLASSES)
        # if self.cfg.MODEL.MLP:
        #     self.prototypes = torch.cat((self.prototypes, torch.zeros(cfg.MODEL.NUM_CLASSES, cfg.VIDEO.HEAD.MLP.OUT_DIM - cfg.MODEL.NUM_CLASSES)), dim=1).cuda()
        # else:
        #     self.prototypes = torch.cat((self.prototypes, torch.zeros(cfg.MODEL.NUM_CLASSES, 2048 - cfg.MODEL.NUM_CLASSES)), dim=1).cuda()
        self.prototypes = torch.normal(0.005, 0.005, (cfg.MODEL.NUM_CLASSES, cfg.VIDEO.HEAD.MLP.OUT_DIM)).cuda()
        if hasattr(cfg.MODEL, 'NUM_UNKNOWN'):
            self.prototypes = torch.normal(0.005, 0.005, (cfg.MODEL.NUM_CLASSES+cfg.MODEL.NUM_UNKNOWN, cfg.VIDEO.HEAD.MLP.OUT_DIM)).cuda()
        if cfg.MODEL.LEARNABLE_PROTOTYPE:
            self.prototypes = Variable(self.prototypes, requires_grad=True)

    
    def forward(self, x):
        if isinstance(x, dict):
            x_data = x["video"]
        else:
            x_data = x
        if len(x_data.shape) == 6:
            b, n, c, t, h, w = x_data.shape
            x_data = x_data.reshape(b*n, c, t, h, w)
        if self.cfg.MODEL.MLP:
            features = self.backbone(x_data)
            logits_nm = self.head(features)
        else:
            logits = self.backbone(x_data)
            logits_nm = F.normalize(logits,p=2,dim=1)
        prototypes = F.normalize(self.prototypes[:self.cfg.MODEL.NUM_CLASSES,:], p=2, dim=1)
        sim = torch.matmul(logits_nm, prototypes.transpose(0,1))
        sim_mtx = torch.exp(sim/self.cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
        pred = sim_mtx / sim_mtx.sum(dim=1, keepdim=True)
        pred = (pred, self.prototypes)

        return pred, logits_nm


@MODEL_REGISTRY.register()
class VideoTextModel(BaseMultiModalModel):
    def __init__(self, cfg):
        super(VideoTextModel, self).__init__(cfg)

    def forward(self, x):
        if self.training:
            if len(x["video"].shape) == 6:
                b,n,c,t,h,w = x["video"].shape
                x["video"] = x["video"].reshape(b*n, c, t, h, w)
            xv_mid, xv, pv, xt, pt, zt, _ = super(VideoTextModel, self).forward(
                x["video"],
                xt=x["text_embedding"]
            )
            logits = {
                "video_premlp": xv_mid,
                "video": xv,
                "text": xt,
                "text_attentive_pooled": zt,
                "video_prototypes": pv,
                "text_prototypes": pt
            }
            pred = {}
        else:
            logits = {}
            pred = {}
            if "video" in x.keys():
                xv_mid, xv = super(VideoTextModel, self).forward_video(x["video"])
                logits["xv_mid"] = xv_mid
                logits["xv"] = xv
            if "text_embedding" in x.keys():
                xt = super(VideoTextModel, self).forward_text(x["text_embedding"])
                logits["xt"] = xt
            
            # xv, pv, xt, pt, zt, _ = super(VideoTextModel, self).forward(
            #     x["video"],
            #     xt=x["text_embedding"],
            #     xt_mask=x["text_validity"] if "text_validity" in x.keys() else None
            # )
            # if len(xt.shape) == 2:
            #     pred = torch.matmul(xv, xt.transpose(0,1)).softmax(-1)
            # else:
            #     pred = {}
            # logits = {
            #     "video": xv,
            #     "text": xt,
            #     "video_prototypes": pv,
            #     "text_prototypes": pt
            # }
        return pred, logits

@MODEL_REGISTRY.register()
class VideoSimSiamModel(BaseVideoSiamModel):
    def __init__(self, cfg):
        super(VideoSimSiamModel, self).__init__(cfg)
    
    def forward(self, x):
        if isinstance(x, dict):
            x_data = x["video"]
        else:
            x_data = x
        x1 = x_data[:, 0]
        x2 = x_data[:, 1]
        z1, p1 = super(VideoSimSiamModel, self).forward(x1)
        z2, p2 = super(VideoSimSiamModel, self).forward(x2)
        logits = (z1, z2, p1, p2)
        pred = {}
        return pred, logits

@MODEL_REGISTRY.register()
class VideoFlowTwoStreamModel(BaseVideoModel):
    def __init__(self, cfg):
        super(VideoFlowTwoStreamModel, self).__init__(cfg)
        self.flow_backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
    
    def forward(self, x):
        x_rgb  = x["video"]
        x_flow = x["flow"]
        x_rgb = self.backbone(x_rgb)
        x_flow = self.flow_backbone(x_flow)
        return self.head(x_rgb, x_flow)

@MODEL_REGISTRY.register()
class BaseVideoModelObj(nn.Module):

    def __init__(self, cfg):
        super(BaseVideoModelObj, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
        
    def forward(self, x):
        if isinstance(x, dict):
            x_vid = x["video"]
            x_obj = x["obj_feat"]
        x_vid = self.backbone(x_vid)
        x = self.head(x_vid, x_obj)
        return x
    
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(BaseVideoModelObj, self).train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self

@MODEL_REGISTRY.register()
class TransformerModelObj(nn.Module):

    def __init__(self, cfg):
        super(TransformerModelObj, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
        
    def forward(self, x):
        if isinstance(x, dict):
            x_vid = x["video"]
            x_obj = x["obj_feat"]
        x = self.backbone(x_vid, x_obj)
        x = self.head(x)
        return x
    
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(TransformerModelObj, self).train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self


@MODEL_REGISTRY.register()
class TwoStreamVerbNounObj(nn.Module):

    def __init__(self, cfg):
        super(TwoStreamVerbNounObj, self).__init__()
        self.cfg = cfg
        self.verb_backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
        self.noun_backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
        
    def forward(self, x):
        if isinstance(x, dict):
            x_vid = x["video"]
            x_obj = x["obj_feat"]
        x_verb = self.verb_backbone(x_vid)
        x_noun = self.noun_backbone(x_vid)
        x = self.head(x_verb, x_noun, x_obj)
        return x
    
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(TwoStreamVerbNounObj, self).train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self


@MODEL_REGISTRY.register()
class TwoStreamVerbNounObjAsym(nn.Module):

    def __init__(self, cfg):
        super(TwoStreamVerbNounObjAsym, self).__init__()
        self.cfg = cfg
        cfg_verb = cfg.deep_copy()
        cfg_noun = cfg.deep_copy()
        cfg_verb.VIDEO.BACKBONE = cfg_verb.VIDEO.BACKBONE_VERB
        cfg_noun.VIDEO.BACKBONE = cfg_noun.VIDEO.BACKBONE_NOUN
        self.verb_backbone = BACKBONE_REGISTRY.get(cfg_verb.VIDEO.BACKBONE.META_ARCH)(cfg=cfg_verb)
        self.noun_backbone = BACKBONE_REGISTRY.get(cfg_noun.VIDEO.BACKBONE.META_ARCH)(cfg=cfg_noun)
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
        
    def forward(self, x):
        if isinstance(x, dict):
            x_vid = x["video"]
            if "obj_feat" in x.keys():
                x_obj = x["obj_feat"]
            else:
                x_obj = None
        x_verb = self.verb_backbone(x_vid)
        x_noun = self.noun_backbone(x_vid)
        x = self.head(x_verb, x_noun, x_obj)
        return x
    
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(TwoStreamVerbNounObjAsym, self).train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self


@MODEL_REGISTRY.register()
class BaseVideoModelLFB(nn.Module):

    def __init__(self, cfg):
        super(BaseVideoModelLFB, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONE_REGISTRY.get(cfg.VIDEO.BACKBONE.META_ARCH)(cfg=cfg)
        self.head = HEAD_REGISTRY.get(cfg.VIDEO.HEAD.NAME)(cfg=cfg)
        
    def forward(self, x):
        if isinstance(x, dict):
            x_vid = x["video"]
            x_lfb = x["lfb_feat"]
        x_vid = self.backbone(x_vid)
        x = self.head(x_vid, x_lfb)
        return x
    
    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        super(BaseVideoModelLFB, self).train(mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.cfg.BN.FREEZE:
                module.train(False)
        return self