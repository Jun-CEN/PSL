#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Losses. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import Registry

import utils.misc as misc
import utils.distributed as du

import models.utils.contrastive_losses as cont
import models.utils.clustering_losses as clst

from datasets.utils.mixup import label_smoothing

SSL_LOSSES = Registry("SSL_Losses")

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, reduction=None):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class FocalCrossEntropy(nn.Module):

    def __init__(self, reduction=None):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(FocalCrossEntropy, self).__init__()
        self.gamma = 1.5
    
    def forward(self, x, target):
        pred = x.softmax(-1)
        loss = ((1-pred)**self.gamma) * (-torch.log(pred))
        loss = loss[torch.linspace(0, x.shape[0]-1, x.shape[0]).long(), target].mean()
        return loss

class BayesianNNLoss(nn.Module):

    def __init__(self, reduction=None):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(BayesianNNLoss, self).__init__()

    def forward(cfg, output_dict, labels):
        """Bayesian NN Loss."""

        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            prior (torch.Tensor): The log prior
            posterior (torch.Tensor): The log variational posterior
            kwargs: Any keyword argument to be used to calculate
                Bayesian NN loss.

        Returns:
            torch.Tensor: The returned Bayesian NN loss.
        """
        cls_score = output_dict['pred_mean']
        beta=0
        losses = dict()

        # negative log-likelihood (BCE loss)
        loss_cls = F.cross_entropy(cls_score, labels)
        # parse the output
        log_prior = output_dict['log_prior']
        log_posterior = output_dict['log_posterior']

        # complexity regularizer
        loss_complexity = beta * (log_posterior - log_prior)
        # total loss
        loss = loss_cls + loss_complexity
        losses = {'loss_cls': loss_cls, 'loss_complexity': loss_complexity,  # items to be backwarded
                    'LOSS_total': loss,  # items for monitoring
                    'log_posterior': beta * log_posterior,
                    'log_prior': beta * log_prior
                    }
        if 'aleatoric' in output_dict: losses.update({'aleatoric': output_dict['aleatoric']})
        if 'epistemic' in output_dict: losses.update({'epistemic': output_dict['epistemic']})
        return loss

class RPLoss(nn.Module):
    """Reciprocal Point Learning Loss."""
    def __init__(self, reduction=None, temperature=1, weight_pl=0.1, radius_init=1):
        super().__init__()
        self.temperature = temperature
        self.weight_pl = weight_pl
        

    def forward(self, head_outs, labels):
        """Forward function.

        Args:
            head_outs (torch.Tensor): outputs of the RPL head
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        dist, feature, centers, radius = head_outs['dist'], head_outs['feature'], head_outs['centers'], head_outs['radius']
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        # compute losses
        logits = F.softmax(dist, dim=1)
        loss_closed = F.cross_entropy(dist / self.temperature, labels)
        center_batch = centers[labels, :]
        _dis = (feature - center_batch).pow(2).mean(1)
        loss_r = F.mse_loss(_dis, radius[labels].cuda()) / 2
        # gather losses
        loss = loss_closed + self.weight_pl * loss_r

        return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "soft_target": SoftTargetCrossEntropy,
    "focal_ce": FocalCrossEntropy,
    "bnn_loss": BayesianNNLoss,
    "rpl_loss": RPLoss
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

def calculate_loss(cfg, preds, logits, labels, cur_epoch):
    """
    Calculates loss according to cfg.
    For pre-training, losses are defined and registered in `SSL_LOSSES`.
    Different losses can be combined by specifying in the `cfg.PRETRAIN.LOSS` by
    connecting different loss names with `+`.
    
    For supervised training, this function supports cross entropy with mixup,
    label smoothing and the plain cross entropy.
    """
    loss_in_parts = {}
    weight = None
    if cfg.PRETRAIN.ENABLE or cfg.MM_RETRIEVAL.ENABLE:
        loss = 0
        loss_parts = cfg.PRETRAIN.LOSS.split('+')
        loss_weights = cfg.PRETRAIN.LOSS_WEIGHTS
        # sum up all loss items
        for loss_idx, loss_item in enumerate(loss_parts):
            if loss_item == 'CE':
                loss_cur, weight = SSL_LOSSES.get("Loss_"+loss_item)(cfg, preds, logits, labels["supervised"], cur_epoch)
            elif loss_item == 'Super_Contrastive' or loss_item == 'PROTOTYPE':
                loss_cur, weight = SSL_LOSSES.get("Loss_"+loss_item)(cfg, preds, logits, labels, cur_epoch)
            else:
                loss_cur, weight = SSL_LOSSES.get("Loss_"+loss_item)(cfg, preds, logits, labels["self-supervised"], cur_epoch)
            if isinstance(loss_cur, dict):
                for k, v in loss_cur.items():
                    loss_in_parts[k] = v
                    if "debug" not in k:
                        loss += loss_weights[loss_idx]*loss_in_parts[k]
            else:
                loss_in_parts[loss_item] = loss_cur
                loss += loss_weights[loss_idx]*loss_in_parts[loss_item]
    else:
        # Explicitly declare reduction to mean.
        loss_fun = get_loss_func(cfg.TRAIN.LOSS_FUNC)(reduction="mean")
        
        # Compute the loss.
        if "supervised_mixup" in labels.keys():
            if isinstance(labels["supervised_mixup"], dict):
                loss = 0
                for k, v in labels["supervised_mixup"].items():
                    loss_in_parts["loss_"+k] = loss_fun(preds[k], v)
                    loss += loss_in_parts["loss_"+k]
            else:
                loss = loss_fun(preds, labels["supervised_mixup"])
        else:
            if cfg.AUGMENTATION.LABEL_SMOOTHING > 0.0:
                labels_ = label_smoothing(cfg, labels["supervised"])
            else:
                labels_ = labels["supervised"]
            if isinstance(labels_, dict):
                loss = 0
                for k, v in labels_.items():
                    loss_in_parts["loss_"+k] = loss_fun(preds[k], v)
                    loss += loss_in_parts["loss_"+k]
            else:
                loss = loss_fun(preds, labels_)

    return loss, loss_in_parts, weight

@SSL_LOSSES.register()
def Loss_MoSIXY(cfg, preds, logits, labels, cur_epoch=0): 
    b, c = preds["move_x"].shape
    pred_move_x = preds["move_x"]
    pred_move_y = preds["move_y"]
    loss_func = get_loss_func("cross_entropy")(reduction="mean")
    loss = {}
    loss["loss_move_x"] = loss_func(pred_move_x, labels["move_x"].reshape(pred_move_x.shape[0]))
    loss["loss_move_y"] = loss_func(pred_move_y, labels["move_y"].reshape(pred_move_y.shape[0]))
    return loss, None

@SSL_LOSSES.register()
def Loss_MoSIX(cfg, preds, logits, labels, cur_epoch=0): # Camera Movement Spatial Transform
    b, c = preds["move_x"].shape
    pred_move_x = preds["move_x"]
    loss_func = get_loss_func("cross_entropy")(reduction="mean")
    loss = {}
    loss["loss_move_x"] = loss_func(pred_move_x, labels["move_joint"].reshape(pred_move_x.shape[0]))
    return loss, None

@SSL_LOSSES.register()
def Loss_MoSIY(cfg, preds, logits, labels, cur_epoch=0): # Camera Movement Spatial Transform
    b, c = preds["move_y"].shape
    pred_move_y = preds["move_y"]
    loss_func = get_loss_func("cross_entropy")(reduction="mean")
    loss = {}
    loss["loss_move_y"] = loss_func(pred_move_y, labels["move_joint"].reshape(pred_move_y.shape[0]))
    return loss, None

@SSL_LOSSES.register()
def Loss_MoSIJoint(cfg, preds, logits, labels, cur_epoch=0):
    """
    Computes joint MoSI loss.
    See Ziyuan Huang et al.
    Self-supervised Motion Learning from Static Images.
    https://arxiv.org/pdf/2104.00240

    Args:
        cfg (Config): global config object. 
        preds (Tensor): predictions for the joint movement.
        logits (Tensor): the defined so that the function has the same form
            as other losses.
        labels (Dict): labels for the joint movement.
        cur_epoch (float): the current epoch number. defined for adaptively changing 
            some parameters.
    Returns:
        loss (Tensor): the calculated loss.
    """
    b, c = preds["move_joint"].shape
    pred_move_joint = preds["move_joint"]
    loss_func = get_loss_func("cross_entropy")(reduction="mean")
    loss = {}
    loss["loss_joint"] = loss_func(pred_move_joint, labels["move_joint"].reshape(pred_move_joint.shape[0]))
    return loss, None

@SSL_LOSSES.register()
def Loss_Contrastive(cfg, preds, logits, labels={}, cur_epoch=0):
    """
    Computes contrastive loss across different devices.
    Each device computes the loss locally, over the logit gathered from all devices.
    The only part that is with gradient is the part generated on the local device.
    However, it the gradient are aggregated together and averaged as in DDP, the only 
    difference is that this function requires the loss to be mutiplied by the number 
    of devices. By doing so, the gradients computed in this way is equal to the gradients
    computed together on a single device.

    Args:
        cfg (Config): global config object. 
        preds (Tensor): predictions, defined so that the function has the same form
            as other losses.
        logits (Tensor): the logits for contrastive learning.
        labels (Dict): labels for the samples, which is used here to indicate the size 
            of the samples.
        cur_epoch (float): the current epoch number. defined for adaptively changing 
            some parameters.
    Returns:
        loss (Tensor): the calculated loss.
    """
    loss = {}
    batch_size_per_gpu, samples = labels["move_joint"].shape if "move_joint" in labels.keys() else labels["contrastive"].shape
    
    # gather all the logits
    if misc.get_num_gpus(cfg) > 1:
        all_logits = du.all_gather([logits])[0]
    else:
        all_logits = logits
    batch_size = all_logits.shape[0]//samples

    # construct the logits so that the logits generated on the current device
    # replaces the all gathered logits with no gradients.
    # in this way, the gradients can be back propagated.
    logits = construct_logits_with_gradient(logits, all_logits, batch_size_per_gpu, samples)
    if cfg.PRETRAIN.DEBUG:
        loss = cont.debug_nce(cfg, logits, batch_size)
    
    # compute the loss, and multiply the number of devices 
    # for the gradients computed to be identical 
    # to the ones computed on a single device.
    if hasattr(cfg.PRETRAIN, "SHORT_NEGTIVE") and cfg.PRETRAIN.SHORT_NEGTIVE:
        loss["loss_contrastive"] = cont.nce_triple(
            cfg, logits, batch_size, samples
        )*du.get_world_size()
    else:
        loss["loss_contrastive"] = cont.nce(
            cfg, logits, batch_size, samples
        )*du.get_world_size()
    return loss, None

@SSL_LOSSES.register()
def Loss_Super_Contrastive(cfg, preds, logits, labels={}, cur_epoch=0):
    """
    Computes contrastive loss across different devices.
    Each device computes the loss locally, over the logit gathered from all devices.
    The only part that is with gradient is the part generated on the local device.
    However, it the gradient are aggregated together and averaged as in DDP, the only 
    difference is that this function requires the loss to be mutiplied by the number 
    of devices. By doing so, the gradients computed in this way is equal to the gradients
    computed together on a single device.

    Args:
        cfg (Config): global config object. 
        preds (Tensor): predictions, defined so that the function has the same form
            as other losses.
        logits (Tensor): the logits for contrastive learning.
        labels (Dict): labels for the samples, which is used here to indicate the size 
            of the samples.
        cur_epoch (float): the current epoch number. defined for adaptively changing 
            some parameters.
    Returns:
        loss (Tensor): the calculated loss.
    """
    loss = {}
    batch_size_per_gpu, samples = labels["move_joint"].shape if "move_joint" in labels.keys() else labels['self-supervised']["contrastive"].shape
    
    # gather all the logits
    if misc.get_num_gpus(cfg) > 1:
        all_logits = du.all_gather([logits])[0]
        labels['supervised'] = du.all_gather([labels['supervised']])[0]
    else:
        all_logits = logits
    batch_size = all_logits.shape[0]//samples

    # construct the logits so that the logits generated on the current device
    # replaces the all gathered logits with no gradients.
    # in this way, the gradients can be back propagated.
    logits = construct_logits_with_gradient(logits, all_logits, batch_size_per_gpu, samples)
    if cfg.PRETRAIN.DEBUG:
        loss = cont.debug_nce(cfg, logits, batch_size)
    
    # compute the loss, and multiply the number of devices 
    # for the gradients computed to be identical 
    # to the ones computed on a single device.
    loss["loss_contrastive"] = cont.super_nce(
        cfg, logits, batch_size, labels['supervised']
    )*du.get_world_size()
    return loss, None

@SSL_LOSSES.register()
def Loss_PROTOTYPE(cfg, preds, logits, labels={}, cur_epoch=0):
    """
    Computes contrastive loss across different devices.
    Each device computes the loss locally, over the logit gathered from all devices.
    The only part that is with gradient is the part generated on the local device.
    However, it the gradient are aggregated together and averaged as in DDP, the only 
    difference is that this function requires the loss to be mutiplied by the number 
    of devices. By doing so, the gradients computed in this way is equal to the gradients
    computed together on a single device.

    Args:
        cfg (Config): global config object. 
        preds (Tensor): predictions, defined so that the function has the same form
            as other losses.
        logits (Tensor): the logits for contrastive learning.
        labels (Dict): labels for the samples, which is used here to indicate the size 
            of the samples.
        cur_epoch (float): the current epoch number. defined for adaptively changing 
            some parameters.
    Returns:
        loss (Tensor): the calculated loss.
    """
    loss = {}
    batch_size_per_gpu, samples = labels["move_joint"].shape if "move_joint" in labels.keys() else labels['self-supervised']["contrastive"].shape
    
    # gather all the logits
    if misc.get_num_gpus(cfg) > 1:
        all_logits = du.all_gather([logits])[0]
        labels_supervised = du.all_gather([labels['supervised']])[0]
    else:
        all_logits = logits
    batch_size = all_logits.shape[0]//samples

    # construct the logits so that the logits generated on the current device
    # replaces the all gathered logits with no gradients.
    # in this way, the gradients can be back propagated.
    logits = construct_logits_with_gradient(logits, all_logits, batch_size_per_gpu, samples)
    if cfg.PRETRAIN.DEBUG:
        loss = cont.debug_nce(cfg, logits, batch_size)
    
    # compute the loss, and multiply the number of devices 
    # for the gradients computed to be identical 
    # to the ones computed on a single device.
    prototypes = preds[1]
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "THRE") and cfg.PRETRAIN.CONTRASTIVE.THRE:
        if cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 1:
            loss["loss_contrastive"] = cont.nce_prototype_thre_7(
            cfg, logits, batch_size, labels_supervised, prototypes
            )*du.get_world_size()
        elif cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 2:
            loss["loss_contrastive"] = cont.nce_prototype_thre_double_4(
            cfg, logits, batch_size, labels_supervised, prototypes
            )*du.get_world_size()
    elif hasattr(cfg.PRETRAIN.CONTRASTIVE, "NO_CONTRASTIVE") and cfg.PRETRAIN.CONTRASTIVE.NO_CONTRASTIVE:
        loss["loss_contrastive"] = cont.nce_prototype_5(
            cfg, logits, batch_size, labels_supervised, prototypes
        )*du.get_world_size()
    elif hasattr(cfg.PRETRAIN.CONTRASTIVE, "SCNS") and cfg.PRETRAIN.CONTRASTIVE.SCNS:
        if cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 1:
            loss["loss_contrastive"] = cont.nce_prototype_scns_2(
                cfg, logits, batch_size, labels_supervised, prototypes
            )*du.get_world_size()
        elif cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 2:
            loss["loss_contrastive"] = cont.nce_prototype_scns_double_2(
                cfg, logits, batch_size, labels_supervised, prototypes
            )*du.get_world_size()
    elif hasattr(cfg.MODEL, "NUM_UNKNOWN"):
        if cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 1:
            loss["loss_contrastive"] = cont.nce_prototype_unknown_2(
                cfg, logits, batch_size, labels_supervised, prototypes
            )*du.get_world_size()
        elif cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 2:
            loss["loss_contrastive"] = cont.nce_prototype_unknown_double_2(
                cfg, logits, batch_size, labels_supervised, prototypes
            )*du.get_world_size()
    else:
        if cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 1:
            loss["loss_contrastive"] = cont.nce_prototype_5(
                cfg, logits, batch_size, labels_supervised, prototypes
            )*du.get_world_size()
        elif cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 2:
            loss["loss_contrastive"] = cont.nce_prototype_double_2_2(
                cfg, logits, batch_size, labels_supervised, prototypes
            )*du.get_world_size()

    return loss, None


def construct_logits_with_gradient(cur_logits, all_logits, batch_size_per_gpu, samples):
    """
    Replaces the corresponding parts of the all-gathered logits with the ones generated
    by the local device with gradients.
    
    Args:
        cur_logits (Tensor): the logits generated by the model on the local device.
        all_logits (Tensor): the logits gathered from all the devices.
        batch_size_per_gpu (int): used for calculation of the index of the cur_logits 
            in the all_logits.
        samples (Tensor): for a batch size of N, there can be N*{samples} samples, as 
            the batch size is a indicator of the number of videos, and each video can
            generate multiple samples.
    Returns:
        logits (Tensor): all_logits with gradients.
    """

    num_nodes = du.get_world_size()
    rank = du.get_rank()
    num_samples_per_gpu = batch_size_per_gpu * samples
    if rank == 0:
        logits_post = all_logits[num_samples_per_gpu*(rank+1):, :]
        return torch.cat((cur_logits, logits_post), dim=0)
    elif rank == num_nodes-1:
        logits_prev = all_logits[:num_samples_per_gpu*rank, :]
        return torch.cat((logits_prev, cur_logits), dim=0)
    else:
        logits_prev = all_logits[:num_samples_per_gpu*rank, :]
        logits_post = all_logits[num_samples_per_gpu*(rank+1):, :]
        return torch.cat((logits_prev, cur_logits, logits_post), dim=0)

@SSL_LOSSES.register()
def Loss_MILNCE(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    logits_v = logits["video"]
    logits_t = logits["text"]
    batch_size_per_gpu = logits_v.shape[0]
    if misc.get_num_gpus(cfg) > 1:
        all_logits_v, all_logits_t = du.all_gather(
            [logits_v, logits_t]
        )
        logits_v = construct_logits_with_gradient(
            logits["video"], all_logits_v, batch_size_per_gpu, 1
        )
        logits_t = construct_logits_with_gradient(
            logits["text"], all_logits_t, batch_size_per_gpu, 1
        )
    batch_size = logits_v.shape[0]
    if cfg.PRETRAIN.DEBUG:
        loss = cont.debug_milnce(cfg, logits_v, logits_t, batch_size)
    if hasattr(cfg, "EXP") and hasattr(cfg.EXP, "FAKE_MILNCE") and cfg.EXP.FAKE_MILNCE:
        loss["MIL_NCE"] = cont.fake_milnce(cfg, logits_v, logits_t, batch_size) * du.get_world_size()
    else:
        loss["MIL_NCE"] = cont.mil_nce(cfg, logits_v, logits_t, batch_size) * du.get_world_size()
    return loss, None

@SSL_LOSSES.register()
def Loss_Triplet(cfg, preds, logits, labels={}, cur_epoch=0):
    margin = cfg.MM_RETRIEVAL.TRIPLET.MARGIN
    loss = {}
    logits_v = logits["video"]
    logits_t = logits["text"]
    batch_size_per_gpu = logits_v.shape[0]
    if misc.get_num_gpus(cfg) > 1:
        all_logits_v, all_logits_t = du.all_gather(
            [logits_v, logits_t]
        )
        logits_v = construct_logits_with_gradient(
            logits["video"], all_logits_v, batch_size_per_gpu, 1
        )
        logits_t = construct_logits_with_gradient(
            logits["text"], all_logits_t, batch_size_per_gpu, 1
        )
    batch_size = logits_v.shape[0]
    loss['triplet'] = cont.triplet(cfg, logits_v, logits_t, batch_size, margin)  * du.get_world_size()
    return loss, None


@SSL_LOSSES.register()
def Loss_MILNCEV2(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    if cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.SEPARATE:
        logits_v  = logits["video"][0]
        logits_v1 = logits["video"][1]
    else:
        logits_v = logits["video"]
    batch_size_per_gpu = logits_v.shape[0]
    logits_t = logits["text"]
    if misc.get_num_gpus(cfg) > 1:
        if cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.SEPARATE:
            all_logits_v, all_logits_v1, all_logits_t = du.all_gather([logits_v, logits_v1, logits_t])
            logits_v  = construct_logits_with_gradient(logits["video"][0], all_logits_v, batch_size_per_gpu, 1)
            logits_v1 = construct_logits_with_gradient(logits["video"][1], all_logits_v1, batch_size_per_gpu, 1)
        else:
            all_logits_v, all_logits_t = du.all_gather([logits_v, logits_t])
            logits_v = construct_logits_with_gradient(logits["video"], all_logits_v, batch_size_per_gpu, 1)
        logits_t = construct_logits_with_gradient(logits["text"], all_logits_t, batch_size_per_gpu, 1)
    batch_size = logits_v.shape[0]
    loss["MIL_NCE"] = cont.mil_nce(
        cfg, logits_v, logits_t, batch_size
    ) * du.get_world_size()
    loss["MIL_NCE_V2"] = cont.mil_nce_v2(
        cfg, logits_v1 if cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.SEPARATE else logits_v, logits_t, batch_size
    ) * du.get_world_size()
    return loss, None

@SSL_LOSSES.register()
def Loss_MILNCEV3(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    logits_v = logits["video"]
    batch_size_per_gpu = logits_v.shape[0]
    logits_t = logits["text"]
    if misc.get_num_gpus(cfg) > 1:
        all_logits_v, all_logits_t = du.all_gather([logits_v, logits_t])
        logits_v = construct_logits_with_gradient(logits["video"], all_logits_v, batch_size_per_gpu, 1)
        logits_t = construct_logits_with_gradient(logits["text"], all_logits_t, batch_size_per_gpu, 1)
    batch_size = logits_v.shape[0]
    # if cur_epoch < cfg.OPTIMIZER.WARMUP_EPOCHS:
    #     loss["MIL_NCE"] = cont.mil_nce(
    #         cfg, logits_v, logits_t, batch_size
    #     ) * du.get_world_size()
    # else:
    #     loss["MIL_NCE_V2"] = cont.mil_nce_v3(
    #         cfg, logits_v, logits_t, batch_size
    #     ) * du.get_world_size()
    loss["MIL_NCE_V3"] = cont.mil_nce_v3(
        cfg, logits_v, logits_t, batch_size, cur_epoch
    ) * du.get_world_size()
    return loss, None

@SSL_LOSSES.register()
def Loss_TSNCE(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    logits_v = logits["video"]
    logits_t = logits["text"]
    batch_size_per_gpu = logits_v.shape[0]
    if misc.get_num_gpus(cfg) > 1:
        all_logits_v, all_logits_t = du.all_gather(
            [logits_v, logits_t]
        )
        logits_v = construct_logits_with_gradient(
            logits["video"], all_logits_v, batch_size_per_gpu, 1
        )
        logits_t = construct_logits_with_gradient(
            logits["text"], all_logits_t, batch_size_per_gpu//2, 1
        )
    batch_size = logits_v.shape[0]
    loss["ts-nce"] = cont.ts_nce(cfg, logits_v, logits_t, batch_size) * du.get_world_size()
    return loss, None

@SSL_LOSSES.register()
def Loss_MILNCE_SEL(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    logits_v = logits["video"]
    logits_t = logits["text"]
    batch_size_per_gpu = logits_v.shape[0]
    if misc.get_num_gpus(cfg) > 1:
        all_logits_v, all_logits_t = du.all_gather(
            [logits_v, logits_t]
        )
        logits_v = construct_logits_with_gradient(
            logits["video"], all_logits_v, batch_size_per_gpu, 1
        )
        logits_t = construct_logits_with_gradient(
            logits["text"], all_logits_t, batch_size_per_gpu//2, 1
        )
    batch_size = logits_v.shape[0]
    loss["MIL_NCE_SEL"] = cont.mil_nce_sel(cfg, logits_v, logits_t, batch_size) * du.get_world_size()
    return loss, None

@SSL_LOSSES.register()
def Loss_FG_MILNCE(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    logits_v_abs = logits["video"]["abs"]
    logits_v_fg = logits["video"]["fg"]
    logits_t_abs = logits["text"]["abs"]
    logits_t_fg = logits["text"]["fg"]
    text_validity = labels["text_validity"]
    batch_size_per_gpu = logits_v_abs.shape[0]
    if misc.get_num_gpus(cfg) > 1:
        all_logits_v_abs, all_logits_v_fg, all_logits_t_abs, all_logits_t_fg, text_validity = du.all_gather(
            [logits_v_abs, logits_v_fg, logits_t_abs, logits_t_fg, text_validity]
        )
        logits_v_abs = construct_logits_with_gradient(
            logits["video"]["abs"], all_logits_v_abs, batch_size_per_gpu, 1
        )
        logits_v_fg = construct_logits_with_gradient(
            logits["video"]["fg"], all_logits_v_fg, batch_size_per_gpu, 1
        )
        logits_t_abs = construct_logits_with_gradient(
            logits["text"]["abs"], all_logits_t_abs, batch_size_per_gpu, 1
        )
        logits_t_fg = construct_logits_with_gradient(
            logits["text"]["fg"], all_logits_t_fg, batch_size_per_gpu, 1
        )
    batch_size = logits_v_abs.shape[0]
    loss["FG_MIL_NCE"] = cont.fg_mil_nce(
        cfg, logits_v_abs, logits_v_fg, logits_t_abs, logits_t_fg, text_validity, batch_size
    ) * du.get_world_size()
    return loss, None

@SSL_LOSSES.register()
def Loss_SimSiam(cfg, preds, logits, labels={}, cur_epoch=0):
    loss = {}
    z, p = logits
    samples = z.shape[0]
    if misc.get_num_gpus(cfg) > 1:
        all_z, all_p = du.all_gather(
            [z, p]
        )
    else:
        all_z, all_p = z, p
    all_z = construct_logits_with_gradient(z, all_z, samples//2, 2)
    all_p = construct_logits_with_gradient(p, all_p, samples//2, 2)

    all_z = all_z.reshape(all_z.shape[0]//2, 2, all_z.shape[1])
    all_p = all_p.reshape(all_p.shape[0]//2, 2, all_p.shape[1])

    z1, z2 = all_z[:, 0], all_z[:, 1]
    p1, p2 = all_p[:, 0], all_p[:, 1]
    
    loss["Loss_SimSiam"] = (D(p1, z2)/2 + D(p2, z1)/2) * du.get_world_size()

    # z1, z2, p1, p2 = logits
    # loss["Loss_SimSiam_Sep"] = (D(p1, z2)/2 + D(p2, z1)/2)
    return loss, None

@SSL_LOSSES.register()
def Loss_Asym(cfg, preds, logits, labels={}, cur_epoch=0):
    asymmetric = cfg.PRETRAIN.PROTOTYPE.ASYMMETRIC

    loss = {}
    if asymmetric is not None:
        logits_v = logits["video"]
        logits_t = logits["text"]
        batch_size_per_gpu = logits_v.shape[0]
        if misc.get_num_gpus(cfg) > 1:
            all_logits_v, all_logits_t = du.all_gather(
                [logits_v, logits_t]
            )
            if asymmetric == "video":
                logits_v = all_logits_v
                logits_t = construct_logits_with_gradient(
                    logits["text"], all_logits_t, batch_size_per_gpu, 1
                )
            elif asymmetric == "text":
                logits_v = construct_logits_with_gradient(
                    logits["video"], 
                    all_logits_v, 
                    batch_size_per_gpu, 1
                )
                logits_t = all_logits_t
            elif asymmetric == "video+nce":
                logits_v = construct_logits_with_gradient(
                    logits["video"], 
                    all_logits_v, 
                    batch_size_per_gpu, 1
                )
                logits_t = construct_logits_with_gradient(
                    logits["text"], all_logits_t, batch_size_per_gpu, 1
                )
            elif asymmetric == "text+nce":
                logits_v = construct_logits_with_gradient(
                    logits["video"], 
                    all_logits_v, 
                    batch_size_per_gpu, 1
                )
                logits_t = construct_logits_with_gradient(
                    logits["text"], all_logits_t, batch_size_per_gpu, 1
                )
            elif asymmetric == "video+text+nce":
                logits_v = construct_logits_with_gradient(
                    logits["video"], 
                    all_logits_v, 
                    batch_size_per_gpu, 1
                )
                logits_t = construct_logits_with_gradient(
                    logits["text"], all_logits_t, batch_size_per_gpu, 1
                )
        else:
            if asymmetric == "video":
                logits_v = logits_v.detach()
            elif asymmetric == "text":
                logits_t = logits_t.detach()
        batch_size = logits_v.shape[0]

        # NCE loss calculation
        loss["Loss_nce"] = cfg.PRETRAIN.CLUSTERING.NCE_CLUSTERING_WEIGHT[0] * \
            cont.mil_nce(cfg, logits_v, logits_t, batch_size) * du.get_world_size()

    # Cluster prediction loss
    loss["Loss_clp"], weight = clst.clustering_loss(cfg, logits)
    loss["Loss_clp"] *= cfg.PRETRAIN.CLUSTERING.NCE_CLUSTERING_WEIGHT[1]
    return loss, weight

def D(p, z):
    z = z.detach()
    z = torch.nn.functional.normalize(z, p=2, dim=-1)
    return -(p*z).sum(dim=1).mean()

@SSL_LOSSES.register()
def Loss_CE(cfg, preds, logits, labels={}, cur_epoch=0):

    labels_ce = labels
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss_ce = loss_fn(preds, labels_ce)
    

    return loss_ce, None

    
if __name__ == '__main__':
    Loss_Contrastive('a', 'd', torch.tensor(
        [[2,4,2,4,4,4], [4,2,4,2,2,2.,]]
    ).transpose(0,1), labels={"move_joint":torch.randn([2,3])})