#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Contrastive losses. """

import sys
import torch
import utils.distributed as du
import torch.nn.functional as F
import random

def nce(cfg, logits, batch_size, samples):
    """
    Computes NCE loss as in SimCLR. 
    
    See Ting Chen et al.
    A Simple Framework for Contrastive Learning of Visual Representations.
    """
    device = logits.device

    mask_ins = torch.eye(batch_size, device=device).repeat_interleave(samples, dim=1).repeat_interleave(samples, dim=0)
    pos_mask = 1-torch.eye(batch_size*samples, device=device)
    
    sim = torch.matmul(logits, logits.transpose(0,1))
    sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)

    pos = sim_mtx[(mask_ins*pos_mask)!=0].reshape(-1, samples-1)
    neg = ((1-mask_ins)*sim_mtx).sum(0).unsqueeze(1)

    N = pos.shape[1]

    loss = -((1/N) * torch.log(pos/(pos+neg)).sum())

    loss /= (batch_size*samples)

    return loss

def nce_triple(cfg, logits, batch_size, samples):
    """
    Computes NCE loss as in SimCLR. 
    
    See Ting Chen et al.
    A Simple Framework for Contrastive Learning of Visual Representations.
    """
    device = logits.device

    mask_ins = torch.eye(batch_size, device=device).repeat_interleave(samples, dim=1).repeat_interleave(samples, dim=0)
    pos_mask = 1-torch.eye(batch_size*samples, device=device)
    for i in range(batch_size):
        mask_ins[1+3*i, 2+3*i] = 0
        mask_ins[2+3*i, 1+3*i] = 0
    
    sim = torch.matmul(logits, logits.transpose(0,1))
    sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)

    pos_mask_final = mask_ins*pos_mask
    neg = ((1-mask_ins)*sim_mtx).sum(0).unsqueeze(1)

    N = pos_mask_final.sum()

    res_matrix = sim_mtx/(sim_mtx+neg)

    loss = -torch.log(res_matrix[pos_mask_final != 0]).sum()

    loss /= N

    return loss

def super_nce(cfg, features, batch_size, labels):

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)

    labels = labels.repeat_interleave(2,dim=0)

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels, labels.T).float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = features

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def super_nce_thre(cfg, features, batch_size, labels):

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)

    labels = labels.repeat_interleave(2,dim=0)

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels, labels.T).float().to(device)
    mask_ins = torch.eye(int(batch_size/2), device=device).repeat_interleave(2, dim=1).repeat_interleave(2, dim=0)
    mask_same = mask - mask_ins
    mask_same = mask_same.bool()

    contrast_count = features.shape[1]
    contrast_feature = features

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    sim = torch.matmul(anchor_feature, contrast_feature.T)
    sim[mask_same] = 1 - torch.abs(sim[mask_same] - cfg.PRETRAIN.CONTRASTIVE.THRE)
    anchor_dot_contrast = torch.div(
        sim,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_no_contrastive(cfg, features, batch_size, labels, prototypes):

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous()

    prototypes = F.normalize(prototypes, p=2, dim=1)

    if labels.shape[-1] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features, prototypes.T),
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    mask = torch.zeros_like(anchor_dot_contrast).float().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels))
    mask[idx[0],idx[1]] = 1

    # compute log_prob
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    labels_all = torch.cat((labels, labels_prototype),dim=0)

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels_all, labels_all.T).float().to(device)

    # compute logits
    sim = torch.matmul(features, features_all.T)
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        sim[:,:batch_size] = sim[:,:batch_size] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE
    anchor_dot_contrast = torch.div(
        sim,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # mask-out self-contrast cases
    mask = mask[:batch_size]
    mask[:,:batch_size] = 0

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_2(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        sim_labels[mask_labels_ma] = sim_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_2_2(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    logits_mask_labels = logits_mask_labels.float()
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        logits_mask_labels[mask_labels_ma] = logits_mask_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_2_3(cfg, features, batch_size, labels, prototypes):

    device = features.device
    batch_size_0 = features.shape[0]

    prototypes = F.normalize(prototypes, p=2, dim=1)

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    index_shuffle = [i for i in range(features.shape[0])]
    random.shuffle(index_shuffle)
    features_shuffle = features[index_shuffle]
    features_mix = cfg.PRETRAIN.CONTRASTIVE.MIXUP_ALPHA_F * features + (1 - cfg.PRETRAIN.CONTRASTIVE.MIXUP_ALPHA_F) * features_shuffle
    features_mix = F.normalize(features_mix, dim=1)
    index_select = []
    for i in range(features_mix.shape[0]):
        if random.random() < cfg.PRETRAIN.CONTRASTIVE.MIXUP_P_F:
            index_select.append(i)
    features = torch.cat((features, features_mix[index_select]), dim=0)

    labels_shuffle = labels[index_shuffle]
    labels = torch.cat((labels, labels_shuffle[index_select]), dim=0)

    batch_size = features.shape[0]

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    logits_mask_labels = logits_mask_labels.float()
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        logits_mask_labels[mask_labels_ma] = logits_mask_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[:batch_size_0]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_2_4(cfg, features, batch_size, labels, prototypes):

    device = features.device
    batch_size_0 = features.shape[0]

    prototypes = F.normalize(prototypes, p=2, dim=1)

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    index_shuffle = [i for i in range(features.shape[0])]
    random.shuffle(index_shuffle)
    features_shuffle = features[index_shuffle]
    features_mix = cfg.PRETRAIN.CONTRASTIVE.MIXUP_ALPHA_F * features + (1 - cfg.PRETRAIN.CONTRASTIVE.MIXUP_ALPHA_F) * features_shuffle
    features_mix = F.normalize(features_mix, dim=1)
    index_select = []
    for i in range(features_mix.shape[0]):
        if random.random() < cfg.PRETRAIN.CONTRASTIVE.MIXUP_P_F:
            index_select.append(i)
    features = torch.cat((features, features_mix[index_select]), dim=0)

    labels_shuffle = labels[index_shuffle]
    labels_neg = -1 * torch.ones_like(labels).cuda()
    labels_0 = torch.cat((labels, labels_neg[index_select]), dim=0)

    batch_size = features.shape[0]

    if labels_0.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels_0, labels_0.T).to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    logits_mask_labels = logits_mask_labels.float()
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        logits_mask_labels[mask_labels_ma] = logits_mask_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    labels = torch.cat((labels, labels_shuffle[index_select]), dim=0)
    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((~mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[:batch_size_0]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss


def nce_prototype_3(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        sim_labels[mask_labels_ma] = sim_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((mask_labels_ma, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_4(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        sim_labels[mask_labels_ma] = sim_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((~mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_5(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        sim_labels[mask_labels_ma] = sim_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((mask_labels_ma, mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss


def nce_prototype_double(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat_interleave(2,dim=0)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    labels_all = torch.cat((labels, labels_prototype),dim=0)

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels_all, labels_all.T).float().to(device)

    # compute logits
    sim = torch.matmul(features, features_all.T)
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        sim[:,:batch_size] = sim[:,:batch_size] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE
    anchor_dot_contrast = torch.div(
        sim,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # mask-out self-contrast cases
    mask = mask[:batch_size]
    mask[:,:batch_size] = 0

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_double_2(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat_interleave(2,dim=0)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        sim_labels[mask_labels_ma] = sim_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_double_2_2(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat_interleave(2,dim=0)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    logits_mask_labels = logits_mask_labels.float()
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        logits_mask_labels[mask_labels_ma] = logits_mask_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss



def nce_prototype_double_3(cfg, features, batch_size, labels, prototypes):

    device = features.device

    prototypes = F.normalize(prototypes, p=2, dim=1)

    batch_size = features.shape[0]

    features_all = torch.cat((features, prototypes),dim=0)

    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat_interleave(2,dim=0)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    if hasattr(cfg.PRETRAIN.CONTRASTIVE, "CONTRASTIVE_THRE"):
        sim_labels[mask_labels_ma] = sim_labels[mask_labels_ma] * cfg.PRETRAIN.CONTRASTIVE.CONTRASTIVE_THRE

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((mask_labels_ma, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss


def nce_prototype_thre(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels[mask_labels_ma] = torch.abs(sim_labels[mask_labels_ma] - cfg.PRETRAIN.CONTRASTIVE.THRE_1)

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_2(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels_positive = sim_labels[mask_labels_ma]
    sim_labels_positive = cfg.PRETRAIN.CONTRASTIVE.THRE_1 - sim_labels_positive
    sim_labels_positive[sim_labels_positive < 0] = 0
    sim_labels[mask_labels_ma] = sim_labels_positive

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes_positive = sim_prototypes[mask_prototypes]
    sim_prototypes_positive[sim_prototypes_positive > cfg.PRETRAIN.CONTRASTIVE.THRE_2] = cfg.PRETRAIN.CONTRASTIVE.THRE_2
    sim_prototypes[mask_prototypes] = sim_prototypes_positive
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_3(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels_positive = sim_labels[mask_labels_ma]
    sim_labels_positive = cfg.PRETRAIN.CONTRASTIVE.THRE_1 - sim_labels_positive
    sim_labels_positive[sim_labels_positive < 0] = 0
    sim_labels[mask_labels_ma] = sim_labels_positive

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes_positive = sim_prototypes[mask_prototypes]
    sim_prototypes_positive[sim_prototypes_positive > cfg.PRETRAIN.CONTRASTIVE.THRE_2] = cfg.PRETRAIN.CONTRASTIVE.THRE_2
    sim_prototypes[mask_prototypes] = sim_prototypes_positive
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_4(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels_positive = sim_labels[mask_labels_ma]
    sim_labels_positive = cfg.PRETRAIN.CONTRASTIVE.THRE_1 - sim_labels_positive
    sim_labels_positive[sim_labels_positive < 0] = 0
    sim_labels[mask_labels_ma] = sim_labels_positive

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes_positive = sim_prototypes[mask_prototypes]
    sim_prototypes_positive[sim_prototypes_positive > cfg.PRETRAIN.CONTRASTIVE.THRE_2] = cfg.PRETRAIN.CONTRASTIVE.THRE_2
    sim_prototypes[mask_prototypes] = sim_prototypes_positive
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((~mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_5(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels[mask_labels_ma] = torch.abs(sim_labels[mask_labels_ma] - cfg.PRETRAIN.CONTRASTIVE.THRE_1)

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((~mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_6(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels[mask_labels_ma] = torch.abs(sim_labels[mask_labels_ma] - cfg.PRETRAIN.CONTRASTIVE.THRE_1)

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((torch.zeros_like(logits_mask_labels).float().cuda(), logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_7(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels[mask_labels_ma] = 1 - torch.abs(sim_labels[mask_labels_ma] - cfg.PRETRAIN.CONTRASTIVE.THRE_1)

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((mask_labels_ma, mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss


def nce_prototype_thre_double(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat_interleave(2,dim=0)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels[mask_labels_ma] = torch.abs(sim_labels[mask_labels_ma] - cfg.PRETRAIN.CONTRASTIVE.THRE_1)

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_double_2(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels[mask_labels_ma] = torch.abs(sim_labels[mask_labels_ma] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    labels[1::2] = labels[0::2]
    mask_labels_2 = torch.eq(labels, labels.T).bool().to(device)
    mask_labels_ma_2 = mask_labels_2 * logits_mask_labels
    mask_labels_mix = torch.logical_xor(mask_labels_ma, mask_labels_ma_2)
    sim_labels[mask_labels_mix] = torch.abs(sim_labels[mask_labels_mix] - cfg.PRETRAIN.CONTRASTIVE.THRE_1)

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_double_3(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_2 = labels.clone()
    labels_2[1::2] = -1
    labels_3 = labels.clone()
    labels_3[::2] = -2
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels_2 = torch.eq(labels_2, labels_2.T).bool().to(device)
    mask_labels_3 = torch.eq(labels_3, labels_3.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels_2),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels_2 * logits_mask_labels
    logits_mask_labels_2 = mask_labels_3 * logits_mask_labels
    sim_labels[mask_labels_ma] = 1 - torch.abs(sim_labels[mask_labels_ma] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    mask_labels_ma_2 = mask_labels_ma.clone()
    for i in range(0, labels.shape[0], 2):
        if labels[i+1] == -1:
            mask_labels_ma_2[i][i+1] = 1
            logits_mask_labels_2[i][i+1] = 1
            sim_labels[i][i+1] = 1 - torch.abs(sim_labels[i][i+1] - cfg.PRETRAIN.CONTRASTIVE.THRE_1)

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels_2, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((mask_labels_ma_2, mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_thre_double_4(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_2 = labels.clone()
    labels_2[1::2] = -1
    labels_3 = labels.clone()
    labels_3[::2] = -2
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels_2 = torch.eq(labels_2, labels_2.T).bool().to(device)
    mask_labels_3 = torch.eq(labels_3, labels_3.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels_2),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels_2 * logits_mask_labels
    logits_mask_labels_2 = mask_labels_3 * logits_mask_labels
    sim_labels[mask_labels_ma] = torch.abs(sim_labels[mask_labels_ma] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    for i in range(0, labels.shape[0], 2):
        if labels[i+1] == -1:
            logits_mask_labels_2[i][i+1] = 1
            sim_labels[i][i+1] = torch.abs(sim_labels[i][i+1] - cfg.PRETRAIN.CONTRASTIVE.THRE_1)

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels_2, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels_ma).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss


def nce_prototype_scns(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    features_minus_prototype = features - prototypes[labels].squeeze()
    features_minus_prototype = F.normalize(features_minus_prototype, p=2, dim=1)
    sim_labels = cfg.PRETRAIN.CONTRASTIVE.MINUS_ALPHA * torch.matmul(features_minus_prototype, features_minus_prototype.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_scns_2(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    features_minus_prototype = features - prototypes[labels].squeeze()
    features_minus_prototype = F.normalize(features_minus_prototype, p=2, dim=1)
    sim_labels = torch.matmul(features_minus_prototype, features_minus_prototype.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels[mask_labels_ma] = cfg.PRETRAIN.CONTRASTIVE.MINUS_ALPHA * sim_labels[mask_labels_ma]

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((mask_labels_ma, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_scns_double(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat_interleave(2,dim=0)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    features_minus_prototype = features - prototypes[labels].squeeze()
    features_minus_prototype = F.normalize(features_minus_prototype, p=2, dim=1)
    sim_labels = cfg.PRETRAIN.CONTRASTIVE.MINUS_ALPHA * torch.matmul(features_minus_prototype, features_minus_prototype.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((logits_mask_labels, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_scns_double_2(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat_interleave(2,dim=0)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    features_minus_prototype = features - prototypes[labels].squeeze()
    features_minus_prototype = F.normalize(features_minus_prototype, p=2, dim=1)
    sim_labels = cfg.PRETRAIN.CONTRASTIVE.MINUS_ALPHA * torch.matmul(features_minus_prototype, features_minus_prototype.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask_labels_ma = mask_labels * logits_mask_labels
    sim_labels[mask_labels_ma] = cfg.PRETRAIN.CONTRASTIVE.MINUS_ALPHA * sim_labels[mask_labels_ma]

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((mask_labels_ma, logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)
    mean_log_prob_pos = mean_log_prob_pos[::2]

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss


def nce_prototype_unknown(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    mask_prototypes[:,-cfg.MODEL.NUM_UNKNOWN:] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    log_prob[:,-cfg.MODEL.NUM_UNKNOWN:] = log_prob[:,-cfg.MODEL.NUM_UNKNOWN:] * cfg.PRETRAIN.CONTRASTIVE.UNKNOWN_ALPHA

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def nce_prototype_unknown_2(cfg, features, batch_size, labels, prototypes):

    prototypes = F.normalize(prototypes, p=2, dim=1)

    device = features.device

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    labels_prototype = torch.arange(0,101).unsqueeze(-1)
    labels_prototype = labels_prototype.cuda()

    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')

    mask_labels = torch.eq(labels, labels.T).bool().to(device)
    sim_labels = torch.matmul(features, features.T)
    logits_mask_labels = torch.scatter(
        torch.ones_like(mask_labels),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )

    sim_prototypes = torch.matmul(features, prototypes.T)
    mask_prototypes = torch.zeros_like(sim_prototypes).bool().to(device)
    mask_prototypes_part = torch.zeros_like(sim_prototypes).bool().to(device)
    idx = torch.stack((torch.arange(batch_size).to(device),labels.squeeze()))
    mask_prototypes[idx[0],idx[1]] = 1
    mask_prototypes_part[idx[0],idx[1]] = 1
    mask_prototypes_part = ~mask_prototypes_part
    mask_prototypes[:,-cfg.MODEL.NUM_UNKNOWN:] = 1
    sim_prototypes[mask_prototypes] = 1 - torch.abs(sim_prototypes[mask_prototypes] - cfg.PRETRAIN.CONTRASTIVE.THRE_2)
    logits_mask_prototypes = torch.ones_like(sim_prototypes).float().to(device)

    sim_all = torch.cat((sim_labels, sim_prototypes), dim=1)
    logits_mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), logits_mask_prototypes), dim=1)
    mask_all = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes),dim=1)
    logits_mask_all_part = torch.cat((torch.zeros_like(mask_labels).float().cuda(), mask_prototypes_part), dim=1)

    # compute logits
    anchor_dot_contrast = torch.div(
        sim_all,
        cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask_all
    exp_logits_part = torch.exp(logits) * logits_mask_all_part
    log_prob_in = logits - torch.log(exp_logits.sum(1, keepdim=True))
    log_prob_out = logits - torch.log(exp_logits_part.sum(1, keepdim=True))
    log_prob = torch.cat((log_prob_in[:,:-cfg.MODEL.NUM_UNKNOWN], log_prob_out[:,-cfg.MODEL.NUM_UNKNOWN:]), dim=1)

    log_prob[:,-cfg.MODEL.NUM_UNKNOWN:] = log_prob[:,-cfg.MODEL.NUM_UNKNOWN:] * cfg.PRETRAIN.CONTRASTIVE.UNKNOWN_ALPHA

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_all * log_prob).sum(1) / mask_all.sum(1)

    # loss
    # loss = - (cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE / 0.07) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss


def debug_nce(cfg, logits_v, batch_size):
    device = logits_v.device
    logits_v = logits_v.detach()
    # video statistics
    sim_vv = torch.matmul(logits_v, logits_v.t()) 
    sim_vv = sim_vv * (1-torch.eye(sim_vv.shape[0], device=device))
    mask_ins = torch.eye(batch_size, device=device).repeat_interleave(2, dim=1).repeat_interleave(2, dim=0)
    pos_mask = 1-torch.eye(batch_size*2, device=device)

    # -- video intra statistics
    video_intra_maxsim = sim_vv[(pos_mask*mask_ins)!=0].max()
    video_intra_minsim = sim_vv[(pos_mask*mask_ins)!=0].min()
    video_intra_meansim = sim_vv[(pos_mask*mask_ins)!=0].mean()

    video_inter_mean_maxsim = (sim_vv * (1-mask_ins)).max(0)[0].mean()
    video_inter_mean_minsim = (sim_vv * (1-mask_ins) + mask_ins).min(0)[0].mean()

    video_inter_min = (sim_vv * (1-mask_ins) + mask_ins).min()
    video_inter_max = (sim_vv * (1-mask_ins)).max()
    video_inter_meansim = (sim_vv * (1-mask_ins)).sum() / (mask_ins==0).sum()

    sim_vv = torch.matmul(logits_v, logits_v.t()) 
    sim_mtx = torch.exp(sim_vv/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)

    # contrastive loss:
    # -log(e^p / (e^p + sum(e^n)))
    # gradient of contrastive loss for single modality:
    # dl_dtheta = (
    #   dl_dp * dp_dtheta + sum(dl_dn * dn_dtheta)
    # )

    sim_mtx_right = torch.exp((sim_vv+0.0005)/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    sim_mtx_left = torch.exp((sim_vv-0.0005)/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)

    p_right_value = -torch.log(sim_mtx_right[(pos_mask*mask_ins)!=0] / (sim_mtx_right[(pos_mask*mask_ins)!=0] + (sim_mtx*(1-mask_ins)).sum(-1)))
    p_left_value = -torch.log(sim_mtx_left[(pos_mask*mask_ins)!=0] / (sim_mtx_left[(pos_mask*mask_ins)!=0] + (sim_mtx*(1-mask_ins)).sum(-1)))
    dl_dp = (p_right_value - p_left_value) / 0.001

    dl_dp_max = dl_dp.max()
    dl_dp_min = dl_dp.min()
    dl_dp_mean = dl_dp.mean()

    n_right_value = -torch.log(sim_mtx[(pos_mask*mask_ins)!=0] / (sim_mtx[(pos_mask*mask_ins)!=0] + (sim_mtx_right*(1-mask_ins) + (sim_mtx * (1-mask_ins)).sum(-1,keepdim=True) - sim_mtx*(1-mask_ins)) * (1-mask_ins)))
    n_left_value = -torch.log(sim_mtx[(pos_mask*mask_ins)!=0] / (sim_mtx[(pos_mask*mask_ins)!=0] + (sim_mtx_left*(1-mask_ins) + (sim_mtx * (1-mask_ins)).sum(-1,keepdim=True) - sim_mtx*(1-mask_ins)) * (1-mask_ins)))
    dl_dn = (n_right_value - n_left_value) / 0.001
    dl_dn_instance = dl_dn.sum(-1)

    dl_dn_matrix_max = dl_dn.max()
    dl_dn_matrix_min = (dl_dn+mask_ins).min()
    dl_dn_matrix_mean = dl_dn.sum() / (1-mask_ins).sum()
    dl_dn_instance_mean = dl_dn_instance.mean()
    dl_dn_instance_max = dl_dn_instance.max()
    dl_dn_instance_min = dl_dn_instance.min()

    dldp_dldn_ratio = - dl_dp / dl_dn_instance
    dldp_dldn_ratio_max = dldp_dldn_ratio.max()
    dldp_dldn_ratio_min = dldp_dldn_ratio.min()
    dldp_dldn_ratio_mean = dldp_dldn_ratio.mean()

    C_pos = (sim_mtx*(1-mask_ins)).sum(-1).mean()
    C_neg = sim_mtx[(pos_mask*mask_ins)!=0].mean()
    D_neg = ((sim_mtx * (1-mask_ins)) * (batch_size*2-3)).sum() / (1-mask_ins).sum()


    estimated_loss = -torch.log(C_neg / (C_neg + C_pos))

    return {
        # video x 8
        "debug_video_intra_maxsim": video_intra_maxsim,
        "debug_video_intra_minsim": video_intra_minsim,
        "debug_video_intra_meansim": video_intra_meansim,

        "debug_video_inter_mean_maxsim": video_inter_mean_maxsim,
        "debug_video_inter_mean_minsim": video_inter_mean_minsim,

        "debug_video_inter_min": video_inter_min,
        "debug_video_inter_max": video_inter_max,
        "debug_video_inter_meansim": video_inter_meansim,

        "debug_C_pos": C_pos,
        "debug_C_neg": C_neg,
        "debug_D_neg": D_neg,
        
        "debug_estimated_loss": estimated_loss,
        # gradient
        "debug_dl_dp_max":              dl_dp_max,
        "debug_dl_dp_min":              dl_dp_min,
        "debug_dl_dp_mean":             dl_dp_mean,
        "debug_dl_dn_matrix_max":       dl_dn_matrix_max,
        "debug_dl_dn_matrix_min":       dl_dn_matrix_min,
        "debug_dl_dn_matrix_mean":      dl_dn_matrix_mean,
        "debug_dl_dn_instance_mean":    dl_dn_instance_mean,
        "debug_dl_dn_instance_max":     dl_dn_instance_max,
        "debug_dl_dn_instance_min":     dl_dn_instance_min,
        "debug_dldp_dldn_ratio_max":    dldp_dldn_ratio_max,
        "debug_dldp_dldn_ratio_min":    dldp_dldn_ratio_min,
        "debug_dldp_dldn_ratio_mean":   dldp_dldn_ratio_mean,
    }


def contrastive_augmentation_discrimination(cfg, logits, batch_size, samples):
    device = logits.device

    mask_aug = torch.eye(samples, device=device).repeat(batch_size, batch_size)
    pos_mask = 1-torch.eye(batch_size*samples, device=device)
    
    sim = torch.matmul(logits, logits.transpose(0,1))
    sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)

    if cfg.PRETRAIN.CONTRASTIVE.AUG_MIL:
        pos = sim_mtx[(mask_aug*pos_mask)!=0].reshape(-1, batch_size-1).sum(1, keepdim=True)
    else:
        pos = sim_mtx[(mask_aug*pos_mask)!=0].reshape(-1, batch_size-1)
    neg = ((1-mask_aug)*sim_mtx).sum(0).unsqueeze(1)

    N = pos.shape[1]
    return -((1/N) * torch.log(pos/neg).sum()) / (batch_size*samples)   

def mil_nce(cfg, logits_v, logits_t, batch_size):
    device = logits_v.device
    logits_t = logits_t.reshape(-1, logits_t.shape[-1])
    sim = torch.matmul(logits_v, logits_t.transpose(0,1))

    q = torch.eye(batch_size, device=device)
    if hasattr(cfg.EXP, "PARAM_C") and cfg.EXP.PARAM_C > 0.0:
        
        sim = (1-q.unsqueeze(-1)) * cfg.EXP.PARAM_C + sim.reshape(batch_size, batch_size, cfg.TEXT.NUM_SENTENCES)
        sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE).sum(-1)
    else:
        if hasattr(cfg.EXP, "POSITIVE_MULTIPLIER") and cfg.EXP.POSITIVE_MULTIPLIER > 0.0:
            sim = sim * (1-q) + sim * q * cfg.EXP.POSITIVE_MULTIPLIER
        sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE).reshape(
            batch_size, batch_size, cfg.TEXT.NUM_SENTENCES
        ).sum(-1)

    p = sim_mtx / (
        sim_mtx.sum(-1, keepdim=True) + sim_mtx.t().sum(-1,keepdim=True) - (sim_mtx*torch.eye(batch_size, device=device)).sum(-1,keepdim=True)
    )

    if cfg.PRETRAIN.CONTRASTIVE.DROPOUT > 0:
        loss = -torch.log( p[q!=0] ) # calculate q*log(p)
        v,i = loss.topk(batch_size//5)
        rand_mtx = (torch.rand(batch_size//5, device=device)>0.5)*0.9999+0.0001
        loss[i] = v * rand_mtx
        return loss.sum() / batch_size
    elif cfg.PRETRAIN.CONTRASTIVE.SYMMETRIC:
        loss_nce = -torch.log( p[q!=0] ) # calculate q*log(p)
        logq = (1-q)*cfg.PRETRAIN.CONTRASTIVE.SYMMETRIC_A
        loss_rnce = (-p*logq).sum(-1)
        loss = loss_nce * cfg.PRETRAIN.CONTRASTIVE.SYMMETRIC_ALPHA + loss_rnce * cfg.PRETRAIN.CONTRASTIVE.SYMMETRIC_BETA
        return loss.sum() / batch_size
    elif cfg.PRETRAIN.CONTRASTIVE.LABEL_SMOOTH_FACTOR > 0.0:
        q = q * (1-cfg.PRETRAIN.CONTRASTIVE.LABEL_SMOOTH_FACTOR) + cfg.PRETRAIN.CONTRASTIVE.LABEL_SMOOTH_FACTOR/batch_size
        loss = -torch.log(p) * q
        return loss.sum()/batch_size
    else:
        loss = -torch.log( p[q!=0] ) # calculate q*log(p)
        return loss.sum() / batch_size

def triplet(cfg, logits_v, logits_t, batch_size, margin):
    device = logits_v.device
    logits_t = logits_t.reshape(-1, logits_t.shape[-1])
    sim = torch.matmul(logits_v, logits_t.transpose(0,1))
    all_t2v=(-sim.diag().unsqueeze(1) + sim.transpose(0, 1))[(1-torch.eye(batch_size, device=device)).bool()].reshape(batch_size, -1)
    all_v2t=(-sim.diag().unsqueeze(1) + sim)[(1-torch.eye(batch_size, device=device)).bool()].reshape(batch_size, -1)
    loss = torch.clamp(all_t2v.max(dim=1)[0] + margin, min=0) + torch.clamp(all_v2t.max(dim=1)[0] + margin, min=0)
    loss = loss.sum() / batch_size
    return loss



def debug_milnce(cfg, logits_v, logits_t, batch_size):
    device = logits_v.device
    logits_v = logits_v.detach()
    logits_t = logits_t.detach()
    # video statistics
    sim_vv = torch.matmul(logits_v, logits_v.t()) 
    sim_vv = sim_vv * (1-torch.eye(sim_vv.shape[0], device=device))
    video_overall_meansim = sim_vv.sum() / (1-torch.eye(sim_vv.shape[0], device=device)).sum()
    video_overall_maxsim = sim_vv.max()
    video_overall_minsim = (sim_vv + torch.eye(sim_vv.shape[0], device=device)).min()
    video_mean_minsim = (sim_vv + torch.eye(sim_vv.shape[0], device=device)).min(1)[0].mean()
    video_mean_maxsim = sim_vv.max(1)[0].mean()

    # text statistics
    logits_t = logits_t.reshape(-1, logits_t.shape[-1])
    sim_tt = torch.matmul(logits_t, logits_t.t())
    sim_tt = sim_tt * (1-torch.eye(sim_tt.shape[0], device=device))
    sim_tt_reshaped = sim_tt.reshape(
        batch_size, cfg.TEXT.NUM_SENTENCES, batch_size, cfg.TEXT.NUM_SENTENCES
    ).permute(0,2,1,3).reshape(batch_size, batch_size, -1)

    # -- text intra statistics
    if cfg.TEXT.NUM_SENTENCES > 1:
        sim_tt_intra = sim_tt_reshaped[torch.eye(sim_tt_reshaped.shape[0], device=device).long()==1]
        text_intra_mean_meansim = sim_tt_intra.sum() / (batch_size * cfg.TEXT.NUM_SENTENCES * (cfg.TEXT.NUM_SENTENCES-1))
        text_intra_mean_maxsim = sim_tt_intra.max(-1)[0].mean()
        text_intra_mean_minsim = (sim_tt_intra + torch.eye(cfg.TEXT.NUM_SENTENCES, device=device).reshape(-1).unsqueeze(0)).min(-1)[0].mean()
    else:
        text_intra_mean_meansim = torch.tensor(0.)
        text_intra_mean_maxsim  = torch.tensor(0.)
        text_intra_mean_minsim  = torch.tensor(0.)

    # -- text inter statistics
    sim_tt_inter = (sim_tt_reshaped * (1-torch.eye(sim_tt_reshaped.shape[0], device=device).unsqueeze(-1))).mean(-1)
    text_inter_mean_meansim = sim_tt_inter.sum() / (1-torch.eye(sim_tt_reshaped.shape[0], device=device)).sum()
    text_inter_mean_maxsim = sim_tt_inter.max(-1)[0].mean()
    text_inter_mean_minsim = (sim_tt_inter + torch.eye(sim_tt_reshaped.shape[0], device=device)).min(-1)[0].mean()

    # vt statistics
    sim_vt = torch.matmul(logits_v, logits_t.t()).reshape(
        batch_size, batch_size, cfg.TEXT.NUM_SENTENCES
    )
    sim_vt_mean = sim_vt.mean(-1)

    # -- vt intra statistics (same as tv intra statistics)
    sim_vt_mean_intra = sim_vt_mean[torch.eye(sim_vt_mean.shape[0], device=device).long()==1]
    vt_mean_intra_meansim = sim_vt_mean_intra.mean()
    vt_mean_intra_maxsim = sim_vt_mean_intra.max()
    vt_mean_intra_minsim = sim_vt_mean_intra.min()

    # -- vt inter statistics
    sim_vt_mean_inter = sim_vt_mean * (1-torch.eye(sim_vt_mean.shape[0], device=device))
    vt_mean_inter_meansim = sim_vt_mean_inter.sum() / (1-torch.eye(sim_vt_mean.shape[0], device=device)).sum()
    vt_mean_inter_mean_maxsim = sim_vt_mean_inter.max(-1)[0].mean() # average of max video-to-negative-text similarity 
    vt_mean_inter_mean_minsim = (sim_vt_mean_inter + torch.eye(sim_vt_mean.shape[0], device=device)*5).min(-1)[0].mean()

    # -- tv inter statistics
    sim_tv_mean_inter = sim_vt_mean_inter.t()
    tv_mean_inter_meansim = sim_tv_mean_inter.sum() / (1-torch.eye(sim_tv_mean_inter.shape[0], device=device)).sum() # should be same as vt
    tv_mean_inter_mean_maxsim = sim_tv_mean_inter.max(-1)[0].mean() # average of max video-to-negative-text similarity 
    tv_mean_inter_mean_minsim = (sim_tv_mean_inter + torch.eye(sim_tv_mean_inter.shape[0], device=device)*5).min(-1)[0].mean()

    if hasattr(cfg.EXP, "PARAM_C") and cfg.EXP.PARAM_C != 0.0:
        sim_after = (1-torch.eye(sim_vt.shape[0], device=device).unsqueeze(-1)) * cfg.EXP.PARAM_C + sim_vt.reshape(batch_size, batch_size, cfg.TEXT.NUM_SENTENCES)
        sim_mtx_after = torch.exp(sim_after/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE).sum(-1)

        q = torch.eye(batch_size, device=device)
        C_pos_after = sim_mtx_after.sum(-1, keepdim=True) + sim_mtx_after.t().sum(-1,keepdim=True) - 2*(sim_mtx_after*q).sum(-1,keepdim=True)
        C_pos_after_mean = C_pos_after.mean()
        C_neg_after_mean = (sim_mtx_after*q).sum(-1).mean()
        D_neg_after_mean = ((sim_mtx_after*(1-q)) * (batch_size*2-3)).sum() / (batch_size-1) / batch_size
        estimated_loss_after = -torch.log(C_neg_after_mean / (C_neg_after_mean + C_pos_after_mean))
    else:
        C_pos_after_mean = torch.tensor(0.)
        C_neg_after_mean = torch.tensor(0.)
        D_neg_after_mean = torch.tensor(0.)
        estimated_loss_after = torch.tensor(0.)

    sim_mtx_before = torch.exp(sim_vt/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE).reshape(
        batch_size, batch_size, cfg.TEXT.NUM_SENTENCES
    ).sum(-1)

    q = torch.eye(batch_size, device=device)
    C_pos_before = sim_mtx_before.sum(-1, keepdim=True) + sim_mtx_before.t().sum(-1,keepdim=True) - 2*(sim_mtx_before*q).sum(-1,keepdim=True)
    C_pos_before_mean = C_pos_before.mean()
    C_neg_before_mean = (sim_mtx_before*q).sum(-1).mean()
    D_neg_before_mean = ((sim_mtx_before*(1-q)) * (batch_size*2-3)).sum() / (batch_size-1) / batch_size
    estimated_loss_before = -torch.log(C_neg_before_mean / (C_neg_before_mean + C_pos_before_mean))

    # gradient calculation
    if cfg.TEXT.NUM_SENTENCES == 1:
        # only calculate gradient when N=1
        sim_vt = torch.matmul(logits_v, logits_t.t()).reshape(
            batch_size, batch_size, cfg.TEXT.NUM_SENTENCES
        ).sum(-1)
        if hasattr(cfg.EXP, "PARAM_C"):
            sim_vt = ((1-q) * cfg.EXP.PARAM_C + sim_vt)

        sim_mtx = torch.exp(sim_vt/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
        sim_mtx_right = torch.exp((sim_vt+0.0005)/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
        sim_mtx_left = torch.exp((sim_vt-0.0005)/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)

        p_right_value = -torch.log(
            sim_mtx_right[q!=0] / (         # positive samples divided by
                (sim_mtx*(1-q)).sum(0) +    # negative video samples for texts and
                (sim_mtx*(1-q)).sum(1) +    # negative text samples for videos and 
                sim_mtx_right[q!=0]         # positive samples
            )
        )
        p_left_value = -torch.log(
            sim_mtx_left[q!=0] / (          # positive samples divided by
                (sim_mtx*(1-q)).sum(0) +    # negative video samples for texts and 
                (sim_mtx*(1-q)).sum(1) +    # negative text samples for videos and 
                sim_mtx_left[q!=0]          # positive samples
            )
        )
        dl_dp = (p_right_value - p_left_value) / 0.001

        dl_dp_max = dl_dp.max()
        dl_dp_min = dl_dp.min()
        dl_dp_mean = dl_dp.mean()
        
        # negative video samples for text: nt = xv' * xt
        n_right_value_t = -torch.log(
            sim_mtx[q!=0] / (
                sim_mtx[q!=0] + 
                (sim_mtx_right*(1-q) + sim_mtx*(1-q).sum(0) - sim_mtx*(1-q))*(1-q) + 
                (sim_mtx*(1-q)).sum(1)
            )
        )
        n_left_value_t = -torch.log(
            sim_mtx[q!=0] / (
                sim_mtx[q!=0] + 
                (sim_mtx_left*(1-q) + sim_mtx*(1-q).sum(0) - sim_mtx*(1-q))*(1-q) + 
                (sim_mtx*(1-q)).sum(1)
            )
        )
        dl_dnt = (n_right_value_t - n_left_value_t) / 0.001
        dl_dnt_instance = dl_dnt.sum(0)
        dl_dnt_matrix_max = dl_dnt.max()
        dl_dnt_matrix_min = (dl_dnt+q*100).min()
        dl_dnt_matrix_mean = dl_dnt.sum() / (1-q).sum()
        dl_dnt_instance_mean = dl_dnt_instance.mean()
        dl_dnt_instance_max = dl_dnt_instance.max()
        dl_dnt_instance_min = dl_dnt_instance.min()

        # negative text samples for video: nv = xv * xt'
        n_right_value_v = -torch.log(
            sim_mtx[q!=0] / (
                sim_mtx[q!=0] + 
                (sim_mtx_right*(1-q) + sim_mtx*(1-q).sum(1) - sim_mtx*(1-q))*(1-q) + 
                (sim_mtx*(1-q)).sum(0)
            )
        )
        n_left_value_v = -torch.log(
            sim_mtx[q!=0] / (
                sim_mtx[q!=0] + 
                (sim_mtx_left*(1-q) + sim_mtx*(1-q).sum(1) - sim_mtx*(1-q))*(1-q) + 
                (sim_mtx*(1-q)).sum(0)
            )
        )
        dl_dnv = (n_right_value_v - n_left_value_v) / 0.001
        dl_dnv_instance = dl_dnv.sum(1)
        dl_dnv_matrix_max = dl_dnv.max()
        dl_dnv_matrix_min = (dl_dnv+q*100).min()
        dl_dnv_matrix_mean = dl_dnv.sum() / (1-q).sum()
        dl_dnv_instance_mean = dl_dnv_instance.mean()
        dl_dnv_instance_max = dl_dnv_instance.max()
        dl_dnv_instance_min = dl_dnv_instance.min()

        dldp_dldnt_ratio = - dl_dp / dl_dnt_instance
        dldp_dldnv_ratio = - dl_dp / dl_dnv_instance

        dldp_dldnt_ratio_max = dldp_dldnt_ratio.max()
        dldp_dldnt_ratio_min = dldp_dldnt_ratio.min()
        dldp_dldnt_ratio_mean = dldp_dldnt_ratio.mean()
        dldp_dldnv_ratio_max = dldp_dldnv_ratio.max()
        dldp_dldnv_ratio_min = dldp_dldnv_ratio.min()
        dldp_dldnv_ratio_mean = dldp_dldnv_ratio.mean()

    if cfg.TEXT.NUM_SENTENCES > 1:
        # --- max
        sim_vt_max = sim_vt.max(-1)[0]

        sim_vt_max_intra = sim_vt_max[torch.eye(sim_vt_max.shape[0], device=device).long()==1]
        vt_max_intra_meansim = sim_vt_max_intra.mean()
        vt_max_intra_maxsim = sim_vt_max_intra.max()
        vt_max_intra_minsim = sim_vt_max_intra.min()

        sim_vt_max_inter = sim_vt_max * (1-torch.eye(sim_vt_max.shape[0], device=device))
        vt_max_inter_meansim = sim_vt_max_inter.sum() / (1-torch.eye(sim_vt_max.shape[0], device=device)).sum()
        vt_max_inter_mean_maxsim = sim_vt_max_inter.max(-1)[0].mean()
        vt_max_inter_mean_minsim = (sim_vt_max_inter + torch.eye(sim_vt_max.shape[0], device=device)*5).min(-1)[0].mean()

        sim_tv_max_inter = sim_vt_max_inter.t()
        tv_max_inter_meansim = sim_tv_max_inter.sum() / (1-torch.eye(sim_tv_max_inter.shape[0], device=device)).sum()
        tv_max_inter_mean_maxsim = sim_tv_max_inter.max(-1)[0].mean()
        tv_max_inter_mean_minsim = (sim_tv_max_inter + torch.eye(sim_vt_max.shape[0], device=device)*5).min(-1)[0].mean()

        # --- min
        sim_vt_min = sim_vt.min(-1)[0]

        sim_vt_min_intra = sim_vt_min[torch.eye(sim_vt_min.shape[0], device=device).long()==1]
        vt_min_intra_meansim = sim_vt_min_intra.mean()
        vt_min_intra_maxsim = sim_vt_min_intra.max()
        vt_min_intra_minsim = sim_vt_min_intra.min()

        sim_vt_min_inter = sim_vt_min * (1-torch.eye(sim_vt_min.shape[0], device=device))
        vt_min_inter_meansim = sim_vt_min_inter.sum() / (1-torch.eye(sim_vt_min.shape[0], device=device)).sum()
        vt_min_inter_mean_maxsim = sim_vt_min_inter.max(-1)[0].mean()
        vt_min_inter_mean_minsim = (sim_vt_min_inter + torch.eye(sim_vt_min.shape[0], device=device)*5).min(-1)[0].mean()

        sim_tv_min_inter = sim_vt_min_inter.t()
        tv_min_inter_meansim = sim_tv_min_inter.sum() / (1-torch.eye(sim_tv_min_inter.shape[0], device=device)).sum()
        tv_min_inter_mean_maxsim = sim_tv_min_inter.max(-1)[0].mean()
        tv_min_inter_mean_minsim = (sim_tv_min_inter + torch.eye(sim_vt_min.shape[0], device=device)*5).min(-1)[0].mean()
    else:
        vt_max_intra_meansim        = torch.tensor(0.)
        vt_max_intra_maxsim         = torch.tensor(0.)
        vt_max_intra_minsim         = torch.tensor(0.)
        vt_max_inter_meansim        = torch.tensor(0.)
        vt_max_inter_mean_maxsim    = torch.tensor(0.)
        vt_max_inter_mean_minsim    = torch.tensor(0.)
        tv_max_inter_meansim        = torch.tensor(0.)
        tv_max_inter_mean_maxsim    = torch.tensor(0.)
        tv_max_inter_mean_minsim    = torch.tensor(0.)

        vt_min_intra_meansim        = torch.tensor(0.)
        vt_min_intra_maxsim         = torch.tensor(0.)
        vt_min_intra_minsim         = torch.tensor(0.)
        vt_min_inter_meansim        = torch.tensor(0.)
        vt_min_inter_mean_maxsim    = torch.tensor(0.)
        vt_min_inter_mean_minsim    = torch.tensor(0.)
        tv_min_inter_meansim        = torch.tensor(0.)
        tv_min_inter_mean_maxsim    = torch.tensor(0.)
        tv_min_inter_mean_minsim    = torch.tensor(0.)

    return {
        # video x 5
        "debug_video_overall_meansim":          video_overall_meansim,
        "debug_video_overall_maxsim":           video_overall_maxsim,
        "debug_video_overall_minsim":           video_overall_minsim,
        "debug_video_mean_minsim":              video_mean_minsim,
        "debug_video_mean_maxsim":              video_mean_maxsim,
        # text x 6
        "debug_text_intra_mean_meansim":        text_intra_mean_meansim,
        "debug_text_intra_mean_maxsim":         text_intra_mean_maxsim,
        "debug_text_intra_mean_minsim":         text_intra_mean_minsim,
        "debug_text_inter_mean_meansim":        text_inter_mean_meansim,
        "debug_text_inter_mean_maxsim":         text_inter_mean_maxsim,
        "debug_text_inter_mean_minsim":         text_inter_mean_minsim,
        # vt_mean
        "debug_vt_mean_intra_meansim":          vt_mean_intra_meansim,
        "debug_vt_mean_intra_maxsim":           vt_mean_intra_maxsim,
        "debug_vt_mean_intra_minsim":           vt_mean_intra_minsim,
        "debug_vt_mean_inter_meansim":          vt_mean_inter_meansim,
        "debug_vt_mean_inter_mean_maxsim":      vt_mean_inter_mean_maxsim,
        "debug_vt_mean_inter_mean_minsim":      vt_mean_inter_mean_minsim,
        "debug_tv_mean_inter_meansim":          tv_mean_inter_meansim,
        "debug_tv_mean_inter_mean_maxsim":      tv_mean_inter_mean_maxsim,
        "debug_tv_mean_inter_mean_minsim":      tv_mean_inter_mean_minsim,
        # vt_max
        "debug_vt_max_intra_meansim":           vt_max_intra_meansim,
        "debug_vt_max_intra_maxsim":            vt_max_intra_maxsim,
        "debug_vt_max_intra_minsim":            vt_max_intra_minsim,
        "debug_vt_max_inter_meansim":           vt_max_inter_meansim,
        "debug_vt_max_inter_mean_maxsim":       vt_max_inter_mean_maxsim,
        "debug_vt_max_inter_mean_minsim":       vt_max_inter_mean_minsim,
        "debug_tv_max_inter_meansim":           tv_max_inter_meansim,
        "debug_tv_max_inter_mean_maxsim":       tv_max_inter_mean_maxsim,
        "debug_tv_max_inter_mean_minsim":       tv_max_inter_mean_minsim,
        # vt_min
        "debug_vt_min_intra_meansim":           vt_min_intra_meansim,
        "debug_vt_min_intra_maxsim":            vt_min_intra_maxsim,
        "debug_vt_min_intra_minsim":            vt_min_intra_minsim,
        "debug_vt_min_inter_meansim":           vt_min_inter_meansim,
        "debug_vt_min_inter_mean_maxsim":       vt_min_inter_mean_maxsim,
        "debug_vt_min_inter_mean_minsim":       vt_min_inter_mean_minsim,
        "debug_tv_min_inter_meansim":           tv_min_inter_meansim,
        "debug_tv_min_inter_mean_maxsim":       tv_min_inter_mean_maxsim,
        "debug_tv_min_inter_mean_minsim":       tv_min_inter_mean_minsim,
        # C
        "debug_C_pos_after_mean":               C_pos_after_mean,
        "debug_C_neg_after_mean":               C_neg_after_mean,
        "debug_D_neg_after_mean":               D_neg_after_mean,
        "debug_C_pos_before_mean":              C_pos_before_mean,
        "debug_C_neg_before_mean":              C_neg_before_mean,
        "debug_D_neg_before_mean":              D_neg_before_mean,
        # estimated loss
        "debug_estimated_loss_before":          estimated_loss_before,
        "debug_estimated_loss_after":           estimated_loss_after,
        # gradients
        "debug_dl_dp_max":                      dl_dp_max,
        "debug_dl_dp_min":                      dl_dp_min,
        "debug_dl_dp_mean":                     dl_dp_mean,

        "debug_dl_dnt_matrix_max":              dl_dnt_matrix_max,
        "debug_dl_dnt_matrix_min":              dl_dnt_matrix_min,
        "debug_dl_dnt_matrix_mean":             dl_dnt_matrix_mean,
        "debug_dl_dnt_instance_mean":           dl_dnt_instance_mean,
        "debug_dl_dnt_instance_max":            dl_dnt_instance_max,
        "debug_dl_dnt_instance_min":            dl_dnt_instance_min,

        "debug_dl_dnv_matrix_max":              dl_dnv_matrix_max,
        "debug_dl_dnv_matrix_min":              dl_dnv_matrix_min,
        "debug_dl_dnv_matrix_mean":             dl_dnv_matrix_mean,
        "debug_dl_dnv_instance_mean":           dl_dnv_instance_mean,
        "debug_dl_dnv_instance_max":            dl_dnv_instance_max,
        "debug_dl_dnv_instance_min":            dl_dnv_instance_min,

        "debug_dldp_dldnt_ratio_max":           dldp_dldnt_ratio_max,
        "debug_dldp_dldnt_ratio_min":           dldp_dldnt_ratio_min,
        "debug_dldp_dldnt_ratio_mean":          dldp_dldnt_ratio_mean,
        "debug_dldp_dldnv_ratio_max":           dldp_dldnv_ratio_max,
        "debug_dldp_dldnv_ratio_min":           dldp_dldnv_ratio_min,
        "debug_dldp_dldnv_ratio_mean":          dldp_dldnv_ratio_mean,
    }

def fake_milnce(cfg, logits_v, logits_t, batch_size):
    device = logits_v.device
    logits_t = logits_t.reshape(-1, logits_t.shape[-1])
    sim = torch.matmul(logits_v, logits_t.transpose(0,1))

    sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE).reshape(
        batch_size, batch_size, cfg.TEXT.NUM_SENTENCES
    )

    q = torch.eye(batch_size, device=device)
    if cfg.EXP.FAKE_MILNCE_REV:
        p = sim_mtx.sum(-1) / sim_mtx[:, :, max(1, cfg.TEXT.NUM_SENTENCES // 2)].sum(-1,keepdim=True)
    else:
        p = sim_mtx[:, :, max(1, cfg.TEXT.NUM_SENTENCES // 2)] / sim_mtx.sum(-1).sum(-1,keepdim=True)

    if cfg.PRETRAIN.CONTRASTIVE.DROPOUT > 0:
        loss = -torch.log( p[q!=0] ) # calculate q*log(p)
        v,i = loss.topk(batch_size//5)
        rand_mtx = (torch.rand(batch_size//5, device=device)>0.5)*0.9999+0.0001
        loss[i] = v * rand_mtx
        return loss.sum() / batch_size
    elif cfg.PRETRAIN.CONTRASTIVE.SYMMETRIC:
        loss_nce = -torch.log( p[q!=0] ) # calculate q*log(p)
        logq = (1-q)*cfg.PRETRAIN.CONTRASTIVE.SYMMETRIC_A
        loss_rnce = (-p*logq).sum(-1)
        loss = loss_nce * cfg.PRETRAIN.CONTRASTIVE.SYMMETRIC_ALPHA + loss_rnce * cfg.PRETRAIN.CONTRASTIVE.SYMMETRIC_BETA
        return loss.sum() / batch_size
    elif cfg.PRETRAIN.CONTRASTIVE.LABEL_SMOOTH_FACTOR > 0.0:
        q = q * (1-cfg.PRETRAIN.CONTRASTIVE.LABEL_SMOOTH_FACTOR) + cfg.PRETRAIN.CONTRASTIVE.LABEL_SMOOTH_FACTOR/batch_size
        loss = -torch.log(p) * q
        return loss.sum()/batch_size
    else:
        loss = -torch.log( p[q!=0] ) # calculate q*log(p)
        return loss.sum() / batch_size

def mil_nce_v2(cfg, logits_v, logits_t, batch_size):
    device = logits_v.device
    logits_t_detach = logits_t.detach()

    if cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.MODE == 'avg':
        logits_t_detach = logits_t_detach.mean(1)
    elif cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.MODE == 'max':
        logits_t_detach = logits_t_detach.max(1)[0]
    
    logits_t_detach = torch.nn.functional.normalize(logits_t_detach, p=2, dim=-1)

    sim_text = torch.matmul(logits_t_detach, logits_t_detach.transpose(0,1))

    if cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.FUNC == 'linear':
        w = (sim_text + 1)/2
    elif cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.FUNC == 'sigmoid':
        w = (sim_text/0.5).sigmoid()
    elif cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.FUNC == 'clamp':
        w = sim_text.clamp(0,1)
    elif cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.FUNC == 'softmax':
        w = (torch.masked_fill(sim_text/0.1, torch.eye(batch_size, device=device).bool(), float('-inf'))).softmax(1)

        
    sim_video = torch.matmul(logits_v, logits_v.transpose(0,1))
    sim_video_mtx = torch.exp(sim_video/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)
    mask = 1-torch.eye(batch_size, device=device)
    if cfg.PRETRAIN.CONTRASTIVE.MILNCEV2.SOFTMIL:
        pos = (sim_video_mtx*w*mask).sum(1)
        neg = (sim_video_mtx*(1-w)*mask).sum(1)
        loss = pos/(pos+neg)
    else:
        pos = (sim_video_mtx*w*mask)
        neg = (sim_video_mtx*(1-w)*mask).sum(1).unsqueeze(1)
        loss = (pos/(pos+neg) + torch.eye(batch_size, device=device))


    return -(torch.log(loss).sum()) / batch_size

def mil_nce_v3(cfg, logits_v, logits_t, batch_size, cur_epoch):
    # default with one
    device = logits_v.device

    sim = torch.matmul(logits_v, logits_t.reshape(-1, logits_t.shape[-1]).transpose(0,1))
    sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE).reshape(
        batch_size, batch_size, cfg.TEXT.NUM_SENTENCES
    ).sum(-1)

    q = torch.eye(batch_size, device=device)
    sim_vt = sim.reshape(batch_size, batch_size, -1)
    _, vt_ind = sim_vt[q!=0].max(-1)
    logits_t_max = logits_t[torch.linspace(0,batch_size-1,batch_size, dtype=torch.long, device=device), vt_ind].detach()
    sim_tt_max = torch.matmul(logits_t_max, logits_t_max.transpose(0,1)).clamp(0,1)
    p = sim_mtx / sim_mtx.sum(-1,keepdim=True)

    if cfg.PRETRAIN.CONTRASTIVE.MILNCEV3.WEIGHT_WARMUP:
        if cur_epoch < cfg.OPTIMIZER.WARMUP_EPOCHS:
            weight_start = cfg.PRETRAIN.CONTRASTIVE.MILNCEV3.WARMUP_START_WEIGHT
            weight_end = cfg.PRETRAIN.CONTRASTIVE.MILNCEV3.WEIGHT_MTX
            alpha = (weight_end - weight_start) / cfg.OPTIMIZER.WARMUP_EPOCHS
            weight = cur_epoch * alpha + weight_start
        else:
            weight = cfg.PRETRAIN.CONTRASTIVE.MILNCEV3.WEIGHT_MTX
    else:
        weight = cfg.PRETRAIN.CONTRASTIVE.MILNCEV3.WEIGHT_MTX
    weight_matrix = (1-torch.eye(batch_size, device=device)) * weight + torch.eye(batch_size, device=device)

    return -(torch.log(p)*sim_tt_max*weight_matrix).sum() / batch_size

def ts_nce(cfg, logits_v, logits_t, batch_size):
    device = logits_v.device
    logits_t = logits_t.reshape(-1, logits_t.shape[-1])
    sim = torch.matmul(logits_v, logits_t.transpose(0,1))
    sim_mtx = sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE).reshape(
        batch_size//2, batch_size, cfg.TEXT.NUM_SENTENCES
    ).sum(-1)


    q = torch.cat((torch.eye(batch_size//2, device=device), torch.zeros(batch_size//2, batch_size//2, device=device)), dim=-1)
    p = sim_mtx / sim_mtx.sum(-1,keepdim=True)
    loss = -torch.log(p[q!=0])
    return loss.sum() / batch_size

def mil_nce_sel(cfg, logits_v, logits_t, batch_size):
    device = logits_v.device
    logits_t = logits_t.reshape(-1, logits_t.shape[-1])
    logits_v_pos = logits_v[0:batch_size:2]
    logits_v_rev = logits_v[1:batch_size:2]
    sim = torch.matmul(logits_v_pos, logits_t.transpose(0,1)).reshape(batch_size//2, batch_size//2, cfg.TEXT.NUM_SENTENCES)
    sim_rev = torch.matmul(logits_v_rev, logits_t.transpose(0,1)).reshape(batch_size//2, batch_size//2, cfg.TEXT.NUM_SENTENCES)
    sim_mtx = torch.exp(sim/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE).sum(-1)


    if cfg.PRETRAIN.CONTRASTIVE.WITH_ONE:
        q = torch.eye(batch_size//2, device=device)
        p = sim_mtx / sim_mtx.sum(-1,keepdim=True)

        loss = -torch.log( p[q!=0] ) # calculate q*log(p)
        diff = (sim.sum(-1) - sim_rev.sum(-1))[q!=0].abs()
        if cfg.PRETRAIN.CONTRASTIVE.SEL.DETACH:
            diff = diff.detach().clone()
        loss = diff / diff.sum() * loss
        # return loss.sum() / batch_size
        return loss.sum()
    else:
        raise NotImplementedError

def fg_mil_nce(
    cfg, logits_v_abs, logits_v_fg, logits_t_abs, logits_t_fg, text_validity, batch_size
):
    device = logits_v_abs.device

    # ---- fine grained text with abstract video
    sim_vt = torch.matmul(logits_v_abs, logits_t_fg.reshape(-1, logits_t_fg.shape[-1]).transpose(0,1))
    positive_pairs_vt = torch.eye(batch_size, device=device).repeat_interleave(
        cfg.TEXT.NUM_SENTENCES*cfg.TEXT.MAX_WORDS, dim=1
    )

    sim_mtx = torch.exp(sim_vt/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)

    pos_vt = (sim_mtx*positive_pairs_vt*text_validity.reshape(1,-1)).sum(1)
    neg_vt = (sim_mtx*(1-positive_pairs_vt)*text_validity.reshape(1,-1)).sum(1)

    num_sup_pixels = torch.prod(torch.tensor(logits_v_fg.shape[2:])).item()
    sim_tv = torch.matmul( logits_t_abs.reshape(-1, logits_t_abs.shape[-1]), logits_v_fg.permute(0,2,3,4,1).reshape(-1, logits_v_fg.shape[1]).transpose(0,1))
    positive_pairs_tv = torch.eye(batch_size, device=device).repeat_interleave(5, dim=0).repeat_interleave(num_sup_pixels, dim=1)

    sim_mtx = torch.exp(sim_tv/cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE)

    pos_tv = (sim_mtx*positive_pairs_tv).reshape(batch_size, cfg.TEXT.NUM_SENTENCES, -1).sum([-1,-2])
    neg_tv = (sim_mtx*(1-positive_pairs_tv)).reshape(batch_size, cfg.TEXT.NUM_SENTENCES, -1).sum([-1,-2])

    return -(torch.log(
        (pos_vt+pos_tv)/(pos_vt+pos_tv+neg_vt+neg_tv)
    )).sum() / batch_size