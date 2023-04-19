import sys
import math
import torch
import utils.misc as misc
import utils.distributed as du

def clustering_loss(cfg, logits):
    loss = 0
    epsilon = cfg.PRETRAIN.CLUSTERING.EPSILON

    pv = logits["video_prototypes"]
    pt = logits["text_prototypes"]
    if cfg.PRETRAIN.PROTOTYPE.POOLING == "max":
        zt = logits["text"]
        pt = pt.max(dim=1)[0]
    elif cfg.PRETRAIN.PROTOTYPE.POOLING == "avg":
        zt = logits["text"]
        pt = pt.mean(dim=1)
    elif cfg.PRETRAIN.PROTOTYPE.POOLING == "sim-max":
        xv = logits["video"].unsqueeze(1)
        zt = logits["text"]
        _, vt_ind = (xv*zt).sum(-1).max(-1)
        batch_size = vt_ind.shape[0]
        device = vt_ind.device
        pt = pt[torch.linspace(0,batch_size-1,batch_size, dtype=torch.long, device=device), vt_ind]
    elif cfg.PRETRAIN.PROTOTYPE.POOLING == "attentive-pool":
        pass
    if cfg.PRETRAIN.PROTOTYPE.ASYMMETRIC is None or cfg.PRETRAIN.PROTOTYPE.ASYMMETRIC == "video+text+nce":
        qv = distributed_sinkhorn(cfg, torch.exp(pv / epsilon).t(), 3)
        qt = distributed_sinkhorn(cfg, torch.exp(pt / epsilon).t(), 3)

        pv = torch.nn.functional.softmax(pv / cfg.PRETRAIN.CLUSTERING.TEMPERATURE, dim=1)
        pt = torch.nn.functional.softmax(pt / cfg.PRETRAIN.CLUSTERING.TEMPERATURE, dim=1)

        loss -= torch.mean(torch.sum(qt * torch.log(pv), dim=1))
        loss -= torch.mean(torch.sum(qv * torch.log(pt), dim=1))
        weight = None
    elif cfg.PRETRAIN.PROTOTYPE.ASYMMETRIC == "video" or cfg.PRETRAIN.PROTOTYPE.ASYMMETRIC == "video+nce":
        qt = distributed_sinkhorn(cfg, torch.exp(pt / epsilon).t(), 3)
        pv = torch.nn.functional.softmax(pv / cfg.PRETRAIN.CLUSTERING.TEMPERATURE, dim=1)
        loss -= torch.mean(torch.sum(qt * torch.log(pv), dim=1))
        if cfg.PRETRAIN.PROTOTYPE.REGULARIZATION:
            weight = qt
        elif cfg.PRETRAIN.PROTOTYPE.MOMENTUM < 1 and cfg.PRETRAIN.PROTOTYPE.MOMENTUM > 0:
            zt = logits["text_attentive_pooled"]
            weight = torch.matmul(qt.transpose(0,1), zt)
            if misc.get_num_gpus(cfg) > 1:
                weight = du.all_gather(
                    [weight]
                )[0]
        else:
            weight = None
    elif cfg.PRETRAIN.PROTOTYPE.ASYMMETRIC == "text" or cfg.PRETRAIN.PROTOTYPE.ASYMMETRIC == "text+nce":
        qv = distributed_sinkhorn(cfg, torch.exp(pv / epsilon).t(), 3)
        pt = torch.nn.functional.softmax(pt / cfg.PRETRAIN.CLUSTERING.TEMPERATURE, dim=1)
        loss -= torch.mean(torch.sum(qv * torch.log(pt), dim=1))
        if cfg.PRETRAIN.PROTOTYPE.REGULARIZATION:
            weight = qt
        elif cfg.PRETRAIN.PROTOTYPE.MOMENTUM < 1 and cfg.PRETRAIN.PROTOTYPE.MOMENTUM > 0:
            zt = logits["text_attentive_pooled"]
            weight = torch.matmul(qv.transpose(0,1), zt)
            if misc.get_num_gpus(cfg) > 1:
                weight = du.all_gather(
                    [weight]
                )[0]
        else:
            weight = None
    return loss, weight

def distributed_sinkhorn(cfg, Q, nmb_iters):
    with torch.no_grad():
        num_gpus = misc.get_num_gpus(cfg)
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        if num_gpus > 1:
            sum_Q = du.all_reduce([sum_Q], average=False)[0]
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (num_gpus * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            if num_gpus > 1:
                u = du.all_reduce([u], average=False)[0]
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor