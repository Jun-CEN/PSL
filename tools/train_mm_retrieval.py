#!/usr/bin/env python3

"""Train a video classification model."""
import numpy as np
import pprint
import torch

import os
import oss2 as oss

import models.utils.losses as losses
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.bucket as bu
# import slowfast.visualization.tensorboard_vis as tb
from utils.meters import TrainMeter, RetrievalVideoTextMeter

from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, indexes, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.        
        if misc.get_num_gpus(cfg):
            if not cfg.AUGMENTATION.USE_GPU:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                elif isinstance(inputs, (dict,)):
                    for k, v in inputs.items():
                        if k == "sentences":
                            continue
                        inputs[k] = v.cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            labels["supervised"] = labels["supervised"].cuda()
            labels["self-supervised"] = {}
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        preds, logits = model(inputs)

        loss, loss_in_parts, weight = losses.calculate_loss(cfg, preds, logits, labels, cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        if hasattr(cfg, "MULTI_MODAL") and\
            cfg.PRETRAIN.PROTOTYPE.ENABLE and\
            cfg.PRETRAIN.PROTOTYPE.REGULARIZATION:
                loss_in_parts["loss_regularization"] = cfg.PRETRAIN.PROTOTYPE.REGULARIZATION_WEIGHT \
                * model.module.regularization(weight, logits["text_attentive_pooled"]) \
                    if misc.get_num_gpus(cfg) > 1 \
                    else model.regularization(weight, logits["text_attentive_pooled"])
                loss += loss_in_parts["loss_regularization"]
        loss.backward()
        if hasattr(cfg, "MULTI_MODAL") and\
            cfg.PRETRAIN.PROTOTYPE.ENABLE and\
            cur_epoch < cfg.PRETRAIN.PROTOTYPE.FREEZE_EPOCHS:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # Update the parameters.
        optimizer.step()
        if hasattr(cfg, "MULTI_MODAL") and hasattr(model, "m"):
            model.update_prototypes(weight)


        if misc.get_num_gpus(cfg) > 1:
            loss = du.all_reduce([loss])[0]
        loss = loss.item()

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            None, None, loss, lr, inputs["video"].shape[0] if isinstance(inputs, dict) else inputs.shape[0]
        )
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {"Train/loss": loss, "Train/lr": lr},
                global_step=data_size * cur_epoch + cur_iter,
            )
        train_meter.update_custom_stats(loss_in_parts)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch+cfg.TRAIN.NUM_FOLDS-1)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(val_loader):
        if misc.get_num_gpus(cfg):
            # Transferthe data to the current GPU device.
            if not cfg.AUGMENTATION.USE_GPU:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                elif isinstance(inputs, (dict,)):
                    for k, v in inputs.items():
                        inputs[k] = v.cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            labels["supervised"] = labels["supervised"].cuda()
            labels["self-supervised"] = {}
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        preds, logits = model(inputs)
        text_feature = logits['xt'].squeeze(1)
        video_feature = logits['xv']
        if misc.get_num_gpus(cfg) > 1:
            video_idx = video_idx.cuda()
            text_feature, video_feature, video_idx= du.all_gather(
                [text_feature, video_feature, video_idx]
            )
        else:
            labels_supervised = labels["supervised"]
        videos_name_id = val_loader.dataset.get_videos_name(video_idx)
        val_meter.update_stats(
            video_feature.cpu(), text_feature.cpu(), video_idx.cpu(), videos_name_id
        )
        val_meter.log_iter_stats(cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.finalize_metrics()
    val_meter.reset()

def train_mm_retrieval(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
    model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)
    
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, model_bucket)

    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = RetrievalVideoTextMeter(
                cfg, 
                len(val_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                len(val_loader),
                cfg.DATA.ENSEMBLE_METHOD,)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        misc.get_num_gpus(cfg)
    ):
        # writer = tb.TensorboardWriter(cfg)
        pass
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.OPTIMIZER.MAX_EPOCH-start_epoch)%cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    for cur_epoch in range(start_epoch, cfg.OPTIMIZER.MAX_EPOCH, cfg.TRAIN.NUM_FOLDS):

        # Shuffle the dataset.
        shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
        )
        torch.cuda.empty_cache()

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cfg, cur_epoch+cfg.TRAIN.NUM_FOLDS-1):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch+cfg.TRAIN.NUM_FOLDS-1, cfg, model_bucket)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch+cfg.TRAIN.NUM_FOLDS-1):
            eval_epoch(val_loader, model, val_meter, cur_epoch+cfg.TRAIN.NUM_FOLDS-1, cfg, writer)

    if writer is not None:
        writer.close()
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

