#!/usr/bin/env python3

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import json

import utils.bucket as bu
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets.base.builder import build_loader
from models.base.builder import build_model
from utils.meters import TestMeter

from vis.gradcam import GradCAM

logger = logging.get_logger(__name__)


# @torch.no_grad()
def perform_visualization(visualization_loader, model, test_meter, cfg, writer=None, gradcam=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    Args:
        visualization_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    # model.eval()
    test_meter.iter_tic()
    res_dic = {}
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(visualization_loader):
        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            elif isinstance(inputs, (dict,)):
                for i in range(len(inputs["video"])):
                    inputs["video"][i] = inputs["video"][i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels["supervised"] = labels["supervised"].cuda()
            video_idx = video_idx.cuda()
            if cfg.PRETRAIN.ENABLE:
                for k, v in labels["self-supervised"].items():
                    labels["self-supervised"][k] = v.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        if cfg.VISUALIZATION.GRAD_CAM.ENABLE:
            # Perform the forward pass.
            # inputs, preds = gradcam(inputs, labels=labels["supervised"])
            inputs, preds = gradcam(inputs["video"], labels["supervised"], index=video_idx, use_labels=cfg.VISUALIZATION.GRAD_CAM.USE_LABELS)
        else:
            preds = model(inputs["video"])

        # Gather all the predictions across all the devices to perform ensemble.
        if misc.get_num_gpus(cfg) > 1:
            # print("to gather, shape: {}".format(preds.shape))
            preds, labels_supervised, video_idx = du.all_gather(
                [preds, labels["supervised"], video_idx]
            )
        else:
            labels_supervised = labels["supervised"]
        if hasattr(cfg.TEST, "RECORD_SSL_TEST") and cfg.TEST.RECORD_SSL_TEST:
            for elem_idx, elem in enumerate(video_idx):
                res_dic[int(elem.cpu())] = int(preds.max(1).indices.cpu()[elem_idx])
        if misc.get_num_gpus(cfg):
            preds = preds.cpu()
            labels_supervised = labels_supervised.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    test_meter.reset()


def vis(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    # Setup logging format.
    logging.setup_logging(cfg, "test")

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
    model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)

    cu.load_test_checkpoint(cfg, model, model_bucket)

    # Create video testing loaders.
    visualization_loader = build_loader(cfg, "test")
    logger.info("Visualizing model for {} iterations".format(len(visualization_loader)))

    assert (
        len(visualization_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        cfg,
        len(visualization_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.VIDEO.HEAD.NUM_CLASSES,
        len(visualization_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        misc.get_num_gpus(cfg)
    ):
        # writer = tb.TensorboardWriter(cfg)
        pass
    else:
        writer = None

    if cfg.VISUALIZATION.GRAD_CAM.ENABLE:
        gradcam = GradCAM(
                cfg,
                model,
                target_layers=cfg.VISUALIZATION.GRAD_CAM.LAYERS,
                data_mean=cfg.DATA.MEAN,
                data_std=cfg.DATA.STD,
                colormap=cfg.VISUALIZATION.GRAD_CAM.COLOR_MAP
            )
    else:
        gradcam = None

    # # Perform multi-view test on the entire dataset.
    perform_visualization(visualization_loader, model, test_meter, cfg, writer, gradcam)
    if writer is not None:
        writer.close()

