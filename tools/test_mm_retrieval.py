#!/usr/bin/env python3

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import json
# from fvcore.common.file_io import PathManager

import utils.bucket as bu
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
# import slowfast.visualization.tensorboard_vis as tb
from datasets.base.builder import build_loader
from models.base.builder import build_model
from utils.meters import RetrievalVideoTextMeter
from tools.train_mm_retrieval import eval_epoch
logger = logging.get_logger(__name__)


def test_mm_retrieval(cfg):
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
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
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
    test_loader = build_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    cfg.LOG_PERIOD = 2
    test_meter = RetrievalVideoTextMeter(
            cfg, 
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            len(test_loader),
            cfg.DATA.ENSEMBLE_METHOD,)

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        misc.get_num_gpus(cfg)
    ):
        # writer = tb.TensorboardWriter(cfg)
        pass
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    # perform_test(test_loader, model, test_meter, cfg, writer)
    eval_epoch(test_loader, model, test_meter, -1, cfg, writer)
    if writer is not None:
        writer.close()
    
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )
        if cfg.TEST.SAVE_RESULTS_PATH != "":
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
            if hasattr(cfg.TEST, "RECORD_SSL_TEST") and cfg.TEST.RECORD_SSL_TEST:
                filename = os.path.join(cfg.OUTPUT_DIR, "{}_ssl".format(cfg.TEST.SAVE_RESULTS_PATH))
                bu.put_to_bucket(
                    model_bucket, 
                    cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                    filename,
                    cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
                )
