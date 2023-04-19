


import numpy as np
import os
import pickle
import torch
import json

import utils.bucket as bu
import utils.checkpoint as cu
import utils.distributed as du
import utils.feature as fu
import utils.logging as logging
import utils.misc as misc
from datasets.base.builder import build_loader
from models.base.builder import build_model
from utils.meters import GroundingFeatureExtractMeter

logger = logging.get_logger(__name__)

@torch.no_grad()
def extract_features(loader, model, feature_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        retrieval_set_loader (loader): video retrieval_set loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()
    feature_meter.iter_tic()
    # file_name = "frames_16_interval_4.txt"
    # lists = ""
    # overall_num_clips = 0
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(loader):
    #     ---- generate list ----
    #     name = inputs["name"][0]
    #     num_clips = inputs["list"].shape[1]
    #     video_h = inputs["original"].shape[2]
    #     video_w = inputs["original"].shape[3]
    #     for i in range(num_clips):
    #         lists += "{}.mp4,{},{},{}\n".format(name, i, video_h, video_w)
    #     overall_num_clips += num_clips
    #     if cur_iter % 500 == 0:
    #         print(cur_iter)
    # with open("/home/ziyuan/data/Charades/annos/video_list.txt", "w") as f:
    #     f.writelines(lists)

        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            if "video" in inputs.keys():
                inputs["video"] = inputs["video"].cuda(non_blocking=True)
            if "text" in inputs.keys():
                inputs["text"] = inputs["text"].cuda(non_blocking=True)
                inputs["text_validity"] = inputs["text_validity"].cuda(non_blocking=True)
            video_idx = video_idx.cuda(non_blocking=True)

        # Perform the forward pass.
        _, logits = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if misc.get_num_gpus(cfg) > 1:
            if "xv" in logits.keys():
                video_idx, logits["xv"], logits["xv_mid"] = du.all_gather(
                    [video_idx, logits["xv"], logits["xv_mid"]]
                )
        
        if misc.get_num_gpus(cfg):
            video_idx = video_idx.cpu()
            for k, v in logits.items():
                logits[k] = v.cpu()
        
        if "xv" in logits.keys():
            video_idx, video_idx_args = video_idx.sort()
            for k, v in logits.items():
                logits[k] = v[video_idx_args]

        feature_meter.iter_toc()
        # log stats.
        feature_meter.update_stats(
            logits, video_idx, 
            names=None if "text" not in inputs.keys() else inputs["name"],
            sentences=None if "text" not in inputs.keys() else inputs["sentence"]
        )
        feature_meter.log_iter_stats(cur_iter)

        feature_meter.iter_tic()

    feature_meter.finalize_metrics()

def grounding_feature_extraction(cfg):
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
    logging.setup_logging(cfg, cfg.LINEAR_EVALUATION.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Grounding feature extraction with config:")
        logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
    model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)

    cu.load_test_checkpoint(cfg, model, model_bucket)

    # --- extracting / loading text features
    if cfg.LINEAR_EVALUATION.VIDEO_GROUNDING_TEXT:
        train_loader = build_loader(cfg, "grounding_text")
        logger.info("Extracting text features for {} iterations".format(len(train_loader)))
        train_feature_meter = GroundingFeatureExtractMeter(
            cfg, 
            split="grounding_text",
            num_videos=len(train_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            overall_iters=len(train_loader),
            samples=train_loader.dataset._samples,
            ensemble_method=None,
        )
        extract_features(train_loader, model, train_feature_meter, cfg)

    # --- extracting / loading video features
    if cfg.LINEAR_EVALUATION.VIDEO_GROUNDING_VIDEO:
        train_loader = build_loader(cfg, "grounding_video")
        logger.info("Extracting video features for {} iterations".format(len(train_loader)))
        train_feature_meter = GroundingFeatureExtractMeter(
            cfg, 
            split="grounding_video",
            num_videos=len(train_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            overall_iters=len(train_loader),
            samples=train_loader.dataset._samples,
            ensemble_method=None,
        )
        extract_features(train_loader, model, train_feature_meter, cfg)