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
import utils.feature as fu
import utils.logging as logging
import utils.misc as misc
# import slowfast.visualization.tensorboard_vis as tb
from datasets.base.builder import build_loader
from models.base.builder import build_model
from utils.meters import TestMeter, LinearFeatureExtractMeter, LinearTestMeter, RetrievalVideoTextMeter

from sklearn import preprocessing
from sklearn.svm import LinearSVC

logger = logging.get_logger(__name__)

@torch.no_grad()
def extract_video_features(loader, model, feature_meter, cfg):
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
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(loader):
        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            elif isinstance(inputs, (dict,)):
                for k, v in inputs.items():
                    inputs[k] = v.cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels["supervised"] = labels["supervised"].cuda()
            video_idx = video_idx.cuda()

        # Perform the forward pass.
        preds, logits = model(inputs)
        # logits = torch.nn.functional.normalize(logits, dim=-1, p=2)

        # Gather all the predictions across all the devices to perform ensemble.
        if misc.get_num_gpus(cfg) > 1:
            preds, labels_supervised, video_idx, logits = du.all_gather(
                [preds, labels["supervised"], video_idx, logits]
            )
        else:
            labels_supervised = labels["supervised"]
        
        if misc.get_num_gpus(cfg):
            preds = preds.cpu()
            logits = logits.cpu()
            labels_supervised = labels_supervised.cpu()
            video_idx = video_idx.cpu()

        feature_meter.iter_toc()
        # log stats.
        feature_meter.update_stats(
            logits.detach(), labels_supervised.detach(), video_idx.detach()
        )
        feature_meter.log_iter_stats(cur_iter)

        feature_meter.iter_tic()

    feature_meter.finalize_metrics()
    return feature_meter.get_feature_and_labels()


@torch.no_grad()
def perform_retrival(retrieval_loader, model, retrieval_meter, retrieval_set_features, retrieval_set_labels, cfg):
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
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()
    retrieval_meter.iter_tic()
    retrieval_set_features = torch.nn.functional.normalize(retrieval_set_features, dim=-1, p=2)
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(retrieval_loader):
        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            elif isinstance(inputs, (dict,)):
                for k, v in inputs.items():
                    inputs[k] = v.cuda(non_blocking=True)
                    if retrieval_set_features.device != inputs[k].device:
                        retrieval_set_features = retrieval_set_features.to(inputs[k].device)
            else:
                inputs = inputs.cuda(non_blocking=True)


            # Transfer the data to the current GPU device.
            labels["supervised"] = labels["supervised"].cuda()
            video_idx = video_idx.cuda()

        # Perform the forward pass.
        preds, logits = model(inputs)
        similarity = torch.matmul(torch.nn.functional.normalize(logits, dim=-1, p=2), retrieval_set_features.transpose(0,1))

        # Gather all the predictions across all the devices to perform ensemble.
        if misc.get_num_gpus(cfg) > 1:
            preds, labels_supervised, video_idx, similarity, logits = du.all_gather(
                [preds, labels["supervised"], video_idx, similarity, logits]
            )
        else:
            labels_supervised = labels["supervised"]
        

        if misc.get_num_gpus(cfg):
            similarity = similarity.cpu()
            labels_supervised = labels_supervised.cpu()
            video_idx = video_idx.cpu()
            logits = logits.cpu()

        retrieval_meter.iter_toc()
        # Update and log stats.
        retrieval_meter.update_stats(
            logits.detach(), similarity.detach(), labels_supervised.detach(), video_idx.detach(), retrieval_set_labels
        )
        retrieval_meter.log_iter_stats(cur_iter)

        retrieval_meter.iter_tic()

    retrieval_meter.finalize_metrics()

    return retrieval_meter.get_feature_and_labels()

@torch.no_grad()
def perform_video_text_retrival(retrieval_loader, model, retrieval_meter, cfg):
    # Enable eval mode.
    model.eval()
    retrieval_meter.iter_tic()
    from models.utils.clustering_losses import distributed_sinkhorn
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(retrieval_loader):
        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            elif isinstance(inputs, (dict,)):
                for k, v in inputs.items():
                    inputs[k] = v.cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)


            # Transfer the data to the current GPU device.
            labels["supervised"] = labels["supervised"].cuda()
            video_idx = video_idx.cuda()

        # Perform the forward pass.
        preds, logits = model(inputs)
        if cfg.RETRIEVAL.MULTI_MODAL.CLUSTER:
            text_q = logits['text_prototypes']
            video_p = logits['video_prototypes']
            epsilon = cfg.PRETRAIN.CLUSTERING.EPSILON
            # text_feature = distributed_sinkhorn(cfg, torch.exp(text_feature / epsilon).t(), 3)
            video_p = torch.nn.functional.softmax(video_p / cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE, dim=1)
            text_q = torch.nn.functional.softmax(text_q / cfg.PRETRAIN.CONTRASTIVE.TEMPERATURE, dim=1)
        else:
            video_p = None
            text_q = None
        text_feature = logits['text'].squeeze(1)
        video_feature = logits['video']

        # Gather all the predictions across all the devices to perform ensemble.
        videos_name_id = retrieval_loader.dataset.get_videos_name(video_idx).cuda()
        if misc.get_num_gpus(cfg) > 1:
            preds, labels_supervised, video_idx, videos_name_id, text_feature, video_feature, video_p, text_q = du.all_gather(
                [preds, labels["supervised"], video_idx, videos_name_id, text_feature, video_feature, video_p, text_q]
            )
        else:
            labels_supervised = labels["supervised"]
        

        if misc.get_num_gpus(cfg):
            labels_supervised = labels_supervised.cpu()
            video_idx = video_idx.cpu()
            video_feature = video_feature.cpu()
            text_feature = text_feature.cpu()
            videos_name_id = videos_name_id.cpu()
            if video_p is not None:
                video_p = video_p.cpu()
                text_q = text_q.cpu()

        retrieval_meter.iter_toc()
        # Update and log stats.video_features, text_features, clip_ids
        retrieval_meter.update_stats(
            video_feature, text_feature, video_p, text_q, video_idx.detach(), videos_name_id
        )
        retrieval_meter.log_iter_stats(cur_iter)

        retrieval_meter.iter_tic()

    retrieval_meter.finalize_metrics()

    return retrieval_meter.get_feature_and_labels()

def eval_classification(train_set_features, train_set_labels, test_set_features, test_set_labels, cfg):
    num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

    le = preprocessing.LabelEncoder()
    train_set_labels = le.fit_transform(train_set_labels.numpy())
    test_set_labels = le.fit_transform(test_set_labels.numpy())
    train_set_features = torch.nn.functional.normalize(train_set_features, dim=-1, p=2)
    test_set_features = torch.nn.functional.normalize(test_set_features, dim=-1, p=2)
    # scaler = preprocessing.StandardScaler().fit(train_set_features)
    # test_set_features = scaler.transform(test_set_features)
    for reg in [0.05, 0.1, 0.5, 1.0]:
        c = LinearSVC(C=reg, max_iter=1000, dual=False)
        c.fit(train_set_features.numpy(), train_set_labels)
        preds = c.decision_function(test_set_features)
        preds = np.reshape(preds, (len(test_set_labels), num_clips, -1))
        preds = preds.sum(axis=1)
        preds = np.argmax(preds, axis=1)
        acc = np.sum(preds == test_set_labels) / float(len(preds))
        logger.info("C: {}. Top 1 acc: {}.".format(
            reg, acc
        ))

def linear_classification(cfg):
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
        logger.info("Retrival with config:")
        logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
    model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)

    cu.load_test_checkpoint(cfg, model, model_bucket)

    # --- extracting / loading train set features
    if fu.feature_exists(cfg, "linear_train_cls"):
        train_set_features, train_set_labels = fu.load_features(cfg, "linear_train_cls")
    else:
        train_loader = build_loader(cfg, "linear_train_cls")
        logger.info("Extracting train set features for {} iterations".format(len(train_loader)))
        train_feature_meter = LinearFeatureExtractMeter(
            cfg, 
            split="linear_train_cls",
            num_videos=len(train_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            overall_iters=len(train_loader),
            ensemble_method=None,
        )
        train_set_features, train_set_labels = extract_video_features(train_loader, model, train_feature_meter, cfg)
        fu.save_video_features(cfg, "linear_train_cls", train_set_features, train_set_labels)

    # --- extracting / loading test set features
    if fu.feature_exists(cfg, "linear_test_cls"):
        test_set_features, test_set_labels = fu.load_features(cfg, "linear_test_cls")
    else:
        test_loader = build_loader(cfg, "linear_test_cls")
        logger.info("Retrieving for {} iterations".format(len(test_loader)))

        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_feature_meter = LinearFeatureExtractMeter(
            cfg,
            split="linear_test_cls",
            num_videos=len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            overall_iters=len(test_loader),
            ensemble_method=None
        )
        test_set_features, test_set_labels = extract_video_features(test_loader, model, test_feature_meter, cfg)
        fu.save_video_features(cfg, "linear_test_cls", test_set_features, test_set_labels)

    eval_classification(
        train_set_features, train_set_labels, test_set_features, test_set_labels, cfg
    )

    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.LINEAR_EVALUATION.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

def eval_retrieval(train_set_features, train_set_labels, test_set_features, test_set_labels, cfg):
    train_set_features = train_set_features.cuda(non_blocking=True)
    test_set_features = test_set_features.cuda(non_blocking=True)
    train_set_features = torch.nn.functional.normalize(train_set_features, dim=-1, p=2)
    test_set_features = torch.nn.functional.normalize(test_set_features, dim=-1, p=2)
    sim = torch.matmul(test_set_features, train_set_features.t())
    topks = [1, 5, 10, 20]
    stats = {}
    for topk in topks:
        correct_map = (train_set_labels[sim.topk(20)[1]] == test_set_labels.unsqueeze(1))
        acc = (correct_map[:, :topk].sum(-1) > 0).sum()*100.0/test_set_labels.size(0)
        stats["top{}".format(topk, prec=2)] = float(acc)
    logging.log_json_stats(stats)


def linear_retrieval(cfg):
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
        logger.info("Retrival with config:")
        logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
    model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)

    cu.load_test_checkpoint(cfg, model, model_bucket)

    # if cfg.LINEAR_EVALUATION.MULTI_MODAL.ENABLE:
    #     retrieval_set_loader = build_loader(cfg, "linear_train")
    #     retrieval_set_meter = RetrievalVideoTextMeter(
    #         cfg, 
    #         len(retrieval_set_loader.dataset)
    #         // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
    #         cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
    #         len(retrieval_set_loader),
    #         cfg.LINEAR_EVALUATION.TRAIN_ENSEMBLE_METHOD,
    #     )
    #     logger.info("Extracting video text retrieval set features for {} iterations".format(len(retrieval_set_loader)))
    #     video_features, text_features, videos_name_id = perform_video_text_retrival(retrieval_set_loader, model, retrieval_set_meter, cfg)
    #     fu.save_video_text_retrieval_set_statics(video_features, text_features, videos_name_id, cfg)
    #     return

    # --- extracting / loading train set features
    if fu.feature_exists(cfg, "linear_train_ret"):
        train_set_features, train_set_labels = fu.load_features(cfg, "linear_train_ret")
    else:
        train_loader = build_loader(cfg, "linear_train_ret")
        logger.info("Extracting train set features for {} iterations".format(len(train_loader)))
        train_feature_meter = LinearFeatureExtractMeter(
            cfg, 
            split="linear_train_ret",
            num_videos=len(train_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            overall_iters=len(train_loader),
            ensemble_method=cfg.LINEAR_EVALUATION.TRAIN_ENSEMBLE_METHOD,
        )
        train_set_features, train_set_labels = extract_video_features(train_loader, model, train_feature_meter, cfg)
        fu.save_video_features(cfg, "linear_train_ret", train_set_features, train_set_labels)

    # --- extracting / loading test set features
    if fu.feature_exists(cfg, "linear_test_ret"):
        test_set_features, test_set_labels = fu.load_features(cfg, "linear_test_ret")
    else:
        test_loader = build_loader(cfg, "linear_test_ret")
        logger.info("Retrieving for {} iterations".format(len(test_loader)))

        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_feature_meter = LinearFeatureExtractMeter(
            cfg,
            split="linear_test_ret",
            num_videos=len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            overall_iters=len(test_loader),
            ensemble_method=cfg.LINEAR_EVALUATION.TEST_ENSEMBLE_METHOD
        )
        test_set_features, test_set_labels = extract_video_features(test_loader, model, test_feature_meter, cfg)
        fu.save_video_features(cfg, "linear_test_ret", test_set_features, test_set_labels)

    eval_retrieval(
        train_set_features, train_set_labels, test_set_features, test_set_labels, cfg
    )

    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.LINEAR_EVALUATION.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )
