#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

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
from utils.meters import TestMeter, EpicKitchenMeter
from scipy.special import xlogy

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    Perform multi-view test on the specified test set, where {cfg.TEST.NUM_ENSEMBLE_VIEWS}
    clips and {cfg.TEST.NUM_SPATIAL_CROPS} crops are sampled temporally and spatially, forming 
    in total cfg.TEST.NUM_ENSEMBLE_VIEWS x cfg.TEST.NUM_SPATIAL_CROPS views.
    The softmax scores are aggregated by summation. 
    The predictions are compared with the ground-truth labels and the accuracy is logged.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (Config): The global config object.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    res_dic = {}
    all_confidences, all_uncertainties, all_results, all_gts = [], [], [], []
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            for k, v in inputs.items():
                if not isinstance(v, (torch.Tensor, list)):
                    continue
                if isinstance(inputs[k], list):
                    for i in range(len(inputs[k])):
                        inputs[k][i] = v[i].cuda(non_blocking=True)
                elif isinstance(inputs[k], torch.Tensor):
                    inputs[k] = v.cuda(non_blocking=True)

            # Transfer the labels to the current GPU device.
            if isinstance(labels["supervised"], dict):
                for k, v in labels["supervised"].items():
                    labels["supervised"][k] = v.cuda()
            else:
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

        if 0:
            # currently supports MoSI test.
            preds, _ = model(inputs)
            if misc.get_num_gpus(cfg) > 1:
                pred, label, video_idx = du.all_gather(
                    [preds["move_joint"], labels["self-supervised"]['move_joint'].reshape(preds["move_joint"].shape[0]), video_idx]
                )
            if misc.get_num_gpus(cfg):
                pred = pred.cpu()
                label = label.cpu()
                video_idx = video_idx.cpu()
            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                pred.detach(), label.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)
        else:
            # Perform the forward pass.
            if cfg.TEST.OPEN_METHOD == "dropout":
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()
            preds, logits = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                # Mainly for the EPIC-KITCHENS dataset.
                if misc.get_num_gpus(cfg) > 1:
                    preds_verb, preds_noun, labels_verb, labels_noun, video_idx = du.all_gather(
                        [
                            preds["verb_class"], 
                            preds["noun_class"], 
                            labels["supervised"]["verb_class"], 
                            labels["supervised"]["noun_class"], 
                            video_idx
                        ]
                    )
                else:
                    preds_verb  = preds["verb_class"]
                    preds_noun  = preds["noun_class"]
                    labels_verb = labels["supervised"]["verb_class"]
                    labels_noun = labels["supervised"]["noun_class"]
                if misc.get_num_gpus(cfg):
                    preds_verb  = preds_verb.cpu()
                    preds_noun  = preds_noun.cpu()
                    labels_verb = labels_verb.cpu()
                    labels_noun = labels_noun.cpu()
                    video_idx   = video_idx.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds_verb.detach(),
                    preds_noun.detach(),
                    labels_verb.detach(),
                    labels_noun.detach(),
                    video_idx.detach(),
                    [test_loader.dataset._get_sample_info(i)["name"] for i in video_idx.tolist()] if "name" in test_loader.dataset._get_sample_info(0).keys() else []
                )
                test_meter.log_iter_stats(cur_iter)
            else:
                if cfg.MODEL.NAME == "PrototypeModel":
                    preds = preds[0]
                    if cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 2:
                        preds = preds[::2]
                elif cfg.VIDEO.HEAD.NAME == 'BaseHeadRPL':
                    preds = preds['dist']
                # Gather all the predictions across all the devices to perform ensemble.
                if misc.get_num_gpus(cfg) > 1:
                    preds, logits, labels_supervised, video_idx = du.all_gather(
                        [preds.contiguous(), logits, labels["supervised"], video_idx]
                    )
                else:
                    labels_supervised = labels["supervised"]
                if misc.get_num_gpus(cfg):
                    preds = preds.cpu()
                    labels_supervised = labels_supervised.cpu()
                    video_idx = video_idx.cpu()
                    logits = logits.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach(), labels_supervised.detach(), video_idx.detach()
                )
                test_meter.log_iter_stats(cur_iter)

                preds_tmp = np.argmax(preds.numpy(), axis=1)
                labels_tmp = labels_supervised.numpy()
                confidences_tmp = np.max(preds.numpy(), axis=1)
                if cfg.TEST.OPEN_METHOD == "softmax_probability":
                    uncertainties_tmp = 1 - np.max(preds.numpy(), axis=1)
                    print(uncertainties_tmp.shape)
                elif cfg.TEST.OPEN_METHOD == "softmax_entropy":
                    uncertainties_tmp = - np.sum(xlogy(preds.numpy(), preds.numpy()), axis=1)
                elif cfg.TEST.OPEN_METHOD == "maha_distance":
                    print(logits.shape)
                    uncertainties_tmp = logits.numpy()
                elif cfg.TEST.OPEN_METHOD == "dropout":
                    # uncertainties_tmp = torch.sum(logits, dim=1).numpy()
                    uncertainties_tmp = logits[(range(logits.shape[0]), preds_tmp)]
                all_results.append(preds_tmp)
                all_confidences.append(confidences_tmp)
                all_uncertainties.append(uncertainties_tmp)
                all_gts.append(labels_tmp)
        test_meter.iter_tic()

    all_confidences = np.concatenate(all_confidences, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_results = np.concatenate(all_results, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    print(all_confidences.shape, all_uncertainties.shape, all_results.shape, all_gts.shape)

    # save epic-kitchens statistics
    if "epickitchen100" in cfg.TEST.DATASET:
        if cfg.DATA.MULTI_LABEL or not hasattr(cfg.DATA, "TRAIN_VERSION"):
            verb = test_meter.video_preds["verb_class"]
            noun = test_meter.video_preds["noun_class"]
            file_name_verb = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            file_name_noun = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(verb, file_name_verb)
            torch.save(noun, file_name_noun)

            logger.info(
                "Successfully saved verb and noun results to {} and {}.".format(file_name_verb, file_name_noun)
            )
        elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION == "only_train_verb":
            file_name = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(test_meter.video_preds, file_name)
            logger.info(
                "Successfully saved verb results to {}.".format(file_name)
            )
        elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION == "only_train_noun":
            file_name = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(test_meter.video_preds, file_name)
            logger.info(
                "Successfully saved noun results to {}.".format(file_name)
            )
    elif cfg.TEST.SAVE_PREDS: 
        file_name = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_PREDS_PATH.split('.')[0] + "_{}clipsx{}crops.pyth".format(
            cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
        ))
        torch.save(test_meter.video_preds, file_name)
        logger.info(f"Successfully saved preds to {file_name}.")

    test_meter.finalize_metrics()
    test_meter.reset()

    return all_confidences, all_uncertainties, all_results, all_gts

def test_open(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (Config): The global config object.
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
    cfg.DATA = cfg.DATA_IN
    model, model_ema = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
    model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)

    cu.load_test_checkpoint(cfg, model, model_ema, model_bucket)

    # Create video testing loaders.
    cfg_in = cfg.deep_copy()
    cfg_in.DATA = cfg_in.DATA_IN
    cfg_in.TEST.DATASET = cfg_in.TEST.DATASET_IN
    cfg_ood = cfg.deep_copy()
    cfg_ood.DATA = cfg_in.DATA_OOD
    cfg_ood.TEST.DATASET = cfg_ood.TEST.DATASET_OOD
    cfg_train = cfg.deep_copy()
    cfg_train.TEST.DATASET = cfg_train.TEST.DATASET_IN
    test_loader_in = build_loader(cfg_in, "test_in")
    test_loader_ood = build_loader(cfg_ood, "test_ood")
    test_loader_train = build_loader(cfg_train, "test_train")
    logger.info("Testing model for {} iterations".format(len(test_loader_in)))
    logger.info("Testing model for {} iterations".format(len(test_loader_ood)))
    logger.info("Testing model for {} iterations".format(len(test_loader_train)))


    assert (
        len(test_loader_in.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    assert (
        len(test_loader_ood.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    assert (
        len(test_loader_train.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    # cfg.LOG_PERIOD = max(len(test_loader) // 10, 5)
    if cfg.DATA.MULTI_LABEL or hasattr(cfg.DATA, "TRAIN_VERSION"):
        test_meter = EpicKitchenMeter(
            cfg,
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.VIDEO.HEAD.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.ENSEMBLE_METHOD,
        )
    else:
        test_meter_in = TestMeter(
            cfg_in,
            len(test_loader_in.dataset)
            // (cfg_in.TEST.NUM_ENSEMBLE_VIEWS * cfg_in.TEST.NUM_SPATIAL_CROPS),
            cfg_in.TEST.NUM_ENSEMBLE_VIEWS * cfg_in.TEST.NUM_SPATIAL_CROPS,
            cfg_in.VIDEO.HEAD.NUM_CLASSES,
            len(test_loader_in),
            cfg_in.DATA.ENSEMBLE_METHOD,
        )
        test_meter_ood = TestMeter(
            cfg_ood,
            len(test_loader_ood.dataset)
            // (cfg_ood.TEST.NUM_ENSEMBLE_VIEWS * cfg_ood.TEST.NUM_SPATIAL_CROPS),
            cfg_ood.TEST.NUM_ENSEMBLE_VIEWS * cfg_ood.TEST.NUM_SPATIAL_CROPS,
            cfg_ood.VIDEO.HEAD.NUM_CLASSES,
            len(test_loader_ood),
            cfg_ood.DATA.ENSEMBLE_METHOD,
        )
        test_meter_train = TestMeter(
            cfg_train,
            len(test_loader_train.dataset)
            // (cfg_train.TEST.NUM_ENSEMBLE_VIEWS * cfg_train.TEST.NUM_SPATIAL_CROPS),
            cfg_train.TEST.NUM_ENSEMBLE_VIEWS * cfg_train.TEST.NUM_SPATIAL_CROPS,
            cfg_train.VIDEO.HEAD.NUM_CLASSES,
            len(test_loader_train),
            cfg_train.DATA.ENSEMBLE_METHOD,
        )

    # Perform multi-view test on the entire dataset.
    if cfg.TEST.OPEN_METHOD == "maha_distance":
        train_confidences, train_uncertainties, train_results, train_labels = perform_test(test_loader_train, model, test_meter_train, cfg_train)
    
    test_meter_in.set_model_ema_enabled(False)
    ind_confidences, ind_uncertainties, ind_results, ind_labels = perform_test(test_loader_in, model, test_meter_in, cfg_in)
    test_meter_ood.set_model_ema_enabled(False)
    ood_confidences, ood_uncertainties, ood_results, ood_labels = perform_test(test_loader_ood, model, test_meter_ood, cfg_ood)
    
    filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.RESULT_FILE_NAME)
    if cfg.TEST.OPEN_METHOD == "maha_distance":
        if cfg.MODEL.NAME == "PrototypeModel":
           np.savez(filename, ind_conf=ind_confidences, ood_conf=ood_confidences,
                                   ind_unctt=ind_uncertainties, ood_unctt=ood_uncertainties, 
                                   ind_pred=ind_results, ood_pred=ood_results,
                                   ind_label=ind_labels, ood_label=ood_labels,
                                   train_conf=train_confidences, train_unctt=train_uncertainties,
                                   train_pred=train_results, train_label=train_labels, prototypes=model.module.prototypes.cpu().detach().numpy()) 
        else:
            np.savez(filename, ind_conf=ind_confidences, ood_conf=ood_confidences,
                                   ind_unctt=ind_uncertainties, ood_unctt=ood_uncertainties, 
                                   ind_pred=ind_results, ood_pred=ood_results,
                                   ind_label=ind_labels, ood_label=ood_labels,
                                   train_conf=train_confidences, train_unctt=train_uncertainties,
                                   train_pred=train_results, train_label=train_labels)
    else:
        np.savez(filename, ind_conf=ind_confidences, ood_conf=ood_confidences,
                                   ind_unctt=ind_uncertainties, ood_unctt=ood_uncertainties, 
                                   ind_pred=ind_results, ood_pred=ood_results,
                                   ind_label=ind_labels, ood_label=ood_labels)
    if model_ema is not None:
        test_meter.set_model_ema_enabled(True)
        perform_test(test_loader, model_ema.module, test_meter, cfg)
    
    # upload results to bucket
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.RESULT_FILE_NAME)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'test_open/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

        result_file_name = cfg.TEST.LOG_FILE
        result_file_name = result_file_name.split('.')[0] + "_res" + ".json"
        filename = os.path.join(cfg.OUTPUT_DIR, result_file_name)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

        result_file_name = cfg.TEST.LOG_FILE
        result_file_name = result_file_name.split('.')[0] + "_res_ema" + ".json"
        filename = os.path.join(cfg.OUTPUT_DIR, result_file_name)
        if os.path.exists(filename):
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
        
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb_ema.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb_ema.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )
        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun_ema.pyth")):
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun_ema.pyth")
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )

        if cfg.TEST.SAVE_PREDS: 
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_PREDS_PATH.split('.')[0] + "_{}clipsx{}crops.pyth".format(
                cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
            ))
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )

    # synchronize all processes on different GPUs to prevent collapsing
    du.synchronize()