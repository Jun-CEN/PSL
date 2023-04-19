#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Entry file for training, evaluating and testing a video model."""

import os
import sys
import time
# sys.path.append(os.path.abspath(os.curdir))
# sys.path.append(os.path.abspath(os.curdir)+"/PSL")

sys.path = [os.path.abspath(os.curdir)] + sys.path
sys.path = [os.path.abspath(os.curdir)+"/PSL"] + sys.path

from utils.multiprocessing import launch_task

from test_net import test
from test_net_open import test_open
from train_net import train
from train_mm_retrieval import train_mm_retrieval
from test_mm_retrieval import test_mm_retrieval
from vis_net import vis
from linear_evaluation import linear_classification, linear_retrieval
from grounding import grounding_feature_extraction
from submission_test_net import submission_test

from utils.config import Config

def _prepare_data(cfg):
    if cfg.PRE_DOWNLOAD.ENABLE:
        print("OSS configurating...")
        os.system('chmod +777 ./PSL/ossutil64')
        os.system('./PSL/ossutil64 config -e {} -i {} -k {}'.format(
            cfg.OSS.ENDPOINT, cfg.OSS.KEY, cfg.OSS.SECRET
        ))
        os.system('./PSL/ossutil64 cp -r {} ./data -j 64'.format(
            cfg.DATA.DATA_ROOT_DIR
        ))
        cfg.DATA.DATA_ROOT_DIR = './data'

    if cfg.TASK_TYPE in ['mm_retrieval']:
        train_func = train_mm_retrieval
        test_func = test_mm_retrieval
    elif cfg.TASK_TYPE in ['classification']:
        train_func = train
        test_func = test
        test_func_open = test_open
    elif cfg.TASK_TYPE in ["submission"]:
        cfg.TRAIN.ENABLE = False
        cfg.TEST.ENABLE = False
        train_func = None
        test_func = None
        submission_func = submission_test
    else:
        raise ValueError("unknown TASK_TYPE {}".format(cfg.TASK_TYPE))
    
    run_list = []
    if cfg.TRAIN.ENABLE:
        # Training process is performed by the entry function defined above.
        run_list.append([cfg.deep_copy(), train_func])
    
    if cfg.TEST.ENABLE:
        # Test is performed by the entry function defined above.
        # run_list.append([cfg.deep_copy(), test_func])
        if cfg.TEST.AUTOMATIC_MULTI_SCALE_TEST:
            """
            By default, test_func performs single view test. 
            AUTOMATIC_MULTI_SCALE_TEST automatically performs multi-view test after the single view test.
            """
            cfg.LOG_MODEL_INFO = False
            cfg.LOG_CONFIG_INFO = False

            cfg.TEST.NUM_ENSEMBLE_VIEWS = 10
            cfg.TEST.NUM_SPATIAL_CROPS = 1

            if "kinetics" in cfg.TEST.DATASET or "epickitchen" in cfg.TEST.DATASET:
                cfg.TEST.NUM_SPATIAL_CROPS = 3
                cfg.DATA.SAMPLING_MODE = "interval_based"
                if "Transformer" in cfg.VIDEO.BACKBONE.META_ARCH:
                    cfg.TEST.NUM_ENSEMBLE_VIEWS = 4
            if "UCF" in cfg.TEST.DATASET:
                cfg.TEST.NUM_ENSEMBLE_VIEWS = 2
                cfg.TEST.NUM_SPATIAL_CROPS = 3
                cfg.DATA.SAMPLING_MODE = "interval_based"
            if "imagenet" in cfg.TEST.DATASET and not cfg.PRETRAIN.ENABLE:
                cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
                cfg.TEST.NUM_SPATIAL_CROPS = 3
            if "ssv2" in cfg.TEST.DATASET:
                cfg.TEST.NUM_ENSEMBLE_VIEWS = 2
                cfg.TEST.NUM_SPATIAL_CROPS = 3
            cfg.TEST.LOG_FILE = "val_{}clipsx{}crops.log".format(
                cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
            )
            run_list.append([cfg.deep_copy(), test_func])

    if hasattr(cfg.TEST, "ENABLE_OPEN") and cfg.TEST.ENABLE_OPEN:
        # Test is performed by the entry function defined above.
        run_list.append([cfg.deep_copy(), test_func_open])

    if cfg.SUBMISSION.ENABLE:
        # currently only supports epic kitchen submission
        cfg.LOG_MODEL_INFO = False
        cfg.TEST.SAVE_RESULTS_PATH = "test.json"
        cfg.TEST.NUM_ENSEMBLE_VIEWS = 10
        cfg.TEST.NUM_SPATIAL_CROPS = 3

        cfg.TEST.LOG_FILE = "test_{}clipsx{}crops.log".format(
            cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
        )
        run_list.append([cfg.deep_copy(), submission_func])

    if cfg.VISUALIZATION.ENABLE:
        run_list.append([cfg.deep_copy(), vis])
  
    return run_list

def main():
    """
    Entry function for spawning all the function processes. 
    """
    cfg = Config(load=True)

    # get the list of configs and functions for running
    run_list = _prepare_data(cfg)

    for run in run_list:
        launch_task(cfg=run[0], init_method=run[0].get_args().init_method, func=run[1])

    print("Finish running with config: {}".format(cfg.args.cfg_file))


if __name__ == "__main__":
    main()
