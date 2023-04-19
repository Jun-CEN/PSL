import os
import torch
import utils.checkpoint as cu
import utils.distributed as du

def _save_video_text_retrieval_set_statics(video_features, text_features, video_name_id, cfg):
    # Save retrieval statics only from the master process.
    path_to_job = cfg.OUTPUT_DIR

    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) and not cfg.PAI:
        return
    
    if not os.path.exists(cu.get_checkpoint_dir(path_to_job)):
        os.mkdir(cu.get_checkpoint_dir(path_to_job))

    statistics = {
        "video_features": video_features,
        "text_features": text_features,
        "video_name_id": video_name_id
    }

    path_to_statistics = os.path.join(path_to_job, "video-text-features.pyth")
    with open(path_to_statistics, "wb") as f:
        torch.save(statistics, f)

    return path_to_statistics

def _load_video_features(cfg, split="linear_train_cls"):
    assert split in ["linear_train_cls", "linear_train_ret", "linear_test_cls", "linear_test_ret"], "Feature saving only configured for 'linear_train_cls' and 'linear_test_cls' sets."
    if "train" in split:
        crop_size = cfg.DATA.TRAIN_CROP_SIZE
    elif "test" in split:
        crop_size = cfg.DATA.TEST_CROP_SIZE

    path_to_features = os.path.join(cfg.OUTPUT_DIR, "features_{}_{}x{}.pyth".format(split, cfg.DATA.NUM_INPUT_FRAMES, crop_size))
    with open(path_to_features, "rb") as f:
        statistics = torch.load(f, map_location="cpu")
    retrieval_set_features = statistics["features"]
    retrieval_set_labels = statistics["labels"]
    return retrieval_set_features, retrieval_set_labels

def feature_exists(cfg, split="linear_train_cls"):
    assert split in ["linear_train_cls", "linear_train_ret", "linear_test_cls", "linear_test_ret"], "Feature saving only configured for 'linear_train_cls' and 'linear_test_cls' sets."
    if "train" in split:
        crop_size = cfg.DATA.TRAIN_CROP_SIZE
    elif "test" in split:
        crop_size = cfg.DATA.TEST_CROP_SIZE
    path_to_features = os.path.join(cfg.OUTPUT_DIR, "features_{}_{}x{}.pyth".format(split, cfg.DATA.NUM_INPUT_FRAMES, crop_size))
    if os.path.exists(path_to_features):
        return True
    return False

def load_features(cfg, split="linear_train_cls"):
    if split in ["linear_train_cls", "linear_train_ret", "linear_test_cls", "linear_test_ret"]:
        return _load_video_features(cfg, split)
    else:
        raise NotImplementedError

def save_video_features(cfg, split="linear_train_cls", features=None, labels=None):
    # Save features only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) and not cfg.PAI:
        return
    assert split in ["linear_train_cls", "linear_train_ret", "linear_test_cls", "linear_test_ret"], "Feature saving only configured for 'linear_train_cls' and 'linear_test_cls' sets."
    
    if "train" in split:
        crop_size = cfg.DATA.TRAIN_CROP_SIZE
    elif "test" in split:
        crop_size = cfg.DATA.TEST_CROP_SIZE

    save_path = cfg.OUTPUT_DIR
    if not os.path.exists(cu.get_checkpoint_dir(save_path)):
        os.mkdir(cu.get_checkpoint_dir(save_path))
    
    statistics = {
        "features": features,
        "labels": labels
    }

    path_to_features = os.path.join(save_path, "features_{}_{}x{}.pyth".format(split, cfg.DATA.NUM_INPUT_FRAMES, crop_size))

    with open(path_to_features, "wb") as f:
        torch.save(statistics, f)

    return path_to_features
    