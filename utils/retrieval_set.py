import os
import torch
import utils.checkpoint as cu
import utils.distributed as du

def save_video_text_retrieval_set_statics(video_features, text_features, video_name_id, cfg):
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



def save_retrieval_set_statics(retrieval_set_features, retrieval_set_labels, cfg):
    # Save retrieval statics only from the master process.
    path_to_job = cfg.OUTPUT_DIR

    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) and not cfg.PAI:
        return
    
    if not os.path.exists(cu.get_checkpoint_dir(path_to_job)):
        os.mkdir(cu.get_checkpoint_dir(path_to_job))

    statistics = {
        "features": retrieval_set_features,
        "labels": retrieval_set_labels
    }

    path_to_statistics = os.path.join(path_to_job, "features.pyth")
    with open(path_to_statistics, "wb") as f:
        torch.save(statistics, f)

    return path_to_statistics

def load_retrieval_set_statics(cfg):
    path_to_statistics = os.path.join(cfg.OUTPUT_DIR, "features.pyth")
    with open(path_to_statistics, "rb") as f:
        statistics = torch.load(f, map_location="cpu")
    retrieval_set_features = statistics["features"]
    retrieval_set_labels = statistics["labels"]
    return retrieval_set_features, retrieval_set_labels

def statistics_file_exists(cfg):
    path_to_statistics = os.path.join(cfg.OUTPUT_DIR, "features.pyth")
    if os.path.exists(path_to_statistics):
        return True
    return False

def save_test_set_statistics(test_set_features, test_set_labels, cfg):
    path_to_job = cfg.OUTPUT_DIR

    # Save retrieval statics only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) and not cfg.PAI:
        return
    
    if not os.path.exists(cu.get_checkpoint_dir(path_to_job)):
        os.mkdir(cu.get_checkpoint_dir(path_to_job))

    statistics = {
        "features": test_set_features,
        "labels": test_set_labels
    }

    path_to_statistics = os.path.join(path_to_job, "features_test.pyth")
    with open(path_to_statistics, "wb") as f:
        torch.save(statistics, f)

    return path_to_statistics

def load_test_set_statics(cfg):
    path_to_statistics = os.path.join(cfg.OUTPUT_DIR, "features_test.pyth")
    with open(path_to_statistics, "rb") as f:
        statistics = torch.load(f, map_location="cpu")
    retrieval_set_features = statistics["features"]
    retrieval_set_labels = statistics["labels"]
    return retrieval_set_features, retrieval_set_labels

def test_feature_exists(cfg):
    path_to_statistics = os.path.join(cfg.OUTPUT_DIR, "features_test.pyth")
    if os.path.exists(path_to_statistics):
        return True
    return False