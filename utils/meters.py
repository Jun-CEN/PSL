#!/usr/bin/env python3
# Copyright (C) Alibaba Group H volding Limited. 

"""
Meters.
Modifed from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/meters.py.
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from utils.timer import Timer
from sklearn.metrics import average_precision_score

import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.distributed as du

logger = logging.get_logger(__name__)

class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        cfg,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.cfg = cfg
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))

        self.video_labels = (
            torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.clip_indices = torch.linspace(0, num_videos-1, num_videos).long()
        self.model_ema_enabled = False
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        print(self.cfg.LOG_PERIOD)
        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter" if not self.model_ema_enabled else "ema_test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            # "{}: {}".format(i, k)
                            # for i, k in enumerate(self.clip_count.tolist())
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final" if not self.model_ema_enabled else "ema_test_final"}
        num_topks_correct = metrics.topks_correct(
            self.video_preds, self.video_labels, ks
        )
        topks = [
            (x / self.video_preds.size(0)) * 100.0
            for x in num_topks_correct
        ]
        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )
        logging.log_json_stats(stats)
    
    def set_model_ema_enabled(self, model_ema_enabled):
        self.model_ema_enabled = model_ema_enabled

class EpicKitchenMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.

    For the EpicKitchenMeter specifically, it caters to the need of the EpicKitchens
    dataset, where both verbs and nouns are predicted before actions are predicted using
    those predictions.
    """

    def __init__(
        self,
        cfg,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            cfg (Config): the global config object.
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.cfg = cfg
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.num_videos = num_videos
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method

        assert self.ensemble_method in ["sum", "max"], f"Ensemble Method {ensemble_method} is not supported"
        
        if cfg.DATA.MULTI_LABEL or not hasattr(cfg.DATA, "TRAIN_VERSION"):
            # Initialize tensors.
            self.video_preds = {
                "verb_class": torch.zeros((num_videos, self.num_clips, num_cls[0])),
                "noun_class": torch.zeros((num_videos, self.num_clips, num_cls[1])),
                "action_class_ind_pred": torch.zeros((num_videos, self.num_clips, num_cls[0]*num_cls[1]))
            }

            self.video_labels = {
                "verb_class": torch.zeros((num_videos)),  # verb
                "noun_class": torch.zeros((num_videos)),  # noun
                "action_class_ind_pred": torch.zeros((num_videos)),
            }
            self.update_stats = self.update_stats_multi_label
            self.finalize_metrics = self.finalize_metrics_multi_label
        elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION in ["only_train_verb", "only_train_noun"]:
            self.video_preds = torch.zeros((num_videos, self.num_clips, num_cls))
            self.video_labels = torch.zeros((num_videos))
            self.update_stats = self.update_stats_separate_label
            self.finalize_metrics = self.finalize_metrics_separate_label
        else: raise NotImplementedError
        self.video_names = {i: "" for i in range(num_videos)}
        self.clip_count = torch.zeros((num_videos)).long()
        self.clip_indices = torch.linspace(0, num_videos-1, num_videos).long()
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        if isinstance(self.video_preds, dict):
            for k, v in self.video_preds.items():
                v.zero_()
            for k, v in self.video_labels.items():
                v.zero_()
        else:
            self.video_preds.zero_()
            self.video_labels.zero_()

    def update_stats_separate_label(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble, for separate verb and noun training.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            view_id = int(clip_ids[ind]) % self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )

            self.video_labels[vid_id] = labels[ind]

            self.video_preds[vid_id][view_id] = preds[ind]

            self.clip_count[vid_id] += 1

    def update_stats_multi_label(self, preds_verb, preds_noun, labels_verb, labels_noun, clip_ids, names=[]):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble, for joint verb and noun training.
        Args:
            preds_verb (tensor): verb predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls[0]).
            preds_noun (tensor): noun predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls[1]).
            labels_verb (tensor): the corresponding verb labels of the current batch.
                Dimension is N.
            labels_noun (tensor): the corresponding noun labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
            names (list): list of video names.
        """
        for ind in range(preds_verb.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            view_id = int(clip_ids[ind]) % self.num_clips
            if self.video_labels["verb_class"][vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels["verb_class"][vid_id].type(torch.FloatTensor),
                    labels_verb[ind].type(torch.FloatTensor),
                )
                assert torch.equal(
                    self.video_labels["noun_class"][vid_id].type(torch.FloatTensor),
                    labels_noun[ind].type(torch.FloatTensor),
                )
            if len(names) > 0:
                if self.video_names[vid_id] != "":
                    assert self.video_names[vid_id] == names[ind], \
                        f"For {vid_id}, its name {self.video_names[vid_id]} should be equal to {names[ind]}"
                else:
                    self.video_names[vid_id] = names[ind]

            self.video_labels["verb_class"][vid_id] = labels_verb[ind]
            self.video_labels["noun_class"][vid_id] = labels_noun[ind]
            self.video_labels["action_class_ind_pred"][vid_id] = labels_verb[ind] * preds_noun.shape[1] + labels_noun[ind]

            self.video_preds["verb_class"][vid_id][view_id] = preds_verb[ind]
            self.video_preds["noun_class"][vid_id][view_id] = preds_noun[ind]
            self.video_preds["action_class_ind_pred"][vid_id][view_id] = (preds_verb[ind].unsqueeze(-1) * preds_noun[ind].unsqueeze(-2)).reshape(-1)

            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter" if not self.model_ema_enabled else "ema_test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics_multi_label(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics for joint verb and 
        noun training.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            # "{}: {}".format(i, k)
                            # for i, k in enumerate(self.clip_count.tolist())
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final" if not self.model_ema_enabled else "ema_test_final"}
        video_preds = {}
        if self.ensemble_method == "sum":
            video_preds["verb_class"] = self.video_preds["verb_class"].sum(1)
            video_preds["noun_class"] = self.video_preds["noun_class"].sum(1)
            video_preds["action_class_ind_pred"] = self.video_preds["action_class_ind_pred"].sum(1)
        elif self.ensemble_method == "max":
            video_preds["verb_class"] = self.video_preds["verb_class"].max(1)[0]
            video_preds["noun_class"] = self.video_preds["noun_class"].max(1)[0]
            video_preds["action_class_ind_pred"] = self.video_preds["action_class_ind_pred"].max(1)[0]
        num_topks_correct, b = metrics.joint_topks_correct(
            video_preds, self.video_labels, ks
        )
        for name, v in num_topks_correct.items():
            topks = [ (x / b) * 100.0 for x in v ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                stats["top_{}_acc_{}".format(name, k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        logging.log_json_stats(stats)

    def finalize_metrics_separate_label(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics, for separate verb 
        and noun training.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final" if not self.model_ema_enabled else "ema_test_final"}
        if self.ensemble_method == "sum":
            video_preds = self.video_preds.sum(1)
        elif self.ensemble_method == "max":
            video_preds = self.video_preds.max(1)[0]
        num_topks_correct = metrics.topks_correct(
            video_preds, self.video_labels, ks
        )
        topks = [
            (x / self.video_preds.size(0)) * 100.0
            for x in num_topks_correct
        ]
        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )
        logging.log_json_stats(stats)

    def set_model_ema_enabled(self, model_ema_enabled):
        """
        Whether the meter logs for ema models or not.
        Args:
            model_ema_enabled (bool): indicator of whether ema model 
                is enabled.
        """
        self.model_ema_enabled = model_ema_enabled

    def get_video_preds(self):
        """
        Returns the saved video predictions.
        """
        video_preds = {}
        if self.ensemble_method == "sum":
            video_preds["verb_class"] = self.video_preds["verb_class"].sum(1)
            video_preds["noun_class"] = self.video_preds["noun_class"].sum(1)
            video_preds["action_class_ind_pred"] = self.video_preds["action_class_ind_pred"].sum(1)
        elif self.ensemble_method == "max":
            video_preds["verb_class"] = self.video_preds["verb_class"].max(1)[0]
            video_preds["noun_class"] = self.video_preds["noun_class"].max(1)[0]
            video_preds["action_class_ind_pred"] = self.video_preds["action_class_ind_pred"].max(1)[0]
        return video_preds

class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size=10):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (Config): the global config object.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.OPTIMIZER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.opts = defaultdict(ScalarMeter)

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.opts = defaultdict(ScalarMeter)
        

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size, **kwargs):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.opts[k].add_value(v)
        
        if self._cfg.MODEL.NAME == 'Contrastive_CE_Model' or self._cfg.MODEL.NAME == 'PrototypeModel':
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size

        elif not self._cfg.PRETRAIN.ENABLE and not self._cfg.MM_RETRIEVAL.ENABLE:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size
    def update_custom_stats(self, stats):
        """
        Update stats using custom keys.
        Args:
            stats (dict): additional stats to be updated.
        """
        for k,v in stats.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.opts[k].add_value(v)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.OPTIMIZER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            # "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        for k,v in self.opts.items():
            stats[k] = v.get_win_median()
        if self._cfg.MODEL.NAME == 'Contrastive_CE_Model' or self._cfg.MODEL.NAME == 'PrototypeModel':
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        elif not self._cfg.PRETRAIN.ENABLE and not self._cfg.MM_RETRIEVAL.ENABLE:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.OPTIMIZER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }
        for k,v in self.opts.items():
            stats[k] = v.get_global_avg()
        if self._cfg.MODEL.NAME == 'Contrastive_CE_Model' or self._cfg.MODEL.NAME == 'PrototypeModel':
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        elif not self._cfg.PRETRAIN.ENABLE and not self._cfg.MM_RETRIEVAL.ENABLE:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (Config): the global config object.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.model_ema_enabled = False
        self.opts = defaultdict(ScalarMeter)

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.opts = defaultdict(ScalarMeter)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size, **kwargs):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.opts[k].add_value(v)
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size
    
    def update_custom_stats(self, stats):
        """
        Update stats using custom keys.
        Args:
            stats (dict): additional stats to be updated.
        """
        for k,v in stats.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.opts[k].add_value(v)
    
    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter" if not self.model_ema_enabled else "ema_val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.OPTIMIZER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        for k,v in self.opts.items():
            stats[k] = v.get_win_median()
        stats["top1_err"] = self.mb_top1_err.get_win_median()
        stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch" if not self.model_ema_enabled else "ema_val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.OPTIMIZER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }
        for k,v in self.opts.items():
            if "top1_err" in k or "top5_err" in k:
                stats[k] = v.get_win_median()
            else:
                stats[k] = v.get_global_avg()
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        self.min_top5_err = min(self.min_top5_err, top5_err)

        stats["top1_err"] = top1_err
        stats["top5_err"] = top5_err
        stats["min_top1_err"] = self.min_top1_err
        stats["min_top5_err"] = self.min_top5_err

        logging.log_json_stats(stats)
    
    def set_model_ema_enabled(self, model_ema_enabled):
        self.model_ema_enabled = model_ema_enabled

class LinearTestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        cfg,
        num_videos,
        num_clips,
        overall_iters,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.cfg = cfg
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.num_features = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES

        # Initialize tensors.
        self.video_labels = (torch.zeros((num_videos)).long())
        self.clip_count = torch.zeros((num_videos)).long()
        self.clip_indices = torch.linspace(0, num_videos-1, num_videos).long()

        self.video_retrieved_top20_indexes = torch.zeros(num_videos, num_clips, 20).long()
        self.video_retrieved_top20_labels = torch.zeros(num_videos, num_clips, 20).long()

        self.video_features = torch.zeros(num_videos, num_clips, self.num_features)

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_labels.zero_()
        self.video_retrieved_top20_indexes.zero_()
        self.video_retrieved_top20_labels.zero_()

    def update_stats(self, features, similarity, labels, clip_ids, retrieval_set_labels):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(similarity.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            self.video_features[vid_id][self.clip_count[vid_id]] = features[ind]
            self.video_retrieved_top20_indexes[vid_id][self.clip_count[vid_id]] = similarity[ind].topk(20)[1]
            self.video_retrieved_top20_labels[vid_id][self.clip_count[vid_id]] = retrieval_set_labels[similarity[ind].topk(20)[1]]
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "retrieval_test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5, 10, 20)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "retrieval_test_final"}
        
        # compute topks
        num_topks_correct = metrics.topk_retrieved_correct_video_level_loose(
            self.video_retrieved_top20_labels, self.video_labels, ks
        )
        topks_video_level_loose = [
            (x.sum().float() / self.video_retrieved_top20_labels.size(0)) * 100.0
            for x in num_topks_correct
        ]
        assert len({len(ks), len(topks_video_level_loose)}) == 1
        for k, topk in zip(ks, topks_video_level_loose):
            stats["video_top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )

        num_topks_correct = metrics.topk_retrieved_correct_clip_level_tight(
            self.video_retrieved_top20_labels, self.video_labels, ks
        )
        topks_video_level_loose = [
            (x.sum().float() / self.video_retrieved_top20_labels.size(0) / self.video_retrieved_top20_labels.size(1)) * 100.0
            for x in num_topks_correct
        ]
        assert len({len(ks), len(topks_video_level_loose)}) == 1
        for k, topk in zip(ks, topks_video_level_loose):
            stats["clip_top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )

        # finalize features and labels
        self.video_labels = self.video_labels
        self.video_features = self.video_features.reshape(-1, self.num_features)

        logging.log_json_stats(stats)

    def get_feature_and_labels(self):
        return self.video_features, self.video_labels

class LinearFeatureExtractMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        cfg,
        split,
        num_videos,
        num_clips,
        overall_iters,
        ensemble_method=None,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "None" and "avg".
        """

        self.cfg = cfg
        self.split = split
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method

        self.num_features = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        # Initialize tensors.

        self.video_labels = (torch.zeros((num_videos)).long())
        self.clip_count = torch.zeros((num_videos)).long()
        self.clip_indices = torch.linspace(0, num_videos-1, num_videos).long()

        self.video_features = torch.zeros(num_videos, num_clips, self.num_features)
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_labels.zero_()
        self.video_features.zero_()

    def update_stats(self, features, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(features.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            self.video_features[vid_id][self.clip_count[vid_id]] = features[ind]
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "{}_iter".format(self.split),
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self):
        """
        Calculate and log the final ensembled features.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "{}_final".format(self.split)}
        
        if self.ensemble_method == None:
            if "train" in self.split:
                self.video_labels = self.video_labels.unsqueeze(1).repeat_interleave(self.num_clips, dim=1).view(-1)
            self.video_features = self.video_features.reshape(-1, self.num_features)
        elif self.ensemble_method == "avg":
            self.video_features = self.video_features.mean(1)
        
        logging.log_json_stats(stats)

    def get_feature_and_labels(self):
        return self.video_features, self.video_labels

class RetrievalVideoTextMeter(object):

    def __init__(
        self,
        cfg,
        num_videos,
        num_clips,
        overall_iters,
        ensemble_method=None,
    ):
        self.cfg = cfg
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method
        assert self.ensemble_method in ['mean']
        # if not cfg.MM_RETRIEVAL.MULTI_MODAL.CLUSTER:
        #     self.num_features = cfg.PRETRAIN.CONTRASTIVE.HEAD_OUT_DIM
        # else:
        #     self.num_features = cfg.PRETRAIN.PROTOTYPE.NUM_PROTOTYPES
        # Initialize tensors.

        self.clip_count = torch.zeros((num_videos)).long()
        self.videos_name_id = torch.zeros((num_videos, num_clips)).long()
        self.clip_indices = torch.linspace(0, num_videos-1, num_videos).long()

        self.video_features = torch.zeros(num_videos, num_clips, cfg.VIDEO.HEAD.MLP.OUT_DIM)
        self.text_features = torch.zeros(num_videos, num_clips, cfg.TEXT.HEAD.OUT_DIM)
        self.video_p = torch.zeros(num_videos, num_clips, cfg.VIDEO.HEAD.MLP.OUT_DIM)
        self.text_q = torch.zeros(num_videos, num_clips, cfg.TEXT.HEAD.OUT_DIM)
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_features.zero_()
        self.text_features.zero_()
        self.video_p.zero_()
        self.text_q.zero_()
        self.videos_name_id.fill_(-1)

    def update_stats(self, video_features, text_features, clip_ids, videos_name_id):
        for ind in range(video_features.shape[0]):
            # vid_id = int(clip_ids[ind]) // self.num_clips
            vid_id = videos_name_id[ind]
            self.video_features[vid_id][self.clip_count[vid_id]] = video_features[ind]
            self.text_features[vid_id][self.clip_count[vid_id]] = text_features[ind]
            self.videos_name_id[vid_id][self.clip_count[vid_id]] = videos_name_id[ind]
            self.clip_count[vid_id] += 1


    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "retrieval_train_iter",
            "cur_iter": "{}/{}".format(cur_iter + 1, self.overall_iters),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self):
        """
        Calculate and log the final ensembled features.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(ind, self.clip_count[ind]) for idx, ind in enumerate(self.clip_indices[self.clip_count!=self.num_clips].tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "retrieval_train_final"}
        ks = [1, 5, 10, 20]
        text_features = self.text_features
        video_features = self.video_features
        video_p = self.video_p
        text_q = self.text_q
        video_name_idx = self.videos_name_id[:, 0]
        '''
        reduce_video_features = []
        for i in range(video_name_idx.max() + 1):
            tmpv = video_features[video_name_idx == i, :, :]
            tmpv = tmpv.mean(0)
            reduce_video_features.append(tmpv)
        reduce_video_features = torch.stack(reduce_video_features, dim=0)
        reduce_video_features = reduce_video_features.mean(1)
        '''
        if self.ensemble_method == 'mean':
            video_features = video_features.mean(1)
            text_features = text_features.mean(1)
            video_p = video_p.mean(1)
            text_q = text_q.mean(1)
        else:
            raise ValueError("unknown ensemble_method:{}".format(self.ensemble_method))
        
        label = torch.arange(0, len(video_name_idx))
        t2v_similarity_martix = torch.matmul(text_features[:, :], video_features.transpose(0,1))
        t2vres = metrics.topk_accuracies(t2v_similarity_martix, label, ks=ks)

        v2t_similarity_martix = torch.matmul(video_features[:, :], text_features.transpose(0,1))
        v2tres = metrics.topk_accuracies(v2t_similarity_martix, label, ks=ks)
        t2vkey = "t2v" + ','.join(["R@{}".format(k) for k in ks])
        v2tkey = "v2t" + ','.join(["R@{}".format(k) for k in ks])
        stats[t2vkey] = ','.join(['{:.4f}'.format(t2vres[i].item()) for i in range(len(ks))])
        stats[v2tkey] = ','.join(['{:.4f}'.format(v2tres[i].item()) for i in range(len(ks))])
        logging.log_json_stats(stats)

    def get_feature_and_labels(self):
        return self.video_features, self.text_features, self.videos_name_id

class GroundingFeatureExtractMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        cfg,
        split, 
        num_videos,
        num_clips,
        overall_iters,
        samples,
        ensemble_method=None,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "None" and "avg".
        """

        self.cfg = cfg
        self.split = split
        self.samples = samples
        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method

        self.num_features = cfg.VIDEO.BACKBONE.NUM_OUT_FEATURES
        # Initialize tensors.
        self.clip_count = {}
        self.clip_indices = torch.linspace(0, num_videos-1, num_videos).long()

        self.video_features = []
        self.video_features_mid = []
        self.last_idx = -1
        self.error_videos = []
        self.reset_save = False
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count = {}
        self.video_features = []
        self.video_features_mid = []
        self.last_idx = -1
        self.error_videos = []
        self.reset_save = False

    def update_stats(self, features, clip_ids, names=None, sentences=None):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        if "xv" in features.keys():
            xv = features["xv"]
            xv_mid = features["xv_mid"]
            # logger.info("Batch size: {}, MinID: {}, Name Min ID: {} MaxID: {}, Name Max ID: {}".format(
            #     xv.shape[0],
            #     clip_ids[0],
            #     self.samples[int(clip_ids[0])],
            #     clip_ids[-1],
            #     self.samples[int(clip_ids[-1])]
            # ))
            for ind in range(xv.shape[0]):
                vid_id = int(clip_ids[ind])
                if vid_id == self.last_idx + 1 or self.reset_save:
                    self.reset_save = False
                    if vid_id > 0:
                        name_curr, name_prev = self.samples[vid_id], self.samples[vid_id-1]
                    else:
                        name_curr, name_prev = self.samples[vid_id], self.samples[vid_id]
                    name_curr = name_curr.split(".")[0]
                    name_prev = name_prev.split(".")[0]
                    if self.clip_count == {}:
                        self.clip_count[name_curr] = 0
                    if name_curr != name_prev:
                        if du.is_master_proc():
                            if not os.path.exists("./features_vid_full"):
                                os.makedirs("./features_vid_full")
                                os.makedirs("./features_vid_mid")
                            if not os.path.exists("./features_vid_full/{}.npy".format(name_prev)):
                                np.save("./features_vid_full/{}.npy".format(name_prev), torch.stack(self.video_features))
                                np.save("./features_vid_mid/{}.npy".format(name_prev), torch.stack(self.video_features_mid))
                                logger.info("Saved: {}".format(name_prev))
                        self.video_features = []
                        self.video_features_mid = []
                        self.clip_count[name_curr] = 0
                    self.video_features.append(xv[ind])
                    self.video_features_mid.append(xv_mid[ind])
                    
                    self.clip_count[name_curr] += 1
                    self.last_idx = vid_id
                else: 
                    name_prev = self.samples[vid_id-1]
                    self.error_videos.append(self.samples[vid_id-1])
                    logger.info("Last idx: {}. Cur idx: {}. Error name: {}.".format(
                        self.last_idx, vid_id, name_prev
                    ))
                    self.reset_save = True
        elif "xt" in features.keys():
            xt = features["xt"]
            for ind in range(xt.shape[0]):
                if not os.path.exists("./features_text"):
                    os.makedirs("./features_text")
                sentence = sentences[ind].replace('/', " ")
                file_name = "./features_text/{}_{}".format(names[ind], sentence).replace(" ", "_") + "npy"
                np.save(file_name, xt[ind])


    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "{}_iter".format(self.split),
            "cur_iter": "{}/{}".format(cur_iter + 1, self.overall_iters),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self):
        """
        Calculate and log the final ensembled features.
        """
        stats = {"split": "{}_final".format(self.split)}
        logger.info("ERROR Videos: {}".format(self.error_videos))
        logger.info("Finished uploading for {}.".format(
            self.cfg.TRAIN.CHECKPOINT_FILE_PATH.split('/')[-3] if self.cfg.TRAIN.CHECKPOINT_FILE_PATH[:3] == "oss" else "test_run -u"
        ))
        logging.log_json_stats(stats)

def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap