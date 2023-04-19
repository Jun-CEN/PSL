import os
import random
import torch
import torch.nn as nn
import torch.utils.data
import utils.logging as logging
import torch.utils.dlpack as dlpack

import time
import json
import oss2 as oss
import traceback

import decord
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge('native')

import numpy as np
from datasets.base.base_dataset import BaseVideoDataset

import utils.bucket as bu

from datasets.base.builder import DATASET_REGISTRY

from datasets.utils.transformations import AutoResizedCropVideo
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Charadesvideo(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Charadesvideo, self).__init__(cfg, split) 

        self.decode = self._sequentially_decode_video
        # self._initialize_text_loader()
    
    def _get_dataset_list_name(self):
        """
            Returns:
                dataset_list_name (string)
        """
        # name = "valid_videos.txt"
        name = "video_list.txt"
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
            Input: 
                index (int): video index
            Returns:
                sample_info (dict): contains different informations to be used later
                    Things that must be included are:
                    "video_path" indicating the video's path w.r.t. index
                    "supervised_label" indicating the class of the video 
        """
        # video_name = self._samples[index].strip()
        # video_path = os.path.join(self.data_root_dir, video_name + ".mp4")
        # sample_info = {
        #     "name": video_name,
        #     "path": video_path,
        #     "supervised_label": 0
        # }
        video_name, video_id, _, _ = self._samples[index].strip().split(',')
        video_id = int(video_id)
        video_path = os.path.join(self.data_root_dir, video_name)
        sample_info = {
            "name": video_name,
            "path": video_path,
            "id": video_id
        }
        return sample_info

    def _sequentially_decode_video(self, sample_info, index, num_clips_per_video=1):
        # ------------------------ decode video ------------------------
        vid_path = sample_info["path"]
        retries = 1000
        for retry in range(retries):
            vr, tmp_file_vid, success =  self._read_video(vid_path, index)
            if success:
                break

        if not success:
            logger.info("Failed. ID: {}".format(index))
            return vr, tmp_file_vid, success

        start_idx = sample_info["id"] * self.cfg.LINEAR_EVALUATION.GROUNDING_CLIP_INTERVAL
        clip_length = self._num_frames * self._sampling_rate
        end_idx = start_idx + clip_length - 1
        list_ = torch.linspace(start_idx, end_idx, self._num_frames)
        list_ = torch.clamp(list_, 0, len(vr)-1).long()
        frames = dlpack.from_dlpack(vr.get_batch(list_).to_dlpack()).clone()

        frames = {
            "video": frames,
            "name": sample_info["name"],
            # "list": torch.stack(list_),
            # "original": frames
        }

        del vr

        file_to_remove = []
        if vid_path[:3] == 'oss':
            file_to_remove += tmp_file_vid
        
        return frames, file_to_remove, True

    # def _generate_lists(self, vid_length, clip_interval):
    #     clip_length = self._num_frames * self._sampling_rate * self.cfg.DATA.FPS / self.cfg.DATA.TARGET_FPS
    #     # max_idx = max(vid_length-clip_length, 0)
    #     indexes = []
    #     for start_idx in range(0, int(vid_length-clip_length), clip_interval):
    #         end_idx = start_idx + clip_length - 1
    #         index = torch.linspace(start_idx, end_idx, self._num_frames)
    #         index = torch.clamp(index, 0, vid_length-1).long()
    #         indexes.append(index)
    #     return indexes

    def _config_transform(self):
        self.transform = None
        self.resize_video = AutoResizedCropVideo(
                size=self.cfg.DATA.TEST_CROP_SIZE,
                scale=[
                    1.0, 
                    1.0
                ],
                mode=self.cfg.TEST.SPATIAL_CROPS,
            )
        std_transform_list = [
            transforms.ToTensorVideo(),
            self.resize_video,
            transforms.NormalizeVideo(
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
                inplace=True
            ),
        ]
        self.transform = Compose(std_transform_list)
    
    def _get_caption(self, caption_path):
        for tmp in range(10):
            try:
                if self.caption_download:
                    tmp_file = os.path.join(self.local_caption_root, caption_path.split('/')[-1])
                    if not os.path.exists(tmp_file):
                        tmp_file = self._get_object_to_file(caption_path, tmp_file)
                else:
                    tmp_file = self._get_object_to_file(caption_path, None, read_from_buffer=True)
                caption = np.load(tmp_file, allow_pickle=True).item()
                break
            except:
                if tmp == 9:
                    return caption, tmp_file, False
        return caption, tmp_file, True

    def _get_sentence_idx(self, text_idx, max_text_idx, num_sentences):
        left = text_idx - num_sentences // 2
        right = text_idx + num_sentences // 2
        if left < 0:
            right += abs(left)
            left += abs(left)
        return torch.linspace(left, right, num_sentences, dtype=torch.long).clamp_(0, max_text_idx)
    
    def _join_text(self, text_list):
        all_text = ""
        for word in text_list:
            all_text += word
            all_text += " "
        return [all_text[:-1]]


@DATASET_REGISTRY.register()
class Charadestext(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Charadestext, self).__init__(cfg, split) 

        self.decode = self._decode_text
        self._initialize_text_loader()
    
    def _get_dataset_list_name(self):
        """
            Returns:
                dataset_list_name (string)
        """
        # name = "valid_videos.txt"
        name = "charades_text_all.txt"
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
            Input: 
                index (int): video index
            Returns:
                sample_info (dict): contains different informations to be used later
                    Things that must be included are:
                    "video_path" indicating the video's path w.r.t. index
                    "supervised_label" indicating the class of the video 
        """
        # video_name = self._samples[index].strip()
        # video_path = os.path.join(self.data_root_dir, video_name + ".mp4")
        # sample_info = {
        #     "name": video_name,
        #     "path": video_path,
        #     "supervised_label": 0
        # }
        video_name = self._samples[index].strip().split(' ')[0]
        text = self._samples[index].strip().split("##")[1]
        sample_info = {
            "name": video_name,
            "text": text
        }
        return sample_info

    def _decode_text(self, sample_info, index, num_clips_per_video=1):
        # ------------------------ decode video ------------------------

        name = sample_info["name"]
        text = sample_info["text"]

        word_ids = self.words_to_ids([text])
        word_ids_existence_map = 1-(word_ids==0)*1
        word_embd = self.word_embd(word_ids)

        frames = {
            "text": word_embd,
            "sentence": text,
            "name": sample_info["name"],
            "text_validity": word_ids_existence_map
        }

        file_to_remove=[]
        
        return frames, file_to_remove, True

    # def _generate_lists(self, vid_length, clip_interval):
    #     clip_length = self._num_frames * self._sampling_rate * self.cfg.DATA.FPS / self.cfg.DATA.TARGET_FPS
    #     # max_idx = max(vid_length-clip_length, 0)
    #     indexes = []
    #     for start_idx in range(0, int(vid_length-clip_length), clip_interval):
    #         end_idx = start_idx + clip_length - 1
    #         index = torch.linspace(start_idx, end_idx, self._num_frames)
    #         index = torch.clamp(index, 0, vid_length-1).long()
    #         indexes.append(index)
    #     return indexes

    def _config_transform(self):
        self.transform = None
        self.resize_video = AutoResizedCropVideo(
                size=self.cfg.DATA.TEST_CROP_SIZE,
                scale=[
                    1.0, 
                    1.0
                ],
                mode=self.cfg.TEST.SPATIAL_CROPS,
            )
        std_transform_list = [
            transforms.ToTensorVideo(),
            self.resize_video,
            transforms.NormalizeVideo(
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
                inplace=True
            ),
        ]
        self.transform = Compose(std_transform_list)
    
    def _get_caption(self, caption_path):
        for tmp in range(10):
            try:
                if self.caption_download:
                    tmp_file = os.path.join(self.local_caption_root, caption_path.split('/')[-1])
                    if not os.path.exists(tmp_file):
                        tmp_file = self._get_object_to_file(caption_path, tmp_file)
                else:
                    tmp_file = self._get_object_to_file(caption_path, None, read_from_buffer=True)
                caption = np.load(tmp_file, allow_pickle=True).item()
                break
            except:
                if tmp == 9:
                    return caption, tmp_file, False
        return caption, tmp_file, True

    def _get_sentence_idx(self, text_idx, max_text_idx, num_sentences):
        left = text_idx - num_sentences // 2
        right = text_idx + num_sentences // 2
        if left < 0:
            right += abs(left)
            left += abs(left)
        return torch.linspace(left, right, num_sentences, dtype=torch.long).clamp_(0, max_text_idx)
    
    def _join_text(self, text_list):
        all_text = ""
        for word in text_list:
            all_text += word
            all_text += " "
        return [all_text[:-1]]
