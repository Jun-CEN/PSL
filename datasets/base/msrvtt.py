
import os
import utils.bucket as bu
import torch.utils.dlpack as dlpack
from datasets.base.builder import DATASET_REGISTRY
from datasets.base.base_dataset import BaseVideoDataset
import json
import torch
import utils.logging as logging
import pandas as pd
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from datasets.utils.transformations import (
    ColorJitter, CustomResizedCropVideo, 
    AutoResizedCropVideo,
    RandomResizedCropVideo
)
import numpy as np

logger = logging.get_logger(__name__)
@DATASET_REGISTRY.register()
class Msrvtt(BaseVideoDataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            cfg,
            split
    ):
        """
        Args:
        """
        super(Msrvtt, self).__init__(cfg, split)
        self.decode = self._decode_video_text
        self._initialize_text_loader()
        self.construct_video_names()
    
    def _get_dataset_list_name(self):
        """
            Returns:
                dataset_list_name (string)
        """

        if self.split in ['train', 'val']:
            name = "{}_meta.txt".format(self.split)
        elif self.split in ['test']:
            name = "{}_meta.txt".format("val")
        else:
            raise ValueError("Unkown MM_RETRIEVAL SPLIT {} .".format(split))
        
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
        video_id, sentence = self._samples[index].strip().split(",")
        video_path = os.path.join(self.data_root_dir, video_id)
        sample_info = {
            "path": video_path,
            "id": video_id,
            "sentence": sentence,
            "supervised_label": 0
        }
        return sample_info

    def _config_transform(self):
        self.transform = None
        if self.split == 'train':
            std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomResizedCropVideo(
                    size=self.cfg.DATA.TRAIN_CROP_SIZE,
                    scale=[
                        self.cfg.DATA.TRAIN_JITTER_SCALES[0]*self.cfg.DATA.TRAIN_JITTER_SCALES[0]/256.0/340.0,
                        self.cfg.DATA.TRAIN_JITTER_SCALES[1]*self.cfg.DATA.TRAIN_JITTER_SCALES[1]/256.0/340.0
                    ],
                    ratio=self.cfg.AUGMENTATION.RATIO
                ),
                transforms.RandomHorizontalFlipVideo()
            ]
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        ),
                )
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
            ]
            self.transform = Compose(std_transform_list)
        elif self.split == 'val' or self.split == 'test':
            self.resize_video = AutoResizedCropVideo(
                    size=self.cfg.DATA.TEST_CROP_SIZE,
                    scale=[
                        self.cfg.DATA.TEST_SCALE/256.0,
                        self.cfg.DATA.TEST_SCALE/256.0  
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

    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, vid_fps, clip_idx, num_clips, num_frames, interval)

    def _decode_video_text(self, sample_info, index, num_clips_per_video=1):
        """
            Decode video and extract the embedding of the related text in the given captions.
            By default, the param 'num_clips_per_video' has no use here.
        """
        # ------------------------ decode text ------------------------
        sentence = np.array([sample_info['sentence']])
        word_ids = self.words_to_ids(sentence)
        word_ids_existence_map = 1-(word_ids==0)*1
        word_embd = self.word_embd(word_ids)

        # ------------------------ decode video ------------------------
        vid_path = sample_info['path']
        vr, tmp_file_vid, success =  self._read_video(vid_path, index)

        if not success:
            return vr, tmp_file_vid, success

        if self.split == "train":
            clip_idx = -1
            self.spatial_idx = -1
        elif self.split == "val":
            clip_idx = -1
            self.spatial_idx = 0
        elif self.split in ["test", 'retrieval_set']:
            clip_idx = self._spatial_temporal_index[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                self.spatial_idx = 0
            else:
                self.spatial_idx = self._spatial_temporal_index[index] % self.cfg.TEST.NUM_SPATIAL_CROPS

        list_ = self._get_video_frames_list(
            len(vr),
            clip_idx,
            random_sample=True if self.split=="train" else False 
        )
        frames = None
        frames = dlpack.from_dlpack(vr.get_batch(list_).to_dlpack()).clone()

        frames = {"video": frames, "text_embedding": word_embd, "text_validity": word_ids_existence_map}

        del vr

        file_to_remove = []
        if vid_path[:3] == 'oss':
            file_to_remove += tmp_file_vid
        
        return frames, file_to_remove, True

    def construct_video_names(self):
        self._videos_name_list = []
        for index in range(len(self)):
            video_name = self._get_sample_info(index)['id']
            if video_name not in self._videos_name_list:
                self._videos_name_list.append(video_name)
   
    def get_videos_name(self, index_tensor):
        video_list = []
        for idx in index_tensor:
            video_name = self._get_sample_info(idx)['id']
            video_list.append(self._videos_name_list.index(video_name))
        return torch.Tensor(video_list).long()
