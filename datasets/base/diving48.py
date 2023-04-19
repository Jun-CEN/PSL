import os
import random
import torch
import torch.utils.data
import utils.logging as logging

import time
import oss2 as oss

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from datasets.utils.transformations import (
    ColorJitter, CustomResizedCropVideo, 
    AutoResizedCropVideo
)

from datasets.base.base_dataset import BaseVideoDataset

import utils.bucket as bu

from datasets.base.builder import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Diving48(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Diving48, self).__init__(cfg, split) 
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
    
    def _get_dataset_list_name(self):
        """
            Returns:
                dataset_list_name (string)
        """
        name = "diving48_{}.txt".format(
            "train" if self.split == "train" else "test",
        )
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
        video_path, class_, = self._samples[index].strip().split(" ")
        class_ = int(class_)
        video_path = os.path.join(self.data_root_dir, video_path)
        sample_info = {
            "path": video_path,
            "supervised_label": class_,
        }
        return sample_info
    
    def _config_transform(self):
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
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