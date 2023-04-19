import os
import cv2
import random
import torch
import numpy as np
import torch.utils.data
import utils.logging as logging

import time
import json

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch.nn.functional as F
from datasets.utils.transformations import (
    ColorJitter, CustomResizedCropVideo, 
    AutoResizedCropVideo,
    KineticsResizedCrop
)

from datasets.base.base_dataset import BaseVideoDataset

import utils.bucket as bu

from datasets.base.builder import DATASET_REGISTRY

from base64 import b64decode

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Kinetics400flow(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Kinetics400flow, self).__init__(cfg, split) 
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
        self.decode = self._decode_flow
        
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        name = "kinetics400_{}_list.txt".format(
            self.split,
        )
        # name = "./k400_val_list_new.txt"
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                "name": the name of the video
                "path": indicating the target's path w.r.t. the index
                "supervised_label": indicating the class of the target 
        """
        video_name, class_, = self._samples[index].strip().split(" ")
        class_ = int(class_)
        video_path = os.path.join(self.data_root_dir, f"of256_{self.cfg.DATA.OPTICAL_FLOW}_info", video_name).split('.')[0] + ".json"
        sample_info = {
            "name": video_name.split('.')[0],
            "path": video_path,
            "supervised_label": class_,
        }
        return sample_info

    def _get_flow_info(self, info_path, index):
        tmp_file = str(round(time.time() * 1000)) + info_path.split('/')[-1]
        try:
            info = None
            tmp_file = self._get_object_to_file(info_path, tmp_file, read_from_buffer=True, num_retries=1 if self.split == "train" else 20)
            info = json.load(tmp_file)
            assert isinstance(info, (list,)) and len(info) > 0
            success = True
        except:
            success = False
        file_to_remove = [tmp_file] if info_path[:3] == "oss" else [None]
        return info, file_to_remove, success

    def _get_flow_batch(self, frame_list, video_name, flow_info):
        file_to_remove = []
        split_list = []
        for info in flow_info:
            split_list.append(info["start"])
        split_list.append(flow_info[-1]["end"])
        try:
            flow = []
            download = False
            for idx in range(len(split_list)-1):
                # print(idx)
                if split_list[idx+1] > frame_list[0] and split_list[idx] <= frame_list[0]:
                    download = True
                if split_list[idx] > frame_list[-1] and split_list[idx-1] <= frame_list[-1]:
                    download = False 
                    break
                if download:
                    # print(f"[{frame_list[0]}, {frame_list[-1]}] in [{split_list[idx]}, {split_list[idx+1]}], yah")
                    select_list = frame_list[(frame_list<split_list[idx+1]) * (frame_list>=split_list[idx])]-split_list[idx]
                    flow_path = os.path.join(self.data_root_dir, f"of256_{self.cfg.DATA.OPTICAL_FLOW}", video_name, flow_info[idx]["up_name"])
                    flow_frames = json.load(self._get_object_to_file(flow_path, None, read_from_buffer=True, num_retries=5))
                    if self.cfg.DATA.OPTICAL_FLOW == "raft":
                        for frame_idx in select_list.tolist():
                            data = flow_frames[frame_idx].encode(encoding='utf-8')
                            data = b64decode(data)
                            flow.append(cv2.imdecode(np.asarray(bytearray(data),dtype='uint8'), cv2.IMREAD_COLOR))
                    elif self.cfg.DATA.OPTICAL_FLOW == "tvl1":
                        for frame_idx in select_list.tolist():
                            data = [flow_frames[frame_idx*2].encode(encoding='utf-8'), flow_frames[frame_idx*2+1].encode(encoding='utf-8')]
                            data = [b64decode(d) for d in data]
                            flow.append([
                                cv2.imdecode(np.asarray(bytearray(d),dtype='uint8'), cv2.IMREAD_GRAYSCALE) for d in data
                            ])
                    else: raise NotImplementedError
            flow = torch.tensor(np.stack(flow))
            if self.cfg.DATA.OPTICAL_FLOW == "tvl1":
                flow = flow.permute(0,2,3,1)
                flow_third = torch.sqrt((flow.float()**2).sum(-1, keepdim=True))
                flow_third = (flow_third / flow_third.max() * 255).to(torch.uint8)
                flow = torch.cat((flow, flow_third), dim=-1)
            success = True
        except:
            success = False
        return flow, success

    def _decode_flow(self, sample_info, index, num_clips_per_video=1):
        path = sample_info["path"]
        flow_info, file_to_remove, success = self._get_flow_info(path, index)

        if not success:
            return flow_info, file_to_remove, success

        if self.split == "train" or self.split == "linear_train_cls":
            clip_idx = -1
            self.spatial_idx = -1
        elif self.split == "val":
            clip_idx = -1
            self.spatial_idx = 0
        elif self.split == "test" or self.split == "linear_test_cls" or self.split == "linear_test_ret":
            clip_idx = self._spatial_temporal_index[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                self.spatial_idx = 0
            else:
                self.spatial_idx = self._spatial_temporal_index[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
        elif self.split == "linear_train_ret": 
            clip_idx = self._spatial_temporal_index[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                self.spatial_idx = 0
            else:
                self.spatial_idx = self._spatial_temporal_index[index] % self.cfg.TEST.NUM_SPATIAL_CROPS

        length = flow_info[0]["duration"]
        fps = flow_info[0]["fps"]

        list_ = self._get_video_frames_list(
            length,
            fps,
            clip_idx,
            random_sample=True if self.split=="train" else False 
        )
        frames = None
        frames, success = self._get_flow_batch(list_, sample_info["name"], flow_info)
        return {"video": frames}, file_to_remove, True
    
    def _config_transform(self):
        """
        Configs the transform for the dataset.
        For train, we apply random cropping, random color jitter (optionally),
            normalization and random erasing (optionally).
        For val and test, we apply controlled spatial cropping and normalization.
        The transformations are stored as a callable function to "self.transforms".
        
        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list = [
                transforms.ToTensorVideo(),
                KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),
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
            self.resize_video = KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)


    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, vid_fps, clip_idx, num_clips, num_frames, interval)