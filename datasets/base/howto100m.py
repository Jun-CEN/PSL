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

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Howto100m(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Howto100m, self).__init__(cfg, split) 
        # self._samples.pop(0)

        self.caption_root_dir = self.cfg.TEXT.CAPTION_ROOT_DIR
        self.caption_download = self.cfg.TEXT.CAPTION_DOWNLOAD
        if self.caption_download:
            self.local_caption_root = "./captions/"
            if not os.path.exists(self.local_caption_root):
                os.makedirs(self.local_caption_root)

        self.decode = self._decode_video_text
        self._initialize_text_loader()
    
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        name = "HowTo100M_v4.csv"
        # name = "HowTo100M_v4_debug.csv"
        # name = "HowTo100M_v4_debug_50k.csv"
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
        video_path, cat1, cat2, rank, taskid = self._samples[index].strip().split(',')
        video_id = video_path.split('.')[0]
        caption_path = os.path.join(self.caption_root_dir, video_id+'.npy')
        video_path = os.path.join(self.data_root_dir, video_path)
        sample_info = {
            "path": video_path,
            "type": video_path.split('.')[-1],
            "id": video_id,
            "caption_path": caption_path,
            "supervised_label": 0
        }
        return sample_info

    def _decode_video_text(self, sample_info, index, num_clips_per_video=1):
        """
        Decode video and extract the embedding of the related text in the given captions.
        By default, the param 'num_clips_per_video' has no use here.
        """
        # ------------------------ decode text ------------------------

        caption, tmp_file_cap, success = self._get_caption(sample_info['caption_path'])
        overall_num_clips = len(caption["start"])
        rand_clip_id = random.randint(0,overall_num_clips-1)
        si,sd,ei,ed = ( # start/end, integer/decimal
            "{:04d}".format(int(caption["start"][rand_clip_id])),
            "{:03d}".format(int((caption["start"][rand_clip_id]-int(caption["start"][rand_clip_id]))*1000)),
            "{:04d}".format(int(caption["end"][rand_clip_id])),
            "{:03d}".format(int((caption["end"][rand_clip_id]-int(caption["end"][rand_clip_id]))*1000))
        )

        max_text_idx = len(caption['start'])-1
        sentence_idx = self._get_sentence_idx(rand_clip_id, max_text_idx, self.cfg.TEXT.NUM_SENTENCES).tolist()
        sentences = np.array(caption['text'])[sentence_idx].tolist()
        word_ids = self.words_to_ids(sentences)
        word_ids_existence_map = 1-(word_ids==0)*1
        word_embd = self.word_embd(word_ids)

        # ------------------------ decode video ------------------------
        vid_path = os.path.join(self.data_root_dir, "{}_{}_{}-{}_{}.{}".format(
            sample_info["id"], si, sd, ei, ed, sample_info["type"]
        ))
        vr, tmp_file_vid, success =  self._read_video(vid_path, index)

        if not success:
            return vr, tmp_file_vid, success

        if self.split == "train":
            clip_idx = -1
            self.spatial_idx = -1
        elif self.split == "val":
            clip_idx = -1
            self.spatial_idx = 0
        elif self.split == "test":
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
        if sample_info['caption_path'][:3]=='oss' and not self.caption_download:
            file_to_remove.append(tmp_file_cap)
        if vid_path[:3] == 'oss':
            file_to_remove += tmp_file_vid
        
        return frames, file_to_remove, True
    
    def _config_transform(self):
        self.transform = None
        
    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, vid_fps, clip_idx, num_clips, num_frames, interval)
    
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