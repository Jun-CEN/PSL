import os
import random
import torch
import torch.utils.data
import utils.logging as logging
from PIL import Image
import torchvision
import torchvision.transforms._functional_video as F
import numpy as np

import time
import oss2 as oss

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from datasets.utils.transformations import (
    ColorJitter, CustomResizedCropVideo, 
    AutoResizedCropVideo,
    RandomResizedCropVideo,
    KineticsResizedCrop
)

from datasets.base.base_dataset import BaseVideoDataset

import utils.bucket as bu

from datasets.base.builder import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Ucf101(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Ucf101, self).__init__(cfg, split) 
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
    
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        if hasattr(self.cfg.PRETRAIN, "UCF_CUT") and self.cfg.PRETRAIN.UCF_CUT and self.split == "train":
            name = "ucf101_train_list_cut.txt"
        elif hasattr(self.cfg.TRAIN, "FEW_SHOT") and self.cfg.TRAIN.FEW_SHOT and self.split == "train":
            name = "ucf101_train_list_fewshot.txt"
            print("FEW SHOT!!")
        elif hasattr(self.cfg.TRAIN, "OPENSET") and self.cfg.TRAIN.OPENSET:
            if "train" in self.split:
                name = "ucf101_train_split_1_videos.txt"
                print("OPENSET!! Load ucf101_train_split_1_videos.txt.")
            else:
                name = "ucf101_val_split_1_videos.txt"
                print("OPENSET!! Load ucf101_val_split_1_videos.txt.")
        else:
            name = "ucf101_{}_list.txt".format(
                "train" if "train" in self.split else "test",
            )
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                "path": indicating the target's path w.r.t. index
                "supervised_label": indicating the class of the target 
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
        """
        Configs the transform for the dataset.
        For train, we apply random cropping, random horizontal flip, random color jitter (optionally),
            normalization and random erasing (optionally).
        For val and test, we apply controlled spatial cropping and normalization.
        The transformations are stored as a callable function to "self.transforms".
        
        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = None

        # if self.split == 'train' and self.cfg.AUGMENTATION.UCF:
        #     train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(self.cfg.DATA.TRAIN_CROP_SIZE, [1, .875, .75, .66]),
        #                                                transforms.RandomHorizontalFlipVideo()])
        #     self.transform = torchvision.transforms.Compose([
        #                train_augmentation,
        #                ToTorchFormatTensor(div=(True))
        #            ])

        # elif self.split == 'train' and not self.cfg.PRETRAIN.ENABLE or self.split == 'linear_train_cls':
        #     std_transform_list = [
        #         transforms.ToTensorVideo(),
        #         transforms.RandomResizedCropVideo(
        #             size=self.cfg.DATA.TRAIN_CROP_SIZE,
        #             scale=[
        #                 self.cfg.DATA.TRAIN_JITTER_SCALES[0]*self.cfg.DATA.TRAIN_JITTER_SCALES[0]/256.0/340.0,
        #                 self.cfg.DATA.TRAIN_JITTER_SCALES[1]*self.cfg.DATA.TRAIN_JITTER_SCALES[1]/256.0/340.0
        #             ],
        #             ratio=self.cfg.AUGMENTATION.RATIO
        #         ),
        #         transforms.RandomHorizontalFlipVideo()
        #     ]
        #     # Add color aug
        #     if self.cfg.AUGMENTATION.COLOR_AUG:
        #         std_transform_list.append(
        #             ColorJitter(
        #                 brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
        #                 contrast=self.cfg.AUGMENTATION.CONTRAST,
        #                 saturation=self.cfg.AUGMENTATION.SATURATION,
        #                 hue=self.cfg.AUGMENTATION.HUE,
        #                 grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
        #                 consistent=self.cfg.AUGMENTATION.CONSISTENT,
        #                 shuffle=self.cfg.AUGMENTATION.SHUFFLE,
        #                 gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
        #                 ),
        #         )
        #     std_transform_list += [
        #         transforms.NormalizeVideo(
        #             mean=self.cfg.DATA.MEAN,
        #             std=self.cfg.DATA.STD,
        #             inplace=True
        #         ),
        #     ]
        #     self.transform = Compose(std_transform_list)

        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE or self.split == 'linear_train_cls':
            std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomResizedCropVideo(
                    size=self.cfg.DATA.TRAIN_CROP_SIZE,
                    scale=[
                        0.3, 1
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
            # std_transform_list += [
            #     transforms.NormalizeVideo(
            #         mean=self.cfg.DATA.MEAN,
            #         std=self.cfg.DATA.STD,
            #         inplace=True
            #     ),
            # ]
            self.transform = Compose(std_transform_list)


        elif self.split == 'val' or self.split == 'test' or\
            self.split == 'linear_train_ret' or "test" in self.split:
            self.resize_video = KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
            ]
            self.transform = Compose(std_transform_list)
        # elif self.split == 'val' or self.split == 'test' or\
        #     self.split == 'linear_train_ret' or "test" in self.split:
        #     self.resize_video = KineticsResizedCrop(
        #             short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
        #             crop_size = self.cfg.DATA.TEST_CROP_SIZE,
        #             num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
        #         )
        #     std_transform_list = [
        #         transforms.ToTensorVideo(),
        #         self.resize_video,
        #         transforms.NormalizeVideo(
        #             mean=self.cfg.DATA.MEAN,
        #             std=self.cfg.DATA.STD,
        #             inplace=True
        #         )
        #     ]
        #     self.transform = Compose(std_transform_list)
        # elif self.split == 'val' or self.split == 'test' or\
        #     self.split == 'linear_train_ret' or "test" in self.split:
            # self.resize_video = AutoResizedCropVideo(
        #             size=self.cfg.DATA.TEST_CROP_SIZE,
        #             scale=[
        #                 self.cfg.DATA.TEST_SCALE/256.0,
        #                 self.cfg.DATA.TEST_SCALE/256.0  
        #             ],
        #             mode=self.cfg.TEST.SPATIAL_CROPS,
        #         )
        #     std_transform_list = [
        #         transforms.ToTensorVideo(),
        #         self.resize_video,
        #         transforms.NormalizeVideo(
        #             mean=self.cfg.DATA.MEAN,
        #             std=self.cfg.DATA.STD,
        #             inplace=True
        #         ),
        #     ]
        #     self.transform = Compose(std_transform_list)

    def _pre_transformation_config(self):
        """
        Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, vid_fps, clip_idx, num_clips, num_frames, interval)
    
    def _get_ssl_label(self, frames):
        """
        Uses cfg to obtain ssl label.
        Returns:
            ssl_label (dict): self-supervised labels
        """
        raise NotImplementedError

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        
        return tensor

class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation="bilinear"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        return F.resize(img_group.float().permute(3, 0, 1, 2), target_size = self.size, interpolation_mode = self.interpolation)

class GroupCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        return F.center_crop(img_group, self.size)

class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size()

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img_group.permute(3,0,1,2)
        # crop_img_group = torchvision.transforms.functional.crop(img_group, offset_h, offset_w, crop_h, crop_w)
        ret_img_group = F.resized_crop(crop_img_group.float(), offset_h, offset_w, crop_h, crop_w, (self.input_size[0],self.input_size[1]))
        # ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                        #  for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        elif isinstance(pic, torch.cuda.FloatTensor):
            img = pic
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()