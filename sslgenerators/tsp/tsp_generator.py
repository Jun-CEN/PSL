import os
import torch
import random
import utils.logging as logging
import torchvision.transforms._functional_video as F
import matplotlib.pyplot as plt

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from sslgenerators.builder import SSL_GENERATOR_REGISTRY
from sslgenerators.utils.augmentations import RandomColorJitter

logger = logging.get_logger(__name__)

@SSL_GENERATOR_REGISTRY.register()
class TSPGenerator(object):
    """
    Generator for pseudo camera motions.
    """
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        
        self.config_transform()
        self.labels = {"tsp": torch.tensor([0, 1])}
        # 0 denotes regular speed
        # 1 denotes reverse speed

    def sample_generator(self, frames, index):
        if self.cfg.PRETRAIN.CONTRASTIVE.TSP.SPATIAL_ALIGN:
            frames_video = self.transform(frames["video"])
            frames["video"] = torch.stack(
                (frames_video, frames_video.flip(1))
            )
        else:
            frames["video"] = torch.stack(
                (self.transform(frames["video"]), self.transform(frames["video"]).flip(1))
            )
        
        self.labels["text_validity"] = frames["text_validity"]
        return frames
    
    def config_transform(self):
        std_transform_list = []
        if self.split == 'train' or self.split == 'val':
            # To tensor and normalize
            std_transform_list += [
                transforms.ToTensorVideo(),
                transforms.RandomResizedCropVideo(
                    size=self.cfg.DATA.TRAIN_CROP_SIZE,
                    scale=[
                        self.cfg.DATA.TRAIN_JITTER_SCALES[0]*self.cfg.DATA.TRAIN_JITTER_SCALES[0]/256.0/340.0,
                        self.cfg.DATA.TRAIN_JITTER_SCALES[1]*self.cfg.DATA.TRAIN_JITTER_SCALES[1]/256.0/340.0
                    ],
                    ratio=self.cfg.AUGMENTATION.RATIO
                ),
            ]
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    RandomColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        color=self.cfg.AUGMENTATION.COLOR,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                    )
                )
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                transforms.RandomHorizontalFlipVideo(),
            ]
            self.transform = Compose(std_transform_list)
        elif self.split == 'test':
            std_transform_list += [
                transforms.ToTensorVideo(),
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)
            
    def visualize_frames(self, frames, index, move_dir):
        if not os.path.exists("output/visualization/pseudo_motion"):
            os.makedirs("output/visualization/pseudo_motion")
        if len(frames.shape) == 4:
            t,h,w,c = frames.shape
            for frame_idx in range(t):
                if not os.path.exists("output/visualization/pseudo_motion/{}/move_{}_{}/".format(index, move_dir[0], move_dir[1])):
                    os.makedirs("output/visualization/pseudo_motion/{}/move_{}_{}/".format(index, move_dir[0], move_dir[1]))
                frames_vis = frames[frame_idx].detach().cpu().numpy()
                plt.imsave("output/visualization/pseudo_motion/{}/move_{}_{}/{:02d}.jpg".format(index, move_dir[0], move_dir[1], frame_idx), frames_vis)
            
    def __call__(self, frames, index):
        return self.sample_generator(frames, index), self.labels