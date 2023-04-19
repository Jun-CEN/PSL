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
from datasets.utils.transformations import (
    ColorJitter, CustomResizedCropVideo, 
    AutoResizedCropVideo,
    RandomResizedCropVideo,
    KineticsResizedCrop
)

logger = logging.get_logger(__name__)

@SSL_GENERATOR_REGISTRY.register()
class ContrastiveGenerator(object):
    """
    Generator for pseudo camera motions.
    """
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.loss = cfg.PRETRAIN.LOSS
        self.crop_size = cfg.DATA.TRAIN_CROP_SIZE
        self.num_speeds = cfg.VIDEO.HEAD.NUM_CLASSES
        self.split = split

        if type(self.crop_size) is list:
            assert len(self.crop_size) <= 2
            if len(self.crop_size) == 2:
                assert self.crop_size[0] == self.crop_size[1]
                self.crop_size = self.crop_size[0]   
        
        self.config_transform()
        if self.cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 3:
            self.labels = {"contrastive": torch.tensor([0, 1, 2])}
        elif self.cfg.PRETRAIN.NUM_CLIPS_PER_VIDEO == 2:
            self.labels = {"contrastive": torch.tensor([0, 1])}
        else:
            self.labels = {"contrastive": torch.tensor([0])}

    def sample_generator(self, frames, index):
        out = []
        if len(frames["video"].shape) == 4:
            frames["video"] = frames["video"].unsqueeze(0)
        for i in range(frames["video"].shape[0]):
            out.append(self.transform(frames["video"][i]))
        # out.append(self.transform(frames["video"]))
        # out.append(self.transform(frames["video"]))
        frames["video"] = torch.stack(out)
        return frames
    
    def config_transform(self):
        std_transform_list = []
        if self.split == 'train':
            # To tensor and normalize
            std_transform_list += [
                transforms.ToTensorVideo(),
                transforms.RandomResizedCropVideo(
                    size=self.cfg.DATA.TRAIN_CROP_SIZE,
                ),
                transforms.RandomHorizontalFlipVideo(),
            ]
            # Add color aug
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
            # std_transform_list += [
            #     transforms.NormalizeVideo(
            #         mean=self.cfg.DATA.MEAN,
            #         std=self.cfg.DATA.STD,
            #         inplace=True
            #     ),
            #     transforms.RandomHorizontalFlipVideo(),
            # ]
            self.transform = Compose(std_transform_list)
        else:
            # std_transform_list += [
            #     transforms.ToTensorVideo(),
            #     transforms.NormalizeVideo(
            #         mean=self.cfg.DATA.MEAN,
            #         std=self.cfg.DATA.STD,
            #         inplace=True
            #     )
            # ]
            self.resize_video = KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            std_transform_list += [
                transforms.ToTensorVideo(),
                self.resize_video
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


if __name__ == '__main__':
    from utils.config import Config
    cfg = Config(load=True)
    ssl_generator = MoSIGenerator(cfg, "train")
    frames = torch.rand(3, 1, 224, 224)
    out, labels = ssl_generator(frames)
    print(out.shape)
    print(labels)