
import torch
import random
from datasets.utils.transformations import ColorJitter
import torchvision.transforms._functional_video as F

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0

class ToTensorTwoStream(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, data):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        data["video"] = to_tensor(data["video"])
        data["flow"] = to_tensor(data["flow"])
        return data

    def __repr__(self):
        return self.__class__.__name__

class TwoStreamResizedCrop(object):
    def __init__(
        self,
        short_side_range,
        crop_size,
        num_spatial_crops=1,
    ):  
        self.idx = -1
        self.short_side_range = short_side_range
        self.crop_size = int(crop_size)
        self.num_spatial_crops = num_spatial_crops
    
    def _get_controlled_crop(self, data):
        assert data["video"].shape == data["flow"].shape, f"Shape not aligned. Video: {data['video'].shape}, Flow: {data['flow'].shape}"
        _, _, clip_height, clip_width = data["video"].shape

        length = self.short_side_range[0]

        if clip_height < clip_width:
            new_clip_height = int(length)
            new_clip_width = int(clip_width / clip_height * new_clip_height)
            new_clip = torch.nn.functional.interpolate(
                data["video"], size=(new_clip_height, new_clip_width), mode="bilinear"
            )
            new_flow = torch.nn.functional.interpolate(
                data["flow"], size=(new_clip_height, new_clip_width), mode="bilinear"
            )
        else:
            new_clip_width = int(length)
            new_clip_height = int(clip_height / clip_width * new_clip_width)
            new_clip = torch.nn.functional.interpolate(
                data["video"], size=(new_clip_height, new_clip_width), mode="bilinear"
            )
            new_flow = torch.nn.functional.interpolate(
                data["flow"], size=(new_clip_height, new_clip_width), mode="bilinear"
            )
        x_max = int(new_clip_width - self.crop_size)
        y_max = int(new_clip_height - self.crop_size)
        if self.num_spatial_crops == 1:
            x = x_max // 2
            y = y_max // 2
        elif self.num_spatial_crops == 3:
            if self.idx == 0:
                if new_clip_width == length:
                    x = x_max // 2
                    y = 0
                elif new_clip_height == length:
                    x = 0
                    y = y_max // 2
            elif self.idx == 1:
                x = x_max // 2
                y = y_max // 2
            elif self.idx == 2:
                if new_clip_width == length:
                    x = x_max // 2
                    y = y_max
                elif new_clip_height == length:
                    x = x_max
                    y = y_max // 2
        data["video"] = new_clip[:, :, y:y+self.crop_size, x:x+self.crop_size]
        data["flow"] = new_flow[:, :, y:y+self.crop_size, x:x+self.crop_size]
        return data

    def _get_random_crop(self, data):
        assert data["video"].shape == data["flow"].shape, f"Shape not aligned. Video: {data['video'].shape}, Flow: {data['flow'].shape}"
        _, _, clip_height, clip_width = data["video"].shape

        if clip_height < clip_width:
            new_clip_height = int(random.uniform(*self.short_side_range))
            new_clip_width = int(clip_width / clip_height * new_clip_height)
            new_clip = torch.nn.functional.interpolate(
                data["video"], size=(new_clip_height, new_clip_width), mode="bilinear"
            )
            new_flow = torch.nn.functional.interpolate(
                data["flow"], size=(new_clip_height, new_clip_width), mode="bilinear"
            )
        else:
            new_clip_width = int(random.uniform(*self.short_side_range))
            new_clip_height = int(clip_height / clip_width * new_clip_width)
            new_clip = torch.nn.functional.interpolate(
                data["video"], size=(new_clip_height, new_clip_width), mode="bilinear"
            )
            new_flow = torch.nn.functional.interpolate(
                data["flow"], size=(new_clip_height, new_clip_width), mode="bilinear"
            )
        x_max = int(new_clip_width - self.crop_size)
        y_max = int(new_clip_height - self.crop_size)
        x = int(random.uniform(0, x_max))
        y = int(random.uniform(0, y_max))
        data["video"] = new_clip[:, :, y:y+self.crop_size, x:x+self.crop_size]
        data["flow"] = new_flow[:, :, y:y+self.crop_size, x:x+self.crop_size]
        return data

    def set_spatial_index(self, idx):
        self.idx = idx

    def __call__(self, data):
        if self.idx == -1:
            return self._get_random_crop(data)
        else:
            return self._get_controlled_crop(data)

class ColorJitterTwoStream(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, grayscale=0, consistent=False, shuffle=True, gray_first=True, is_split=False):
        self.color_jitter = ColorJitter(
            brightness  = brightness,
            contrast    = contrast,
            saturation  = saturation,
            hue         = hue,
            grayscale   = grayscale,
            consistent  = consistent,
            shuffle     = shuffle,
            gray_first  = gray_first,
            is_split    = is_split
        )
    
    def __call__(self, data):
        data["video"] = self.color_jitter(data["video"])
        return data
    
    def __repr__(self):
        self.color_jitter.__repr__()


class NormalizeTwoStream(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        data["video"] = F.normalize(data["video"], self.mean, self.std, self.inplace)
        data["flow"] = F.normalize(data["flow"], self.mean, self.std, self.inplace)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)

class RandomHorizontalFlipTwoStream(object):
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            data["video"] = F.hflip(data["video"])
            data["flow"] = F.hflip(data["flow"])
        return data

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)
