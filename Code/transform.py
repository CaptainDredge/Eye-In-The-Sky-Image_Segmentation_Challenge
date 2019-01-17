import torch
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors and Normalize input image"""
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        mean = [206.75503457700066, 282.4707482803494, 174.54941130734392, 346.48021589897877]
        std = [77.24070766692714, 129.3841999574279, 109.96606967828127, 216.19068282723546]
        image = (image - mean)/std
        image = np.ascontiguousarray(image.transpose((2, 0, 1)), dtype=np.float32)
        mask = np.ascontiguousarray(mask.transpose((2, 0, 1)), dtype=np.uint8)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}



