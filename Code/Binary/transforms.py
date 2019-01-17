import torch
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.ascontiguousarray(image.transpose((2, 0, 1)), dtype=np.float32)
        mask = np.ascontiguousarray(mask.transpose((2,0,1)), dtype=np.float32)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class ImgAugTransform(object):
    """Apply augmentation to Image.
    Args:
        aug_type (sequential or one of or sometimes): Augmentation type
        random (bool): Apply augmentation in random order
        percent(float): Percentage of total images on which augmentation is applied
    """
    def __init__(self, nb_classes = 2, aug_type = 'sequential', mode = 'light', random = True, percent = 0.5):
        assert isinstance(aug_type, str)
        assert isinstance(random, bool)
        assert isinstance(percent, float)
        self.nb_classes = nb_classes
        self.type = aug_type
        self.random = random
        self.percent = percent
        self.mode = mode
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        if self.mode == 'light':
            if self.type == 'sequential':
                self.aug = iaa.Sequential([
                                    iaa.Fliplr(0.5),
                                    iaa.Flipud(0.5),
                                    sometimes(iaa.Affine(rotate=(-20, 20), mode='symmetric')),
                ])
            elif self.type == 'one of':
                self.aug = iaa.OneOf([
                                    iaa.Fliplr(0.5),
                                    iaa.Flipud(0.5),
                                    iaa.Affine(rotate=(-45, 45), mode='symmetric'),
                                    iaa.OneOf([iaa.ElasticTransformation(alpha=50, sigma=5),
                                               iaa.Add((-10, 10), per_channel=0.5),
                                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)])
                                    ])
            else:
                self.aug = iaa.Sometimes(self.percent, [
                                    iaa.Fliplr(0.5),
                                    iaa.Flipud(0.5),
                                    iaa.Affine(rotate=(-45, 45)),
                                    iaa.ElasticTransformation(alpha=50, sigma=5)])
        else:
            self.aug = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.3), # horizontally flip 50% of all images
        iaa.Flipud(0.3), # vertically flip 20% of all images

        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            mode='symmetric' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 2),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
            ],
            random_order=True
        )
    ],
    random_order=True
)

    def __call__(self, sample):
        image = np.array(sample['image'])
        mask = np.array(sample['mask']).astype(np.uint16)
        dim = mask.ndim
        if dim > 2:
            categ_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
            no_classes = mask.shape[-1]
            for i in range(no_classes):
                categ_mask += i * mask[:, :, -(i + 1)]
            mask = categ_mask

        segmap = ia.SegmentationMapOnImage(mask, shape=mask.shape, nb_classes=self.nb_classes)
        seq_det = self.aug.to_deterministic()
        aug_img = seq_det.augment_image(image)
        aug_segmap = seq_det.augment_segmentation_maps([segmap])[0]
        aug_mask = ia.SegmentationMapOnImage.get_arr_int(aug_segmap)
        if dim > 2:
            masks = []
            for clas in range(no_classes - 1, -1, -1):
                bool_mask = aug_mask == clas
                masks.append(bool_mask)
            aug_mask = np.array(masks).astype(np.uint16)
            aug_mask = np.rollaxis(aug_mask, 0, start=3)
        return {'image': aug_img, 'mask': aug_mask}

