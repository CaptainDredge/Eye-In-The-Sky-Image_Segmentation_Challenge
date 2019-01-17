import tifffile
from scipy.ndimage import rotate
import skimage.io as io
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Augument Images')
parser.add_argument('--path', default='train', type=str)
parser.add_argument('--dest', default='image', type=str)


def hr_flip(img, mask):
    img = np.fliplr(img)
    mask = np.fliplr(mask)
    return img, mask


def vr_flip(img, mask):
    img = np.flipud(img)
    mask = np.flipud(mask)
    return img, mask


def rotate90(img, mask):
    img = rotate(img, angle=90)
    mask = rotate(mask, angle=90)
    return img, mask


def rotate180(img, mask):
    img = rotate(img, angle=180)
    mask = rotate(mask, angle=180)
    return img, mask


def rotate270(img, mask):
    img = rotate(img, angle=270)
    mask = rotate(mask, angle=270)
    return img, mask


def main():
    args = parser.parse_args()
    files = os.listdir(os.path.join(args.path, 'sat'))
    val_split = ['5.tif']
    if not os.path.isdir(args.dest):
        os.mkdir(args.dest)
        os.mkdir(os.path.join(args.dest, 'train'))
        os.mkdir(os.path.join(args.dest, 'val'))
        os.mkdir(os.path.join(args.dest, 'train', 'image'))
        os.mkdir(os.path.join(args.dest, 'train', 'mask'))
        os.mkdir(os.path.join(args.dest, 'val', 'image'))
        os.mkdir(os.path.join(args.dest, 'val', 'mask'))

    for i, file in enumerate(files):
        img = io.imread(os.path.join(args.path, 'sat', file))
        mask = io.imread(os.path.join(args.path, 'gt', file))
        if file in val_split:
            tifffile.imsave(os.path.join(args.dest, 'val', 'image', file), img, dtype=np.uint16)
            tifffile.imsave(os.path.join(args.dest, 'val', 'mask', file), mask, dtype=np.uint8)
        else:
            tifffile.imsave(os.path.join(args.dest, 'train', 'image', file), img, dtype=np.uint16)
            tifffile.imsave(os.path.join(args.dest, 'train', 'mask', file), mask, dtype=np.uint8)
            aug_img, aug_mask = hr_flip(img, mask)
            tifffile.imsave(os.path.join(args.dest, 'train', 'image', file.split('.')[0] + '_hr.tif'), aug_img,
                            dtype=np.uint16)
            tifffile.imsave(os.path.join(args.dest, 'train', 'mask', file.split('.')[0] + '_hr.tif'), aug_mask,
                            dtype=np.uint8)
            aug_img, aug_mask = vr_flip(img, mask)

            tifffile.imsave(os.path.join(args.dest, 'train', 'image', file.split('.')[0] + '_vr.tif'), aug_img,
                            dtype=np.uint16)
            tifffile.imsave(os.path.join(args.dest, 'train', 'mask', file.split('.')[0] + '_vr.tif'), aug_mask,
                            dtype=np.uint8)
            aug_img, aug_mask = rotate90(img, mask)
            tifffile.imsave(os.path.join(args.dest, 'train', 'image', file.split('.')[0] + '_90.tif'), aug_img,
                            dtype=np.uint16)
            tifffile.imsave(os.path.join(args.dest, 'train', 'mask', file.split('.')[0] + '_90.tif'), aug_mask,
                            dtype=np.uint8)
            aug_img, aug_mask = rotate180(img, mask)
            tifffile.imsave(os.path.join(args.dest, 'train', 'image', file.split('.')[0] + '_180.tif'), aug_img,
                            dtype=np.uint16)
            tifffile.imsave(os.path.join(args.dest, 'train', 'mask', file.split('.')[0] + '_180.tif'), aug_mask,
                            dtype=np.uint8)
            aug_img, aug_mask = rotate270(img, mask)
            tifffile.imsave(os.path.join(args.dest, 'train', 'image', file.split('.')[0] + '_270.tif'), aug_img,
                            dtype=np.uint16)
            tifffile.imsave(os.path.join(args.dest, 'train', 'mask', file.split('.')[0] + '_270.tif'), aug_mask,
                            dtype=np.uint8)
        print(i)


if __name__ == '__main__':
    main()

