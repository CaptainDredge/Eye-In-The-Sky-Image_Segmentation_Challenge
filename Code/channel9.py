import numpy as np
import skimage.io as io
import argparse
import os
import tifffile

parser = argparse.ArgumentParser(description='Create 9 Channel Image')
parser.add_argument('--path', default='image/', type=str)
# Unique pixel values for masks
PIXELS = [np.array([0, 0, 0]),
          np.array([0, 125, 0]),
          np.array([0, 255, 0]),
          np.array([100, 100, 100]),
          np.array([150, 80, 0]),
          np.array([150, 150, 255]),
          np.array([0, 0, 150]),
          np.array([255, 255, 0]),
          np.array([255, 255, 255]),
          ]


def rgb_to_mask(img):
    """
        converts an rgb mask into 9 channel training mask
    """
    # image to be returned
    out_img = np.zeros((img.shape[0], img.shape[1], 9), dtype=np.uint8)
    for i, pixel in enumerate(PIXELS):
        a = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
        for j, pix in enumerate(pixel):
            x = (img[:, :, j] == pix)
            a = np.logical_and(a, x)
        out_img[:, :, i] = a
    return out_img


def main():
    args = parser.parse_args()
    train_files = os.listdir(os.path.join(args.path, 'train', 'mask'))
    val_files = os.listdir(os.path.join(args.path, 'val', 'mask'))
    if not os.path.isdir(os.path.join(args.path, 'train', '9ch')):
        os.mkdir(os.path.join(args.path, 'train', '9ch'))
    if not os.path.isdir(os.path.join(args.path, 'val', '9ch')):
        os.mkdir(os.path.join(args.path, 'val', '9ch'))

    for i, file in enumerate(train_files):
        mask = io.imread(os.path.join(args.path, 'train', 'mask', file))
        mask = rgb_to_mask(mask)
        tifffile.imsave(os.path.join(args.path, 'train', '9ch', file), mask, dtype=np.uint8)
        print(i)
    print('Training Images Mask Done')
    for i, file in enumerate(val_files):
        mask = io.imread(os.path.join(args.path, 'val', 'mask', file))
        mask = rgb_to_mask(mask)
        tifffile.imsave(os.path.join(args.path, 'val', '9ch', file), mask, dtype=np.uint8)
    print('Validation Image Masks Done')


if __name__ == '__main__':
    main()
