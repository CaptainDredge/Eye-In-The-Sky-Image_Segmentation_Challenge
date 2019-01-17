import argparse
import numpy as np
import tifffile
from skimage import io
import os

parser = argparse.ArgumentParser(description='Create Patches for Images')
parser.add_argument('--source', default='image', type=str)
parser.add_argument('--dest', default='data', type=str)
parser.add_argument('--height', default=256, type=int)
parser.add_argument('--width', default=256, type=int)
parser.add_argument('--stride', default=0.5, type=int)
parser.add_argument('--mode', default='train', type=str, help='train for training and test during prediction')


def main():
    args = parser.parse_args()
    if args.mode == 'train':
        if not os.path.isdir(args.source):
            raise Exception('source file does not exist')
        if not os.path.isdir(args.dest):
            print('creating destination folder')
            os.mkdir(args.dest)
        print('source -> {}\ndestination -> {}'.format(args.source, args.dest))
        create_dataset(args.source, args.dest, args.height, args.width, args.stride)
        print('Done ..........')
    else:
        args.source = 'test'
        args.dest = 'test' + args.dest
        if not os.path.isdir(args.dest):
            print('creating destination folder')
            os.mkdir(args.dest)
        print('source -> {}\ndestination -> {}'.format(args.source, args.dest))
        create_dataset_test(args.source, args.dest, args.height, args.width, args.stride)


def create_patches(img, patch_height=144, patch_width=144, stride=0.5):
    """ input -> image (none,none,4), patch dimensions
        outpout -> patches of desired dimensions (patch_height, patch_width, 4)
    """
    h_stride = int(max(1, patch_height * stride))
    w_stride = int(max(1, patch_width * stride))

    h = 0
    w = 0
    img = pad_image(img, patch_height, patch_width)
    patches = []

    while h <= img.shape[0] - patch_height:
        w = 0
        while w <= img.shape[1] - patch_width:
            patches.append(img[h:h + patch_height, w:w + patch_width, :])
            w = w + w_stride
        h = h + h_stride

    return patches


def pad_image(img, height, width):
    """Pads img to make it fit for extracting patches of
    shape height*width from it
    mode -> constant, reflect
    constant -> pads ith 0's
    reflect -> pads with reflection of image
    """
    print('input shape {}'.format(img.shape))
    h = 0 if img.shape[0] % height == 0 else height - img.shape[0] % height
    w = 0 if img.shape[1] % width == 0 else width - img.shape[1] % width
    pad_shape = [0, 0, 0]
    pad_shape[0] = (0, h)
    pad_shape[1] = (0, w)
    pad_shape[2] = (0, 0)
    print(pad_shape)
    img = np.pad(img, pad_shape, mode='reflect')
    print('output shape {}'.format(img.shape))
    return img


def create_dataset(source, destination, height, width, stride):
    """
        Reads TIFF files from soure and extracts patches
        from it and saves it to destination
    """
    print('destination -> {}'.format(destination))
    train_destination = os.path.join(destination, 'train')
    valid_destination = os.path.join(destination, 'valid')
    print(train_destination, valid_destination)
    if not os.path.isdir(train_destination):
        os.mkdir(train_destination)
    if not os.path.isdir(valid_destination):
        os.mkdir(valid_destination)

    # target dir name
    train_path_img = os.path.join(train_destination, 'images')
    train_path_mask = os.path.join(train_destination, 'masks')

    valid_path_img = os.path.join(valid_destination, 'images')
    valid_path_mask = os.path.join(valid_destination, 'masks')

    # If target dir not present create it
    if not os.path.isdir(train_path_img):
        os.mkdir(train_path_img)
    if not os.path.isdir(train_path_mask):
        os.mkdir(train_path_mask)

    if not os.path.isdir(valid_path_img):
        os.mkdir(valid_path_img)
    if not os.path.isdir(valid_path_mask):
        os.mkdir(valid_path_mask)

    train_images = os.listdir(os.path.join(source, 'train', 'image'))
    val_images = os.listdir(os.path.join(source, 'val', 'image'))

    for i, file in enumerate(train_images):
        print('processing {}'.format(file))
        image = tifffile.imread(os.path.join(source, 'train', 'image', file))
        mask = tifffile.imread(os.path.join(source, 'train', '9ch', file))
        img_patches = create_patches(image, patch_height=height, patch_width=width, stride=stride)
        mask_patches = create_patches(mask, patch_height=height, patch_width=width, stride=stride)
        for j, (img_patch, mask_patch) in enumerate(
                zip(img_patches, mask_patches)):
            tifffile.imsave(
                train_path_img + '/patch_{}_{}_{}_{}_{}_{}.tif'.format(file.split('.')[0], j, image.shape[0],
                                                                       image.shape[1], int(height * stride),
                                                                       height), img_patch)
            tifffile.imsave(
                train_path_mask + '/patch_{}_{}_{}_{}_{}_{}.tif'.format(file.split('.')[0], j, image.shape[0],
                                                                        image.shape[1], int(height * stride),
                                                                        height), mask_patch)

    for i, file in enumerate(val_images):
        print('processing {}'.format(file))
        image = tifffile.imread(os.path.join(source, 'val', 'image', file))
        mask = tifffile.imread(os.path.join(source, 'val', '9ch', file))
        img_patches = create_patches(image, patch_height=height, patch_width=width, stride=stride)
        mask_patches = create_patches(mask, patch_height=height, patch_width=width, stride=stride)
        for j, (img_patch, mask_patch) in enumerate(
                zip(img_patches, mask_patches)):
            tifffile.imsave(valid_path_img + '/patch_{}_{}_{}_{}_{}_{}.tif'.format(
                file.split('.')[0], j, image.shape[0], image.shape[1], int(height * stride), height),
                            img_patch)
            tifffile.imsave(valid_path_mask + '/patch_{}_{}_{}_{}_{}_{}.tif'.format(
                file.split('.')[0], j, image.shape[0], image.shape[1], int(height * stride), height),
                            mask_patch)


def create_dataset_test(source, destination, height, width, stride):
    """
        Reads TIFF files from soure and extracts patches
        from it and saves it to destination
    """
    print('destination -> {}'.format(destination))
    # target dir name

    test_images = os.listdir(source)

    for i, file in enumerate(test_images):
        print('processing {}'.format(file))
        image = tifffile.imread(os.path.join(source, file))
        img_patches = create_patches(image, patch_height=height, patch_width=width, stride=stride)
        for j, (img_patch) in enumerate(img_patches):
            tifffile.imsave(
                destination + '/patch_{}_{}_{}_{}_{}_{}.tif'.format(file.split('.')[0], j, image.shape[0],
                                                                    image.shape[1], int(height * stride),
                                                                    height), img_patch)


if __name__ == '__main__':
    main()

