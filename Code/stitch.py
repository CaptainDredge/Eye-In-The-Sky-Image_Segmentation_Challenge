import skimage.io as io
import os
import argparse
import numpy as np
import glob
import warnings
io.use_plugin('tifffile')
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser(description='Stiching Patches')
parser.add_argument('--id', default=None, type=str)
parser.add_argument('--sub_id', default=None, type=str)
parser.add_argument('--dist', default='TestStichedid_', type=str)
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

class_to_idx = {str(pixel): i for i, pixel in enumerate(PIXELS)}
idx_to_class = {i: pixel for i, pixel in enumerate(PIXELS)}


def main():
    args = parser.parse_args()
    if args.id is None:
        raise Exception('Mention Source path of Patches')
    source = 'Testid_' + args.id
    destination = os.path.join(args.dist + args.sub_id)
    if not os.path.isdir(source):
        raise Exception('source file does not exist')
    if not os.path.isdir(destination):
        print('creating destination folder')
        os.mkdir(destination)

    print('source -> {}\ndestination -> {}'.format(source, destination))
    stitch_patch(source, destination, channel=9)
    print('Done ..........')


def mask_to_rgb(img):
    """
        converts 9 channel mask to 3 channel rgb mask
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    ret_img = np.zeros((img_height, img_width, 3))
    for i in range(img_height):
        for j in range(img_width):
            pixel = idx_to_class[np.argmax(img[i][j])]
            ret_img[i][j] = pixel

    print(ret_img.shape)
    return ret_img


def return_padding(img, height, width):
    """
    Return padding given image and height width of patch
    """
    h = 0 if img.shape[0] % height == 0 else height - img.shape[0] % height
    w = 0 if img.shape[1] % width == 0 else width - img.shape[1] % width
    pad_shape = [0, 0, 0]
    pad_shape[0] = (0, h)
    pad_shape[1] = (0, w)
    pad_shape[2] = (0, 0)
    return pad_shape


def pad_zeros(img, height, width):
    """
    Pads img (with 0's) to make it fit for extracting patches of
    shape height*width from it
    """
    print('input shape {}'.format(img.shape))
    pad_shape = return_padding(img, height, width)
    img = np.pad(img, pad_shape, mode='constant')
    print('output shape {}'.format(img.shape))
    return img


def make_divisor_mask(mask_height, mask_width, step):
    """
    Create a mask array defining the overlap extent of patches
    """
    mask = np.empty([mask_height, mask_width], dtype=np.uint8)
    for i in range(1, mask_height + 1):
        for j in range(1, mask_width + 1):
            mask[i - 1][j - 1] = min(i, mask_height - i + 1, step) * min(j, mask_width - j + 1, step)
    return mask


def sortKeyFunc(s):
    return int(os.path.basename(s).split('_')[2])


def stitch_patch(patch_path, recon_img_path, channel=4):
    patch_list = []
    for i in range(1, 15):
        patches = sorted(glob.glob(patch_path + '/patch_{}_*.tif'.format(i)), key=sortKeyFunc)
        patch_list.append(patches)
    for files in patch_list:
        if not files:
            continue
        else:
            stitch_specs = files[0].split('/')[-1].split('_')
            img_id = int(stitch_specs[1])
            orig_img_height = int(stitch_specs[3])
            orig_img_width = int(stitch_specs[4])
            h_stride = int(stitch_specs[5])
            patch_height = int(stitch_specs[6].split('.')[0])
            print(patch_height, h_stride)
            image = np.zeros((orig_img_height, orig_img_width, channel), dtype=np.float32)
            padding = return_padding(image, patch_height, patch_height)
            image = pad_zeros(image, patch_height, patch_height)
            h = 0
            w = 0
            patches = []
            patch_id = 0
            for name in files:
                try:
                    io.use_plugin('tifffile')
                    patch = io.imread(name)
                    patches.append(patch)
                    if image.dtype != patch.dtype:
                        image = image.astype(patch.dtype, copy=False)
                except OSError as e:
                    print(e.errno)
                    print("Some of the patches are corrupted")
            while h <= image.shape[0] - patch_height:
                w = 0
                while w <= image.shape[1] - patch_height:
                    a = image[h:h + patch_height, w:w + patch_height, :]
                    b = patches[patch_id]
                    image[h:h + patch_height, w:w + patch_height, :] += patches[patch_id]
                    w = w + h_stride
                    patch_id += 1
                h = h + h_stride
            step = patch_height // h_stride
            mask_height = image.shape[0] // h_stride
            mask_width = image.shape[1] // h_stride
            divisor_mask = make_divisor_mask(mask_height, mask_width, step)
            print("Divisor mask shape {}".format(divisor_mask.shape))

            h = 0
            w = 0
            mask_h = 0
            mask_w = 0
            print("Image shape {}".format(image.shape))

            while h <= image.shape[0] - h_stride:
                w = 0
                mask_w = 0
                while w <= image.shape[1] - h_stride:
                    image[h:h + h_stride, w:w + h_stride, :] /= divisor_mask[mask_h, mask_w]
                    w += h_stride
                    mask_w += 1
                h += h_stride
                mask_h += 1

            img = image[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], :]
            print("FinalImage shape{}".format(img.shape))
            assert img.shape == (orig_img_height, orig_img_width, channel)
            if not os.path.isdir(recon_img_path):
                os.mkdir(recon_img_path)
            if not os.path.isdir(recon_img_path + '/binary'):
                os.mkdir(recon_img_path + '/binary')
            io.imsave(recon_img_path + '/binary/' + str(img_id) + '.tif', img.astype(np.uint8), plugin='tifffile')
            rgb_mask = mask_to_rgb(img)
            io.imsave(recon_img_path + '/' + str(img_id) + '.tif', rgb_mask.astype(np.uint8), plugin='tifffile')


if __name__ == '__main__':
    main()

