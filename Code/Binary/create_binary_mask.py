import skimage.io as io
io.use_plugin('tifffile')
import os
import shutil
import json
import sys
import numpy as np
import glob
import gflags
import warnings
warnings.simplefilter("ignore", UserWarning)

def create_binary_mask(patch_source, patch_destination, mask_source, mask_destination, threshold, dict_map = None):
    """
    Takes mask source folder contating rgb ground truth masks 
    and patch source folder containing image patches and converts
    them to binary(grayscale) mask of each class
    destination folders will look like :
    ./destination/
        class-1/
            img1/
            img2/
            ..
            ..
            img9/
        class-2/
            img1/
            ..
            ..
            img9/
        ..
    """
    if not dict_map:
        dict_map = {
        'Roads': [  0,   0,   0],
        'Trees': [  0, 125,   0],
        'Grass': [  0, 255,   0],
        'Water': [0, 0, 150],
        'Building': [100, 100, 100],
        'Soil' : [150,  80,   0],
        'Pool' : [150, 150, 255],
        'Railway':  [255, 255,   0],
        'None' : [255, 255, 255] 
         }
        
    if mask_source[-1] != '/':
        mask_source = mask_source + '/'
    if mask_destination[-1] != '/':
        mask_destination = mask_destination + '/'
        
    if patch_source[-1] != '/':
        patch_source = patch_source + '/'
    if patch_destination[-1] != '/':
        patch_destination = patch_destination + '/'
        
    masks = glob.glob(mask_source + '*.tif')
    class_weights = {}
    for key, rgb in dict_map.items():
        positive_class = 0
        negative_class = 0
        maskpath = mask_destination + key + '/'
        patchpath = patch_destination + key + '/'
        if not os.path.isdir(maskpath):
            os.mkdir(maskpath)
        if not os.path.isdir(patchpath):
            os.mkdir(patchpath)
        for mask in masks:
            img = io.imread(mask)
            img_id = mask.split('/')[-1].split('.')[0]
            
            class_mask = img[:,:,:] == np.array(rgb)
            final_mask = class_mask[...,0] * class_mask[...,1] * class_mask[...,2]
            final_mask = final_mask.astype(np.uint8)
            positive_class += (final_mask == 1).sum()
            negative_class += (final_mask == 0).sum()
            class_percent = (final_mask == 1).sum()/((final_mask ==1).sum() + (final_mask == 0).sum())
            if (class_percent > float(threshold)):
                io.imsave(maskpath + str(img_id)+'.tif', final_mask, plugin='tifffile')
                shutil.copy((patch_source + str(img_id)+'.tif'), patchpath)
        class_weights[key] = positive_class/negative_class
    
    with open('class_weights.json', 'w') as fp:
        json.dump(class_weights, fp, sort_keys=True, indent=4)
                
            
if __name__ == '__main__':
    gflags.DEFINE_string(
        'patch_source', 'data/train/images_256x256/', 'image patch source folder')
    gflags.DEFINE_string('patch_destination', 'data/train/BinImages_256x256', 'image patch destination folder')
    gflags.DEFINE_string(
        'mask_source', 'data/train/train-mask_256x256', 'image patch source folder')
    gflags.DEFINE_string('mask_destination', 'data/train/binmasks_256x256', 'binary mask destination folder')
    gflags.DEFINE_string('threshold', -0.01, 'threshold percentage for class occurence')
    gflags.FLAGS(sys.argv)

    patch_source = gflags.FLAGS.patch_source
    patch_destination = gflags.FLAGS.patch_destination
    
    mask_source = gflags.FLAGS.mask_source
    mask_destination = gflags.FLAGS.mask_destination
    
    threshold = gflags.FLAGS.threshold

    if not os.path.isdir(patch_source):
        raise Exception('patch source file does not exist')
    if not os.path.isdir(patch_destination):
        print('creating patch destination folder')
        os.mkdir(patch_destination)
        
    if not os.path.isdir(mask_source):
        raise Exception('mask source file does not exist')
    if not os.path.isdir(mask_destination):
        print('creating mask destination folder')
        os.mkdir(mask_destination)

    print('patch source -> {}\n patch destination -> {}'.format(patch_source, patch_destination))
    print('mask source -> {}\n mask destination -> {}'.format(mask_source, mask_destination))

    create_binary_mask(patch_source, patch_destination, mask_source, mask_destination, threshold)

    print('Done ..........')