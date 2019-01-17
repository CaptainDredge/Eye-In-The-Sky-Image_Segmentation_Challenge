import numpy as np
import pandas as pd
from skimage.external import tifffile
import skimage.io as io
io.use_plugin('tifffile')
import PIL as Image
import glob
import gflags
import sys
import os
import warnings
warnings.simplefilter("ignore", UserWarning)

# Unique pixel values for masks
PIXELS = [\

np.array([0,0,0]),

np.array([0,0,150]),

np.array([0,125,0]),

np.array([0,255,0]),

np.array([100,100,100]),

np.array([150,80,0]),

np.array([150,150,255]),

np.array([255,255,0]),

np.array([255,255,255]),

]

class_to_idx  = {str(pixel):i for i,pixel in enumerate(PIXELS)}

idx_to_class = {i:pixel for i,pixel in enumerate(PIXELS)}

def rgb_to_mask(img):

    """
        converts an rgb mask into 9 channel training mask
    """

    img_height = img.shape[0]
    img_width = img.shape[1]

    # Flatten imgage spatially so it becomes list of pixels([R,G,B])

    img = img.reshape(img_height*img_width,img.shape[2])

    # image to be returned

    ret_img = np.zeros((img_height*img_width, 9))

    for i,pixel in enumerate(img):
        global class_to_idx
        # idx of true class
        idx = class_to_idx[str(pixel)]
        # set thr idx of true class 1
        ret_img[i][idx] = 1
        
    # Reshape image back to its oiginal shape
    ret_img = ret_img.reshape(img_height, img_width,9)

    print(ret_img.shape)
    return ret_img

def create_patches(img, patch_height=144, patch_width=144, stride=0.5):
    """ input -> image (none,none,4), patch dimensions
    outpout -> patches of desired dimensions (patch_height, patch_width, 4)
    """
    h_stride = int(max(1, patch_height * stride))
    w_stride = int(max(1, patch_width * stride))

    patch_param = {}
    patch_param['image_height'] = img.shape[0]
    patch_param['image_width'] = img.shape[1]
    patch_param['h_stride'] = h_stride
    patch_param['w_stride'] = w_stride
    patch_param['patch_height'] = patch_height
    patch_param['patch_width'] = patch_width

    h = 0
    w = 0

    img = pad_image(img, patch_height, patch_width)

    patches = []

    while h < img.shape[0] - h_stride:
        w = 0
        while w < img.shape[1] - w_stride:
            patches.append(img[h:h+patch_height, w:w+patch_width, :])
            w = w + w_stride
        h = h+h_stride

    # for (i,patch) in enumerate(patches):
    #tiffile.imsave('patch{}.tif'.format(i), patch)
    return patches, patch_param

def return_padding(img, height, width):
    " Return padding given image and height, width of patch"
    h = 0 if img.shape[0]%height == 0 else height - img.shape[0]%height
    w = 0 if img.shape[1]%width == 0 else width - img.shape[1]%width
    pad_shape = tuple(np.zeros((len(img.shape),2),dtype=np.uint16))
    pad_shape = [tuple(x) for x in pad_shape]
    h_left  = h//2
    h_right = h - h_left
    w_left  = w//2
    w_right = w - w_left
    pad_shape[0] = (int(h_left),int(h_right))
    pad_shape[1] = (int(w_left),int(w_right))
    
    print("pad shape is {}".format(pad_shape))
    return pad_shape

def pad_image(img, height, width, channels=4, mode='symmetric'):
    """Pads img to make it fit for extracting patches of 
    shape height*width from it
    mode -> constant, reflect 
    constant -> pads ith 0's
    reflect -> pads with reflection of image
    """
    print('input shape {}'.format(img.shape))
    pad_shape = return_padding(img, height, width)
    img = np.pad(img,pad_shape,mode=mode)
    print('output shape {}'.format(img.shape))
    return img  

def create_binary_mask(source, destination, dict_map = None):
    """
    Takes source folder contating rgb ground truth masks
    and converts them to binary(grayscale) mask of each class
    destination folder will look like :
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
        
    if source[-1] != '/':
        source = source + '/'
    if destination[-1] != '/':
        destination = destination + '/'
    masks = glob.glob(source + '*.tif')
    for key, rgb in dict_map.items():
        path = destination + key
        if not os.path.isdir(path):
            os.mkdir(path)
        for mask in masks:
            img = io.imread(mask)
            img_id = mask.split('/')[-1].split('.')[0]
            
            class_mask = img[:,:,:] == np.array(rgb)
            final_mask = class_mask[...,0] * class_mask[...,1] * class_mask[...,2]
            final_mask = final_mask.astype(np.uint8)
            io.imsave(path+'/'+str(img_id)+'.tif', final_mask, plugin='tifffile')
            
def create_dataset(source, destination, height, width, VAL_IMGS):
    """
        Reads TIFF files from soure and extracts patches
        from it and saves it to destination
    """
    train_destination = os.path.join(destination, 'train')
    valid_destination = os.path.join(destination, 'valid')
    print(train_destination, valid_destination)
    if not os.path.isdir(train_destination):
        os.mkdir(train_destination)
    if not os.path.isdir(valid_destination):
        os.mkdir(valid_destination)
        
    #target dir name
    TRAIN_PATH_IMG = "images_{}x{}".format(height, width)
    TRAIN_PATH_MASK = "train-mask_{}x{}".format(height, width)
    TRAIN_PATH_MULTI_MASK = "masks"
    TRAIN_PATH_BINARY_MASK = "binmasks"
    TRAIN_PATH_IMG = os.path.join(train_destination, TRAIN_PATH_IMG)
    TRAIN_PATH_MASK = os.path.join(train_destination, TRAIN_PATH_MASK)
    TRAIN_PATH_MULTI_MASK = os.path.join(train_destination, TRAIN_PATH_MULTI_MASK)
    TRAIN_PATH_BINARY_MASK = os.path.join(train_destination, TRAIN_PATH_BINARY_MASK)

    VALID_PATH_IMG = "images_{}x{}".format(height, width)
    VALID_PATH_MASK = "valid-mask_{}x{}".format(height, width)
    VALID_PATH_MULTI_MASK = "masks"
    VALID_PATH_BINARY_MASK = "binmasks"
    VALID_PATH_IMG = os.path.join(valid_destination, VALID_PATH_IMG)
    VALID_PATH_MASK = os.path.join(valid_destination, VALID_PATH_MASK)
    VALID_PATH_MULTI_MASK = os.path.join(valid_destination, VALID_PATH_MULTI_MASK)
    VALID_PATH_BINARY_MASK = os.path.join(valid_destination, VALID_PATH_BINARY_MASK)
    
    # If target dir not present create it
    if not os.path.isdir(TRAIN_PATH_IMG):
        os.mkdir(TRAIN_PATH_IMG)
    if not os.path.isdir(TRAIN_PATH_MASK):
        os.mkdir(TRAIN_PATH_MASK)
    if not os.path.isdir(TRAIN_PATH_MULTI_MASK):
        os.mkdir(TRAIN_PATH_MULTI_MASK)
    if not os.path.isdir(TRAIN_PATH_BINARY_MASK):
        os.mkdir(TRAIN_PATH_BINARY_MASK)

    if not os.path.isdir(VALID_PATH_IMG):
        os.mkdir(VALID_PATH_IMG)
    if not os.path.isdir(VALID_PATH_MASK):
        os.mkdir(VALID_PATH_MASK)
    if not os.path.isdir(VALID_PATH_MULTI_MASK):
        os.mkdir(VALID_PATH_MULTI_MASK)
    if not os.path.isdir(VALID_PATH_BINARY_MASK):
        os.mkdir(VALID_PATH_BINARY_MASK)
    

    img_source = os.path.join(source, 'sat')
    mask_source = os.path.join(source, 'gt')
    imgs = os.listdir(img_source)
    masks = os.listdir(mask_source)

    for i, (img, mask) in enumerate(zip(imgs, masks), 1):
        print('processing img {}'.format(i))
        print(img, mask)
        img_patches, img_params = create_patches(
            io.imread(os.path.join(img_source, img)), patch_height=height, patch_width=width)
        mask_patches, mask_params = create_patches(
            io.imread(os.path.join(mask_source, mask)), patch_height=height, patch_width=width)
        print('creating multi channel mask ......') 
        multi_channel_patches, multi_mask_params = create_patches(rgb_to_mask(io.imread(os.path.join(mask_source,mask))), patch_height=height, patch_width=width)

        
        for j, (img_patch, mask_patch, multi_mask_patch) in enumerate(zip(img_patches, mask_patches, multi_channel_patches)):
            if int(img[:-4]) not in VAL_IMGS:
                print('train image {}'.format(img))
                io.imsave(TRAIN_PATH_IMG+'/patch_{}_{}_{}_{}_{}_{}_{}_{}.tif'.format(
                img[:-4], j, img_params['image_height'], img_params['image_width'], img_params['h_stride'], img_params['w_stride'], img_params['patch_height'], img_params['patch_width']), img_patch,  plugin='tifffile')
                
                io.imsave(TRAIN_PATH_MASK+'/patch_{}_{}_{}_{}_{}_{}_{}_{}.tif'.format(
                img[:-4], j, mask_params['image_height'], mask_params['image_width'], mask_params['h_stride'], mask_params['w_stride'], mask_params['patch_height'], mask_params['patch_width']), mask_patch,  plugin='tifffile')
                
                io.imsave(TRAIN_PATH_MULTI_MASK+ '/patch_{}_{}_{}_{}_{}_{}_{}_{}.tif'.format(
                img[:-4], j, multi_mask_params['image_height'], multi_mask_params['image_width'], multi_mask_params['h_stride'], multi_mask_params['w_stride'], multi_mask_params['patch_height'], multi_mask_params['patch_width']), multi_mask_patch,  plugin='tifffile')
                
            else:
                print('valid image {}'.format(img))
                io.imsave(VALID_PATH_IMG+'/patch_{}_{}_{}_{}_{}_{}_{}_{}.tif'.format(
                img[:-4], j, img_params['image_height'], img_params['image_width'], img_params['h_stride'], img_params['w_stride'], img_params['patch_height'], img_params['patch_width']), img_patch,  plugin='tifffile')
                
                io.imsave(VALID_PATH_MASK+'/patch_{}_{}_{}_{}_{}_{}_{}_{}.tif'.format(
                img[:-4], j, mask_params['image_height'], mask_params['image_width'], mask_params['h_stride'], mask_params['w_stride'], mask_params['patch_height'], mask_params['patch_width']), mask_patch,  plugin='tifffile')
                
                io.imsave(VALID_PATH_MULTI_MASK+ '/patch_{}_{}_{}_{}_{}_{}_{}_{}.tif'.format(
                img[:-4], j, multi_mask_params['image_height'], multi_mask_params['image_width'], multi_mask_params['h_stride'], multi_mask_params['w_stride'], multi_mask_params['patch_height'], multi_mask_params['patch_width']), multi_mask_patch,  plugin='tifffile')
        
        
    create_binary_mask(TRAIN_PATH_MASK, TRAIN_PATH_BINARY_MASK)
    create_binary_mask(VALID_PATH_MASK, VALID_PATH_BINARY_MASK)
        
if __name__ == '__main__':
    gflags.DEFINE_string(
        'source', './The-Eye-in-the-Sky-dataset', 'source folder')
    gflags.DEFINE_string('destination', './data', 'destination folder')
    gflags.DEFINE_string('height', 256, 'height of patches')
    gflags.DEFINE_string('width', 256, 'width of patches')
    gflags.FLAGS(sys.argv)

    source = gflags.FLAGS.source
    destination = gflags.FLAGS.destination
    height = gflags.FLAGS.height
    width = gflags.FLAGS.width

    if not os.path.isdir(source):
        #print('source file do not exist')
        raise Exception('source file does not exies')
    if not os.path.isdir(destination):
        print('creating destination folder')
        os.mkdir(destination)
        
    VALID_IMGS = [1, 2, 5, 12]

    print('source -> {}\ndestination -> {}'.format(source, destination))

    create_dataset(source, destination, height, width, VALID_IMGS)

    print('Done ..........')
