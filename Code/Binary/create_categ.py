import skimage.io as io
io.use_plugin('tifffile')
import glob
import numpy as np
import os
import gflags
import sys
import matplotlib.pyplot as plt

def create_categorical_mask(source, destination, dict_map = None):
    """
    Takes source folder contating rgb ground truth masks
    and converts them to categorical mask(0,1,2,...) of each class
    """
    if not dict_map:
        dict_map = {
        'None' : [255, 255, 255],
        'Roads': [  0,   0,   0],
        'Trees': [  0, 125,   0],
        'Grass': [  0, 255,   0],
        'Water': [0, 0, 150],
        'Building': [100, 100, 100],
        'Soil' : [150,  80,   0],
        'Pool' : [150, 150, 255],
        'Railway':  [255, 255,   0]
         }
        
    if source[-1] != '/':
        source = source + '/'
    if destination[-1] != '/':
        destination = destination + '/'
            
    masks = glob.glob(source + '*.tif')
    
    for mask in masks:
        img = io.imread(mask)
        img_id = mask.split('/')[-1].split('.')[0]
        
        idx = 0
        categorical_mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
        for key, rgb in dict_map.items():
            class_mask = img[:,:,:] == np.array(rgb)
            final_mask = class_mask[...,0] * class_mask[...,1] * class_mask[...,2]
            final_mask = final_mask.astype(np.uint8)
            categorical_mask += final_mask*idx
            idx += 1
        assert np.any(categorical_mask > 8) == False
        io.imsave(destination+str(img_id)+'.tif', categorical_mask, plugin='tifffile')
        print("categorical mask of image {} saved".format(img_id))
        

if __name__ == '__main__':
    gflags.DEFINE_string(
        'source', 'The-Eye-in-the-Sky-dataset/gt/', 'ground_truth source folder')
    gflags.DEFINE_string('destination', 'The-Eye-in-the-Sky-dataset/categ_gt', 'binary mask destination folder')
    gflags.FLAGS(sys.argv)

    source = gflags.FLAGS.source
    destination = gflags.FLAGS.destination

    if not os.path.isdir(source):
        raise Exception('source file does not exist')
    if not os.path.isdir(destination):
        print('creating destination folder')
        os.mkdir(destination)

    print('source -> {}\ndestination -> {}'.format(source, destination))

    create_categorical_mask(source, destination)

    print('Done ..........')