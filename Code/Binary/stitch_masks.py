import skimage.io as io
import os
import json
import gflags
import sys
import numpy as np
import glob
io.use_plugin('tifffile')
import errno
import warnings
warnings.simplefilter("ignore", UserWarning)

# Unique pixel values for masks
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

def mask_to_rgb(img, class_names, threshold):
    """
        converts arbitrary channel mask to 3 channel rgb mask
    """
    print("converting mask to rgb")
    print("multi channel mask shape{}".format(img.shape))
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_channel_count = img.shape[2]
    
    #img = np.moveaxis(img, 0,-1)
    # Flatten imgage spatially so it becomes list of pixels([R,G,B])
    img = img.reshape(img_height*img_width, img_channel_count)

    # image to be returned
    ret_img = np.zeros((img_height*img_width, 3))
    for k, mask in enumerate(img):
        #print('k = {}\nmask shape {}'.format(k,mask.shape))
        #global idx_to_class
        if(np.max(mask)< threshold):
            pixel = np.array(dict_map['None'])
        else:
            class_name = class_names[np.argmax(mask)]
            pixel = np.array(dict_map[class_name])

        # set thr idx of true class 1
        ret_img[k] = pixel
    # Reshape image back to its oiginal shape
    ret_img = ret_img.reshape(img_height, img_width,3)
    print("return mask shape {}".format(ret_img.shape))
    return ret_img

def return_padding(img, height, width):
    " Return padding given image and height width of patch"
    h = 0 if img.shape[0]%height == 0 else height - img.shape[0]%height
    w = 0 if img.shape[1]%width == 0 else width - img.shape[1]%width
    pad_shape = tuple(np.zeros((len(img.shape),2),dtype=np.uint16))
    pad_shape = [tuple(x) for x in pad_shape]
    h_left  = int(h//2)
    h_right = int(h - h_left)
    w_left  = int(w//2)
    w_right = int(w - w_left)
    pad_shape[0] = (h_left,h_right)
    pad_shape[1] = (w_left,w_right)
    
    return pad_shape

def pad_zeros(img, height, width, channels=4):
    """Pads img (with 0's) to make it fit for extracting patches of 
    shape height*width from it
    """
    print('input shape {}'.format(img.shape))
    pad_shape = return_padding(img, height, width)
    img = np.pad(img,pad_shape,mode='constant')
    print('output shape {}'.format(img.shape))
    return img 

def make_divisor_mask(mask_height, mask_width, step):
    
    """ Create a mask array defining the overlap extent of patches"""
    mask = np.empty([mask_height, mask_width], dtype=np.uint16)
    for i in range(1,mask_height+1):
        for j in range(1,mask_width+1):
            mask[i-1][j-1] = min(i,mask_height-i+1,step)*min(j,mask_width-j+1,step)
    return mask

def sortKeyFunc(s):
    return int(os.path.basename(s).split('_')[2])

def stitch_patch(patch_path, recon_img_path, class_names, threshold, channel=4):
    
    patch_list = []
    for i in range(1,15):
        patches = sorted(glob.glob(patch_path+'/patch_{}_*.tif'.format(i)), key=sortKeyFunc)
        patch_list.append(patches)
    for files in patch_list:
        if not files:
            continue
        else:
            stitch_specs = files[0].split('/')[-1].split('_')
            img_id = int(stitch_specs[1])
            orig_img_height = int(stitch_specs[3])
            orig_img_width  = int(stitch_specs[4])
            h_stride = int(stitch_specs[5])
            w_stride = int(stitch_specs[6])
            patch_height = int(stitch_specs[7])
            patch_width = int(stitch_specs[8].split('.')[0])
            img_dtype = np.float32
            image     = np.zeros((orig_img_height, orig_img_width, channel), dtype = img_dtype)
            padding   = return_padding(image, patch_height, patch_width)
            image     = pad_zeros(image, patch_height, patch_width, channel)
            h = 0
            w = 0
            patches = []
            patch_id =0
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

            while h <= image.shape[0]-patch_height:
                w = 0
                while w <= image.shape[1]-patch_width:
                    image[h:h+patch_height, w:w+patch_width, :] += patches[patch_id]
                    w = w + w_stride
                    patch_id+=1
                h = h+h_stride
            if(h_stride==w_stride):
                step = patch_height//h_stride
            else:
                print("Unequal strides are not yet suppported")

            mask_height = image.shape[0]//h_stride
            mask_width  = image.shape[1]//w_stride
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
                while w <= image.shape[1] - w_stride:
                    image[h:h+h_stride, w:w+w_stride,:] /= divisor_mask[mask_h,mask_w]
                    w += w_stride
                    mask_w +=1
                h += h_stride
                mask_h +=1

            img = image[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1],:]
            print("FinalImage shape{}".format(img.shape))
            assert img.shape == (orig_img_height, orig_img_width, channel)
            
            binary_recon_img_path = recon_img_path+'binary'
            
            if not os.path.isdir(binary_recon_img_path):
                os.mkdir(binary_recon_img_path)
            
            io.imsave(binary_recon_img_path+'/'+str(img_id)+'.tif', img.astype(np.float32), plugin='tifffile')

            if not os.path.isdir(recon_img_path):
                os.mkdir(recon_img_path)
            rgb_mask = mask_to_rgb(img, class_names, threshold)

            io.imsave(recon_img_path+'/'+str(img_id)+'.tif', rgb_mask.astype(np.uint16), plugin='tifffile')
            
if __name__ == '__main__':
    gflags.DEFINE_string('sub_id', 'testing', 'submission id')
    gflags.DEFINE_string('threshold', 0.5, 'threshold for deciding background')
    gflags.FLAGS(sys.argv)
    
    for mode in ['train','valid','testing']:
        source = 'ProbTest_id' + str(gflags.FLAGS.sub_id) + '/' + mode + '/patch_256x256/'
        destination = 'Test_id' + str(gflags.FLAGS.sub_id) + 'stitched_patch/' + mode + '/'
        threshold = float(gflags.FLAGS.threshold)

        if not os.path.isdir(source):
            continue
        if not os.path.isdir(destination):
            print('creating destination folder')
            os.makedirs(destination)

        with open('./predict_jsons/' + str(gflags.FLAGS.sub_id)+'.json', 'r') as fp:
            class_names = json.load(fp)
        assert isinstance(class_names, (list,))

        print('source -> {}\ndestination -> {}'.format(source, destination))
        print("Number of classes {}".format(len(class_names)))

        stitch_patch(source, destination, class_names, threshold, channel=len(class_names))
    

    print('Done ..........')