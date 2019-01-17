"""
This file contains Utility Functions such as Dataloader, Average Meter, Metrics.
"""
import os
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import torch.nn.functional as F



def to_image(image):
    batch_size = image.size(0)
    out_channels = 3
    h = image.size(1)
    w = image.size(2)
    img = np.zeros((batch_size, out_channels, h, w), dtype=np.uint8)
    pixels = [np.array([0, 0, 0]), np.array([0, 0, 150]), np.array([0, 125, 0]), np.array([0, 255, 0]),
              np.array([100, 100, 100]), np.array([150, 80, 0]), np.array([150, 150, 255]), np.array([255, 255, 0]),
              np.array([255, 255, 255])]
    for k in range(batch_size):
        for i in range(h):
            for j in range(w):
                img[k, :, i, j] = pixels[image[k, i, j]]

    return img


def save_checkpoint(state, filename):
    torch.save(state, filename)


def add_image(image, writer, global_step, ground_truth=True,name='Validation GT'):
    if ground_truth:
        #image = to_image(image)
        writer.add_image(name, image, global_step)

    else:
        #image = to_image(image)
        writer.add_image('Validation Output', image, global_step)


def accuracy(target, output):
    return accuracy_score(target,output)


def f1score(target, output):
    labels = [0, 1]
    return f1_score(target, output, labels=labels)


def conf_matrix(target, output):
    labels = [0,1]
    return confusion_matrix(target, output, labels=labels)

def kappa_score(target, output):
    return cohen_kappa_score(target, output)

def plot_confusion_matrix(conf_mat,
                          figsize=(8, 8)):
    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    cmap = plt.cm.Blues
    matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    fig.colorbar(matshow)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = format(normed_conf_mat[i, j], '.8f')
            ax.text(x=j, y=i, s=cell_text, va='center', ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax


class TrainDataset(Dataset):
    """
    Dataset Class for Train and Valid
    The Dataset should be arranged in this way:

    Training Samples:
    path/images/x.ext
    path/images/y.ext
    Mask:
    path/masks/x.ext
    path/masks/y.ext

    Note: Name of Mask and Input Image should be same for one training sample

    Returns :,C,H,W and dtype of Image is float32 , dtype of Masks is double
    """

    def __init__(self, path, transform, class_name):
        super(TrainDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.class_name = class_name
        self.difference = 0
        self.list = os.listdir(os.path.join(self.path)
        self.mask_source = os.path.join(self.path, 'binmasks_256x256/{}'.format(class_name))
        self.weights = make_weights_for_balanced_classes(self.list, 2, 0.01, self.mask_source)
        #print("Weights {}".format(self.weights))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        image = tiff.imread(os.path.join(self.path, 'BinImages_256x256/{}'.format(self.class_name), self.list[item])).astype(np.float32))
                               
        mean = np.array([206.75503457700066, 282.4707482803494, 174.54941130734392, 346.48021589897877])
        std = np.array([77.24070766692714, 129.3841999574279, 109.96606967828127, 216.19068282723546])
        image = (image - mean) / std
        mask = tiff.imread(os.path.join(self.path, 'binmasks_256x256/{}'.format(self.class_name), self.list[item]))
        #print(mask.shape)
        sample = {'image': image, 'mask': mask}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class TestDataset(Dataset):
    """
    Dataset Class for Test
    Dataset should be arranged in this way:
    test/x.ext
    test/y.ext
    Returns Test Image and its Name
    """

    def __init__(self, path, transform):
        super(TestDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.list = os.listdir(self.path)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        image = tiff.imread(os.path.join(self.path, self.list[item])).astype(np.float32)
        image = (image - np.mean(image)) / np.std(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.list[item]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageCM(object):
    def __init__(self):
        self.count = 0
        self.cm = np.zeros((9, 9), dtype=np.float32)

    def update(self, val, n):
        self.count += n
        self.cm += val
        self.avg = self.cm

def make_weights_for_balanced_classes(images, nclasses, threshold, mask_source):                        
    eps = 1e-3
    count = [0] * nclasses
    img_classes = []
    for item in images:
        mask = tiff.imread('{}/{}'.format(mask_source,item))
        if ((mask == 1).sum()/((mask==0).sum() + (mask==1).sum())) > threshold:                                            
            count[1] += 1
            img_classes.append(1)
        else:
            count[0] += 1
            img_classes.append(0)
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/(float(count[i]) + eps)
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[img_classes[idx]]                                  
    return weight



def plot_confusion_matrix_values(conf_mat,
                          figsize=(8, 8)):
    total_samples = conf_mat.sum(axis=1)[:, np.newaxis].sum()
    normed_conf_mat = conf_mat.astype('float') / total_samples
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    cmap = plt.cm.Blues
    matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    fig.colorbar(matshow)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = format(normed_conf_mat[i, j], '.8f')
            ax.text(x=j, y=i, s=cell_text, va='center', ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax