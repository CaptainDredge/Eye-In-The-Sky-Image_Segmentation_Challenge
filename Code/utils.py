"""
This file contains Utility Functions such as Dataloader, Average Meter, Metrics.
"""
import os
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt


def to_image(image):
    batch_size = image.size(0)
    out_channels = 3
    h = image.size(1)
    w = image.size(2)
    img = np.zeros((batch_size, out_channels, h, w), dtype=np.uint8)
    pixels = [np.array([0, 0, 0]),
              np.array([0, 125, 0]),
              np.array([0, 255, 0]),
              np.array([100, 100, 100]),
              np.array([150, 80, 0]),
              np.array([150, 150, 255]),
              np.array([0, 0, 150]),
              np.array([255, 255, 0]),
              np.array([255, 255, 255]),
              ]
    for k in range(batch_size):
        for i in range(h):
            for j in range(w):
                img[k, :, i, j] = pixels[image[k, i, j]]

    return img


def save_checkpoint(state, filename):
    torch.save(state, filename)


def add_image(image, writer, global_step, name):
    image = to_image(image)
    writer.add_image(name, image, global_step)


def accuracy(target, output):
    total_pixel_without_none = (target != 8).sum().item()
    acc = 0.
    for i in range(8):
        x1 = (target == i)
        x2 = (output == i)
        acc += (x1 * x2).sum().item()
    return acc / total_pixel_without_none


def conf_matrix(target, output):
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    return confusion_matrix(target, output, labels=labels)


def kappa_score(target, output):
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    return cohen_kappa_score(target, output, labels=labels)


def plot_confusion_matrix(conf_mat,
                          figsize=(13, 13)):
    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    cmap = plt.cm.Blues
    matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    fig.colorbar(matshow)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = format(normed_conf_mat[i, j], '.5f')
            ax.text(x=j, y=i, s=cell_text, va='center', ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax


def plot_confusion_matrix_nm(conf_mat,
                             figsize=(13, 13)):
    total_samples = conf_mat.sum()
    normed_conf_mat = conf_mat.astype('float') / total_samples
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    cmap = plt.cm.Blues
    matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    fig.colorbar(matshow)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = format(normed_conf_mat[i, j], '.5f')
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

    def __init__(self, path, transform, finetune=False):
        super(TrainDataset, self).__init__()
        self.path = path
        self.transform = transform
        if finetune:
            self.list = get_finetune_list(os.listdir(os.path.join(self.path, 'images')), threshold=0.4)
        else:
            self.list = os.listdir((os.path.join(self.path, 'images')))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        image = tiff.imread(os.path.join(self.path, 'images', self.list[item])).astype(np.float32)
        mask = tiff.imread(os.path.join(self.path, 'masks', self.list[item]))
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

    def __init__(self, path, transform=None):
        super(TestDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.list = os.listdir(self.path)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        image = tiff.imread(os.path.join(self.path, self.list[item])).astype(np.float32)
        mean = [206.75503457700066, 282.4707482803494, 174.54941130734392, 346.48021589897877]
        std = [77.24070766692714, 129.3841999574279, 109.96606967828127, 216.19068282723546]
        image = (image - mean) / std
        image = np.ascontiguousarray(image.transpose((2, 0, 1)), dtype=np.float32)
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
        self.cm = np.zeros((8, 8), dtype=np.float32)

    def update(self, val):
        self.cm += val


def get_finetune_list(files, threshold=0.5, class1=0, class2=3):
    ret_files = []
    for file in files:
        gt = tiff.imread('./data/train/masks/{}'.format(file))
        sum = gt[:, :, class1].sum() + gt[:, :, class2].sum()
        tot = gt.shape[0] * gt.shape[1]
        if sum / tot > threshold:
            ret_files.append(file)
    if len(ret_files):
        raise Exception('Threshold too Much')
    return ret_files

