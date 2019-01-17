import argparse
import os
import numpy as np
from skimage import io
import torch

from torch.utils.data import DataLoader
from utils import TestDataset
from Models import Res

parser = argparse.ArgumentParser(description='Predicting on Patches')
parser.add_argument('--id', default=None, type=str)
parser.add_argument('--sub_id', default=None, type=str)
parser.add_argument('--mode', default='valid', type=str)


def main():
    args = parser.parse_args()
    assert args.id is not None, "No train ID provided !"
    load_path = 'Trainid_' + args.id
    destination = 'Testid_' + args.sub_id
    if args.mode == 'valid':
        test_dataset_path = './data/valid/images'
    else:
        test_dataset_path = 'testdata/'
    model = Res(n_ch=4, n_classes=9)
    batch_size = 8
    # loading chekpoint
    weight_path = sorted(os.listdir(load_path + '/Checkpoint/'), key=lambda x: float(x[:-8]))[0]
    checkpoint = torch.load(load_path + '/Checkpoint/' + weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded Checkpoint of loss: {}'.format(weight_path[:-8]))
    predict_patches(model, test_dataset_path, batch_size, destination)
    print('Done ....!')


def predict_patches(model, test_path, batch_size, destination):
    # Creating destination
    if not os.path.isdir(destination):
        os.mkdir(destination)
    # Load dataset
    test_dataset = TestDataset(path=test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    size_test = len(test_dataloader)
    print('Number of Test Images {}'.format(size_test))
    #model = model.cuda()
    soft = torch.nn.Softmax2d()
    with torch.no_grad():
        for i, (input, name) in enumerate(test_dataloader):
            input = input.cuda()
            # print('input shape {},\ntarget shape {}'.format(input.shape, target.shape))
            output = model(input)
            output = soft(output)
            output = output.numpy()
            output = np.moveaxis(output, 1, -1)
            print('output shape {}'.format(output.shape))
            print('output mask shape {}'.format(output.shape))
            print('destination {}\n name {}'.format(destination, name))
            for j, img in enumerate(output):
                io.imsave(os.path.join(destination, name[j]), img)
    print('Patches Predicted')


if __name__ == '__main__':
    main()

