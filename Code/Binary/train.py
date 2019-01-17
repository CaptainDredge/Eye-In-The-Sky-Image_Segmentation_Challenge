import gflags
import sys
import os
import json
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils import TestDataset, TrainDataset, AverageMeter, AverageCM, add_image, save_checkpoint, f1score, accuracy, \
    conf_matrix, plot_confusion_matrix, kappa_score, make_weights_for_balanced_classes, plot_confusion_matrix_values
from torchvision import transforms
from transforms import  ImgAugTransform,ToTensor
from Unet import Unet
from networks.shivaunet import ConvNet
from networks.dinknet import DinkNet34
from loss import SoftDiceLoss, LogDiceLoss, FocalLoss, f2loss
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')


def main():
    gflags.DEFINE_string('id', None, 'ID for Training')
    gflags.DEFINE_string('epoch', 25, 'Number of Epochs')
    gflags.DEFINE_string('pretrained', None, 'Pretrained for Resuming Training')
    gflags.DEFINE_string('threshold', 0.5, 'Threshold probability for predicting class')
    gflags.DEFINE_string('batchsize', 128, 'Batch Size')
    gflags.DEFINE_string('gpu', True, 'Use GPU or Not')
    gflags.DEFINE_string('lr', 0.001, 'Learning Rate')
    gflags.DEFINE_string('class_name', 'None', 'class name')
    gflags.FLAGS(sys.argv)
    
    # Directory Path for saving weights of Trained Model
    save_path = 'Train_id' + str(gflags.FLAGS.id)
    class_name = gflags.FLAGS.class_name
    threshold = gflags.FLAGS.threshold
    class_name = gflags.FLAGS.class_name
    writer = SummaryWriter('./runs/{}'.format(gflags.FLAGS.id))
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + '/Checkpoint')

    train_dataset_path = 'data/train'
    val_dataset_path = 'data/valid'
    train_transform = transforms.Compose([
        ImgAugTransform(),
        ToTensor()
    ])
    valid_transform = transforms.Compose([
        ToTensor()
    ])
    
    train_dataset = TrainDataset(path=train_dataset_path, transform=valid_transform, class_name=class_name)
    val_dataset = TrainDataset(path=val_dataset_path, transform=valid_transform, class_name=class_name)
    
    sampler = WeightedRandomSampler(torch.DoubleTensor(train_dataset.weights), len(train_dataset.weights))
    
    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                                pin_memory=True, num_workers=4)

    size_train = len(train_dataloader)
    size_val = len(val_dataloader)
    
    print('Number of Training Images: {}'.format(size_train))
    print('Number of Validation Images: {}'.format(size_val))
    
    # Reads class weights from a Json file
    with open('class_weights.json', 'r') as fp:
        class_weights = json.load(fp)
        
    weight = torch.tensor([1/class_weights[class_name]])
    start_epoch = 0
    
    if class_name in ['Roads', 'Railway']:
        model = DinkNet34(num_classes = 1)
    else:   
        model = Unet(n_ch=4, n_classes=1)
    
    if pretrained is not None:
        criterion = FocalLoss()
    else:
        criterion = LogDiceLoss()
    criterion1 = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(gflags.FLAGS.lr))

    if gflags.FLAGS.gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion1 = criterion1.cuda()

    if gflags.FLAGS.pretrained is not None:
        weight_path = sorted(os.listdir('./weights/' + save_path+ '/Checkpoint/'), key=lambda x:float(x[:-8]))[0]
        checkpoint = torch.load('./weights/' + save_path + '/Checkpoint/' + weight_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded Checkpoint of Epoch: {}'.format(gflags.FLAGS.weight))

    for epoch in range(start_epoch, int(gflags.FLAGS.epoch) + start_epoch):
        print("epoch {}".format(epoch))
        train(model, train_dataloader, criterion, criterion1, optimizer, epoch, writer, size_train, threshold)
        print('')
        val_loss = val(model, val_dataloader, criterion, criterion1, epoch, writer, size_val, threshold)
        print('')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename= save_path + '/Checkpoint/' + str(val_loss) + '.pth.tar')
    writer.export_scalars_to_json(save_path + 'log.json')


def train(model, train_dataloader, criterion, criterion1, optimizer, epoch, writer, size_train, threshold):
    total_loss = AverageMeter()
    dice_loss = AverageMeter()
    ce_loss = AverageMeter()
    acc = AverageMeter()
    eps = 1e-2

    for i, sample in enumerate(train_dataloader):
        input, target = sample['image'].cuda(), sample['mask'].long().cuda()
        output = model(input)
        
        loss1 = criterion(output, target.float())
        target = torch.argmax(target, 1)
        loss2 = criterion1(output, target)
        loss = loss1 + loss2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss.update(loss.item(), input.size(0))
        dice_loss.update(loss1.item(), input.size(0))
        ce_loss.update(loss2.item(), input.size(0))
        
        output = torch.argmax(output, 1)
        flat_target = target.view(-1)
        flat_output = output.view(-1)
        
        acc.update(accuracy(flat_target, flat_output), input.size(0))
        kappa = 2*(acc.avg - 0.5) # kappa for binary case
        
        sys.stdout.write('\r')
        sys.stdout.write('Training Epoch: [{0}][{1}/{2}]\t'
                         'Total Loss  ({total_loss.avg:.4f})\t'
                         'Dice Loss ({dice_loss.avg:.4f})\t'
                         'Cross Entropy ({ce_loss.avg:.4f})\t'
                         'Acc ({acc.avg:.4f})\t'
                         'Kappa ({kappa:.4f})'.format(epoch + 1, i, size_train, total_loss=total_loss,
                                                          dice_loss=dice_loss, ce_loss=ce_loss, acc=acc, kappa=kappa))
        sys.stdout.flush()
        global_step = (epoch * size_train) + i
        writer.add_scalar('Training Total Loss', total_loss.avg, global_step)
        writer.add_scalar('Training Dice Loss', dice_loss.avg, global_step)
        writer.add_scalar('Training Cross Entropy Loss', ce_loss.avg, global_step)
        writer.add_scalar('Training Accuracy', acc.avg, global_step)
        writer.add_scalar('Training kappa Score', kappa, global_step)


def val(model, val_dataloader, criterion, criterion1, epoch, writer, size_val, threshold):
    total_loss = AverageMeter()
    dice_loss = AverageMeter()
    ce_loss = AverageMeter()
    acc = AverageMeter()
    cm = AverageCM()
    
    with torch.no_grad():
        for i, sample in enumerate(val_dataloader):
            input, target = sample['image'].cuda(), sample['mask'].float().cuda()
            output = model(input)
            
            loss1 = criterion(output, target)
            target = torch.argmax(target, dim=1)
            loss2 = criterion1(output, target.long())
            output = torch.argmax(output, dim=1)
            loss =  loss1 + loss2
            
            total_loss.update(loss.item(), input.size(0))
            dice_loss.update(loss1.item(), input.size(0))
            ce_loss.update(loss2.item(), input.size(0))
            
            flat_target = target.view(-1)
            flat_output = output.view(-1)
            
            acc.update(accuracy(flat_target.data, flat_output.data), input.size(0))
            cm.update(conf_matrix((flat_target.data).cpu().numpy(), (flat_output.data).cpu().numpy()), input.size(0))
            
            kappa = 2*(acc.avg-0.5)
            
            sys.stdout.write('\r')
            sys.stdout.write('Validation Epoch: [{0}][{1}/{2}]\t'
                             'Total Loss  ({total_loss.avg:.4f})\t'
                             'Dice Loss ({dice_loss.avg:.4f})\t'
                             'Cross Entropy  ({ce_loss.avg:.4f})\t'
                             'Acc ({acc.avg:.4f})\t'
                             'Kappa ({kappa:.4f})'.format(epoch + 1, i, size_val, total_loss=total_loss,
                                                              dice_loss=dice_loss, ce_loss=ce_loss, acc=acc, kappa=kappa))
            sys.stdout.flush()
            global_step = (epoch * size_val) + i
            writer.add_scalar('Validation Total Loss', total_loss.avg, global_step)
            writer.add_scalar('Validation Dice Loss', dice_loss.avg, global_step)
            writer.add_scalar('Validation Cross Entropy Loss', ce_loss.avg, global_step)
            writer.add_scalar('Validation Accuracy', acc.avg, global_step)
            writer.add_scalar('Validation kappa Score', 2*(acc.avg-0.5), global_step)
            if (i+1) % 5 == 0:
                add_image(target, writer, global_step, ground_truth=True)
                add_image(output, writer, global_step, ground_truth=False)
    fig, ax = plot_confusion_matrix(cm.cm)
    writer.add_figure('Validation Confusion Matrix', fig, epoch)
    fig, ax = plot_confusion_matrix_values(cm.cm)
    writer.add_figure('Validation Confusion Matrix Values', fig, epoch)

    return total_loss.avg


if __name__ == '__main__':
    main()
