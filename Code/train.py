import argparse
import os
import warnings
import time
import sys

import torch
from torch.utils.data import DataLoader
from utils import TestDataset, TrainDataset, AverageMeter, AverageCM, add_image, save_checkpoint, accuracy, \
    conf_matrix, plot_confusion_matrix, kappa_score, plot_confusion_matrix_nm
from torchvision import transforms
from transform import ToTensor
from Models import Res
from loss import DiceLoss, FocalLoss
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--id', default='none', type=str)
parser.add_argument('--epoch', default=30, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=0.001, type=int)
parser.add_argument('--tag', default=None, type=str)
parser.add_argument('--gpu', default=True, type=bool)


def main():
    args = parser.parse_args()
    save_path = 'Trainid_' + args.id
    writer = SummaryWriter(log_dir='runs/' + args.tag + str(time.time()))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + '/Checkpoint')

    train_dataset_path = 'data/train'
    val_dataset_path = 'data/valid'
    train_transform = transforms.Compose([
        ToTensor()
    ])
    val_transform = transforms.Compose([
        ToTensor()
    ])
    train_dataset = TrainDataset(path=train_dataset_path, transform=train_transform)
    val_dataset = TrainDataset(path=val_dataset_path, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, num_workers=4)

    size_train = len(train_dataloader)
    size_val = len(val_dataloader)
    print('Number of Training Images: {}'.format(size_train))
    print('Number of Validation Images: {}'.format(size_val))
    start_epoch = 0
    model = Res(n_ch=4, n_classes=9)
    class_weights = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1,0]).cuda()
    criterion = DiceLoss()
    criterion1 = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion1 = criterion1.cuda()

    if args.resume is not None:
        weight_path = sorted(os.listdir(save_path + '/Checkpoint/'), key=lambda x: float(x[:-8]))[0]
        checkpoint = torch.load(save_path + '/Checkpoint/' + weight_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded Checkpoint of Epoch: {}'.format(args.resume))

    for epoch in range(start_epoch, int(args.epoch) + start_epoch):
        adjust_learning_rate(optimizer, epoch)
        train(model, train_dataloader, criterion, criterion1, optimizer, epoch, writer, size_train)
        print('')
        val_loss = val(model, val_dataloader, criterion, criterion1, epoch, writer, size_val)
        print('')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=save_path + '/Checkpoint/' + str(val_loss) + '.pth.tar')
    writer.export_scalars_to_json(save_path + '/log.json')


def train(model, train_dataloader, criterion, criterion1, optimizer, epoch, writer, size_train):
    model = model.train()
    total_loss = AverageMeter()
    dice_loss = AverageMeter()
    ce_loss = AverageMeter()
    acc = AverageMeter()
    for i, sample in enumerate(train_dataloader):
        input, target = sample['image'].cuda(), sample['mask'].long().cuda()
        output = model(input)
        loss1 = criterion(output, target)
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
        sys.stdout.write('\r')
        sys.stdout.write('Training Epoch: [{0}][{1}/{2}]\t'
                         'Total Loss  ({total_loss.avg:.4f})\t'
                         'Dice Loss ({dice_loss.avg:.4f})\t'
                         'Cross Entropy Loss ({ce_loss.avg:.4f})\t'
                         'Accuracy ({acc.avg:.4f})\t'.format(epoch + 1, i, size_train, total_loss=total_loss,
                                                             dice_loss=dice_loss, ce_loss=ce_loss, acc=acc))
        sys.stdout.flush()
        global_step = (epoch * size_train) + i
        writer.add_scalar('Training Total Loss', total_loss.avg, global_step)
        writer.add_scalar('Training Dice Loss', dice_loss.avg, global_step)
        writer.add_scalar('Training Cross Entropy Loss', ce_loss.avg, global_step)
        writer.add_scalar('Training Accuracy', acc.avg, global_step)


def val(model, val_dataloader, criterion, criterion1, epoch, writer, size_val):
    model = model.eval()
    total_loss = AverageMeter()
    dice_loss = AverageMeter()
    ce_loss = AverageMeter()
    acc = AverageMeter()
    cm = AverageCM()
    with torch.no_grad():
        for i, sample in enumerate(val_dataloader):
            input, target = sample['image'].cuda(), sample['mask'].long().cuda()
            output = model(input)
            loss1 = criterion(output, target)
            target = torch.argmax(target, 1)
            loss2 = criterion1(output, target)
            loss = loss2 + loss1
            total_loss.update(loss.item(), input.size(0))
            dice_loss.update(loss1.item(), input.size(0))
            ce_loss.update(loss2.item(), input.size(0))
            output = torch.argmax(output, 1)
            flat_target = target.view(-1)
            flat_output = output.view(-1)
            acc.update(accuracy(flat_target, flat_output), input.size(0))
            cm.update(conf_matrix(flat_target, flat_output))
            kappa = kappa_score(flat_target, flat_output)
            sys.stdout.write('\r')
            sys.stdout.write('Validation Epoch: [{0}][{1}/{2}]\t'
                             'Total Loss  ({total_loss.avg:.4f})\t'
                             'Dice Loss ({dice_loss.avg:.4f})\t'
                             'Cross Entropy Loss ({ce_loss.avg:.4f})\t'
                             'Accuracy ({acc.avg:.4f})\t'
                             'Kappa score({kappa:.4f})'.format(epoch + 1, i, size_val, total_loss=total_loss,
                                                               dice_loss=dice_loss, ce_loss=ce_loss, acc=acc,
                                                               kappa=kappa))
            sys.stdout.flush()
            global_step = (epoch * size_val) + i
            writer.add_scalar('Validation Total Loss', total_loss.avg, global_step)
            writer.add_scalar('Validation Dice Loss', dice_loss.avg, global_step)
            writer.add_scalar('Validation Cross Entropy Loss', ce_loss.avg, global_step)
            writer.add_scalar('Validation Accuracy', acc.avg, global_step)
            writer.add_scalar('Validation kappa Score', kappa, global_step)
            if (i + 1) % 14 == 0:
                add_image(target, writer, global_step, name='Validation Gt')
                add_image(output, writer, global_step, name='Validation Output')
    fig, ax = plot_confusion_matrix(cm.cm)
    writer.add_figure('Validation Confusion Matrix', fig, epoch)
    fig, ax = plot_confusion_matrix_nm(cm.cm)
    writer.add_figure('Validation Confusion Matrix Normalized', fig, epoch)
    return total_loss.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 15))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

