import torch
import torch.nn as nn
import torch.nn.functional as F


class Res(nn.Module):
    """docstring for Unet"""

    def __init__(self, n_ch, n_classes):
        super(Res, self).__init__()
        self.inpt = InConv(n_ch, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.up1 = UpConv(512, 256)
        self.up2 = UpConv(256, 128)
        self.up3 = UpConv(128, 64)
        self.out = OutConv(64, n_classes)

    def forward(self, X):
        x1 = self.inpt(X)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.out(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride),
            nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


'''
class Dconv(nn.Module):
    """Basic block of Conv2d, BatchNorm2d, and Relu layers conneted togather twice"""

    def __init__(self, In_ch, Out_ch, K_size=3, stride=1, padding=1):
        super(Dconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(In_ch, Out_ch, K_size, padding=1),
            nn.BatchNorm2d(Out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(Out_ch, Out_ch, K_size, padding=1),
            nn.BatchNorm2d(Out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.conv(X)

'''


class Dconv(nn.Module):
    """Basic block of Conv2d, BatchNorm2d, and Relu layers conneted togather twice"""

    def __init__(self, In_ch, Out_ch, K_size=3):
        super(Dconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(In_ch, Out_ch, K_size, padding=1),
            nn.BatchNorm2d(Out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(Out_ch, Out_ch, K_size, padding=1),
            nn.BatchNorm2d(Out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.conv(X)


class InConv(nn.Module):
    """Convolution layer for the input to Unet"""

    def __init__(self, In_ch, Out_ch):
        super(InConv, self).__init__()
        self.conv = Dconv(In_ch, Out_ch)

    def forward(self, X):
        return self.conv(X)


class DownConv(nn.Module):
    """Block of layers stacked up togather for Down Convolution"""

    def __init__(self, In_ch, Out_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(In_ch, Out_ch)
        )

    def forward(self, X):
        return self.conv(X)


class UpConv(nn.Module):
    """Block of layers stacked up togather for Up Convolution"""

    def __init__(self, In_ch, Out_ch, learnable=True):
        super(UpConv, self).__init__()

        # learnable -> parameter to specify if to learn Upsampling or Use extrapolation
        if learnable == False:
            self.up = nn.Sequential(
                nn.Conv2d(In_ch, In_ch // 2, kernel_size=2, padding=2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
        else:
            # In_ch = In_ch//2 since before convloution input has In_ch//2 number of channels
            # Out_ch = In_ch//2 since upsampling or transpose convolution doesnot alter count of channels
            self.up = nn.ConvTranspose2d(In_ch, In_ch // 2, kernel_size=2, stride=2)

        self.conv = Dconv(In_ch, Out_ch)

    def forward(self, X1, X2):
        # X1 input from below X2 input from left
        X1 = self.up(X1)

        # spatial size of X1 < spatial size of X2
        diffX, diffY = (X2.size()[2] - X1.size()[2], X2.size()[3] - X1.size()[3])
        X1 = F.pad(X1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        X = torch.cat([X2, X1], dim=1)
        return self.conv(X)


class OutConv(nn.Module):
    """Final Output layer with kernel size = 1"""

    def __init__(self, In_ch, Out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(In_ch, Out_ch, 1)

    def forward(self, X):
        return self.conv(X)


