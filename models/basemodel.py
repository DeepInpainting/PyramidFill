'''
Copyright 2020 OPPO LLC
This work is licensed under a Creative Commons 
Attribution-NonCommercial 4.0 International License.
The software is for educational and academic research purpose only.
'''
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()

class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()
    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

class GatedConv2dWithActivation(nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:phi(f(I))*sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.activation = activation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.pixNorm = PixelwiseNorm() 
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.pixNorm(self.conv2d(input))
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            out = self.activation(x) * self.gated(mask)
        else:
            out = x * self.gated(mask)
        return out

class SNConvWithActivation(nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class DenseBlock(nn.Module):
    def __init__(self,in_planes, scale=1):
        super(DenseBlock,self).__init__()
        self.scale=scale
        self.branch1=GatedConv2dWithActivation(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.branch2=GatedConv2dWithActivation(2*in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.branch3=GatedConv2dWithActivation(3*in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.branch4=GatedConv2dWithActivation(4*in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        #self.conv=nn.Conv2d(5*in_planes, in_planes, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        x1=self.branch1(x)
        x2=self.branch2(torch.cat((x,x1),dim=1))
        x3=self.branch3(torch.cat((x,x1,x2),dim=1))
        x4=self.branch4(torch.cat((x,x1,x2,x3),dim=1))
        #out=self.conv(torch.cat((x,x1,x2,x3,x4),dim=1))
        return x4*self.scale+x

class RRDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=1):
        super(RRDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseBlock(filters), DenseBlock(filters), DenseBlock(filters)
        )
    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 4*in_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.pixNorm = PixelwiseNorm()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(self.pixNorm(x))
        return x

class Generator1(nn.Module):
    def __init__(self, n_in_channel=4):
        super(Generator1, self).__init__()
        cnum = 32
        self.block1=nn.Sequential(
            GatedConv2dWithActivation(n_in_channel, 2*cnum, 5, 1, padding=2),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(4*cnum, 8*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(8*cnum, 16*cnum, 3,1, padding=1)
            )
        self.block2=nn.Sequential(
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, dilation=2, padding=2),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, dilation=4, padding=4),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, dilation=8, padding=8),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, dilation=12, padding=12)
            )
        self.block3=nn.Sequential(
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, padding=1)
            )
        self.block4=nn.Sequential(
            GatedConv2dWithActivation(16*cnum, 8*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(8*cnum, 4*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(4*cnum, 2*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(2*cnum, 3, kernel_size=5, stride=1, padding=2, activation=nn.Tanh())
            )
    def forward(self, imgs, masks):
        masked_imgs=imgs*(1-masks) + masks
        input=torch.cat([masked_imgs, masks],dim=1)
        x1 = self.block1(input)
        x1_1, x1_2=torch.chunk(x1, 2, dim=1)
        x2 = self.block2(x1_1)
        x3 = self.block3(x1_2)
        x4 = self.block4(torch.cat([x2, x3],dim=1))    
        return x4

class Generator(nn.Module):
    def __init__(self, n_in_channel=3, cnum = 32, num_res_blocks=2):
        super(Generator, self).__init__()
        self.conv1 = GatedConv2dWithActivation(n_in_channel, 2*cnum, 9, 1, padding=4)
        self.res_blocks1 = nn.Sequential(*[RRDenseBlock(2*cnum) for _ in range(num_res_blocks)])
        self.conv2 = GatedConv2dWithActivation(2*cnum,2*cnum, 3, 1, 1)
        self.upsample=UpsampleBlock(2*cnum, 2)
        self.conv3=GatedConv2dWithActivation(2*cnum,2*cnum,3,1,1)
        self.toRGB1=GatedConv2dWithActivation(2*cnum, 3, kernel_size=9, stride=1, padding=4, activation=nn.Tanh())
        self.conv4= GatedConv2dWithActivation(n_in_channel, 2*cnum, 9, 1, padding=4)
        self.block2=nn.Sequential(            
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=1),
            GatedConv2dWithActivation(4*cnum, 2*cnum, 3, 1, padding=1)
            )
        self.toRGB2= GatedConv2dWithActivation(2*cnum, 3, kernel_size=9, stride=1, padding=4, activation=nn.Tanh())
    def forward(self, imgs1, imgs2, masks):
        x0 = self.conv1(imgs1)
        x1 = self.res_blocks1(x0)
        x2 = self.conv2(x1)
        x3 = x0 + x2
        x4 = self.upsample(x3)
        x5 = self.conv3(x4)
        output = self.toRGB1(x5)
        new = imgs2*(1-masks) + output*masks
        new0 = self.conv4(new)
        new1 = self.block2(new0)
        new3 = new0+new1
        new4 = self.toRGB2(new3)
        return output, new4

class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(3, 2*cnum, 3, 1, padding=1),
            SNConvWithActivation(2*cnum, 4*cnum, 3, 1, padding=1),
            SNConvWithActivation(4*cnum, 4*cnum, 3, 1, padding=1),
            SNConvWithActivation(4*cnum, 8*cnum, 3, 1, padding=1),
            SNConvWithActivation(8*cnum, 8*cnum, 3, 1, padding=1),
            SNConvWithActivation(8*cnum, 4*cnum, 3, 1, padding=1),
            SNConvWithActivation(4*cnum, 2*cnum, 3, 1, padding=1),
            SNConvWithActivation(2*cnum, 1, 3, 1, padding=1, activation=None)
        )

    def forward(self, imgs):
        x = self.discriminator_net(imgs)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(3, 2*cnum, 3, 1, padding=1),
            SNConvWithActivation(2*cnum, 4*cnum, 3, 2, padding=1),
            SNConvWithActivation(4*cnum, 4*cnum, 3, 1, padding=1),
            SNConvWithActivation(4*cnum, 8*cnum, 3, 2, padding=1),
            SNConvWithActivation(8*cnum, 8*cnum, 3, 1, padding=1),
            SNConvWithActivation(8*cnum, 8*cnum, 3, 2, padding=1),
            SNConvWithActivation(8*cnum, 8*cnum, 3, 1, padding=1),
            SNConvWithActivation(8*cnum, 8*cnum, 3, 2, padding=1),
            SNConvWithActivation(8*cnum, 1, 3, 1, padding=1, activation=None)
        )

    def forward(self, imgs):
        x = self.discriminator_net(imgs)
        return x


