'''
Copyright 2020 OPPO LLC
This work is licensed under a Creative Commons 
Attribution-NonCommercial 4.0 International License.
The software is for educational and academic research purpose only.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.basemodel import Generator1, Discriminator1, Generator, Discriminator, weights_init
from models.loss import SNDisLoss, SNGenLoss, ReconLoss, TwoReconLoss, PerceptualLoss, StyleLoss, TVLoss, SSIM_Loss, MSSSIM_Loss
from models.vgg import vgg16_bn
from data.inpaint_dataset import InpaintDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import time
import sys
import argparse
import os
from collections import OrderedDict

parser = argparse.ArgumentParser()
# General parameters
parser.add_argument('--multigpu', type = bool, default = True, help = 'nn.Parallel needs or not')
parser.add_argument('--save_folder', type = str, default = 'weights/', help = 'storage of model checkpoints')
# Training parameters
parser.add_argument('--epoches', type = int, default = 30, help = 'number of epochs of training')
parser.add_argument('--batch_size', type = int, default = 8, help = 'size of the batches')
parser.add_argument('--lr_g', type = float, default = 2e-4, help = 'Adam: learning rate for generator')
parser.add_argument('--lr_d', type = float, default = 8e-4, help = 'Adam: learning rate for discriminator')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=0, help="adam: weight decay")
parser.add_argument('--lr_decrease_epoch', type = int, default = 15, help = 'lr decrease at certain epoch and its multiple')
parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
parser.add_argument('--num_workers', type = int, default = 4, help = 'number of cpu threads to use during batch generation')
parser.add_argument('--resume', type = str, default = None, help = 'Checkpoint state_dict file to resume training')
# Dataset parameters
parser.add_argument('--flist', type = str, default = './data/celeba-hq/train1024.txt', help = 'the training folder')
parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
parser.add_argument('--imgsize', type = int, default = 1024, help = 'size of image')
parser.add_argument('--margin', type = int, default = 64, help = 'margin of image')
parser.add_argument('--bbox_shape', type = int, default = 256, help = 'shape of bbox mask')
parser.add_argument('--min_width', type = int, default = 60, help = 'parameter of length for free form mask')
parser.add_argument('--max_width', type = int, default = 160, help = 'parameter of width for free form mask')
opt = parser.parse_args()

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

# Dataset setting
print("Initialize the dataset...")
train_dataset = InpaintDataset(opt)
train_loader = DataLoader(train_dataset,batch_size=opt.batch_size, shuffle=True,
                                        num_workers=opt.num_workers, pin_memory=True)
print("Finish the dataset initialization.")
# Define the Network Structure
print("Define the Network Structure and Losses")
netG = Generator()
netD = Discriminator()
Gen1=Generator1()
Gen2=Generator()
Gen3=Generator()
Gen4=Generator()
#print(netG)
model1_path='weights/G1_V25_random_celeba.pth.tar'
nets1_weights=torch.load(model1_path)
netG1_state_dict=nets1_weights['netG_state_dict']
Gen1.load_state_dict(netG1_state_dict)
Gen1.cuda()
Gen1.eval()

model2_path='weights/G2_V25_random_celeba.pth.tar'
nets2_weights=torch.load(model2_path)
netG2_state_dict=nets2_weights['netG_state_dict']
Gen2.load_state_dict(netG2_state_dict)
Gen2.cuda()
Gen2.eval()

model3_path='weights/G3_V25_random_celeba.pth.tar'
nets3_weights=torch.load(model3_path)
netG3_state_dict=nets3_weights['netG_state_dict']
Gen3.load_state_dict(netG3_state_dict)
Gen3.cuda()
Gen3.eval()


model4_path='weights/G4_V25_random_celeba.pth.tar'
nets4_weights=torch.load(model4_path)
netG4_state_dict=nets4_weights['netG_state_dict']
Gen4.load_state_dict(netG4_state_dict)
Gen4.cuda()
Gen4.eval()

start_epoch = 0
if opt.resume:
    nets = torch.load(opt.resume)
    netG_state_dict, netD_state_dict = nets['netG_state_dict'], nets['netD_state_dict']
    start_epoch = nets['epoch']
    netG.load_state_dict(netG_state_dict)
    netD.load_state_dict(netD_state_dict)
    print("Loading pretrained models from {} ...".format(opt.resume))
else:
    netG.apply(weights_init)
    netD.apply(weights_init)

if opt.multigpu:
    netG=nn.DataParallel(netG)
    netD=nn.DataParallel(netD)
    Gen1=nn.DataParallel(Gen1)
    Gen2=nn.DataParallel(Gen2)
    Gen3=nn.DataParallel(Gen3)
    Gen4=nn.DataParallel(Gen4)
netG=netG.cuda()
netD=netD.cuda()
cudnn.benckmark = True

if not opt.resume:
    print('Initialize weights...')

feature_extractor=vgg16_bn()
feature_extractor.eval()
feature_extractor.cuda()
if opt.multigpu:
    feature_extractor=nn.DataParallel(feature_extractor)
recon_loss = TwoReconLoss(0.1, 0.05, 0.1, 0.05)
#recon_loss = ReconLoss(1, 1)
#ssim_loss = SSIM_Loss(1)
gan_loss = SNGenLoss(0.001)
dis_loss = SNDisLoss()
perceptual_loss=PerceptualLoss(weight=0.1, layers=[0,5,12,22], feat_extractors=feature_extractor)
style_loss=StyleLoss(weight=250, layers=[0,5,12,22], feat_extractors=feature_extractor)
tv_loss=TVLoss(weight=0.1)
optG = torch.optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2),weight_decay = opt.weight_decay)
optD = torch.optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2),weight_decay = opt.weight_decay)    

def img2photo(imgs):
    return ((imgs+1)*127.5).transpose(0,1).transpose(1,2).detach().cpu().numpy()


def train():
    netG.train()
    netD.train()
    for epoch in range(start_epoch, opt.epoches):
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs=imgs.cuda()
            masks=masks.cuda()
            imgs4=F.interpolate(imgs, scale_factor=1/2, mode='area')
            masks4=F.interpolate(masks, scale_factor=1/2, mode='area')
            imgs3=F.interpolate(imgs4, scale_factor=1/2, mode='area')
            masks3=F.interpolate(masks4, scale_factor=1/2, mode='area')
            imgs2=F.interpolate(imgs3, scale_factor=1/2, mode='area')
            masks2=F.interpolate(masks3, scale_factor=1/2, mode='area')
            imgs1=F.interpolate(imgs2, scale_factor=1/2, mode='area')
            masks1=F.interpolate(masks2, scale_factor=1/2, mode='area')
            masked_img =  imgs[0] * (1 - masks[0]) + masks[0]
            input_img = img2photo(masked_img)
            input_img = Image.fromarray(input_img.astype(np.uint8))
            input_img.save(os.path.join('celeba1/','corrupted.jpg'))
            real_img = img2photo(imgs[0])
            real_img = Image.fromarray(real_img.astype(np.uint8))
            real_img.save(os.path.join('celeba1/','real.jpg'))
            # Optimize Discriminator
            optD.zero_grad()
            with torch.no_grad():
                output1=Gen1(imgs1, masks1)
            #recon_imgs=netG(imgs1, masks1)
            output1=imgs1*(1-masks1)+output1*masks1
            with torch.no_grad():
                _,output2=Gen2(output1,imgs2, masks2)
            output2=imgs2*(1-masks2)+output2*masks2
            with torch.no_grad():
                _,output3=Gen3(output2, imgs3, masks3)
            output3=imgs3*(1-masks3)+output3*masks3
            with torch.no_grad():
                _,output4=Gen4(output3, imgs4, masks4)
            output4=imgs4*(1-masks4)+output4*masks4
            coarse, recon_imgs = netG(output4, imgs, masks)
            
            input_img1=img2photo(output1[0])
            input_img1 = Image.fromarray(input_img1.astype(np.uint8))
            input_img1.save(os.path.join('celeba1/','input1.jpg'))
        
            input_img2=img2photo(output2[0])
            input_img2 = Image.fromarray(input_img2.astype(np.uint8))
            input_img2.save(os.path.join('celeba1/','input2.jpg'))
            input_img3=img2photo(output3[0])
            input_img3 = Image.fromarray(input_img3.astype(np.uint8))
            input_img3.save(os.path.join('celeba1/','input3.jpg'))
            input_img4=img2photo(output4[0])
            input_img4 = Image.fromarray(input_img4.astype(np.uint8))
            input_img4.save(os.path.join('celeba1/','input4.jpg'))
        
            gen_img = img2photo(recon_imgs[0])
            gen_img = Image.fromarray(gen_img.astype(np.uint8))
            gen_img.save(os.path.join('celeba1/','gen.jpg'))
            complete_imgs0 = coarse * masks + imgs * (1 - masks)
            complete_imgs = recon_imgs * masks + imgs * (1 - masks)
            com_img = img2photo(complete_imgs[0])
            com_img = Image.fromarray(com_img.astype(np.uint8))
            com_img.save(os.path.join('celeba1/','com.jpg'))
            pos_imgs = imgs
            #mix_imgs = complete_imgs
            neg_imgs = recon_imgs
            pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

            pred_pos_neg = netD(pos_neg_imgs)
            pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
            #mix_pred=pred_pos*(1-masks1)+pred_neg*masks1
            d1_loss = dis_loss(pred_pos, pred_neg)
            #d2_loss = mix_pred
            #pos_mix_imgs = torch.cat([pos_imgs, mix_imgs], dim=0)
            #pred_pos_mix = netD(pos_mix_imgs)
            #pred_pos1, pred_mix = torch.chunk(pred_pos_mix, 2, dim=0)    
            #dmix_loss = pred_mix
            #L2_loss = torch.nn.MSELoss()
            #cons_loss = L2_loss(dmix_loss,d2_loss)
            #d_loss = d1_loss + cons_loss
            d_loss = d1_loss
            d_loss.backward(retain_graph=True)
            optD.step()

            # Optimize Generator
            optG.zero_grad()
            pred_neg = netD(neg_imgs)
            g_loss = gan_loss(pred_neg)
            r_loss = recon_loss(imgs, coarse, recon_imgs, masks)
            #r_loss = recon_loss(imgs2, recon_imgs, masks1)
            #str_loss = ssim_loss(imgs1, recon_imgs)
            com_per_loss = perceptual_loss(imgs, complete_imgs)
            com_sty_loss = style_loss(imgs, complete_imgs)
            com_tv_loss = tv_loss(complete_imgs)
            com0_per_loss = perceptual_loss(imgs, complete_imgs0)
            com0_sty_loss = style_loss(imgs, complete_imgs0)
            com0_tv_loss = tv_loss(complete_imgs0)
            whole_loss = g_loss + r_loss + com_per_loss + com_sty_loss + com_tv_loss + com0_per_loss + com0_sty_loss + com0_tv_loss
            #whole_loss = g_loss + r_loss + str_loss
            whole_loss.backward()
            optG.step()

            if (batch_idx+1)%20 ==0:
                print('\r[Epoch %d/%d] [Batch %d/%d] [D loss:%.5f] [G loss:%.5f] [ReconLoss:%.5f] [PerceptualLoss:%.5f] [StyleLoss:%.5f] [TVLoss:%.5f]'%((epoch+1),\
                    opt.epoches, batch_idx+1, len(train_loader),d_loss.item(),g_loss.item(), r_loss.item(), com_per_loss.item(), com_sty_loss.item(), com_tv_loss.item()))
                #print('\r[Epoch %d/%d] [Batch %d/%d] [D loss:%.5f] [G loss:%.5f] [ReconLoss:%.5f]'%((epoch+1), opt.epoches, batch_idx+1, len(train_loader),d_loss.item(),g_loss.item(), r_loss.item()))
                #print('\r[Epoch %d/%d] [Batch %d/%d] [D loss:%.5f] [G loss:%.5f] [ReconLoss:%.5f] [Str_Loss:%.5f]'%((epoch+1), \
                #    opt.epoches, batch_idx+1, len(train_loader),d_loss.item(),g_loss.item(), r_loss.item(), str_loss.item()))
        saved_model = {
                'epoch': epoch + 1,
                'netG_state_dict': netG.module.state_dict(),
                'netD_state_dict': netD.module.state_dict(),
        }
            #if (epoch+1)%10==0:
        torch.save(saved_model,'{}/G5_V25_random_celeba.pth.tar'.format(opt.save_folder))
        adjust_learning_rate(opt.lr_g, optG, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optD, (epoch + 1), opt)

def adjust_learning_rate(lr_in, optimizer, epoch, opt):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    train()
