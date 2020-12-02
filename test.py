'''
Copyright 2020 OPPO LLC
This work is licensed under a Creative Commons 
Attribution-NonCommercial 4.0 International License.
The software is for educational and academic research purpose only.
'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models.basemodel import Generator1, Generator
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import cv2
import math
import skimage
import time
import sys
import os

real_dir = './test512/real'
fake_dir = './test512/fake'
input_dir= './test512/input'
mask_dir = './test512/mask'
def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def random_form(shape, min_width=120, max_width = 200):
        H = shape
        W = shape
        mean_angle = 2*math.pi / 5
        angle_range = 2*math.pi / 15
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(2, 4)):
            num_vertex = np.random.randint(8, 12)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            vertex.append((int(np.random.randint(0+W/4, W-W/4)), int(np.random.randint(0+H/4, H-H/4))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, W)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, H)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        return mask.reshape(mask.shape+(1,))

def centerbox(shape):
        height=shape
        width=shape
        mask=np.zeros(( height, width), np.float32)
        mask[height//4:3*height//4, width//4:3*width//4]=1.
        return mask.reshape(mask.shape+(1,)).astype(np.float32)

def halfbox(shape):
        height=shape
        width=shape
        mask=np.zeros(( height, width), np.float32)
        mask[height//8:7*height//8, width//8:7*width//8]=1.
        return mask.reshape(mask.shape+(1,)).astype(np.float32)

def read_mask(mask_type, shape):
        """
        Read Masks now only support bbox
        """
        if mask_type == 'random_bbox':
            bboxs = []
            for i in range(5):
                bbox = random_bbox(settings)
                bboxs.append(bbox)
        elif mask_type == 'centerbox':
            mask=centerbox(shape)
            return mask
        elif mask_type == 'halfbox':
            mask=halfbox(shape)
            return mask
        elif mask_type == 'free_form':
            mask = random_form(shape)
            return mask
        mask = bbox2mask(bboxs, settings)
        return mask

def img2photo(imgs):
    return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()

def tranform(img):
    Trans = transforms.Compose([
            #transforms.Resize(1024, Image.BICUBIC),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return Trans(img)

def validate(Gen1, Gen2, Gen3, Gen4, path, mask_type):
    """
    validate phase
    """
    
    img = Image.open(path).convert("RGB")
    w=img.size[0]
    h=img.size[1]
    img = tranform(img)
    img=img.unsqueeze(0)
    mask=read_mask(mask_type, 1024)
    mask=torch.from_numpy(mask.transpose((2,0,1))).unsqueeze(0).float()
    mask=mask.cuda()
    img=img.cuda()
    img4=F.interpolate(img, scale_factor=1/2, mode='area')
    mask4=F.interpolate(mask, scale_factor=1/2, mode='area')
    img3=F.interpolate(img4, scale_factor=1/2, mode='area')
    mask3=F.interpolate(mask4, scale_factor=1/2, mode='area')
    img2=F.interpolate(img3, scale_factor=1/2, mode='area')
    mask2=F.interpolate(mask3, scale_factor=1/2, mode='area')
    img1=F.interpolate(img2, scale_factor=1/2, mode='area')
    mask1=F.interpolate(mask2, scale_factor=1/2, mode='area')
    masked_img =  img4 * (1 - mask4) + mask4
    with torch.no_grad():
        output1=Gen1(img1,mask1)
    comp1=img1*(1-mask1)+output1*mask1
    with torch.no_grad():
        _,output2=Gen2(comp1, img2, mask2)
    comp2=img2*(1-mask2)+output2*mask2
    with torch.no_grad():
        _,output3=Gen3(comp2, img3, mask3)
    comp3=img3*(1-mask3)+output3*mask3
    with torch.no_grad():
        _,output4=Gen4(comp3, img4, mask4)
    comp4=img4*(1-mask4)+output4*mask4
    #with torch.no_grad():
    #    _,output5 = Gen5(comp4, img, mask)
    #comp5=img*(1-mask)+output5*mask
    #recon_img = output1
    #complete_img = recon_img * mask1 + img1 * (1 - mask1)
    #real1_img = img2photo(img1)
    #real2_img = img2photo(img2)
    #real3_img = img2photo(img)
    real4_img = img2photo(img4)
    #real5_img = img2photo(img)
    corrupted_img = img2photo(masked_img)
    mask_img = (mask4.expand(1,3,512,512)*255).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    #gen1_img = img2photo(output1)
    #gen2_img = img2photo(output2)
    #gen3_img = img2photo(output3)
    #gen4_img = img2photo(output4)
    #gen5_img = img2photo(recon_img)
    #comp1_img = img2photo(comp1)
    #comp2_img = img2photo(comp2)
    #comp3_img = img2photo(comp3)
    comp4_img = img2photo(comp4)
    #comp5_img = img2photo(comp5)
    #real1_img = Image.fromarray(real1_img[0].astype(np.uint8))
    #real2_img = Image.fromarray(real2_img[0].astype(np.uint8))
    #real3_img = Image.fromarray(real3_img[0].astype(np.uint8))
    real4_img = Image.fromarray(real4_img[0].astype(np.uint8))
    #real5_img = Image.fromarray(real5_img[0].astype(np.uint8))
    corrupted_img = Image.fromarray(corrupted_img[0].astype(np.uint8))
    mask_img = Image.fromarray(mask_img[0].astype(np.uint8))
    #gen1_img = Image.fromarray(gen1_img[0].astype(np.uint8))
    #gen2_img = Image.fromarray(gen2_img[0].astype(np.uint8))
    #gen3_img = Image.fromarray(gen3_img[0].astype(np.uint8))
    #gen4_img = Image.fromarray(gen4_img[0].astype(np.uint8))
    #gen5_img = Image.fromarray(gen5_img[0].astype(np.uint8))
    #comp1_img = Image.fromarray(comp1_img[0].astype(np.uint8))
    #comp2_img = Image.fromarray(comp2_img[0].astype(np.uint8))
    #comp3_img = Image.fromarray(comp3_img[0].astype(np.uint8))
    comp4_img = Image.fromarray(comp4_img[0].astype(np.uint8))
    #comp5_img = Image.fromarray(comp5_img[0].astype(np.uint8))
    #real1_img.save(os.path.join(real_dir, os.path.basename(path)))
    #real2_img.save(os.path.join(real_dir, os.path.basename(path)))
    #real3_img.save(os.path.join(real_dir, os.path.basename(path)))
    real4_img.save(os.path.join(real_dir, os.path.basename(path)))
    #real5_img.save(os.path.join(real_dir, os.path.basename(path)))
    corrupted_img.save(os.path.join(input_dir, os.path.basename(path)))
    mask_img.save(os.path.join(mask_dir, os.path.basename(path)))
    #gen1_img.save(os.path.join(result_dir,'gen1_' + os.path.basename(path)))
    #gen2_img.save(os.path.join(result_dir,'gen2_' + os.path.basename(path)))
    #gen3_img.save(os.path.join(result_dir,'gen3_' + os.path.basename(path)))
    #gen4_img.save(os.path.join(result_dir,'gen4_' + os.path.basename(path)))
    #gen5_img.save(os.path.join(result_dir,'gen5_' + os.path.basename(path)))
    #comp1_img.save(os.path.join(fake_dir, os.path.basename(path)))
    #comp2_img.save(os.path.join(fake_dir, os.path.basename(path)))
    #comp3_img.save(os.path.join(fake_dir,os.path.basename(path)))
    comp4_img.save(os.path.join(fake_dir, os.path.basename(path)))
    #comp5_img.save(os.path.join(fake_dir, os.path.basename(path)))
    
    m1=cv2.imread(os.path.join(real_dir, os.path.basename(path)))
    m2=cv2.imread(os.path.join(fake_dir, os.path.basename(path)))
    #m1=Image.open(os.path.join(real_dir, os.path.basename(path))).convert("RGB").dtype('float32')
    #m2=Image.open(os.path.join(fake_dir, os.path.basename(path))).convert("RGB").dtype('float32')
    l1 = compare_mae(m1/255.0, m2/255.0)
    psnr = skimage.measure.compare_psnr(m1, m2, 255)
    ssim = skimage.measure.compare_ssim(m1, m2, data_range=255,multichannel=True)
    print('comparisons of two images:')
    print(l1)
    print(psnr)
    print(ssim)
    return l1, psnr, ssim
    #comp_img.save(os.path.join(result_dir,'comp_' + os.path.basename(path)))

def main():
    # Define the Network Structure
    print("Loading pretrained models")
    model1_path = 'weights/G1_V25_random_celeba.pth.tar'
    model2_path = 'weights/G2_V25_random_celeba.pth.tar'
    model3_path = 'weights/G3_V25_random_celeba.pth.tar'
    model4_path = 'weights/G4_V25_random_celeba.pth.tar'
    #model5_path = 'weights/G5_V25PerStyle250_centermask_celeba.pth.tar'
    Gen1=Generator1()
    Gen2=Generator()
    Gen3=Generator()
    Gen4=Generator()
    #Gen5=Generator()
    net1 = torch.load(model1_path)
    netG1_state_dict = net1['netG_state_dict']
    Gen1.load_state_dict(netG1_state_dict)
    Gen1.cuda()
    Gen1.eval()
    
    net2 = torch.load(model2_path)
    netG2_state_dict = net2['netG_state_dict']
    Gen2.load_state_dict(netG2_state_dict)
    Gen2.cuda()
    Gen2.eval()  
      
    net3 = torch.load(model3_path)
    netG3_state_dict = net3['netG_state_dict']
    Gen3.load_state_dict(netG3_state_dict)
    Gen3.cuda()
    Gen3.eval()
    
    net4 = torch.load(model4_path)
    netG4_state_dict = net4['netG_state_dict']
    Gen4.load_state_dict(netG4_state_dict)
    Gen4.cuda()
    Gen4.eval()
    
    #net5 = torch.load(model5_path)
    #netG5_state_dict = net5['netG_state_dict']
    #Gen5.load_state_dict(netG5_state_dict)
    #Gen5.cuda()
    #Gen5.eval()
    
    #img_path='./images'
    #img_list=[os.path.join(img_path, x) for x in os.listdir(img_path)]
    img_flist_path='./data/celeba-hq/val1024.txt'
    with open(img_flist_path, 'r') as f:
            img_list = f.read().splitlines()
    L1=[]
    psnr=[]
    ssim=[]
    for path in img_list:
        l1, m1, m2 = validate(Gen1, Gen2, Gen3, Gen4, path, 'free_form')
        L1.append(l1)
        psnr.append(m1)
        ssim.append(m2)
    LL1=np.mean(L1)
    PSNR=np.mean(psnr)
    SSIM=np.mean(ssim)
    print(LL1, PSNR, SSIM)

if __name__ == '__main__':
    main()
